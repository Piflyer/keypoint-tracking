from typing import Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops import boxes as box_ops, roi_align

from torchvision.models.detection import _utils as det_utils
from .modelutils import BalancedPositiveNegativeSampler


def fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x: torch.Tensor, labels: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for each image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]

def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid



def _onnx_heatmaps_to_keypoints(
    maps, maps_i, roi_map_width, roi_map_height, widths_i, heights_i, offset_x_i, offset_y_i
):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)

    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height

    roi_map = F.interpolate(
        maps_i[:, None], size=(int(roi_map_height), int(roi_map_width)), mode="bicubic", align_corners=False
    )[:, 0]

    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

    x_int = pos % w
    y_int = (pos - x_int) // w

    x = (torch.tensor(0.5, dtype=torch.float32) + x_int.to(dtype=torch.float32)) * width_correction.to(
        dtype=torch.float32
    )
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int.to(dtype=torch.float32)) * height_correction.to(
        dtype=torch.float32
    )

    xy_preds_i_0 = x + offset_x_i.to(dtype=torch.float32)
    xy_preds_i_1 = y + offset_y_i.to(dtype=torch.float32)
    xy_preds_i_2 = torch.ones(xy_preds_i_1.shape, dtype=torch.float32)
    xy_preds_i = torch.stack(
        [
            xy_preds_i_0.to(dtype=torch.float32),
            xy_preds_i_1.to(dtype=torch.float32),
            xy_preds_i_2.to(dtype=torch.float32),
        ],
        0,
    )

    # TODO: simplify when indexing without rank will be supported by ONNX
    base = num_keypoints * num_keypoints + num_keypoints + 1
    ind = torch.arange(num_keypoints)
    ind = ind.to(dtype=torch.int64) * base
    end_scores_i = (
        roi_map.index_select(1, y_int.to(dtype=torch.int64))
        .index_select(2, x_int.to(dtype=torch.int64))
        .view(-1)
        .index_select(0, ind.to(dtype=torch.int64))
    )

    return xy_preds_i, end_scores_i


@torch.jit._script_if_tracing
def _onnx_heatmaps_to_keypoints_loop(
    maps, rois, widths_ceil, heights_ceil, widths, heights, offset_x, offset_y, num_keypoints
):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)

    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(
            maps, maps[i], widths_ceil[i], heights_ceil[i], widths[i], heights[i], offset_x[i], offset_y[i]
        )
        xy_preds = torch.cat((xy_preds.to(dtype=torch.float32), xy_preds_i.unsqueeze(0).to(dtype=torch.float32)), 0)
        end_scores = torch.cat(
            (end_scores.to(dtype=torch.float32), end_scores_i.to(dtype=torch.float32).unsqueeze(0)), 0
        )
    return xy_preds, end_scores


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(
            maps,
            rois,
            widths_ceil,
            heights_ceil,
            widths,
            heights,
            offset_x,
            offset_y,
            torch.scalar_tensor(num_keypoints, dtype=torch.int64),
        )
        return xy_preds.permute(0, 2, 1), end_scores

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    # end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    xy_probs = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    xy_radius = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode="bicubic", align_corners=False
        )[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode="floor")
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        # end_scores[i, :] = roi_map[torch.arange(num_keypoints, device=roi_map.device), y_int, x_int]
        # do a 2d softmax on each roi_map so that each channel adds up to 1, input is (C, H, W)
        c, h, w = roi_map.shape
        prob_map = F.softmax(roi_map.view(c, -1), dim=1).view(c, h, w)
        xy_probs[i, :] = prob_map[torch.arange(num_keypoints, device=x.device), y_int, x_int]
        # bubble out from the each keypoint location to get the radius when the sum of the probabilities is 0.9
        for k in range(num_keypoints):
            center = (x_int[k].item(), y_int[k].item())
            radius = 0
            prob_sum = xy_probs[i, k]
            #visibility network
            xy_preds[i, 2, k] = xy_probs[i, k]
            # xy_preds[i, 2, k] = xy_probs[i, k] # for the sake of conformal prediction
            
            while radius < roi_map.shape[1] // 2 and prob_sum <= 0.9:
                radius += 1
                prob_sum = prob_map[k, center[1] - radius:center[1] + radius + 1, center[0] - radius:center[0] + radius + 1].sum()
            xy_radius[i,k] = radius
    return xy_preds.permute(0, 2, 1), xy_probs, xy_radius

# def conformal_keypointrnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs, selected_labels):
#     N, K, H, W = keypoint_logits.shape
#     if H != W:
#         raise ValueError(
#             f"keypoint_logits height and width (last two elements of shape) should be equal. Instead got H = {H} and W = {W}"
#         )
#     pass

def keypoint_visibility_loss(keypoint_logits, gt_keypoints, keypoint_matched_idxs):
    N, K, _, _ = keypoint_logits.shape
    
    gt_vis_all = []
    for gt_kp_in_image, midx in zip(gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        gt_vis_all.append(kp[:, :, 2])
    
    gt_vis = torch.cat(gt_vis_all, dim=0)
    # Ground truth visibility 0 or 1 -> target 0. Ground truth 2 -> target 1.
    gt_vis_binary = (gt_vis == 2).float()

    # More stable computation: use log_softmax instead of softmax for numerical stability
    pred_heatmaps_log_softmax = F.log_softmax(keypoint_logits.view(N, K, -1), dim=-1)
    pred_heatmaps_softmax = torch.exp(pred_heatmaps_log_softmax)
    pred_vis_scores, _ = torch.max(pred_heatmaps_softmax, dim=-1)
    
    # Clamp the visibility scores to prevent extreme values
    pred_vis_scores = torch.clamp(pred_vis_scores, min=1e-7, max=1.0 - 1e-7)
    
    if gt_vis_binary.numel() == 0:
        return pred_vis_scores.sum() * 0

    return F.binary_cross_entropy(pred_vis_scores.view(-1), gt_vis_binary.view(-1), reduction='mean')


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs, selected_labels):
    # type: (Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]) -> dict[str, Tensor]
    N, K, H, W = keypoint_logits.shape
    if H != W:
        raise ValueError(
            f"keypoint_logits height and width (last two elements of shape) should be equal. Instead got H = {H} and W = {W}"
        )
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
        
    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.where(valid)[0]
    
    keypoint_loss = torch.tensor(0.0, device=keypoint_logits.device)
    if keypoint_targets.numel() > 0 and len(valid) > 0:
        keypoint_logits_flat = keypoint_logits.view(N * K, H * W)
        keypoint_loss = F.cross_entropy(keypoint_logits_flat[valid], keypoint_targets[valid])

    vis_loss = keypoint_visibility_loss(keypoint_logits, gt_keypoints, keypoint_matched_idxs)

    return {"loss_keypoint": keypoint_loss, "loss_keypoint_visibility": vis_loss}

#ToDo add Conformal Prediction using keypointrcnn_loss on calibration dataset

def keypointrcnn_inference(x, boxes):
    # type: (Tensor, list[Tensor]) -> tuple[list[Tensor], list[Tensor]]
    kp_probs = []
    kp_scores = []
    kp_radius = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores, radius = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)
        kp_radius.append(radius)

    return kp_probs, kp_scores, kp_radius


class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (list[Tensor], list[Tensor], list[Tensor]) -> tuple[list[Tensor], list[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (list[Tensor]) -> list[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (list[Tensor], list[Tensor]) -> list[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[list[dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if self.has_mask():
            if not all(["masks" in t for t in targets]):
                raise ValueError("Every element of targets should have a masks key")

    def select_training_samples(
        self,
        proposals,  # type: list[Tensor]
        targets,  # type: Optional[list[dict[str, Tensor]]]
    ):
        # type: (...) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        # append ground-truth bboxes to proposals
        # proposals[0].sum() = tensor(1518528.5000, device='mps:0') verified
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        # proposals[0].sum() = tensor(1527237.5000, device='mps:0') verified
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)   
        # matched_idxs[0].sum() = tensor(113, device='mps:0') verified
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
    
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: list[Tensor]
        image_shapes,  # type: list[tuple[int, int]]
    ):
        # type: (...) -> tuple[list[Tensor], list[Tensor], list[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")
        result: list[dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            train_proposals, matched_idxs, train_labels, regression_targets = self.select_training_samples(proposals, targets) #this is it
            box_features = self.box_roi_pool(features, train_proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
            if train_labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, train_labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            loss_keypoint = {}
            keypoint_proposals = [p["boxes"] for p in result]
            pos_matched_idxs = None
            gt_keypoints = None
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(train_proposals)
                train_keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                # Pass in the gt labels for the keypoints to perform calibration step for 
                selected_labels = []
                
                for img_id in range(num_images):
                    pos = torch.where(train_labels[img_id] > 0)[0]
                    train_keypoint_proposals.append(train_proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
                    selected_labels.append(train_labels[img_id][pos])
                
                #culprits:
                keypoint_features = self.keypoint_roi_pool(features, train_keypoint_proposals, image_shapes)
                keypoint_features = self.keypoint_head(keypoint_features)
                keypoint_logits = self.keypoint_predictor(keypoint_features)
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")
                gt_keypoints = [t["keypoints"] for t in targets]
                # kpt_pred, kp_scores = keypointrcnn_inference(keypoint_logits, train_keypoint_proposals)
                loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, train_keypoint_proposals, gt_keypoints, pos_matched_idxs, selected_labels
                )
            # detection calculation
                losses.update(loss_keypoint)
            else:
                keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
                """
                [tensor([[495.1320,  91.7654, 579.0437, 206.8608],
                [347.4554, 195.9687, 407.6700, 288.5320],
                [212.8076, 200.9418, 291.5740, 324.1768],
                [339.8521, 113.9202, 398.2646, 174.0311]], device='mps:0')]
                """
                keypoint_features = self.keypoint_head(keypoint_features)
                #tensor(181405.5469, device='mps:0')
                keypoint_logits = self.keypoint_predictor(keypoint_features)
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores, kp_radius = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                
                #calcuate the keypoint visibility loss
                for keypoint_prob, kps, r, radius in zip(keypoints_probs, kp_scores, result, kp_radius):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
                    r["keypoints_radius"] = radius
        return result, losses