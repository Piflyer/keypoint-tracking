import torch
from utils.dataloader import RCNNTorch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from model.Keypoint_RCNN import KeypointRCNN
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from model.Keypoint_RCNN import FastRCNNPredictor
from model.Keypoint_RCNN import KeypointRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from collections import OrderedDict


class KeypointRCNN_Model:
    def __init__ (self, model_path=None, device='cpu', num_classes=9, num_keypoints=10, trainable_backbone_layers=3):
        self.device = torch.device(device)
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.trainable_backbone_layers = trainable_backbone_layers
        self.build_model()
    
    def build_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        # backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).features
        # backbone.out_channels = 960
        # anchor_generator = AnchorGenerator(
        #     sizes=((32, 64, 128, 256, 512),),
        #     aspect_ratios=((0.5, 1.0, 2.0),))
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        #     featmap_names=['0'],
        #     output_size=7,
        #     sampling_ratio=2)
        # keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        #     featmap_names=['0'],
        #     output_size=14,
        #     sampling_ratio=2)
        # self.model = KeypointRCNN(
        #     backbone,
        #     num_classes=self.num_classes,
        #     num_keypoints=self.num_keypoints,
        #     image_mean=[0.485, 0.456, 0.406],
        #     image_std=[0.229, 0.224, 0.225],
        #     rpn_anchor_generator=anchor_generator,
        #     box_roi_pool=roi_pooler,
        #     keypoint_roi_pool=keypoint_roi_pooler,
        # )

        backbone = mobilenet_backbone(
                backbone_name='mobilenet_v3_large',
                weights='DEFAULT',  # Corresponds to IMAGENET1K_V2
                fpn=True,
                trainable_layers=self.trainable_backbone_layers
        )
        test_input = torch.randn(1, 3, 800, 800)
        with torch.no_grad():
            features = backbone(test_input)
        
        num_feature_maps = len(features)
        
        # Configure anchors based on the actual number of feature maps
        if num_feature_maps == 5:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        elif num_feature_maps == 4:
            anchor_sizes = ((32,), (64,), (128,), (256,))
        elif num_feature_maps == 3:
            anchor_sizes = ((64,), (128,), (256,))
        else:
            # Fallback: create anchor sizes for whatever number of feature maps we have
            base_sizes = [32, 64, 128, 256, 512, 1024]
            anchor_sizes = tuple((base_sizes[i],) for i in range(num_feature_maps))
        
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        self.model = KeypointRCNN(
            backbone,
            num_classes=91,
            num_keypoints=17,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=anchor_generator,
        )

        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, self.num_classes)
        in_features_keypoints = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_keypoints, self.num_keypoints)

        # --- Step 3: Assemble the Full Keypoint R-CNN Model ---
        # We assemble the model using the custom backbone and anchor generator.
        # We use default numbers for classes/keypoints here, as they will be
        # replaced immediately in the next step.

        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def predict(self, image, target=None):
        #check if image is a tensor
        images = [img.to(self.device) for img in image]
        with torch.no_grad():
            output = self.model(images)
        # clean out bad results
        for j in range(len(output)):
            legible = torch.where(output[j]['scores'] > 0.6)
            output[j]['boxes'] = output[j]['boxes'][legible]
            output[j]['labels'] = output[j]['labels'][legible]
            output[j]['scores'] = output[j]['scores'][legible]
            output[j]['keypoints'] = output[j]['keypoints'][legible]
            output[j]['keypoints_scores'] = output[j]['keypoints_scores'][legible]
            # Check if keypoints_radius exists in output
            if 'keypoints_radius' in output[j]:
                output[j]['keypoints_radius'] = output[j]['keypoints_radius'][legible]
        return output

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    # Don't stack images, KeypointRCNN expects a list of tensors
    return list(images), list(targets)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def target_to_predictions(targets, output, iou_threshold, label_threshold, is_training=False):
    target_to_prediction_idx = {j: None for j in targets['labels'].cpu().numpy()}
    for j in range(len(targets['labels'])):
        target_box = targets['boxes'][j].cpu().numpy()
        target_label = targets['labels'][j].item()

        ious = []
        if is_training:
            for pred_box in output['boxes'].detach().cpu().numpy():
                ious.append(calculate_iou(pred_box, target_box))
        else:
            for pred_box in output['boxes'].cpu().numpy():
                ious.append(calculate_iou(pred_box, target_box))

        # Find the best matching prediction for each target
        if ious:
            best_match = np.argmax(ious)
            best_iou = ious[best_match]
            if best_iou >= iou_threshold:
                target_to_prediction_idx[target_label] = [j, best_match, best_iou]
    return target_to_prediction_idx

# def calculate_non_conformity

def main(model_path, dataset, batch_size=4, num_classes=9, num_keypoints=10):
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset {dataset} does not exist.")
    if model_path is None:
        raise FileNotFoundError("Model path must be specified.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_set = RCNNTorch(
        gt_file = os.path.join(dataset, "val_labels.json"),
        transform=transform,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True,
        num_workers=6, # Set num_workers to 0 for deterministic data loading, as multi-process data loading can introduce non-determinism
        collate_fn=custom_collate_fn
    )

    print(f"Loading model from {model_path}")
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
    model = KeypointRCNN_Model(model_path=model_path, device=device, num_classes=num_classes, num_keypoints=num_keypoints)
    cf = predictions_conformal(model, val_loader, device)
    
def predictions_conformal(model, val_loader, device, score_threshold=0.6, alpha=0.1):
    """
    Generate conformal predictions based on the model's outputs and validation data.
    Args:
        model (torch.nn.Module): The trained Keypoint R-CNN model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
        np.ndarray: Array of non-conformity scores for each keypoint.
    """
    conformal_predictions = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Evaluating..."):
            images = [img.to(device) for img in images]
            outputs = model.predict(images, targets)
            for j in range(len(outputs)):
                target_to_prediction_idx = target_to_predictions(targets[j], outputs[j], iou_threshold=0.5, label_threshold=0.4, is_training=False)
                # Process the target_to_prediction_idx as needed
                for obj_idx, prediction in target_to_prediction_idx.items():
                    tgt_idx, pred_idx, iou = prediction if prediction is not None else (None, None, None)
                    if tgt_idx is not None and pred_idx is not None:
                        tgt_vis = targets[j]['keypoints'][tgt_idx, : , 2].cpu().numpy()
                        pred_vis = outputs[j]['keypoints'][pred_idx, : ,2].cpu().numpy()
                        for keypoint_idx in range(len(tgt_vis)):
                            if tgt_vis[keypoint_idx] == 2:
                                non_conformity_score = 1.0 - pred_vis[keypoint_idx]
                                conformal_predictions.append(non_conformity_score)
                            else:
                                conformal_predictions.append(pred_vis[keypoint_idx])
    conformal_predictions = np.array(conformal_predictions)
    conformal_predictions = np.sort(conformal_predictions)
    conformal_threshold = (1 - alpha) * (1 + 1/(len(conformal_predictions)))
    if len(conformal_predictions) == 0:
        raise ValueError("No conformal predictions were generated. Check the model and data.")
    score = conformal_predictions[int(conformal_threshold * len(conformal_predictions))]
    print(f"Conformal prediction threshold: {1 - score:.4f} for alpha={alpha}")
    return conformal_predictions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Keypoint R-CNN Conformal Prediction")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')

    args = parser.parse_args()

    main(args.model_path, args.dataset, args.batch_size)
    print("Conformal predictions completed.")