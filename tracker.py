import torch
from utils.dataloader import RCNNDataset, RCNNTorch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from model.Keypoint_RCNN import FastRCNNPredictor
from model.Keypoint_RCNN import KeypointRCNNPredictor
import torchvision
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from model.Keypoint_RCNN import KeypointRCNN

import torch.distributed as dist
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import gc

def create_custom_keypoint_rcnn_mobilenet(num_classes, num_keypoints, trainable_backbone_layers=3):
    """
    Constructs a Keypoint R-CNN model with a MobileNetV3-Large FPN backbone,
    customized for a specific number of classes and keypoints.

    Args:
        num_classes (int): The number of classes for the box predictor,
                           including the background class.
        num_keypoints (int): The number of keypoints for the keypoint predictor.
        trainable_backbone_layers (int): Number of trainable (not frozen) layers
                                         in the backbone. Ranges from 0 to 5.

    Returns:
        torch.nn.Module: The configured Keypoint R-CNN model.
    """
    # --- Step 1: Create the MobileNetV3-FPN Backbone ---
    # This utility function handles extracting features from the correct layers
    # of MobileNetV3-Large and building the FPN on top of them.
    print("Using MobileNetV3-FPN backbone with FPN")
    backbone = mobilenet_backbone(
        backbone_name='mobilenet_v3_large',
        weights='DEFAULT',  # Corresponds to IMAGENET1K_V2
        fpn=True,
        trainable_layers=trainable_backbone_layers
    )

    # --- Step 2: Configure Anchor Generator for FPN ---
    # First, let's determine how many feature maps the MobileNetV3 FPN actually produces
    # by doing a test forward pass
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
    
    # --- Step 3: Assemble the Full Keypoint R-CNN Model ---
    # We assemble the model using the custom backbone and anchor generator.
    # We use default numbers for classes/keypoints here, as they will be
    # replaced immediately in the next step.
    model = KeypointRCNN(
        backbone,
        num_classes=91,  # Default COCO number of classes
        num_keypoints=17, # Default COCO number of keypoints
        rpn_anchor_generator=anchor_generator
    )

    # 3a. Replace the box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 3b. Replace the keypoint predictor
    in_features_keypoint = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_keypoint, num_keypoints)
    return model

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

def main(model_path, dataset, batch_size=4):
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset {dataset} does not exist.")
    if model_path is None:
        raise FileNotFoundError("Model path must be specified.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    model = create_custom_keypoint_rcnn_mobilenet(num_classes=8, num_keypoints=10)
    print(f"Loading model from {model_path}")
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    cf = predictions_conformal(model, val_loader, device)
    
def predictions_conformal(model, val_loader, device, score_threshold=0.6, alpha=0.05):
    """    Generate conformal predictions based on the model's outputs and validation data.
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
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images, targets)
            outputs = outputs[0]
            
            acceptable_scores = outputs['scores'] > score_threshold
            outputs['boxes'] = outputs['boxes'][acceptable_scores]
            outputs['labels'] = outputs['labels'][acceptable_scores]
            outputs['keypoints'] = outputs['keypoints'][acceptable_scores]
            target_to_prediction_idx = target_to_predictions(targets[0], outputs, iou_threshold=0.5, label_threshold=0.4, is_training=False)
            for obj_idx, prediction in target_to_prediction_idx.items():
                tgt_idx, pred_idx, iou = prediction if prediction is not None else (None, None, None)
                if tgt_idx is not None and pred_idx is not None:
                    tgt_vis = targets[0]['keypoints'][tgt_idx, : , 2].cpu().numpy()
                    pred_vis = outputs['keypoints'][pred_idx, : ,2].cpu().numpy()
                    for keypoint_idx in range(len(tgt_vis)):
                        if tgt_vis[keypoint_idx] == 2:
                            non_conformity_score = 1.0 - pred_vis[keypoint_idx]
                            conformal_predictions.append(non_conformity_score)
    conformal_predictions = np.array(conformal_predictions)
    conformal_predictions = np.sort(conformal_predictions)
    conformal_threshold = (1 - alpha) * (1 + 1/(val_loader.__len__()))
    score = conformal_predictions[conformal_threshold]
    print(f"Conformal prediction threshold: {score:.4f} for alpha={alpha}")
    return conformal_predictions

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Keypoint R-CNN Conformal Prediction")
    # parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    # parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    
    # args = parser.parse_args()
    
    # main(args.model_path, args.dataset, args.batch_size)
    # print("Conformal predictions completed.")
    # model_path = "/Users/Tim/Documents/GitHub/Spark-25/epoch_90.pth"
    # dataset_path = "/Users/Tim/Documents/GitHub/Spark-25/kpt_rcnn_update/rcnn-processed"
    # model_path = "/home/gridsan/tnguyen1/kpts_detector/runs/2025-07-21_13-02-36_keypoint-RCNN-xtds-visibility/epoch_170.pth"
    model_path = "/home/gridsan/tnguyen1/kpts_detector/runs/2025-08-04_13-33-16_mugs_visibility_FPN_LRPLAT/epoch_50.pth"
    dataset_path = "/home/gridsan/tnguyen1/mugs_nococo/rcnn-processed"
    main(model_path, dataset_path, batch_size=32)
    print("Conformal predictions completed.")
    