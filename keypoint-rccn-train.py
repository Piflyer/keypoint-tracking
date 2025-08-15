import torch
from utils.dataloader import RCNNTorch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from model.Keypoint_RCNN import FastRCNNPredictor
from model.Keypoint_RCNN import KeypointRCNNPredictor
import torchvision
from torchvision.models.detection.backbone_utils import mobilenet_backbone
import argparse
import cv2
from model.Keypoint_RCNN import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from collections import OrderedDict
import gc

torch.backends.cudnn.benchmark = True
def create_custom_keypoint_rcnn_mobilenet(num_classes, num_keypoints, trainable_backbone_layers=3, classical_model=True):
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
    
    if classical_model:
        print("Using classical model")
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        backbone.out_channels = 960
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * len( ((32, 64, 128, 256, 512),))
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )
        # Define the number of keypoints your model should predict per instance
        # if ckpt_path is None and not cont:
        #     default_num_keypoints = num_keypoints
        #     default_num_classes = num_classes  # 8 classes + background
        model = KeypointRCNN(
            backbone,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
        return model
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

        if ious:
            best_match = np.argmax(ious)
            best_iou = ious[best_match]
            if best_iou >= iou_threshold:
                target_to_prediction_idx[target_label] = [j, best_match, best_iou]
    return target_to_prediction_idx

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def train_one_epoch(model, data_loader, optimizer, device, epoch, writer, num_epochs, rank, train_set, warmup_scheduler=None, warmup_epochs=10):
    if rank == 0:
        print(f"Training epoch {epoch+1}/{num_epochs}")
    model.train()
    
    # Keep track of running losses
    total_loss_epoch = 0.0
    loss_classifier_epoch = 0.0
    loss_box_reg_epoch = 0.0
    loss_objectness_epoch = 0.0
    loss_rpn_box_reg_epoch = 0.0
    loss_keypoint_epoch = 0.0
    loss_keypoint_visibility = 0.0

    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # model() returns a dictionary of losses in training mode
        loss_dict = model(images, targets)
        #check if tuple of loss, output or just a dict
        if isinstance(loss_dict, tuple):
            loss_dict = loss_dict[0]
        losses = 0
        for loss_name, loss_value in loss_dict.items():
            if loss_name == "loss_keypoint":
                losses += (loss_value * 5.0)  # Reduced from 10 to 5
            elif loss_name == "loss_keypoint_visibility":
                losses += (loss_value * 3.0)  # Reduced from 5 to 3
            else:
                losses += loss_value
        
        # Check for NaN or infinite losses
        if not torch.isfinite(losses):
            if rank == 0:
                print(f"Warning: Non-finite loss detected: {losses}, skipping batch")
                for name, val in loss_dict.items():
                    print(f"  {name}: {val}")
            continue
        
        # losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        
        # Add gradient clipping to prevent exploding gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Debug: Print gradient norm if it's getting large
        if rank == 0 and grad_norm > 0.5 and epoch > 200:
            print(f"Warning: Large gradient norm detected: {grad_norm:.4f}")
        
        optimizer.step()
        
        # Apply warmup scheduler only during warmup period
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        
        # Accumulate losses for logging
        total_loss_epoch += losses.item()
        loss_classifier_epoch += loss_dict['loss_classifier'].item()
        loss_box_reg_epoch += loss_dict['loss_box_reg'].item()
        loss_objectness_epoch += loss_dict['loss_objectness'].item()
        loss_rpn_box_reg_epoch += loss_dict['loss_rpn_box_reg'].item()
        loss_keypoint_epoch += (loss_dict['loss_keypoint'].item() * 5.0)  # Updated to match training weights
        if 'loss_keypoint_visibility' in loss_dict:
            loss_keypoint_visibility += (loss_dict['loss_keypoint_visibility'].item() * 3.0)  # Updated to match training weights

    #gather all losses across all processes

    # Calculate average losses for the epoch
    num_batches = len(data_loader)
    avg_total_loss = total_loss_epoch / num_batches
    avg_loss_classifier = loss_classifier_epoch / num_batches
    avg_loss_box_reg = loss_box_reg_epoch / num_batches
    avg_loss_objectness = loss_objectness_epoch / num_batches
    avg_loss_rpn_box_reg = loss_rpn_box_reg_epoch / num_batches
    avg_loss_keypoint = loss_keypoint_epoch / num_batches
    if 'loss_keypoint_visibility' in loss_dict:
        avg_loss_keypoint_visibility = loss_keypoint_visibility / num_batches
    
    # Log to TensorBoard
    if rank == 0:
        writer.add_scalar('Loss/train_total', avg_total_loss, epoch)
        writer.add_scalar('Loss/train_classifier', avg_loss_classifier, epoch)
        writer.add_scalar('Loss/train_box_reg', avg_loss_box_reg, epoch)
        writer.add_scalar('Loss/train_objectness', avg_loss_objectness, epoch)
        writer.add_scalar('Loss/train_rpn_box_reg', avg_loss_rpn_box_reg, epoch)
        writer.add_scalar('Loss/train_keypoint', avg_loss_keypoint, epoch)
        # log learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)
        if 'loss_keypoint_visibility' in loss_dict:
            writer.add_scalar('Loss/train_keypoint_visibility', avg_loss_keypoint_visibility, epoch)
    
        print("------- Training Loss -------")
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_total_loss:.4f}")
        print(f"  Classifier: {avg_loss_classifier:.4f}, Box Reg: {avg_loss_box_reg:.4f}, Keypoint: {avg_loss_keypoint:.4f}")
        print(f"  Objectness: {avg_loss_objectness:.4f}, RPN Box Reg: {avg_loss_rpn_box_reg:.4f}")
        if 'loss_keypoint_visibility' in loss_dict:
            print(f"  Keypoint Visibility: {avg_loss_keypoint_visibility:.4f}")
        print("\n")
    
    #clear the cache to avoid memory issues
    del images, targets, loss_dict, losses, avg_total_loss, avg_loss_classifier, avg_loss_box_reg, avg_loss_objectness, avg_loss_rpn_box_reg, avg_loss_keypoint
    gc.collect()
    torch.cuda.empty_cache()

def evaluate(model, data_loader, device, writer, epoch, rank):
    print(f"Evaluating epoch {epoch+1}")
    #set model to train just to get the losses
    # This is a workaround since KeypointRCNN does not have a separate eval mode
    model.train()
    total_loss_epoch = 0.0
    loss_classifier_epoch = 0.0
    loss_box_reg_epoch = 0.0
    loss_objectness_epoch = 0.0
    loss_rpn_box_reg_epoch = 0.0
    loss_keypoint_epoch = 0.0
    loss_keypoint_visibility_epoch = 0.0    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating epoch {epoch+1}"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # model() returns a dictionary of losses in training mode
            loss_dict = model(images, targets)
            if isinstance(loss_dict, tuple):
                loss_dict = loss_dict[0]
            
            losses = sum(loss for loss in loss_dict.values())
            
            # Accumulate losses for logging
            total_loss_epoch += losses.item()
            loss_classifier_epoch += loss_dict['loss_classifier'].item()
            loss_box_reg_epoch += loss_dict['loss_box_reg'].item()
            loss_objectness_epoch += loss_dict['loss_objectness'].item()
            loss_rpn_box_reg_epoch += loss_dict['loss_rpn_box_reg'].item()
            loss_keypoint_epoch += loss_dict['loss_keypoint'].item()
            if 'loss_keypoint_visibility' in loss_dict:
                loss_keypoint_visibility_epoch += loss_dict['loss_keypoint_visibility'].item()

    # Calculate average losses for the epoch
    num_batches = len(data_loader)
    avg_total_loss = total_loss_epoch / num_batches
    avg_loss_classifier = loss_classifier_epoch / num_batches
    avg_loss_box_reg = loss_box_reg_epoch / num_batches
    avg_loss_objectness = loss_objectness_epoch / num_batches
    avg_loss_rpn_box_reg = loss_rpn_box_reg_epoch / num_batches
    avg_loss_keypoint = loss_keypoint_epoch / num_batches
    if 'loss_keypoint_visibility' in loss_dict:
        avg_loss_keypoint_visibility = loss_keypoint_visibility_epoch / num_batches
    # Log to TensorBoard
    if rank == 0:
        writer.add_scalar('Loss/val_total', avg_total_loss, epoch)
        writer.add_scalar('Loss/val_classifier', avg_loss_classifier, epoch)
        writer.add_scalar('Loss/val_box_reg', avg_loss_box_reg, epoch)
        writer.add_scalar('Loss/val_objectness', avg_loss_objectness, epoch)
        writer.add_scalar('Loss/val_rpn_box_reg', avg_loss_rpn_box_reg, epoch)
        writer.add_scalar('Loss/val_keypoint', avg_loss_keypoint, epoch)
        if 'loss_keypoint_visibility' in loss_dict:
            writer.add_scalar('Loss/val_keypoint_visibility', avg_loss_keypoint_visibility, epoch)

    del images, targets, loss_dict, losses, avg_loss_classifier, avg_loss_box_reg, avg_loss_objectness, avg_loss_rpn_box_reg, avg_loss_keypoint
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return the average total validation loss for plateau scheduler
    return avg_total_loss

def visualize_predictions(image, prediction, writer, caption, kp_threshold=0.5, score_threshold=0.5, is_target=False):
    # Convert image tensor back to a displayable format
    image = image.permute(1,2,0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    # Create a copy to draw on
    img_to_draw = image.copy()
    
    boxes = prediction['boxes']
    keypoints = prediction['keypoints']
    
    if not is_target:
        scores = prediction['scores']
    
    for i in range(boxes.shape[0]):
        if not is_target and scores[i] < score_threshold:
            continue
            
        box = boxes[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        kps = keypoints[i].cpu().numpy()
        for j, (x, y, v) in enumerate(kps):
            # For predictions, v is a visibility score. For targets, it's 0, 1, or 2.
            if (not is_target and v < kp_threshold) or (is_target and v == 0):
                continue
            cv2.circle(img_to_draw, (int(x), int(y)), 3, (0, 255, 0), -1)
            
    # Add image to TensorBoard
    writer.add_image(caption, img_to_draw, global_step=0, dataformats='HWC')

def main(rank, dataset, batch_size=4, num_epochs=10, learning_rate=0.001, log_dir='logs', num_workers=4, cont=False, ckpt_path=None, ckpt_tb=None, num_classes=8, num_keypoints=10, classical_model=True, warmup_epochs=10):

    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset {dataset} does not exist.")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(f'cuda:{rank}')
    print(f"Using device: {device}")
    single_class = (num_classes-1 == 1)
    transform_list = [
        transforms.ToTensor()
    ]
    train_set = RCNNTorch(
        gt_file= os.path.join(dataset, 'train_labels.json'),
        transform=transforms.Compose(transform_list),
        augment=False,
        single_class=single_class,
    )
    val_set = RCNNTorch(
        gt_file= os.path.join(dataset, 'val_labels.json'),
        transform=transforms.Compose(transform_list),
        augment=False,
        single_class=single_class,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )
    writer = None
    if rank == 0:
        if ckpt_tb is not None:
            log_dir = ckpt_tb
        else:
            log_dir = f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_keypoint-RCNN-xtds"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

    model = create_custom_keypoint_rcnn_mobilenet(num_classes, num_keypoints, classical_model=classical_model)
    
    if cont and ckpt_path is not None:
        if os.path.exists(ckpt_path):
            print(f"Loading model from {ckpt_path}")
            # convert state_dict keys to match the model's expected keys
            state_dict = torch.load(ckpt_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded successfully.")
        else:
            print(f"Checkpoint {ckpt_path} does not exist. Starting from scratch.")
    print(f"Number of training images: {len(train_set)}")
    print(f"Number of validation images: {len(val_set)}")
    model.to(device)
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay for regularization
    
    # Add warmup scheduler for better training stability
    warmup_epochs = 20
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    # Use ReduceLROnPlateau after warmup period (handled manually in training loop)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    print(f"Starting training for {num_epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")

    for epoch in range(num_epochs):
        epoch += 60  # Start from epoch 130 if continuing training
        train_one_epoch(model, train_loader, optimizer, device, epoch, writer, num_epochs, rank, train_set, warmup_scheduler=None, warmup_epochs=None)
        val_loss = evaluate(model, val_loader, device, writer, epoch, rank)

        # Apply plateau scheduler after warmup period using validation loss
        if epoch >= warmup_epochs and rank == 0:
            # plateau_scheduler.step(val_loss)
            exponential_scheduler.step()

        if rank == 0:
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, f"epoch_{epoch+1}.pth"))
                print(f"Model saved at epoch {epoch+1}")

    print("Training complete.")
    if rank == 0:
        writer.close()

if __name__ == "__main__":
    # dataset_path = '/home/gridsan/tnguyen1/mugs_nococo/rcnn-processed'
    parser = argparse.ArgumentParser(description='Keypoint R-CNN Training Script')
    parser.add_argument('--dataset_path', default='/home/gridsan/tnguyen1/kpt_rcnn_update/rcnn-processed', help='Path to the dataset')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', default=400, type=int, help='Number of epochs to train')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate for training')
    parser.add_argument('--num_classes', default=9, type=int, help='Number of classes including background')
    parser.add_argument('--num_keypoints', default=10, type=int, help='Number of keypoints to detect')
    parser.add_argument('--classical_model', action='store_true', help='Use classical Keypoint R-CNN model without FPN')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of warmup epochs for learning rate scheduling')
    parser.add_argument('--log_dir', default='runs', type=str, help='Directory to save logs and checkpoints')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from the last checkpoint')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the checkpoint file to continue training')
    parser.add_argument('--checkpoint_tensorboard', default=None, type=str, help='Path to the TensorBoard checkpoint file to continue training')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    num_workers = 6
    log_dir = args.log_dir
    cont = args.continue_training
    ckpt_path = args.checkpoint_path
    ckpt_tb = args.checkpoint_tensorboard
    num_classes = args.num_classes
    num_keypoints = args.num_keypoints
    classical_model = args.classical_model
    main(dataset_path, batch_size, num_epochs, learning_rate, log_dir, num_workers, cont, ckpt_path, ckpt_tb, num_classes, num_keypoints, classical_model, warmup_epochs=20)
