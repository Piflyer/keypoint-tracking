import numpy as np
import cv2 as cv
import argparse
from utils.dataloader import RCNNTorch
from PIL import Image
import time
import torchvision
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from model.Keypoint_RCNN import KeypointRCNN
from collections import OrderedDict
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from model.Keypoint_RCNN import FastRCNNPredictor
from model.Keypoint_RCNN import KeypointRCNNPredictor
from ultralytics import YOLO
import json
import glob
import os
import natsort

# Global error tracking dictionary
epoch_error = dict()

def FolderLoader(dataset_parent, NOCS=False):
    if NOCS:
        rgb_paths = natsort.natsorted(glob.glob(f"{dataset_parent}/*_color.png"))
    else:
        rgb_paths = natsort.natsorted(glob.glob(f"{dataset_parent}/*.png") or glob.glob(f"{dataset_parent}/*.jpg"))
    dataset_len = len(rgb_paths)
    return rgb_paths, dataset_len

def errorCalculation(target, kpts, tracking_manager, frame_count, error_list, hallucinate_counter):
    # first argsort the target keypoints by labels
    target_kpts = target['keypoints'].cpu().numpy()
    target_labels = target['labels'].cpu().numpy()
    sorted_indices = np.argsort(target_labels)
    target_kpts = target_kpts[sorted_indices]
    target_labels = target_labels[sorted_indices]
    if len(kpts) > 0 and len(target_kpts) > 0:
        predicted_xy = []
        target_xy = []
        active_iter = 0
        active_tracks = tracking_manager.getActiveTracks()
        hungarian_target_idx = dict()
        hungarian_predicted_id = dict()
        for j in range(len(target_labels)):
            target_label = target_labels[j]
            instances = tracking_manager.getInstanceCounter(target_label)
            if sum(target_labels == target_label) == 1 and instances == 1:
                # we have a unique match
                try:
                    objidx, instanceidx, ptidx = map(int, active_tracks[active_iter].split('_'))
                except:
                    pass
                while objidx == target_label and active_iter < len(active_tracks):
                    # fix
                    predicted_xy.append(tracking_manager.getTrack(active_tracks[active_iter]).getState())
                    target_xy.append(target_kpts[j][ptidx, :2])
                    active_iter += 1
                    if active_iter >= len(active_tracks):
                        break
                    objidx, instanceidx, ptidx = map(int, active_tracks[active_iter].split('_'))

            elif sum(target_labels == target_label) > 1 or instances > 1:
                # multiple instances of this object - skip for now
                # will use hungarian matching with distance thresholding
                print("Multiple instances detected, skipping for now.")
                # create a cost matrix for hungarian matching
                if target_label not in hungarian_target_idx:
                    hungarian_target_idx[target_label] = []
                if target_label not in hungarian_predicted_id:
                    hungarian_predicted_id[target_label] = []
                # append the target index to the target index list
                hungarian_target_idx[target_label].append(j)
                # append the predicted index to the predicted index list
                obj_idx, instance_idx, pt_idx = map(int, active_tracks[active_iter].split('_'))
                while obj_idx == target_label:
                    hungarian_predicted_id[target_label].append(active_tracks[active_iter])
                    active_iter += 1
                    if active_iter >= len(active_tracks):
                        break
                    obj_idx, instance_idx, pt_idx = map(int, active_tracks[active_iter].split('_'))
            else:
                # object not tracked with prediction, false negative
                print("Object not tracked with prediction, false negative.")
                pass
        # now do a check for any remaining unmatched points
        hallucinated_labels = []
        for track_id in active_tracks[active_iter:]:
            objidx, instanceidx, ptidx = map(int, track_id.split('_'))
            if objidx not in target_labels:
                # hallucinated point - false positive
                hallucinated_labels.append(objidx)
        if len(hallucinated_labels) > 0:
            hallucinate_counter += len(hallucinated_labels)
            print(f"Frame {frame_count}: Hallucinated {len(hallucinated_labels)} points - total hallucinations: {hallucinate_counter}") 
        # Calculate euclidean distance for visible points
        distances = []
        # if len(hungarian_target_idx) != 0 and len(hungarian_predicted_id) != 0:
        #     hungarian_matches = hungarian_matching(hungarian_target_idx, hungarian_predicted_id, target, distance_threshold)
        # #skip for now
        # if len(active_iter) != 
        
        for i in range(min(len(predicted_xy), len(target_xy))):
            dist = np.linalg.norm(predicted_xy[i] - target_xy[i])
            distances.append(dist)
        if distances:
            # Use mean distance for this frame to maintain frame-wise indexing
            frame_error = np.mean(distances)
            error_list.append(frame_error)
        else:
            # No valid keypoints for error calculation
            error_list.append(np.nan)
    else:
        # No predictions or targets available
        error_list.append(np.nan)
    return error_list, hallucinate_counter

def hungarian_matching(target_dict, predicted_dict, targets, distance_threshold=3):
    """
    Match predicted keypoints to target keypoints using Hungarian algorithm.

    Args:
        target_dict (dict): Dictionary containing target keypoints grouped by labels.
        predicted_dict (dict): Dictionary containing predicted keypoints grouped by labels.
        targets (list): List of target keypoints.
        distance_threshold (float): Maximum distance threshold for matching keypoints.

    Returns:
        None: This function is currently a placeholder and does not return any value.
    """
    labels = list(target_dict.keys())
    for label in labels:
        # construct cost matrix
        target_idx = target_dict[label]
        predicted_idx = predicted_dict[label]
        cost_matrix = np.zeros((len(target_idx), len(predicted_idx)), dtype=np.float32)
        for i, t_idx in enumerate(target_idx):
            for j, p_idx in enumerate(predicted_idx):
                # breakdown predicted_idx into obj, instance, ptidx
                p_obj, p_instance, p_ptidx = map(int, p_idx.split('_'))

    # """Match predicted keypoints to target keypoints using Hungarian algorithm."""
    # if len(predicted) == 0 or len(target) == 0:
    #     return [], []

    # # Calculate distance matrix
    # distance_matrix = np.linalg.norm(predicted[:, None] - target[None, :], axis=2)

    # # Apply distance threshold
    # distance_matrix[distance_matrix > distance_threshold] = np.inf

    # # Perform Hungarian matching
    # row_ind, col_ind = scipy.optimize.linear_sum_assignment(distance_matrix)

    # # Filter matches based on threshold
    # matches = []
    # for r, c in zip(row_ind, col_ind):
    #     if distance_matrix[r, c] < distance_threshold:
    #         matches.append((r, c))

    # return matches
    pass

class KalmanFilter:
    """Kalman Filter for 2D point tracking.

    This class implements a Kalman filter for tracking 2D points (x, y) over time.
    """
    
    def __init__ (self, initial_measurement=None, process_noise=1e-2, kpt_rcnn_noise=1e-1, optical_flow_noise=1e-3):
        """Initialize the Kalman filter.

        Args:
            initial_measurement (tuple, optional): Initial measurement (x, y). Defaults to None.
            process_noise (float, optional): Process noise covariance. Defaults to 1e-2.
            kpt_rcnn_noise (float, optional): Keypoint R-CNN measurement noise covariance. Defaults to 1e-1.
            optical_flow_noise (float, optional): Optical flow measurement noise covariance. Defaults to 1e-3.
        """
        self.kalman_filter = cv.KalmanFilter(4, 2) #x, y, dx, dy
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # measurements are x, y
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # transition is x, y, dx, dy
        self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise # process noise covariance

        # measurement noise covariance for each source
        self.R_optical_flow = np.eye(2, dtype=np.float32) * optical_flow_noise # optical flow measurement noise covariance
        self.R_kpts_update = np.eye(2, dtype=np.float32) * kpt_rcnn_noise # ground truth update measurement noise covariance

        self.kalman_filter.measurementNoiseCov = self.R_optical_flow # default measurement noise covariance is optical flow

        if initial_measurement is not None:
            x,y = initial_measurement
            self.kalman_filter.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman_filter.statePost = np.array([[x], [y], [0], [0]], np.float32)
        else:
            self.kalman_filter.statePre = np.zeros((4, 1), np.float32)
            self.kalman_filter.statePost = np.zeros((4, 1), np.float32)

    def predict(self):
        """Predict the next state of the Kalman filter.

        Returns:
            np.ndarray: The predicted state (x, y) of the Kalman filter.
        """
        return self.kalman_filter.predict().flatten()[:2]

    def correct(self, measurement=None, source="RCNN"):
        """Correct the Kalman filter with the new measurement.

        Args:
            measurement (tuple, optional): New measurement (x, y). Defaults to None.
            source (str, optional): Source of the measurement ("RCNN" or "optical_flow"). Defaults to "RCNN".
        """
        if measurement is not None:
            if source == "RCNN":
                self.kalman_filter.measurementNoiseCov = self.R_kpts_update
            else:
                self.kalman_filter.measurementNoiseCov = self.R_optical_flow
            x, y = measurement
            self.kalman_filter.correct(np.array([[x], [y]], np.float32))
        # else:
        #     # If no measurement, just predict
        #     return self.predict().flatten()[:2]
        # return self.kalman_filter.statePost[:2].flatten()

    def getState(self):
        """Get the current state of the Kalman filter.

        Returns:
            np.ndarray: The current state (x, y) of the Kalman filter.
        """
        return self.kalman_filter.statePost[:2].flatten()

class PointTrack:
    """
    Implements a point tracking class with a Kalman filter for state estimation.
    """
    def __init__ (self, xy=None, objid=None, ptidx=None, trackId=None, confidence=1.0, process_noise=1e-2, kpt_rcnn_noise=1e-1, optical_flow_noise=1e-3):
        """Initializes a point track.

        Args:
            xy (tuple, optional): Initial coordinates (x, y). Defaults to None.
            objid (int, optional): Object ID. Defaults to None.
            ptidx (int, optional): Point index. Defaults to None.
            trackId (str, optional): Track ID. Defaults to None.
            confidence (float, optional): Confidence score. Defaults to 1.0.
            process_noise (float, optional): Process noise for the Kalman filter. Defaults to 1e-2.
            kpt_rcnn_noise (float, optional): Keypoint R-CNN noise for the Kalman filter. Defaults to 1e-1.
            optical_flow_noise (float, optional): Optical flow noise for the Kalman filter. Defaults to 1e-3.
        """
        if trackId is not None:
            self.id = trackId
        else:
            self.id = f"{objid}_{ptidx}"
        self.pt = KalmanFilter(initial_measurement=xy,process_noise=1e-2, kpt_rcnn_noise=1e-1, optical_flow_noise=1e-3)
        self.confidence = confidence
        self.xy = xy

    def getConfidence(self):
        """Get the confidence of the point track.
        
        Returns:
            float: The confidence score of the point track.
        """
        return self.confidence

    def getID(self):
        """Get the ID of the point track.

        Returns:
            str: The ID of the point track.
        """
        return self.id
    
    def getPoint(self):
        """Get the point.

        Returns:
            tuple: The (x, y) coordinates of the point.
        """
        return self.xy
    
    def getState(self):
        return self.pt.getState()

    def predictPoint(self):
        """Predict the next point."""
        return self.pt.predict()

    def updatePoint(self, xy, source="optical_flow", confidence=1.0):
        """Update the point with the new coordinates."""
        self.pt.correct(measurement=xy, source=source)
        self.confidence = confidence
        self.xy = xy

    def similarityTest(self, xy, threshold = 3):
        """Test if the point is similar to the given coordinates."""
        xy = np.array(xy, dtype=np.float32)
        distance = np.linalg.norm(self.pt.predict() - xy)
        return distance < threshold

class PointTrackManager:
    """
    Manages multiple point tracks, handling their creation, update, and deletion.

    Attributes:
        tracks (list): List of all point tracks.
        active_tracks (list): List of active track IDs.
        tempActiveTracks (list): Temporary list of active tracks.
        instance_counter (dict): Dictionary to count instances for each object ID.
        tempTracks (list): Temporary list of tracks.
    """
    def __init__ (self):
        """
        Initialize the PointTrackManager.
        """
        self.tracks = []
        self.active_tracks = []
        self.tempActiveTracks = []
        self.instance_counter = dict()
        self.tempTracks = []

    def addTrack(self, xy, objid, ptidx, source="RCNN", confidence=1.0, process_noise=1e-2, kpt_rcnn_noise=1e-1, optical_flow_noise=1e-3):
        """
        Add a new point track.

        Args:
            xy (tuple): Coordinates of the point (x, y).
            objid (int): Object ID.
            ptidx (int): Point index.
            source (str): Source of the measurement (e.g., "RCNN").
            confidence (float): Confidence score of the point.
            process_noise (float, optional): Process noise for the Kalman filter. Defaults to 1e-2.
            kpt_rcnn_noise (float, optional): Keypoint R-CNN noise for the Kalman filter. Defaults to 1e-1.
            optical_flow_noise (float, optional): Optical flow noise for the Kalman filter. Defaults to 1e-3.
        """
        trackID = self.createID(objid, ptidx)
        #check if the track already exists
        new_track = None
        isUniquetrack = True
        for track in self.tracks:
            if track.getID() == trackID:
                track.updatePoint(xy, source=source)
                isUniquetrack = False
                new_track = track
        if isUniquetrack:
            new_track = PointTrack(xy, trackId=trackID, confidence=confidence, process_noise=1e-2, kpt_rcnn_noise=1e-1, optical_flow_noise=1e-3)
        self.tempTracks.append(new_track)
        self.active_tracks.append(trackID)

    def mergeTracks(self):
        """
        Merge temporary tracks into the main track list.
        """
        self.tracks = self.tempTracks
        self.tempTracks = []

    def updateTrack(self, track_id, xy, source="optical_flow", confidence=1.0):
        """
        Update the point track with the new coordinates.
Generate doc string with args and return for each function and class if haven't already been implemented
        Args:
            track_id (str): ID of the track to update.
            xy (tuple): New coordinates of the point (x, y).
            source (str): Source of the measurement (e.g., "optical_flow").
            confidence (float): Confidence score of the point.

        Returns:
            bool: True if the track was updated successfully, False otherwise.
        """
        for track in self.tracks:
            if track.getID() == track_id:
                track.updatePoint(xy, source=source, confidence=confidence)
                return True
        return False

    def updateInstanceCounter(self, objid):
        """
        Update the instance counter for the given object ID.

        Args:
            objid (int): Object ID.
        """
        if objid not in self.instance_counter:
            self.instance_counter[objid] = 0
        self.instance_counter[objid] += 1

    def getInstanceCounter(self, objid):
        """
        Get the instance counter for the given object ID.

        Args:
            objid (int): Object ID.

        Returns:
            int: Instance count for the given object ID.
        """
        try:
            return self.instance_counter[objid]
        except KeyError:
            return 0

    def createID(self, objid, ptidx):
        """
        Create a new ID for the point track.

        Args:
            objid (int): Object ID.
            ptidx (int): Point index.

        Returns:
            str: Generated track ID.
        """
        if objid not in self.instance_counter:
            self.instance_counter[objid] = 0
        return f"{objid}_{self.instance_counter[objid]}_{ptidx}"

    def step(self):
        """
        Step through the point tracks and predict the next state.

        Returns:
            list: List of predicted points for active tracks.
        """
        output = []
        for track in self.tracks:
            if track.getID() in self.active_tracks:
                predicted_point = track.predictPoint()
                output.append(predicted_point)
        return output

    def getActiveTracks(self):
        """
        Get the active tracks.

        Returns:
            list: List of active track IDs.
        """
        return self.active_tracks

    def getTrack(self, track_id):
        """
        Get the point track with the given ID.

        Args:
            track_id (str): ID of the track.

        Returns:
            PointTrack: The point track with the given ID, or None if not found.
        """
        for track in self.tracks:
            if track.getID() == track_id:
                return track
        return None

    def updateTracks(self, update_list):
        """
        Update the active tracks with the new active tracks.

        Args:
            update_list (list): List of indices to update.
        """
        self.active_tracks = np.array(self.active_tracks)[update_list].tolist()
        self.tracks = np.array(self.tracks)[update_list].tolist()
        self.instance_counter = dict()
        #recalculate instance counter
        for j in self.active_tracks:
            objid, instanceid, ptidx = map(int, j.split('_'))
            self.instance_counter[objid] = max(instanceid, self.instance_counter.get(objid, 0))

    def softReset(self):
        """
        Perform a soft reset of the tracking manager.

        Clears active tracks and instance counters while keeping track history.
        """
        self.active_tracks = []
        self.instance_counter = dict()

    def hardReset(self):
        """
        Perform a hard reset of the tracking manager.

        Clears all tracks, active tracks, and instance counters.
        """
        self.tracks = []
        self.active_tracks = []
        self.instance_counter = dict()

    def getBatchStates(self):
        """
        Get the current states of all tracks in the batch.

        Returns:
            list: List of current states for all tracks.
        """
        return [track.getState() for track in self.tracks]

class KeypointRCNN_Model:
    def __init__ (self, model_path=None, device='cpu', num_classes=9, num_keypoints=10, no_fpn=False):
        self.device = torch.device(device)
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.no_fpn = no_fpn
        self.trainable_backbone_layers = 5
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
        if self.no_fpn:
            backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).features
            backbone.out_channels = 960
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),))
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)
            keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=14,
                sampling_ratio=2)
            self.model = KeypointRCNN(
                backbone,
                num_classes=self.num_classes,
                num_keypoints=self.num_keypoints,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                keypoint_roi_pool=keypoint_roi_pooler,
            )
        else:
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
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def predict(self, image, target=None, tracking_manager=None, img_np=None, mask=None, conformal_threshold=0.08, kalman_process_noise=None, kalman_rcnn_noise=None, kalman_optical_flow_noise=None):
        #check if image is a tensor
        if not isinstance(image, torch.Tensor):
            image = Image.fromarray(image)
            image = torchvision.transforms.ToTensor()(image)
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model([image])[0]
        # clean out bad results
        legible = torch.where(output['scores'] > 0.6)
        output['boxes'] = output['boxes'][legible]
        output['labels'] = output['labels'][legible]
        output['scores'] = output['scores'][legible]
        output['keypoints'] = output['keypoints'][legible]
        output['keypoints_scores'] = output['keypoints_scores'][legible]
        # Check if keypoints_radius exists in output
        if 'keypoints_radius' in output:
            output['keypoints_radius'] = output['keypoints_radius'][legible]
        
        kpts = output['keypoints'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        sorted_indices = np.argsort(labels)
        kpts = kpts[sorted_indices]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        if len(kpts) > 0:
            visible_mask = kpts[:, :, 2] > conformal_threshold  # Changed from > 1 to > 0 for more permissive visibility
            # Reset tracking manager but keep some trail history
            #tracking_manager.hardReset()
            tracking_manager.softReset()
            # Fade the existing mask instead of completely resetting it
            if mask is not None:
                mask = cv.multiply(mask, 0.85)  # Fade trails by 15% each model refresh
            else:
                mask = np.zeros_like(img_np)
            # Add visible keypoints to tracking manager
            # valid_points = []
            for obj_idx in range(len(kpts)):
                if visible_mask[obj_idx].any():
                    tracking_manager.updateInstanceCounter(labels[obj_idx].item()) # Increment instance counter for this object
                    for kpt_idx in range(len(kpts[obj_idx])):
                        if visible_mask[obj_idx, kpt_idx]:
                            xy = kpts[obj_idx, kpt_idx, :2]
                            tracking_manager.addTrack(xy, labels[obj_idx].item(), kpt_idx, source="gt", confidence=kpts[obj_idx, kpt_idx, 2], process_noise=kalman_process_noise, kpt_rcnn_noise=kalman_rcnn_noise, optical_flow_noise=kalman_optical_flow_noise) # Add track for this keypoint
                            # valid_points.append(xy) # Append valid keypoint for tracking
            tracking_manager.mergeTracks()  # Merge temporary tracks into main tracking manager
            # Initialize optical flow tracking
            state_batch = tracking_manager.getBatchStates()
            if len(state_batch) > 0:
                p0 = np.array(state_batch, dtype=np.float32).reshape(-1, 1, 2)
            else:
                p0 = None
        else:
                p0 = None
                mask = np.zeros_like(img_np)  # Reset mask if no detections
        return kpts, p0, tracking_manager, output, mask

class YOLOPredictions:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        self.device = torch.device(device)
    def predict(self, image, tracking_manager=None, img_np=None, mask=None, conformal_threshold=None, kalman_process_noise=None, kalman_rcnn_noise=None, kalman_optical_flow_noise=None):
        # Perform prediction using the YOLO model
        # if not isinstance(image, torch.Tensor):
        #     image = Image.fromarray(image)
        #     image = torchvision.transforms.ToTensor()(image)
        # # display the image
        # image = image.to(self.device)
        # in_image = image.permute(1, 2, 0).cpu().numpy()
        output = self.model(image)[0]
        label = output.boxes.cls
        confidence = output.keypoints.conf
        keypoints = output.keypoints.xy
        if len(keypoints) > 0:
            predictions = None 
        else:
            predictions = [{
                "boxes": output.boxes.xyxy.cpu().numpy(),
                "labels": label.cpu().numpy(),
                "keypoints": keypoints.cpu().numpy(),
                "keypoints_scores": confidence.cpu().numpy()
            }]
        if len(keypoints) > 0:
            tracking_manager.softReset()
            if mask is not None:
                mask = cv.multiply(mask, 0.85)  # Fade trails by 15% each model refresh
            else:
                mask = np.zeros_like(img_np)
            
            for obj_idx in range(len(keypoints)):
                tracking_manager.updateInstanceCounter(obj_idx)  # Increment instance counter for this object
                for kpt_idx in range(len(keypoints[obj_idx])):
                    if float(sum(keypoints[obj_idx][kpt_idx])) != 0:
                        tracking_manager.addTrack(keypoints[obj_idx][kpt_idx], int(label[obj_idx]), kpt_idx, source="gt", confidence=confidence[obj_idx][kpt_idx], process_noise=kalman_process_noise, kpt_rcnn_noise=kalman_rcnn_noise, optical_flow_noise=kalman_optical_flow_noise)
            tracking_manager.mergeTracks()  # Merge temporary tracks into main tracking manager
            state_batch = tracking_manager.getBatchStates()
            if len(state_batch) > 0:
                p0 = np.array(state_batch, dtype=np.float32).reshape(-1, 1, 2)
            else:
                p0 = None
        else:
            p0 = None
            mask = np.zeros_like(img_np)  # Reset mask if no detections

        return keypoints, p0, tracking_manager, predictions, mask
class DatasetLoader:
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type.lower()
        self.num_frames = 0
        if self.data_type == 'nocs':
            self.loader, self.num_frames = FolderLoader(data_path, NOCS=True)
        elif self.data_type == 'bop':
            transforms_list = [transforms.ToTensor()]
            self.loader = RCNNTorch(
                gt_file=self.data_path,
                transform=transforms.Compose(transforms_list),
                target_transform=None,
                augment=False,
            )
            self.num_frames = self.loader.__len__()
        elif self.data_type == 'folder':
            self.loader, self.num_frames = FolderLoader(data_path, NOCS=False)
        elif self.data_type == "video":
            self.loader = cv.VideoCapture(data_path)

    def getNumFrames(self):
        return self.num_frames

    def load_data(self, i):
        target = None
        img_np = None
        image = None
        if self.data_type == 'nocs' or self.data_type == 'folder':
            image = self.loader[i]
            image = cv.imread(image)
            img_np = image.copy()
        elif self.data_type == 'bop':
            image, target = self.loader.__getitem__(i)
            if isinstance(image, torch.Tensor):
                img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
            else:
                img_np = image
        elif self.data_type == "video":
            ret, frame = self.loader.read()
            if ret:
                img_np = frame
                image = frame
            else:
                img_np = None
                image = None
        return image, target, img_np

    def canErrorCalculate(self):
        return self.data_type == 'bop'

def visualizePredictions(image, predictions=None, targets=None, optical_flow_points=None, frame_num=None, mask=None, show_trails=True, frame_refresh=None):
    """Visualize predictions and optical flow points."""
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Add the tracking trails if mask is provided
    if mask is not None and show_trails:
        image = cv.add(image, mask)

    if predictions is not None:
        for i in range(len(predictions[0]['boxes'])):
            box = predictions[0]['boxes'][i].cpu().numpy().astype(int)
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv.putText(image, f"Class: {predictions[0]['labels'][i].item()}", (box[0], box[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            keypoints = predictions[0]['keypoints'][i].cpu().numpy()
            for j in range(len(keypoints)):
                kp = keypoints[j].astype(int)
                cv.circle(image, (kp[0], kp[1]), 4, (0, 0, 255), -1)
                # get the keypoint score on each point
                score = predictions[0]['keypoints_scores'][i][j].item()
                cv.putText(image, f"{score:.2f}", (kp[0], kp[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    if targets is not None:
        for i in range(len(targets['boxes'])):
            box = targets['boxes'][i].cpu().numpy().astype(int)
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv.putText(image, f"Class: {targets['labels'][i].item()}", (box[0], box[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            keypoints = targets['keypoints'][i].cpu().numpy()
            for j in range(len(keypoints)):
                kp = keypoints[j].astype(int)
                cv.circle(image, (kp[0], kp[1]), 4, (0, 255, 0), -1)

    # Draw optical flow points with distinct colors
    if optical_flow_points is not None:
        for i, point in enumerate(optical_flow_points):
            # Handle both numpy arrays and tensors
            if hasattr(point, 'flatten'):
                point = point.flatten()
            x, y = int(point[0]), int(point[1])
            color = colors[i % len(colors)].tolist()
            cv.circle(image, (x, y), 5, color, -1)
            cv.putText(image, f"P{i}", (x+8, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add frame information with better styling
    if frame_num is not None:
        # Add black background for text readability
        cv.rectangle(image, (5, 5), (400, 80), (0, 0, 0), -1)
        cv.rectangle(image, (5, 5), (400, 80), (255, 255, 255), 2)
        
        if predictions is not None:
            cv.putText(image, f"Frame: {frame_num} (MODEL INFERENCE)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv.putText(image, f"Frame: {frame_num} (OPTICAL FLOW)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        tracked_count = len(optical_flow_points) if optical_flow_points is not None else 0
        cv.putText(image, f"Tracked Points: {tracked_count}", (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv.imshow("Hybrid Keypoint Tracking", image)
    return image

def run_hybrid_tracking(model_path=None, dataset_path=None, data_type=None, model_type=None, model_refresh_interval=5, frame_refresh=10, no_fpn=False, num_classes=9, num_keypoints=10, kalman_process_noise=1e-2, kalman_rcnn_noise=1e-2, kalman_optical_flow_noise=1e-4, conformal_threshold=0.08):
    """
    Run hybrid tracking with model inference every N frames and optical flow in between.

    Args:
        model_path (str): Path to the trained model.
        dataset_path (str): Path to the dataset JSON file.
        data_type (str): Type of dataset (e.g., 'NOCS', 'BOP', 'folder', 'video').
        model_type (str): Type of model to use (e.g., 'kpt_rcnn', 'yolo').
        frame_refresh (int): Frame refresh rate. Default is 10.
        model_refresh_interval (int): Interval for model refresh. Default is 5.
        no_fpn (bool): Whether to use FPN or not. Default is False.
        num_classes (int): Number of classes for detection. This is for Kpt-RCNN ONLY. Default is 9.
        num_keypoints (int): Number of keypoints to track. This is for Kpt-RCNN ONLY. Default is 10.
        kalman_process_noise (float): Process noise for Kalman filter. Default is 1e-2.
        kalman_rcnn_noise (float): R-CNN noise for Kalman filter. Default is 1e-2.
        kalman_optical_flow_noise (float): Optical flow noise for Kalman filter. Default is 1e-4.
        conformal_threshold (float): Conformal threshold for tracking. Default is 0.08.

    Returns:
        list: List containing tracking errors for each dataset if using BOP dataset
    """

    # Initialize error tracking
    hallucinate_counter = 0
    device = 'cpu'
    if data_type.lower() not in ['nocs', 'bop', 'folder', 'video']:
        print(f"Invalid data type specified: {data_type}")
        print(f"Available data types are: nocs, bop, folder, video")
        return
    dataset = DatasetLoader(dataset_path, data_type=data_type)
    num_frames_to_process = dataset.getNumFrames()
    
    if model_type.lower() == 'kpt_rcnn':
        model = KeypointRCNN_Model(model_path=model_path, device=device, no_fpn=no_fpn, num_classes=num_classes, num_keypoints=num_keypoints)
    elif model_type.lower() == 'yolo':
        model = YOLOPredictions(model_path=model_path)
    else:
        print("Invalid model type specified.")
        print(f"Available model types are: kpt_rcnn, yolo")
        return
    print("Model initialized successfully!")
    tracking_manager = PointTrackManager()
    print("Tracking manager initialized!")

    # Optical flow parameters
    lk_params = dict(winSize=(13, 13),
                    maxLevel=1,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
                    minEigThreshold=2.5e-4)

    # Initialize tracking variables
    old_gray = None
    p0 = None
    mask = None
    global colors
    colors = np.random.randint(0, 255, (100, 3))
    frame_count = 0
    distance_threshold = 0.4
    video_sequence_count = 0

    # Initialize video writer for output
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_video = cv.VideoWriter(f'hybrid_tracking_output_dataset.mp4', fourcc, 30.0, (640, 480))

    avg_time = []
    num_frames_processed = 0
    error_list = []  # Track errors for this dataset

    print("=== STARTING INFERENCE ===")
    print("Starting hybrid inference with model every {} frames and optical flow tracking...".format(model_refresh_interval))
    # print(f"Dataset size: {dataset.__len__()} frames")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    
    if frame_refresh:
        print(f"Frame refresh every {frame_refresh} frames")
    print("=== PROCESSING FRAMES ===")
    NOCS_image_path = None
    json_data = []
    j = 0
    while True:
        try:
            needs_refresh = False
            start = time.time()
            image, target, img_np = dataset.load_data(j)
            if image is None or img_np is None:
                print(f"Frame {j}: No valid image found.")
                print("Ending processing.")
                break
            # Check for frame refresh - start new video sequence
            if frame_refresh and j > 0 and j % frame_refresh == 0:
                print(f"\nFrame refresh at frame {j} - starting new video sequence {video_sequence_count + 1}")
                
                # Reset tracking state for new video sequence
                old_gray = None
                p0 = None
                mask = None
                tracking_manager.hardReset()
                video_sequence_count += 1
                
                # Optional: fade existing trails when starting new sequence
                if mask is not None:
                    mask = cv.multiply(mask, 0.5)  # Fade trails by 50% for new sequence
                needs_refresh = True
            frame_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
            # Run model inference every N frames or on first frame
            if frame_count % model_refresh_interval == 0 or old_gray is None or needs_refresh:
                print(f"Frame {frame_count}: Running model inference...")
                # Run model prediction
                kpts, p0, tracking_manager, output, mask = model.predict(image, tracking_manager=tracking_manager, img_np=img_np, mask=mask, conformal_threshold=conformal_threshold, kalman_process_noise=kalman_process_noise, kalman_rcnn_noise=kalman_rcnn_noise, kalman_optical_flow_noise=kalman_optical_flow_noise)
                if output is not None:
                    model_output = [output]
                else:
                    model_output = None
                optical_flow_points = None if p0 is None else p0.reshape(-1, 2)
            else:
                # Use optical flow tracking
                if p0 is not None and old_gray is not None:
                    # Add slight fade to trails for natural decay
                    mask = cv.multiply(mask, 0.98) if mask is not None else np.zeros_like(img_np)

                    # Calculate optical flow
                    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                    # Back-track to check quality
                    p0r, st1, err1 = cv.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)

                    # Calculate distance between original and back-tracked points
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good_mask = d < distance_threshold
                    good_mask = good_mask & st.flatten().astype(bool) & st1.flatten().astype(bool)

                    # Select good points
                    if good_mask.any():
                        good_new = p1[good_mask]
                        good_old = p0[good_mask]

                        # Update tracking manager with optical flow results
                        tracking_manager.updateTracks(good_mask)
                        active_tracks = tracking_manager.getActiveTracks()
                        for i, track_id in enumerate(active_tracks):
                            if i < len(good_new):
                                tracking_manager.updateTrack(track_id, good_new[i].flatten())
                        good_new = np.asarray(tracking_manager.getBatchStates())
                        # Draw tracking trails - use different colors for each point with thinner thickness
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = int(new.ravel()[0]), int(new.ravel()[1])
                            c, d = int(old.ravel()[0]), int(old.ravel()[1])
                            color = colors[i % len(colors)].tolist()
                            mask = cv.line(mask, (a, b), (c, d), color, 1)  # Reduced from 3 to 1
                            # Draw a smaller circle at the current position
                            mask = cv.circle(mask, (a, b), 2, color, -1)  # Reduced from 4 to 2

                        # Update points for next iteration
                        p0 = good_new.reshape(-1, 1, 2)
                        optical_flow_points = good_new
                    else:
                        optical_flow_points = None
                        print(f"Frame {frame_count}: Lost all tracking points")
                else:
                    optical_flow_points = None

                model_output = None

                # For optical flow frames, we can't calculate error since we don't run the model
                # Append NaN to maintain frame-wise indexing
                # error_list.append(np.nan)
            if dataset.canErrorCalculate():
                # Calculate error for this frame by matching using IDs
                if target is not None and 'keypoints' in target:
                    error_list, hallucinate_counter = errorCalculation(target, kpts, tracking_manager, frame_count, error_list, hallucinate_counter)
                else:
                    # No target data available
                    error_list.append(np.nan)
            # Visualize results with improved trail visualization
            if model_output is not None:
                # Model inference frame - show detections with existing trails
                img_display = visualizePredictions(image, predictions=model_output, targets=None, 
                                    optical_flow_points=optical_flow_points, frame_num=frame_count, 
                                    mask=mask, show_trails=True)
            else:
                # Optical flow frame - emphasize the tracking trails
                img_display = visualizePredictions(img_np, predictions=None, targets=None, 
                                    optical_flow_points=optical_flow_points, frame_num=frame_count, 
                                    mask=mask, show_trails=True)

            # Write frame to video output
            if img_display is not None:
                # Resize if needed to match video writer dimensions
                img_resized = cv.resize(img_display, (640, 480))
                output_video.write(img_resized)

            cv.waitKey(1)

            keypoints = np.zeros((43, 2)) # just mugs

            #now go through each point and match by index of the keypoints with the model
            active_tracks = tracking_manager.getActiveTracks()
            for i, track_id in enumerate(active_tracks):
                objid, instanceid, ptidx = map(int, track_id.split('_'))
                keypoints[ptidx] = tracking_manager.getTrack(track_id).getState()

            json_data.append({
                "est_pixel_keypoints": keypoints.tolist(),
                "idx": j
            })

            # Update for next iteration
            old_gray = frame_gray.copy()
            frame_count += 1
            if tracking_manager is not None:
                tracking_manager.step()

            end = time.time()
            avg_time.append(end - start)

            # Collect statistics only when model runs
            if model_output is not None:
                num_frames_processed += 1

            #check if ctrl + c is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            j += 1
            if j >= num_frames_to_process and num_frames_to_process != 0:
                break
        except KeyboardInterrupt:
            print("Inference interrupted by user.")
            break

    cv.destroyAllWindows()
    output_video.release()
    print(f"=== INFERENCE COMPLETE ===")
    print(f"Output video saved as: hybrid_tracking_output_dataset.mp4")
    print(f"Average inference time: {np.mean(avg_time):.4f} seconds")
    print(f"Average FPS: {1/np.mean(avg_time):.2f}")
    print(f"Model ran on {num_frames_processed} out of {num_frames_to_process} frames")
    print(f"Optical flow tracking used on {num_frames_to_process - num_frames_processed} frames")
    if len(error_list) != 0:
        valid_errors = [e for e in error_list if not np.isnan(e)]
        nan_count = sum(np.isnan(error_list))
        print(f"Average tracking error: {np.nanmean(error_list):.2f} pixels")
        print(f"Max tracking error: {np.nanmax(error_list):.2f} pixels")
        print(f"Valid error measurements: {len(valid_errors)} out of {len(error_list)} frames")
        print(f"Frames without error data (NaN): {nan_count}")
    json_data_path = os.path.join(f'{model_type}_kpt_json_data.json')
    with open(json_data_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON Keypoint saved to: {json_data_path}")

    return error_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid tracking with model inference and optical flow.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--data_type", type=str, default=None, help="Type of data (nocs, bop, folder, video).")
    parser.add_argument("--model_type", type=str, default=None, help="Type of model (kpt_rcnn, yolo).")
    parser.add_argument("--model_refresh_interval", type=int, default=10, help="Model refresh interval.")
    parser.add_argument("--frame_refresh", type=int, default=50, help="Frame refresh rate. This hard resets the flow every N frames")
    parser.add_argument("--no_fpn", action="store_true", default=None, help="Disable FPN.")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of object classes for Kpt-RCNN.")
    parser.add_argument("--num_keypoints", type=int, default=10, help="Number of keypoints for Kpt-RCNN.")
    parser.add_argument("--kalman_process_noise", type=float, default=1e-2, help="Kalman process noise.")
    parser.add_argument("--kalman_rcnn_noise", type=float, default=1e-2, help="Kalman R-CNN noise.")
    parser.add_argument("--kalman_optical_flow_noise", type=float, default=1e-4, help="Kalman optical flow noise.")
    parser.add_argument("--conformal_threshold", type=float, default=0.08, help="Conformal threshold.")
    args = parser.parse_args()

    errors = run_hybrid_tracking(
        model_path=args.model_path,
        model_type=args.model_type,
        data_type=args.data_type,
        dataset_path=args.dataset_path,
        model_refresh_interval=args.model_refresh_interval,
        frame_refresh=args.frame_refresh,
        num_classes=args.num_classes,
        num_keypoints=args.num_keypoints,
        no_fpn=args.no_fpn,
        conformal_threshold=args.conformal_threshold
    )
    if len(errors) > 0:
        # Plot error results like in optical-flow-test.py
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        plt.plot(errors, label=f'Dataset', marker='o', markersize=3)
        plt.axhline(np.nanmean(errors), color='red', linestyle='--', 
                    label=f'Avg Error Dataset: {np.nanmean(errors):.2f} PX')
        plt.title('Hybrid Tracking Error per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Error (pixels)')
        plt.legend()
        plt.grid(True)
        plt.savefig('hybrid_tracking_errors.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Error plot saved as: hybrid_tracking_errors.png")