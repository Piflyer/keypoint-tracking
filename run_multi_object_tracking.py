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
import datetime
import time

# Global error tracking dictionary
epoch_error = dict()

def pixel_to_world(kpts_pixel, cam_K, depth):
    """Convert pixel coordinates to world coordinates using depth information."""
    kpts_px = np.hstack([kpts_pixel, np.ones([kpts_pixel.shape[0], 1])])
    kpts_world = depth[:, np.newaxis] * (np.linalg.inv(cam_K) @ kpts_px.T).T
    return kpts_world

def FolderLoader(dataset_parent, NOCS=False, depth=False, meta=False):
    if meta:
        rgb_paths = natsort.natsorted(glob.glob(f"{dataset_parent}/*_meta.txt"))
    elif depth:
        rgb_paths = natsort.natsorted(glob.glob(f"{dataset_parent}/*_depth.png"))
    elif NOCS:
        rgb_paths = natsort.natsorted(glob.glob(f"{dataset_parent}/*_color.png"))
    else:
        rgb_paths = natsort.natsorted(glob.glob(f"{dataset_parent}/*.png") or glob.glob(f"{dataset_parent}/*.jpg"))
    dataset_len = len(rgb_paths)
    return rgb_paths, dataset_len

def errorCalculation(target, kpts, tracking_manager, frame_count, error_list, hallucinate_counter):
    """Calculate tracking error between target and predicted keypoints."""
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
            target_xy.append(target_kpts[j][:2])  # x, y coordinates
            hungarian_target_idx.setdefault(target_label, []).append(j)
            
            # Find corresponding prediction
            found_prediction = False
            for track_id in active_tracks[active_iter:]:
                objid, instanceid, ptidx = map(int, track_id.split('_'))
                if objid == target_label:
                    track = tracking_manager.getTrack(track_id)
                    if track is not None:
                        predicted_xy.append(track.getState())
                        hungarian_predicted_id.setdefault(target_label, []).append(len(predicted_xy) - 1)
                        active_iter += 1
                        found_prediction = True
                        break
            
            if not found_prediction:
                predicted_xy.append([0, 0])  # Default position if no prediction found
                
        # now do a check for any remaining unmatched points
        hallucinated_labels = []
        for track_id in active_tracks[active_iter:]:
            objid, instanceid, ptidx = map(int, track_id.split('_'))
            hallucinated_labels.append(objid)
            
        if len(hallucinated_labels) > 0:
            hallucinate_counter += len(hallucinated_labels)
            
        # Calculate euclidean distance for visible points
        distances = []
        
        for i in range(min(len(predicted_xy), len(target_xy))):
            distance = np.linalg.norm(np.array(predicted_xy[i]) - np.array(target_xy[i]))
            distances.append(distance)
            
        if distances:
            avg_distance = np.mean(distances)
            error_list.append(avg_distance)
        else:
            error_list.append(np.nan)
            
    else:
        # No predictions or targets available
        error_list.append(np.nan)
    return error_list, hallucinate_counter

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
            self.kalman_filter.statePre = np.array([initial_measurement[0], initial_measurement[1], 0, 0], dtype=np.float32).reshape(-1, 1)
            self.kalman_filter.statePost = np.array([initial_measurement[0], initial_measurement[1], 0, 0], dtype=np.float32).reshape(-1, 1)
        else:
            self.kalman_filter.statePre = np.zeros((4, 1), dtype=np.float32)
            self.kalman_filter.statePost = np.zeros((4, 1), dtype=np.float32)

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
            measurement = np.array(measurement, dtype=np.float32).reshape(-1, 1)
            self.kalman_filter.correct(measurement)

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

class MultiObjectTrackManager:
    """
    Enhanced tracking manager that handles multiple objects simultaneously.
    Maintains separate tracking for each object instance.
    """
    def __init__(self):
        """Initialize the MultiObjectTrackManager."""
        self.tracks = []
        self.active_tracks = []
        self.tempActiveTracks = []
        self.instance_counter = dict()
        self.tempTracks = []
        # Multi-object specific attributes
        self.object_tracks = dict()  # objid -> [track_ids]
        self.object_instance_counter = dict()  # objid -> instance_count
        
    def addTrack(self, xy, objid, ptidx, source="RCNN", confidence=1.0, process_noise=1e-2, kpt_rcnn_noise=1e-1, optical_flow_noise=1e-3):
        """Add a new point track for a specific object."""
        trackID = self.createID(objid, ptidx)
        
        # Check if the track already exists
        new_track = None
        isUniquetrack = True
        for track in self.tracks:
            if track.getID() == trackID:
                track.updatePoint(xy, source=source)
                isUniquetrack = False
                new_track = track
                
        if isUniquetrack:
            new_track = PointTrack(xy, trackId=trackID, confidence=confidence, 
                                 process_noise=process_noise, kpt_rcnn_noise=kpt_rcnn_noise, 
                                 optical_flow_noise=optical_flow_noise)
                                 
        self.tempTracks.append(new_track)
        self.active_tracks.append(trackID)
        
        # Multi-object tracking
        if objid not in self.object_tracks:
            self.object_tracks[objid] = []
        if trackID not in self.object_tracks[objid]:
            self.object_tracks[objid].append(trackID)

    def mergeTracks(self):
        """Merge temporary tracks into the main track list."""
        self.tracks = self.tempTracks
        self.tempTracks = []

    def updateTrack(self, track_id, xy, source="optical_flow", confidence=1.0):
        """Update the point track with the new coordinates."""
        for track in self.tracks:
            if track.getID() == track_id:
                track.updatePoint(xy, source=source, confidence=confidence)
                return True
        return False

    def updateInstanceCounter(self, objid):
        """Update the instance counter for the given object ID."""
        if objid not in self.instance_counter:
            self.instance_counter[objid] = 0
        self.instance_counter[objid] += 1
        
        if objid not in self.object_instance_counter:
            self.object_instance_counter[objid] = 0
        self.object_instance_counter[objid] = max(self.object_instance_counter[objid], 
                                                  self.instance_counter[objid])

    def getInstanceCounter(self, objid):
        """Get the instance counter for the given object ID."""
        try:
            return self.instance_counter[objid]
        except KeyError:
            return 0

    def createID(self, objid, ptidx):
        """Create a new ID for the point track."""
        if objid not in self.instance_counter:
            self.instance_counter[objid] = 0
        return f"{objid}_{self.instance_counter[objid]}_{ptidx}"

    def step(self):
        """Step through the point tracks and predict the next state."""
        output = []
        for track in self.tracks:
            if track.getID() in self.active_tracks:
                predicted_point = track.predictPoint()
                output.append(predicted_point)
        return output

    def getActiveTracks(self):
        """Get the active tracks."""
        return self.active_tracks

    def getObjectTracks(self, objid):
        """Get all track IDs for a specific object."""
        return self.object_tracks.get(objid, [])

    def getActiveObjectTracks(self, objid):
        """Get active track IDs for a specific object."""
        object_tracks = self.getObjectTracks(objid)
        return [track_id for track_id in object_tracks if track_id in self.active_tracks]

    def getTrack(self, track_id):
        """Get the point track with the given ID."""
        for track in self.tracks:
            if track.getID() == track_id:
                return track
        return None

    def updateTracks(self, update_list):
        """Update the active tracks with the new active tracks."""
        self.active_tracks = np.array(self.active_tracks)[update_list].tolist()
        self.tracks = np.array(self.tracks)[update_list].tolist()
        
        # Update object_tracks to remove inactive tracks
        for objid in self.object_tracks:
            self.object_tracks[objid] = [track_id for track_id in self.object_tracks[objid] 
                                       if track_id in self.active_tracks]
        
        # Recalculate instance counter
        self.instance_counter = dict()
        for j in self.active_tracks:
            objid, instanceid, ptidx = map(int, j.split('_'))
            self.instance_counter[objid] = max(instanceid, self.instance_counter.get(objid, 0))

    def softReset(self):
        """Perform a soft reset of the tracking manager."""
        self.active_tracks = []
        self.instance_counter = dict()

    def hardReset(self):
        """Perform a hard reset of the tracking manager."""
        self.tracks = []
        self.active_tracks = []
        self.instance_counter = dict()
        self.object_tracks = dict()
        self.object_instance_counter = dict()

    def getBatchStates(self):
        """Get the current states of all tracks in the batch."""
        return [track.getState() for track in self.tracks]
        
    def getObjectBatchStates(self, objid):
        """Get the current states of all tracks for a specific object."""
        object_track_ids = self.getActiveObjectTracks(objid)
        states = []
        for track_id in object_track_ids:
            track = self.getTrack(track_id)
            if track is not None:
                states.append(track.getState())
        return states

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
            if k.startswith("module."):
                name = k[7:]  # Remove 'module.' prefix
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        if self.no_fpn:
            backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).features
            backbone.out_channels = 960
            
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),) * 1
            )
            
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2
            )
            
            keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=14,
                sampling_ratio=2
            )
            
            self.model = KeypointRCNN(
                backbone, 
                num_classes=self.num_classes, 
                num_keypoints=self.num_keypoints,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                keypoint_roi_pool=keypoint_roi_pooler
            )
        else:
            backbone = mobilenet_backbone("mobilenet_v3_large", pretrained=True, trainable_layers=self.trainable_backbone_layers)
            
            anchor_generator = AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,), (512,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )
            
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
            
            keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2
            )
            
            self.model = KeypointRCNN(
                backbone, 
                num_classes=self.num_classes, 
                num_keypoints=self.num_keypoints,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                keypoint_roi_pool=keypoint_roi_pooler
            )
            
            num_classes_for_box = self.num_classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_for_box)
            
            in_features_keypoint = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
            self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_keypoint, self.num_keypoints)
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def predict(self, image, target=None, tracking_manager=None, img_np=None, mask=None, conformal_threshold=0.08, kalman_process_noise=None, kalman_rcnn_noise=None, kalman_optical_flow_noise=None, expected_objects=None):
        """Enhanced prediction method that handles multiple objects."""
        #check if image is a tensor
        if not isinstance(image, torch.Tensor):
            transform = transforms.Compose([transforms.ToTensor()])
            image = transform(image)
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
        boxes = output['boxes'].cpu().numpy()
        
        sorted_indices = np.argsort(scores)[::-1]  # Sort by confidence descending
        kpts = kpts[sorted_indices]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]
        boxes = boxes[sorted_indices]

        # Multi-object processing
        if len(kpts) > 0:
            # Process multiple detections
            processed_objects = self.process_multi_object_detections(
                kpts, scores, labels, boxes, tracking_manager, expected_objects,
                kalman_process_noise, kalman_rcnn_noise, kalman_optical_flow_noise
            )
            
            if mask is None:
                mask = np.zeros(img_np.shape, dtype=np.uint8)
            
            # Create points for optical flow from all objects
            p0_list = []
            for obj_data in processed_objects.values():
                if 'keypoints' in obj_data:
                    p0_list.extend(obj_data['keypoints'])
            
            p0 = np.array(p0_list).reshape(-1, 1, 2) if p0_list else None
            
            tracking_manager.mergeTracks()
            return processed_objects, p0, tracking_manager, output, mask
        else:
            return {}, None, tracking_manager, None, mask
        
    def process_multi_object_detections(self, kpts, scores, labels, boxes, tracking_manager, expected_objects, 
                                      kalman_process_noise, kalman_rcnn_noise, kalman_optical_flow_noise):
        """
        Process multiple object detections and associate them with existing tracks.
        
        Args:
            kpts: Detected keypoints
            scores: Detection confidence scores
            labels: Object class labels
            boxes: Bounding boxes
            tracking_manager: The tracking manager instance
            expected_objects: Dictionary specifying expected objects per scene/frame
            kalman_process_noise: Process noise for Kalman filter
            kalman_rcnn_noise: RCNN noise for Kalman filter
            kalman_optical_flow_noise: Optical flow noise for Kalman filter
            
        Returns:
            Dictionary containing processed object data
        """
        processed_objects = {}
        
        # Group detections by object class
        object_detections = {}
        for i, (kpt, score, label, box) in enumerate(zip(kpts, scores, labels, boxes)):
            if label not in object_detections:
                object_detections[label] = []
            object_detections[label].append({
                'keypoints': kpt,
                'score': score,
                'box': box,
                'detection_idx': i
            })
        
        # Process each object class
        for obj_class, detections in object_detections.items():
            expected_count = expected_objects.get(obj_class, len(detections)) if expected_objects else len(detections)
            
            if expected_count == 0:
                continue
                
            # Limit detections to expected count (keep highest confidence)
            detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:expected_count]
            
            processed_objects[obj_class] = []
            
            for instance_idx, detection in enumerate(detections):
                obj_data = {
                    'keypoints': [],
                    'confidence': detection['score'],
                    'box': detection['box'],
                    'instance_id': instance_idx
                }
                
                # Update tracking manager with keypoints
                tracking_manager.updateInstanceCounter(obj_class)
                
                for kpt_idx, kpt in enumerate(detection['keypoints']):
                    if len(kpt) >= 3 and kpt[2] > 0:  # Check visibility
                        xy = [kpt[0], kpt[1]]
                        tracking_manager.addTrack(
                            xy, obj_class, kpt_idx, source="RCNN", 
                            confidence=detection['score'],
                            process_noise=kalman_process_noise or 1e-2,
                            kpt_rcnn_noise=kalman_rcnn_noise or 1e-2,
                            optical_flow_noise=kalman_optical_flow_noise or 1e-4
                        )
                        obj_data['keypoints'].append(xy)
                
                processed_objects[obj_class].append(obj_data)
        
        return processed_objects

class YOLOPredictions:
    def __init__(self, model_path, device='cpu'):
        self.model = YOLO(model_path)
        self.device = torch.device(device)
        
    def predict(self, image, tracking_manager=None, img_np=None, mask=None, conformal_threshold=None, 
                kalman_process_noise=None, kalman_rcnn_noise=None, kalman_optical_flow_noise=None, expected_objects=None):
        """Enhanced YOLO prediction method that handles multiple objects."""
        # Perform prediction using the YOLO model
        results = self.model(img_np if img_np is not None else image)
        result = results[0]
        
        if not result.keypoints.has_visible:
            return {}, None, tracking_manager, None, mask
        
        kpts = result.keypoints.xy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(kpts))
        boxes = result.boxes.xywh.cpu().numpy() if result.boxes is not None else np.zeros((len(kpts), 4))
        
        # For YOLO, we assume single class for now, but can be extended
        labels = np.zeros(len(kpts), dtype=int)  # Default to class 0
        
        # Process multi-object detections
        processed_objects = self.process_multi_object_detections(
            kpts, scores, labels, boxes, tracking_manager, expected_objects,
            kalman_process_noise, kalman_rcnn_noise, kalman_optical_flow_noise
        )
        
        if mask is None:
            mask = np.zeros(img_np.shape, dtype=np.uint8)
        
        # Create points for optical flow from all objects
        p0_list = []
        for obj_data in processed_objects.values():
            if isinstance(obj_data, list):
                for instance in obj_data:
                    if 'keypoints' in instance:
                        p0_list.extend(instance['keypoints'])
            elif 'keypoints' in obj_data:
                p0_list.extend(obj_data['keypoints'])
        
        p0 = np.array(p0_list).reshape(-1, 1, 2) if p0_list else None
        
        tracking_manager.mergeTracks()
        return processed_objects, p0, tracking_manager, result, mask
    
    def process_multi_object_detections(self, kpts, scores, labels, boxes, tracking_manager, expected_objects,
                                      kalman_process_noise, kalman_rcnn_noise, kalman_optical_flow_noise):
        """Process multiple object detections from YOLO."""
        processed_objects = {}
        
        # For YOLO, assume single object class for now
        obj_class = 0
        expected_count = expected_objects.get(obj_class, len(kpts)) if expected_objects else len(kpts)
        
        if expected_count == 0:
            return processed_objects
            
        # Limit detections to expected count (keep highest confidence)
        detection_indices = np.argsort(scores)[::-1][:expected_count]
        
        processed_objects[obj_class] = []
        
        for instance_idx, det_idx in enumerate(detection_indices):
            obj_data = {
                'keypoints': [],
                'confidence': scores[det_idx],
                'box': boxes[det_idx],
                'instance_id': instance_idx
            }
            
            # Update tracking manager with keypoints
            tracking_manager.updateInstanceCounter(obj_class)
            
            detection_keypoints = kpts[det_idx]
            for kpt_idx, kpt in enumerate(detection_keypoints):
                if len(kpt) >= 2 and (kpt[0] > 0 or kpt[1] > 0):  # Check if valid keypoint
                    xy = [kpt[0], kpt[1]]
                    tracking_manager.addTrack(
                        xy, obj_class, kpt_idx, source="RCNN",
                        confidence=scores[det_idx],
                        process_noise=kalman_process_noise or 1e-2,
                        kpt_rcnn_noise=kalman_rcnn_noise or 1e-2,
                        optical_flow_noise=kalman_optical_flow_noise or 1e-4
                    )
                    obj_data['keypoints'].append(xy)
            
            processed_objects[obj_class].append(obj_data)
        
        return processed_objects

class DatasetLoader:
    def __init__(self, data_path, data_type, depth=False, gt_path=None, scene_id=None):
        """
        Initialize dataset loader for various data types.
        
        Args:
            data_path: Path to the dataset
            data_type: Type of dataset ('nocs', 'bop', 'folder', 'video')
            depth: Whether to load depth data
            gt_path: Path to ground truth data
            scene_id: Scene identifier for NOCS dataset
        """
        self.data_path = data_path
        self.data_type = data_type.lower()
        self.depth = depth
        self.gt_path = gt_path
        self.scene_id = scene_id
        self.dataset_len = 0
        
        if self.data_type == 'folder':
            self.rgb_paths, self.dataset_len = FolderLoader(data_path, NOCS=False)
        elif self.data_type == 'nocs':
            self.rgb_paths, self.dataset_len = FolderLoader(data_path, NOCS=True)
            if depth:
                self.depth_paths, _ = FolderLoader(data_path, depth=True)
                self.meta_paths, _ = FolderLoader(data_path, meta=True)
                if gt_path and scene_id:
                    self.gt_paths = sorted(glob.glob(f"{gt_path}/results_real_test_{scene_id}_*.pkl"))
        elif self.data_type == 'video':
            self.cap = cv.VideoCapture(data_path)
            self.dataset_len = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        
    def getNumFrames(self):
        return self.dataset_len

    def load_data(self, i):
        """Load data for frame i."""
        if self.data_type == 'folder':
            if i >= len(self.rgb_paths):
                return None, None, None, None
            image_path = self.rgb_paths[i]
            image = Image.open(image_path).convert('RGB')
            img_np = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
            filename = os.path.basename(image_path)
            return image, None, img_np, filename
            
        elif self.data_type == 'nocs':
            if i >= len(self.rgb_paths):
                return None, None, None, None
            image_path = self.rgb_paths[i]
            image = Image.open(image_path).convert('RGB')
            img_np = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
            filename = os.path.basename(image_path)
            
            if self.depth:
                depth_data = None
                gt_data = None
                meta_data = None
                
                if hasattr(self, 'depth_paths') and i < len(self.depth_paths):
                    depth_image = Image.open(self.depth_paths[i])
                    depth_data = np.array(depth_image, dtype=float)
                
                if hasattr(self, 'gt_paths') and i < len(self.gt_paths):
                    gt_data = np.load(self.gt_paths[i], allow_pickle=True)['gt_RTs']
                
                if hasattr(self, 'meta_paths') and i < len(self.meta_paths):
                    meta_data = self.meta_paths[i]
                
                return depth_data, gt_data, meta_data, filename
            
            return image, None, img_np, filename
            
        elif self.data_type == 'video':
            ret, frame = self.cap.read()
            if not ret:
                return None, None, None, None
            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            return image, None, frame, f"frame_{i:06d}"
        
        return None, None, None, None

    def canErrorCalculate(self):
        return False  # Simplified for now

def visualizePredictions(image, predictions=None, targets=None, optical_flow_points=None, frame_num=None, mask=None, show_trails=True, frame_refresh=None):
    """Enhanced visualization for multi-object tracking."""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Add the tracking trails if mask is provided
    if mask is not None and show_trails:
        image = cv.add(image, mask)

    if predictions is not None:
        # Handle multi-object predictions
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for obj_class, obj_instances in predictions.items():
            color = colors[obj_class % len(colors)]
            if isinstance(obj_instances, list):
                for instance in obj_instances:
                    if 'keypoints' in instance:
                        for kpt in instance['keypoints']:
                            cv.circle(image, (int(kpt[0]), int(kpt[1])), 3, color, -1)
            elif 'keypoints' in obj_instances:
                for kpt in obj_instances['keypoints']:
                    cv.circle(image, (int(kpt[0]), int(kpt[1])), 3, color, -1)
    
    if targets is not None:
        target_kpts = targets['keypoints'].cpu().numpy()
        for kpt in target_kpts:
            if len(kpt) >= 3 and kpt[2] > 0:
                cv.circle(image, (int(kpt[0]), int(kpt[1])), 5, (0, 255, 255), 2)

    # Draw optical flow points with distinct colors
    if optical_flow_points is not None:
        # Use local colors or initialize if needed
        if 'colors' not in locals():
            colors = np.random.randint(0, 255, (100, 3))
        for i, point in enumerate(optical_flow_points):
            color = colors[i % len(colors)].tolist()
            cv.circle(image, (int(point[0]), int(point[1])), 2, color, -1)

    # Add frame information with better styling
    if frame_num is not None:
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = f"Frame: {frame_num}"
        text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
        text_x = image.shape[1] - text_size[0] - 10
        text_y = 30
        cv.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                    (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    cv.imshow("Multi-Object Hybrid Keypoint Tracking", image)
    return image

def run_multi_object_hybrid_tracking(model_path=None, dataset_path=None, data_type=None, model_type=None, 
                                    model_refresh_interval=5, frame_refresh=10, no_fpn=False, num_classes=9, 
                                    num_keypoints=10, kalman_process_noise=1e-2, kalman_rcnn_noise=1e-2, 
                                    kalman_optical_flow_noise=1e-4, conformal_threshold=0.08, keypoint_path=None, 
                                    enable_visualization=False, NOCS_depth=False, output_name="multi_tracking", 
                                    NOCS_gt_path=None, NOCS_OBJ_ID=None, expected_objects=None, 
                                    save_per_object=True, cam_K=None):
    """
    Run hybrid tracking with support for multiple objects simultaneously.
    Combines the tracking framework from run_tracking with multi-object logic from multiyolo.

    Args:
        model_path (str): Path to the trained model.
        dataset_path (str): Path to the dataset.
        data_type (str): Type of dataset (e.g., 'NOCS', 'BOP', 'folder', 'video').
        model_type (str): Type of model to use (e.g., 'kpt_rcnn', 'yolo').
        model_refresh_interval (int): Interval for model refresh. Default is 5.
        frame_refresh (int): Frame refresh rate. Default is 10.
        no_fpn (bool): Whether to use FPN or not. Default is False.
        num_classes (int): Number of classes for detection. Default is 9.
        num_keypoints (int): Number of keypoints to track. Default is 10.
        kalman_process_noise (float): Process noise for Kalman filter. Default is 1e-2.
        kalman_rcnn_noise (float): R-CNN noise for Kalman filter. Default is 1e-2.
        kalman_optical_flow_noise (float): Optical flow noise for Kalman filter. Default is 1e-4.
        conformal_threshold (float): Conformal threshold for tracking. Default is 0.08.
        keypoint_path (str): Path to the keypoint JSON file. Default is None.
        enable_visualization (bool): Whether to enable visualization. Default is False.
        NOCS_depth (bool): Whether to use NOCS depth processing. Default is False.
        output_name (str): Output file name prefix. Default is "multi_tracking".
        NOCS_gt_path (str): Path to NOCS ground truth data. Default is None.
        NOCS_OBJ_ID (int): Object ID for NOCS dataset. Default is None.
        expected_objects (dict): Dictionary specifying expected objects per frame/scene.
        save_per_object (bool): Whether to save results per object. Default is True.
        cam_K (np.array): Camera intrinsic matrix. Default is None (uses NOCS default).

    Returns:
        dict: Dictionary containing tracking results per object
    """
    # Initialize camera parameters
    if cam_K is None:
        # Default NOCS camera parameters
        cam_K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    
    NOCS_scene = None
    if data_type.lower() == 'nocs' and dataset_path is not None:
        NOCS_scene = os.path.basename(os.path.normpath(dataset_path))
        print(f"NOCS scene detected: {NOCS_scene}")
    
    model_name = os.path.basename(model_path) if model_path is not None else "No model path provided"
    model_name = model_name.replace(".pt", "")
    print(f"Model: {model_name}")
    
    # Initialize error tracking
    hallucinate_counter = 0
    device = 'cpu'
    
    if data_type.lower() not in ['nocs', 'bop', 'folder', 'video']:
        print(f"Invalid data type specified: {data_type}")
        print(f"Available data types are: nocs, bop, folder, video")
        return {}
    
    # Initialize dataset
    dataset = DatasetLoader(dataset_path, data_type=data_type)
    depth_dataset = None
    if NOCS_depth and data_type.lower() == 'nocs':
        depth_dataset = DatasetLoader(dataset_path, data_type=data_type, depth=True, 
                                    gt_path=NOCS_gt_path, scene_id=NOCS_scene)
        print("Depth dataset loaded for NOCS depth processing.")
    
    num_frames_to_process = dataset.getNumFrames()
    
    # Initialize model
    if model_type.lower() == 'kpt_rcnn':
        model = KeypointRCNN_Model(model_path=model_path, device=device, no_fpn=no_fpn, 
                                 num_classes=num_classes, num_keypoints=num_keypoints)
    elif model_type.lower() == 'yolo':
        model = YOLOPredictions(model_path=model_path)
    else:
        print("Invalid model type specified.")
        print(f"Available model types are: kpt_rcnn, yolo")
        return {}
    
    print("Model initialized successfully!")
    
    # Initialize multi-object tracking manager
    tracking_manager = MultiObjectTrackManager()
    print("Multi-object tracking manager initialized!")

    # Optical flow parameters
    lk_params = dict(winSize=(13, 13),
                    maxLevel=1,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
                    minEigThreshold=2.5e-4)

    # Initialize tracking variables
    old_gray = None
    p0 = None
    mask = None
    # Initialize colors for visualization
    colors = np.random.randint(0, 255, (100, 3))
    frame_count = 0
    distance_threshold = 0.4
    video_sequence_count = 0

    # Initialize video writer for output
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_video = cv.VideoWriter(f'{output_name}_output.mp4', fourcc, 30.0, (640, 480))

    avg_time = []
    num_frames_processed = 0
    error_list = []

    print("=== STARTING MULTI-OBJECT INFERENCE ===")
    print("Starting hybrid inference with model every {} frames and optical flow tracking...".format(model_refresh_interval))
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Expected objects: {expected_objects}")
    
    if frame_refresh:
        print(f"Frame refresh every {frame_refresh} frames")
    
    print("=== PROCESSING FRAMES ===")
    
    # Multi-object results storage
    save_results = {}
    scene_objs = {}
    failure_count = 0
    objimg_count = 0
    
    j = 0
    while True:
        try:
            needs_refresh = False
            start = time.time()
            image, target, img_np, file_name = dataset.load_data(j)
            
            if image is None or img_np is None:
                print(f"Frame {j}: No valid image found.")
                print("Ending processing.")
                break

            # Check for frame refresh - start new video sequence
            if frame_refresh != 0:
                if frame_refresh and j > 0 and j % frame_refresh == 0:
                    print(f"\nFrame refresh at frame {j} - starting new video sequence {video_sequence_count + 1}")
                    
                    # Reset tracking state for new video sequence
                    old_gray = None
                    p0 = None
                    mask = None
                    tracking_manager.hardReset()
                    video_sequence_count += 1
                    needs_refresh = True

            frame_gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)

            # Run model inference every N frames or on first frame
            if frame_count % model_refresh_interval == 0 or old_gray is None or needs_refresh:
                print(f"Frame {frame_count}: Running model inference...")
                
                # Run model prediction with multi-object support
                processed_objects, p0, tracking_manager, output, mask = model.predict(
                    image, tracking_manager=tracking_manager, img_np=img_np, mask=mask, 
                    conformal_threshold=conformal_threshold, 
                    kalman_process_noise=kalman_process_noise,
                    kalman_rcnn_noise=kalman_rcnn_noise, 
                    kalman_optical_flow_noise=kalman_optical_flow_noise,
                    expected_objects=expected_objects
                )
                
                model_output = processed_objects if processed_objects else None
                optical_flow_points = None if p0 is None else p0.reshape(-1, 2)
                
                # Process multi-object results
                if processed_objects:
                    process_multi_object_results(processed_objects, j, file_name, 
                                                save_results, scene_objs, cam_K,
                                                img_np, NOCS_depth, depth_dataset, 
                                                NOCS_OBJ_ID, expected_objects, NOCS_scene)
                    
            else:
                # Use optical flow tracking for multiple objects
                if p0 is not None and old_gray is not None:
                    mask = cv.multiply(mask, 0.98) if mask is not None else np.zeros_like(img_np)

                    # Calculate optical flow
                    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                    # Back-track to check quality
                    p0r, st1, err1 = cv.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)

                    # Calculate distance between original and back-tracked points
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good_mask = d < distance_threshold
                    good_mask = good_mask & st.flatten().astype(bool) & st1.flatten().astype(bool)

                    # Select good points and update multi-object tracking
                    if good_mask.any():
                        good_new = p1[good_mask]
                        good_old = p0[good_mask]

                        # Update tracking manager with optical flow results
                        update_multi_object_optical_flow(tracking_manager, good_mask, good_new)
                        
                        # Get updated states from all objects for next iteration
                        all_states = tracking_manager.getBatchStates()
                        if all_states:
                            # Use the Kalman filter states as the new positions
                            updated_positions = np.array(all_states)
                            # Only draw trails if we have matching numbers of old and new points
                            if len(updated_positions) == len(good_old):
                                draw_multi_object_trails(mask, updated_positions, good_old, tracking_manager)
                            p0 = updated_positions.reshape(-1, 1, 2)
                            optical_flow_points = updated_positions
                        else:
                            # Fallback to optical flow positions 
                            draw_multi_object_trails(mask, good_new, good_old, tracking_manager)
                            p0 = good_new.reshape(-1, 1, 2)
                            optical_flow_points = good_new
                    else:
                        optical_flow_points = None
                        print(f"Frame {frame_count}: Lost all tracking points")
                else:
                    optical_flow_points = None

                model_output = None

            # Visualize results
            img_display = None
            if enable_visualization:
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
                img_resized = cv.resize(img_display, (640, 480))
                output_video.write(img_resized)
            
            # Handle keyboard input
            if enable_visualization and cv.waitKey(1) & 0xFF == ord('q'):
                break

            end = time.time()
            elapsed = end - start

            # Update for next iteration
            old_gray = frame_gray.copy()
            frame_count += 1
            
            if tracking_manager is not None:
                tracking_manager.step()

            avg_time.append(end - start)

            # Collect statistics only when model runs
            if model_output is not None:
                num_frames_processed += 1

            j += 1
            if j >= num_frames_to_process and num_frames_to_process != 0:
                break
                
        except KeyboardInterrupt:
            print("Inference interrupted by user.")
            break

    cv.destroyAllWindows()
    output_video.release()
    
    print(f"=== MULTI-OBJECT INFERENCE COMPLETE ===")
    print(f"Output video saved as: {output_name}_output.mp4")
    print(f"Average inference time: {np.mean(avg_time):.4f} seconds")
    print(f"Average FPS: {1/np.mean(avg_time):.2f}")
    print(f"Model ran on {num_frames_processed} out of {num_frames_to_process} frames")
    print(f"Optical flow tracking used on {num_frames_to_process - num_frames_processed} frames")

    # Save results per object if requested
    if save_per_object and save_results:
        save_multi_object_results(save_results, output_name)

    return save_results

def process_multi_object_results(processed_objects, frame_idx, file_name, save_results, scene_objs, 
                               cam_K, img_np, NOCS_depth, depth_dataset, NOCS_OBJ_ID, expected_objects, scene_name=None):
    """Process and store results for multiple objects."""
    # Use provided scene_name or default to frame-based grouping
    if scene_name is None:
        scene_name = f"scene_{frame_idx // 100}"  # Group frames into scenes
    
    if scene_name not in save_results:
        save_results[scene_name] = {}
        scene_objs[scene_name] = []

    for obj_class, instances in processed_objects.items():
        if not isinstance(instances, list):
            instances = [instances]
            
        for instance_idx, instance in enumerate(instances):
            obj_name = f"obj_{obj_class}_{instance_idx}"
            
            if obj_name not in scene_objs[scene_name]:
                scene_objs[scene_name].append(obj_name)
                save_results[scene_name][obj_name] = []

            # Prepare result data
            res_cur = {
                'est_pixel_keypoints': instance.get('keypoints', []),
                'rgb_image_filename': file_name,
                'frame_idx': frame_idx,
                'confidence': instance.get('confidence', 0.0),
                'obj_class': obj_class,
                'instance_id': instance_idx
            }

            # Add depth and world coordinates if available
            if NOCS_depth and depth_dataset is not None:
                depth_data, gt_data, meta_data, _ = depth_dataset.load_data(frame_idx)
                if depth_data is not None and len(instance.get('keypoints', [])) > 0:
                    kpts_pixel = np.array(instance['keypoints'])
                    kpts_pixel_int = np.rint(kpts_pixel).astype(int)
                    
                    # Clamp to image bounds
                    kpts_pixel_int[:,0] = np.clip(kpts_pixel_int[:,0], 0, depth_data.shape[1]-1)
                    kpts_pixel_int[:,1] = np.clip(kpts_pixel_int[:,1], 0, depth_data.shape[0]-1)
                    
                    depth_values = depth_data[kpts_pixel_int[:,1], kpts_pixel_int[:,0]] / 1000.0  # mm to m
                    world_kpts = pixel_to_world(kpts_pixel, cam_K, depth_values)
                    
                    res_cur['est_world_keypoints'] = world_kpts.tolist()

                # Add ground truth pose if available
                if gt_data is not None and meta_data is not None:
                    try:
                        with open(meta_data) as f:
                            for line in f:
                                info = line.split()
                                try:
                                    obj_class_in_meta = int(info[1])
                                    # Process GT for current object class or all objects if NOCS_OBJ_ID not specified
                                    if NOCS_OBJ_ID is None or obj_class_in_meta == NOCS_OBJ_ID or obj_class_in_meta == obj_class:
                                        idx = int(info[0]) - 1
                                        if idx < gt_data.shape[0]:
                                            RT_gt = gt_data[idx, :, :]
                                            # Normalize rotation matrix (NOCS issue)
                                            RT_gt[:,0] = RT_gt[:,0] / np.linalg.norm(RT_gt[:,0])
                                            RT_gt[:,1] = RT_gt[:,1] / np.linalg.norm(RT_gt[:,1])
                                            RT_gt[:,2] = RT_gt[:,2] / np.linalg.norm(RT_gt[:,2])
                                            res_cur['gt_pose'] = RT_gt.tolist()
                                            res_cur['obj_name'] = info[2] if len(info) > 2 else f"obj_{obj_class}"
                                            break  # Found the GT for this object
                                except (ValueError, IndexError) as e:
                                    continue
                    except Exception as e:
                        print(f"Error processing ground truth for frame {frame_idx}: {e}")

            save_results[scene_name][obj_name].append(res_cur)

def update_multi_object_optical_flow(tracking_manager, good_mask, good_new):
    """Update optical flow for multiple objects."""
    # Instead of updating tracks list directly, just update positions
    active_tracks = tracking_manager.getActiveTracks()
    
    # Only update tracks that have corresponding optical flow updates
    num_updates = min(len(active_tracks), len(good_new))
    
    for i in range(num_updates):
        track_id = active_tracks[i]
        if i < len(good_new):
            tracking_manager.updateTrack(track_id, good_new[i].flatten(), source="optical_flow")

def draw_multi_object_trails(mask, good_new, good_old, tracking_manager):
    """Draw tracking trails with object-specific colors."""
    active_tracks = tracking_manager.getActiveTracks()
    object_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255)}
    
    # Ensure good_new and good_old have compatible shapes
    min_length = min(len(good_new), len(good_old))
    
    for i in range(min_length):
        if i < len(active_tracks):
            track_id = active_tracks[i]
            objid = int(track_id.split('_')[0])
            color = object_colors.get(objid, (255, 255, 255))
            
            # Handle different array shapes - flatten to get x,y coordinates
            new_point = good_new[i]
            old_point = good_old[i]
            
            if len(new_point.shape) > 1:
                new_point = new_point.flatten()
            if len(old_point.shape) > 1:
                old_point = old_point.flatten()
                
            a, b = int(new_point[0]), int(new_point[1])
            c, d = int(old_point[0]), int(old_point[1])
            mask = cv.line(mask, (a, b), (c, d), color, 1)
            mask = cv.circle(mask, (a, b), 2, color, -1)

def save_multi_object_results(save_results, output_name):
    """Save results in the same format as multiyolo - per object."""
    print("Saving multi-object results...")
    os.makedirs(f'runs/{output_name}', exist_ok=True)
    
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    for scene in save_results:
        scene_results = save_results[scene]
        for obj in scene_results:
            obj_results = scene_results[obj]
            # Convert to serializable format
            serializable_results = convert_to_serializable(obj_results)
            data = json.dumps(serializable_results, indent=2)
            filename = f"runs/{output_name}/{scene}-{obj}.json"
            with open(filename, 'w') as f:
                f.write(data)
    
    print(f"Multi-object results saved to: runs/{output_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-object hybrid tracking.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--data_type", type=str, default=None, help="Type of data (nocs, bop, folder, video).")
    parser.add_argument("--model_type", type=str, default=None, help="Type of model (kpt_rcnn, yolo).")
    parser.add_argument("--model_refresh_interval", type=int, default=10, help="Model refresh interval.")
    parser.add_argument("--frame_refresh", type=int, default=50, help="Frame refresh rate.")
    parser.add_argument("--no_fpn", action="store_true", default=None, help="Disable FPN.")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of object classes.")
    parser.add_argument("--num_keypoints", type=int, default=10, help="Number of keypoints.")
    parser.add_argument("--kalman_process_noise", type=float, default=1e-2, help="Kalman process noise.")
    parser.add_argument("--kalman_rcnn_noise", type=float, default=1e-2, help="Kalman R-CNN noise.")
    parser.add_argument("--kalman_optical_flow_noise", type=float, default=1e-4, help="Kalman optical flow noise.")
    parser.add_argument("--conformal_threshold", type=float, default=0.08, help="Conformal threshold.")
    parser.add_argument("--keypoint_path", type=str, default=None, help="Path to the keypoint JSON file.")
    parser.add_argument("--enable_visualization", action="store_true", help="Enable visualization display.")
    parser.add_argument("--NOCS_depth", action="store_true", help="Enable NOCS depth processing.")
    parser.add_argument("--output_name", type=str, default="multi_tracking", help="Output file name.")
    parser.add_argument("--NOCS_gt_path", type=str, default=None, help="Path to NOCS ground truth.")
    parser.add_argument("--NOCS_OBJ_ID", type=int, default=None, help="Object ID for NOCS dataset.")
    parser.add_argument("--expected_objects", type=str, default=None, 
                       help="JSON string or path to file specifying expected objects per scene/frame.")
    parser.add_argument("--save_per_object", action="store_true", default=True, 
                       help="Save results per object like multiyolo.")

    args = parser.parse_args()

    # Parse expected objects
    expected_objects = None
    if args.expected_objects:
        try:
            if os.path.exists(args.expected_objects):
                with open(args.expected_objects, 'r') as f:
                    expected_objects = json.load(f)
            else:
                expected_objects = json.loads(args.expected_objects)
        except Exception as e:
            print(f"Error parsing expected_objects: {e}")
            print("Using default single object expectation")
            expected_objects = {0: 1}  # Default to single object of class 0

    results = run_multi_object_hybrid_tracking(
        model_path=args.model_path,
        model_type=args.model_type,
        data_type=args.data_type,
        dataset_path=args.dataset_path,
        model_refresh_interval=args.model_refresh_interval,
        frame_refresh=args.frame_refresh,
        num_classes=args.num_classes,
        num_keypoints=args.num_keypoints,
        no_fpn=args.no_fpn,
        conformal_threshold=args.conformal_threshold,
        keypoint_path=args.keypoint_path,
        enable_visualization=args.enable_visualization,
        NOCS_depth=args.NOCS_depth,
        output_name=args.output_name,
        NOCS_gt_path=args.NOCS_gt_path,
        NOCS_OBJ_ID=args.NOCS_OBJ_ID,
        expected_objects=expected_objects,
        save_per_object=args.save_per_object
    )

    if results:
        print(f"Multi-object tracking completed successfully!")
        print(f"Processed {len(results)} scenes with multiple objects")
    else:
        print("No results generated.")