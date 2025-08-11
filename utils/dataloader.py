import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
import random
from PIL import Image
import torch
import glob
import sys

class BaseDataset():
    def __init__(self, root_dir, model_list=None, kpts3d_path=None):
        if isinstance(root_dir, str):
            if '*' in root_dir:
                self.root_dir = glob.glob(root_dir)
            else:
                self.root_dir = [root_dir]
        else:
            self.root_dir = root_dir
        self.model_list = model_list
        self.scene_gt_info = {}
        self.camera_info = {}
        self.scene_gt_info = {}
        self.kpts3d_path = kpts3d_path
        self.kpts3D = {}
        self.kpts3d_range = {}
        self.dataset = {}
        self.total_kpts = 0
        self.scene_gt = {}
        self.static_obj = False
        self.static_obj_id = 1
    
    def load_kpts3d(self):
        counter = 0
        for obj_name in self.model_list:
            file = os.path.join(self.kpts3d_path, f"obj_{obj_name:06}.csv")
            if not os.path.exists(file):
                print(f"File {file} does not exist. Skipping object {obj_name}.")
                self.kpts3D[obj_name] = np.zeros((3, 0))
                self.kpts3d_range[obj_name] = (0, 0)
            else:
                print(f"Loading keypoints for object {obj_name} from {file}")
                try:
                    points = np.loadtxt(file, delimiter=",", dtype=np.float32)
                    points = points.reshape(-1, 3)
                    self.kpts3D[obj_name] = points
                    self.kpts3d_range[obj_name] = (counter, counter + points.shape[0])
                    counter += points.shape[0]
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    self.kpts3D[obj_name] = np.zeros((3, 0))  # Empty array if loading fails
        self.total_kpts = counter

    def load_scene_gt_info(self):
        for root in self.root_dir:
            try: 
                with open(os.path.join(root, 'scene_gt_info.json'), 'r') as f:
                    scene_gt_info = json.load(f)
            except FileNotFoundError:
                print(f"scene_gt_info.json not found in {root}.")
            self.scene_gt_info[root] = scene_gt_info
    
    def load_camera_info(self):
        for root in self.root_dir:
            try:
                with open(os.path.join(root, 'scene_camera.json'), 'r') as f:
                    camera_info = json.load(f)
            except FileNotFoundError:
                print(f"camera.json not found in {root}.")
            self.camera_info[root] = camera_info

    def load_scene_gt(self):
        for root in self.root_dir:
            try:
                with open(os.path.join(root, 'scene_gt.json'), 'r') as f:
                    scene_gt = json.load(f)
            except FileNotFoundError:
                print(f"scene_gt.json not found in {root}.")
            self.scene_gt[root] = scene_gt
    
    def _load_all_data(self):
        if not self.kpts3d_path:
            raise ValueError("kpts3d_path must be specified to load keypoints.")
        if not self.model_list:
            raise ValueError("model_list must be specified to load models.")
        self.load_kpts3d()
        self.load_scene_gt_info()
        self.load_camera_info()
        self.load_scene_gt()
    
    def processImage(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

class RCNNDataset(BaseDataset):
    def __init__(self, root_dir, model_list=None, kpts3d_path=None):
        super().__init__(root_dir, model_list, kpts3d_path)
        self.dataset = {}
        self.total_kpts = 0
        self.kpts3D = {}
        self.kpts3d_range = {}
        if model_list:
            self.model_id_to_label = {model_id: i for i, model_id in enumerate(self.model_list)}


    def processImage(self, output_path = None, bbox_threshold=0.3, val_split=0.2, seed=42, occulusion_threshold=3, shuffle=True, skip_empty=True, static_obj=False, static_obj_id=1):
        self.static_obj = static_obj
        self.static_obj_id = static_obj_id
        random.seed(seed)
        has_gt_info = True
        if output_path is None:
            raise ValueError("output_path must be specified to save the processed dataset.")
        if not self.camera_info:
            print("No camera_info loaded. Please load the dataset first.")
            return
        if not self.scene_gt_info:
            print("No scene_gt_info loaded. Attempting to calculate bounding boxes manually...")
            has_gt_info = False
        print("Processing images and generating bounding boxes...")
        train_counter = -1
        val_counter = -1
        train_labels = {}
        val_labels = {}
        for root in tqdm(self.root_dir):
            print(f"Processing root directory: {root}")
            if has_gt_info:
                root_scene_gt_info = self.scene_gt_info[root]
            root_scene_gt = self.scene_gt[root]
            root_camera_info = self.camera_info[root]
            scene_ids = list(root_scene_gt.keys())
            if shuffle:
                random.shuffle(scene_ids)
            val_bucket = int(len(scene_ids) * val_split)
            train_bucket = len(scene_ids) - val_bucket
            train_flag = True
            
            for scene_id in tqdm(scene_ids):
                train_bucket -= 1
                if has_gt_info:
                    if scene_id not in root_scene_gt_info:
                        print(f"scene_gt_info missing for scene {scene_id}, skipping...")
                        continue
                    scene_gt_info = root_scene_gt_info[scene_id]
                scene_gt_rt = root_scene_gt[scene_id]
                camera_info = root_camera_info[scene_id]
                K = np.array(camera_info['cam_K']).reshape(3, 3)
                image_path = os.path.join(root, 'rgb', f"{int(scene_id):06}.jpg")
                img_info = cv2.imread(image_path)
                h, w, _ = img_info.shape
                
                depth_scale = camera_info['depth_scale']
                depth_path = os.path.join(root, 'depth', f"{int(scene_id):06}.png")
                depth_map = np.array(Image.open(depth_path)) * depth_scale
                
                if train_bucket < 0:
                    train_flag = False
                    val_counter += 1
                else:
                    train_counter += 1
                
                current_idx = val_counter if not train_flag else train_counter
                current_labels = val_labels if not train_flag else train_labels
                
                current_labels[current_idx] = [image_path, {'boxes' : [], 'labels' : [], 'keypoints' : []}]

                for j in range(len(scene_gt_rt)):
                    if scene_gt_rt[j]['obj_id'] not in self.model_list:
                        continue
                    scene_rt_obj = scene_gt_rt[j]
                    if has_gt_info:
                        if j >= len(scene_gt_info):
                            print(f"scene_gt_info missing for object {j} in scene {scene_id}, skipping...")
                            continue
                        scene_box_obj = scene_gt_info[j]
                        if scene_box_obj['visib_fract'] < bbox_threshold:
                            continue
                    
                    R = np.array(scene_rt_obj['cam_R_m2c']).reshape(3, 3)
                    t = np.array(scene_rt_obj['cam_t_m2c']).reshape(3, 1)
                    pose = np.vstack((np.hstack((R, t)), np.array([[0, 0, 0, 1]])))
                    
                    vertices = self.kpts3D[scene_rt_obj['obj_id']]
                    vertices_world = pose @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T
                    vertices_2d = K @ vertices_world[:3, :]
                    vertices_2d /= vertices_2d[2, :]
                    vertices_2d = np.rint(vertices_2d[:2, :]).astype(int)
                    
                    # Check depth for occlusion
                    kp_x_coords = np.clip(vertices_2d[0, :], 0, w - 1)
                    kp_y_coords = np.clip(vertices_2d[1, :], 0, h - 1)
                    depth_values = depth_map[kp_y_coords, kp_x_coords]
                    diff_map = vertices_world[2] - depth_values
                    
                    outofframe = (vertices_2d[0] < 0) | (vertices_2d[0] >= w) | (vertices_2d[1] < 0) | (vertices_2d[1] >= h)
                    
                    good = np.logical_and(np.logical_not(outofframe), np.abs(diff_map) < occulusion_threshold)
                    occuluded = np.logical_and(np.logical_not(outofframe), diff_map < -occulusion_threshold)
                    bad = np.logical_not(good | occuluded)

                    # *** MODIFIED: Re-integrate the neighborhood check for "bad" points ***
                    if np.sum(bad) > 0:
                        mod = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1], [0,0]])  # Modifiers for neighboring points
                        bad_kpts = vertices_2d[:, bad]
                        bad_kpts = np.repeat(bad_kpts, mod.shape[0], axis=1)
                        bad_kpts += np.tile(mod, (np.sum(bad), 1)).T
                        bad_kpts[0] = np.clip(bad_kpts[0], 0, depth_map.shape[1] - 1)
                        bad_kpts[1] = np.clip(bad_kpts[1], 0, depth_map.shape[0] - 1)
                        diff2 = np.repeat(vertices_world[2, bad], mod.shape[0]) - depth_map[bad_kpts[1], bad_kpts[0]]
                        diff2 = diff2.reshape(-1, mod.shape[0])
                        diff2 = np.min(np.abs(diff2), axis=1)
                        
                        # Re-evaluate based on neighborhood check
                        good2 = np.abs(diff2) < occulusion_threshold
                        occuluded2 = diff2 > occulusion_threshold # Simplified for this logic block
                        
                        idx_update = np.where(bad)[0]
                        good[idx_update] = good2
                        occuluded[idx_update] = occuluded2
                        bad[idx_update] = np.logical_not(good2 | occuluded2)

                    # Check if we have any valid keypoints
                    valid_kpts = good | occuluded
                    if np.sum(valid_kpts) == 0:
                        print(f"Skipping object {scene_rt_obj['obj_id']} in scene {scene_id} due to no valid keypoints.")
                        continue

                    # Clip keypoints to be inside their ground-truth bounding box
                    if has_gt_info:
                        bbox = scene_box_obj['bbox_obj']
                        x1, y1, box_w, box_h = bbox
                        x2, y2 = x1 + box_w, y1 + box_h
                    else:
                        # get the bounding box from the vertices
                        x1 = int(np.min(vertices_2d[0, valid_kpts]))
                        y1 = int(np.min(vertices_2d[1, valid_kpts]))
                        x2 = int(np.max(vertices_2d[0, valid_kpts]))
                        y2 = int(np.max(vertices_2d[1, valid_kpts]))

                    # Skip the loop if more than 80% of keypoints are occluded (fixed logic)
                    if np.sum(occuluded) / np.sum(valid_kpts) >= 0.95:
                        print(f"Skipping object {scene_rt_obj['obj_id']} in scene {scene_id} due to high occlusion.")
                        continue

                    # Add some padding to the bounding box (only once)
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(w - 1, x2 + 10)
                    y2 = min(h - 1, y2 + 10)
                    
                    # Ensure valid bounding box coordinates
                    if x2 <= x1 or y2 <= y1:
                        print(f"Skipping object {scene_rt_obj['obj_id']} in scene {scene_id} due to invalid bounding box: [{x1}, {y1}, {x2}, {y2}]")
                        continue
                    vertices_2d[0, :] = np.clip(vertices_2d[0, :], x1, x2)
                    vertices_2d[1, :] = np.clip(vertices_2d[1, :], y1, y2)
                    # Update visibility flags to follow COCO standard
                    # v=0: not labeled, v=1: occluded, v=2: visible
                    # https://github.com/pytorch/vision/issues/5872#issuecomment-1108506440
                    visibility_array = np.zeros(vertices_2d.shape[1], dtype=np.int32)
                    visibility_array[occuluded] = 1
                    visibility_array[good] = 2
                    
                    final_kps = np.hstack([vertices_2d.T, visibility_array.reshape(-1, 1)]).tolist()
                    bbox_coco = [int(x1), int(y1), int(x2), int(y2)]
                    if self.static_obj:
                        label = int(self.static_obj_id)
                    else:
                        label = int(self.model_id_to_label[scene_rt_obj['obj_id']])
                    
                    current_labels[current_idx][1]['boxes'].append(bbox_coco)
                    current_labels[current_idx][1]['labels'].append(label)
                    current_labels[current_idx][1]['keypoints'].append(final_kps)
                if skip_empty:
                    if len(current_labels[current_idx][1]['boxes']) == 0:
                        del current_labels[current_idx]
                        if train_flag:
                            train_counter -= 1
                        else:
                            val_counter -= 1
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        json.dump(train_labels, open(os.path.join(output_path, 'train_labels.json'), 'w'), indent=4)
        json.dump(val_labels, open(os.path.join(output_path, 'val_labels.json'), 'w'), indent=4)
        print(f"Processed {len(train_labels)} training samples and {len(val_labels)} validation samples.")

class HeatMapDataset(BaseDataset):
    def __init__(self, root_dir, model_list=None, kpts3d_path=None):
        super().__init__(root_dir, model_list, kpts3d_path)


    def gaussian_eq(self, x: list, sigma=1):
        
        x = np.sqrt(x[0]**2 + x[1]**2)  # Convert to distance from origin
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * sigma ** 2))

    def processImage(self, output_path=None, input_dims=(256, 256), output_dims=(64, 64), bbox_threshold=0.3, val_split=0.2, seed=42, evaluate=False):
        random.seed(seed)
        if output_path is None:
            raise ValueError("output_path must be specified to save the processed dataset.")
        #refactor to handle multiple root directories and validation split and save to disk
        if not self.scene_gt_info or not self.camera_info:
            print("No scene_gt_info or camera_info loaded. Please load the dataset first.")
            return
        print("Processing images and generating heatmaps...")
        # counter = 0
        train_counter = -1
        val_counter = -1
        for root in tqdm(self.root_dir):
            print(f"Processing root directory: {root}")
            root_scene_gt_info = self.scene_gt_info[root]
            root_scene_gt = self.scene_gt[root]
            root_camera_info = self.camera_info[root]
            val_bucket = int(len(root_scene_gt_info) * val_split)
            train_bucket = len(root_scene_gt_info) - val_bucket
            train_flag = True
            #shuffle the root_scene_gt_info dictionary, keeping the key-value pairs intact
            
            for scene_id in tqdm(root_scene_gt_info.keys()):
                #split into val and train buckets by the number of scenes
                train_bucket -= 1
                scene_gt_info = root_scene_gt_info[scene_id]
                scene_gt_rt = root_scene_gt[scene_id]
                camera_info = root_camera_info[scene_id]
                K = np.array(camera_info['cam_K']).reshape(3, 3)
                image = cv2.imread(os.path.join(root, 'rgb', "{number:06}".format(number=int(scene_id)) + ".jpg"))
                w, h = image.shape[1], image.shape[0]
                
                # depth_scale = camera_info['depth_scale']
                # depth_path = os.path.join(self.root_dir, 'depth', "{number:06}".format(number=scene_id) + ".png")
                # depth_map = np.array(Image.open(depth_path)) * depth_scale
                                
                for j in range(len(scene_gt_rt)):
                    if scene_gt_rt[j]['obj_id'] not in self.model_list:
                        continue
                    scene_rt_obj = scene_gt_rt[j]
                    scene_box_obj = scene_gt_info[j]
                    #First check of threshold
                    if scene_box_obj['visib_fract'] < bbox_threshold:
                        continue  # Skip this object if visibility fraction is below threshold
                    else:
                        if train_bucket < 0 and not evaluate:
                            train_flag = False
                            val_counter += 1
                        else:
                            train_counter += 1
                        R = np.array(scene_rt_obj['cam_R_m2c']).reshape(3, 3)
                        t = np.array(scene_rt_obj['cam_t_m2c']).reshape(3, 1)
                        pose = np.vstack((np.hstack((R, t)), np.array([[0, 0, 0, 1]])))
                        
                        # project 3D points to 2D
                        vertices = self.kpts3D[scene_rt_obj['obj_id']]
                        vertices_world = pose @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T  # Add homogeneous coordinate
                        vertices_2d = K @ vertices_world[:3, :]
                        vertices_2d /= vertices_2d[2, :]
                        vertices_2d = np.rint(vertices_2d[:2, :]).astype(int)
                        
                        bbox = scene_box_obj['bbox_obj']
                        x, y, w, h = bbox
                        x -= 10
                        y -= 10
                        w += 20
                        h += 20
                        x = np.clip(x, 0, image.shape[1] - 1)
                        y = np.clip(y, 0, image.shape[0] - 1)
                        w = np.clip(w, 0, image.shape[1] - x - 1)
                        h = np.clip(h, 0, image.shape[0] - y - 1)
                        image_cropped = image[y:y+h, x:x+w]
                        cropped_2dpts = vertices_2d.T - np.array([[x, y]])
                        kpts_scale_x =  output_dims[0] * cropped_2dpts[:, 0] / w
                        kpts_scale_y =  output_dims[1] * cropped_2dpts[:, 1] / h
                        scaled_2dpts = np.array([kpts_scale_x, kpts_scale_y])
                        #round to nearest integer
                        vertices_2d = np.rint(scaled_2dpts).astype(int)
                        image_cropped = cv2.resize(image_cropped, input_dims, interpolation=cv2.INTER_LINEAR)
                        mod = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2)) #updated to be a 3x3 grid
                        mod = np.array(mod).reshape(2, -1).T
                        
                        heatmap_array = np.zeros((self.total_kpts, output_dims[1], output_dims[0]), dtype=np.float32)
                        gaus_out = np.zeros(mod.shape[0])
                        for j in range(mod.shape[0]):
                            gaus_out[j] = self.gaussian_eq(mod[j], sigma=1)
                        for j in range(vertices_2d.shape[1]):
                            heatmap = np.zeros(output_dims, dtype=np.float32)
                            x,y = vertices_2d[0, j], vertices_2d[1, j]
                            x = np.clip(x, 0, output_dims[0] - 1)
                            y = np.clip(y, 0, output_dims[1] - 1)
                            for k in range(mod.shape[0]):
                                x_mod = x + mod[k, 0]
                                y_mod = y + mod[k, 1]
                                if 0 <= x_mod < output_dims[0] and 0 <= y_mod < output_dims[1]:
                                    heatmap[y_mod, x_mod] += gaus_out[k]
                            
                            max_val = np.max(heatmap)
                            if max_val > 1e-6: 
                                heatmap /= max_val
                            heatmap_array[self.kpts3d_range[scene_rt_obj['obj_id']][0] + j, :, :] = heatmap           
                        del heatmap # Moved del heatmap here, after it's used
                        #rework logic to save onto disk
                        if not os.path.exists(os.path.join(output_path, "train")) or not os.path.exists(os.path.join(output_path, "val")):
                            os.makedirs(os.path.join(output_path, "train"))
                            os.makedirs(os.path.join(output_path, "val"))
                        if evaluate:
                            if not os.path.exists(os.path.join(output_path, "eval")):
                                os.makedirs(os.path.join(output_path, "eval"))
                            cv2.imwrite(os.path.join(output_path, "eval", f"{scene_id}_{j:06d}.jpg"), image_cropped)
                            np.save(os.path.join(output_path, "eval", f"{scene_id}_{j:06d}_heatmaps.npy"), heatmap_array)
                            # save R and t to a file
                            intrinsics_extrinsics = {
                                'R': R.tolist(),
                                't': t.tolist(),
                                'K': K.tolist(),
                                'obj_id': scene_rt_obj['obj_id'],
                            }
                            with open(os.path.join(output_path, "eval", f"{scene_id}_{j:06d}_intrinsics_extrinsics.json"), 'w') as f:
                                json.dump(intrinsics_extrinsics, f)
                            
                        elif train_flag:
                            cv2.imwrite(os.path.join(output_path, "train", f"{train_counter:06d}.jpg"), image_cropped)
                            np.save(os.path.join(output_path, "train", f"{train_counter:06d}_heatmaps.npy"), heatmap_array)
                        else:
                            cv2.imwrite(os.path.join(output_path, "val", f"{val_counter:06d}.jpg"), image_cropped)
                            np.save(os.path.join(output_path, "val", f"{val_counter:06d}_heatmaps.npy"), heatmap_array)
                        del R, t, pose, vertices, vertices_world, vertices_2d  # Free memory
                    del scene_rt_obj, scene_box_obj  # Free memory
            
class RCNNTorch(Dataset):
    def __init__(self, gt_file, transform=None, target_transform=None, augment=False, crop_size=(480, 640), angle_range=(-10, 10), totensor=True, single_class=False):
        with open(gt_file, 'r') as f:
            self.labels = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
        self.crop_size = crop_size
        self.angle_range = angle_range
        self.totensor = totensor
        self.single_class = single_class  # If True, return a single item instead of a dictionary

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = cv2.imread(self.labels[str(idx)][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        metadata = self.labels[str(idx)][1]
        # Ensure metadata values are numpy arrays for augmentation
        boxes = np.array(metadata['boxes'], dtype=np.float32)
        labels = np.array(metadata['labels'], dtype=np.int64)
        keypoints = np.array(metadata['keypoints'], dtype=np.float32)

        num_kpts_per_instance = keypoints.shape[1] if keypoints.ndim == 3 else 0


        if self.augment and len(boxes) > 0:
            # 1. Rotation
            angle = random.uniform(self.angle_range[0], self.angle_range[1])
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))

            # Transform keypoints
            kpts_xy = keypoints[:, :, :2].reshape(-1, 2)
            kpts_xy_hom = np.hstack([kpts_xy, np.ones((kpts_xy.shape[0], 1))])
            new_kpts_xy = (M @ kpts_xy_hom.T).T.reshape(keypoints.shape[0], keypoints.shape[1], 2)
            keypoints[:, :, :2] = new_kpts_xy

            # Transform bounding boxes
            new_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]])
                new_corners = (M @ corners.T).T
                new_x1, new_y1 = new_corners.min(axis=0)
                new_x2, new_y2 = new_corners.max(axis=0)
                new_boxes.append([new_x1, new_y1, new_x2, new_y2])
            boxes = np.array(new_boxes, dtype=np.float32)

            # 2. Random Crop
            crop_h, crop_w = self.crop_size
            if h > crop_h and w > crop_w:
                y_offset = random.randint(0, h - crop_h)
                x_offset = random.randint(0, w - crop_w)

                image = image[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]

                boxes[:, [0, 2]] -= x_offset
                boxes[:, [1, 3]] -= y_offset
                keypoints[:, :, 0] -= x_offset
                keypoints[:, :, 1] -= y_offset

                # Clip boxes and keypoints to be within crop area
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, crop_w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, crop_h)
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, crop_w)
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, crop_h)

                # Update keypoint visibility if they fall outside crop
                kpts_vis = keypoints[:, :, 2]
                kpts_x = keypoints[:, :, 0]
                kpts_y = keypoints[:, :, 1]
                # Visibility is 0 if outside crop
                kpts_vis[(kpts_x <= 0) | (kpts_x >= crop_w) | (kpts_y <= 0) | (kpts_y >= crop_h)] = 0
                keypoints[:, :, 2] = kpts_vis

                # Filter out boxes that are too small or completely outside
                valid_indices = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
                boxes = boxes[valid_indices]
                labels = labels[valid_indices]
                keypoints = keypoints[valid_indices]

        target = {}
        if len(boxes) > 0:
            # The model's loss function expects background to be class 0.
            # Our mapped labels are [0-7], so we shift them to [1-8].
            if self.totensor:
                if not self.single_class:
                    target['labels'] = torch.tensor(labels, dtype=torch.int64) + 1
                else:
                    target['labels'] = torch.tensor(labels, dtype=torch.int64)
                target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                target['keypoints'] = torch.tensor(keypoints, dtype=torch.float32)
            else:
                if not self.single_class:
                    target['labels'] = labels + 1
                else:
                    target['labels'] = labels
                target['boxes'] = boxes
                target['keypoints'] = keypoints
        else:
            # Return empty tensors if no objects are left after augmentation
            if self.totensor:
                target['labels'] = torch.empty((0,), dtype=torch.int64)
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['keypoints'] = torch.empty((0, num_kpts_per_instance, 3), dtype=torch.float32)
            else:
                target['labels'] = np.array([], dtype=np.int64)
                target['boxes'] = np.empty((0, 4), dtype=np.float32)
                target['keypoints'] = np.empty((0, num_kpts_per_instance, 3), dtype=np.float32)


        if self.transform:
            image = self.transform(image)
        else:
            if self.totensor:
                image = torch.from_numpy(image.transpose((2, 0, 1)))
            else:
                image = image.transpose((2, 0, 1))
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

class HeatMapTorch(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx] 
        image_path = os.path.join(self.path, image_filename)
        base_name = image_filename.replace('.jpg', '')
        heatmap_path = os.path.join(self.path, f"{base_name}_heatmaps.npy")

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise IOError(f"Could not read image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            heatmaps = np.load(heatmap_path)

        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading data for sample index {idx}:")
            print(f"Image path: {image_path}, Heatmap path: {heatmap_path}")
            print(f"Error details: {e}")
            raise # Re-raise the error

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            heatmaps = self.target_transform(heatmaps)
        return image, heatmaps
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

if __name__ == "__main__":
    # dataset = RCNNDataset(
    #             root_dir="/Users/Tim/Documents/GitHub/Spark-25/kpt_rcnn_update/00*",
    #             model_list=[1, 5, 6, 8, 9, 10, 11, 12],
    #             kpts3d_path="/Users/Tim/Documents/GitHub/Spark-25/train_pbr/lmo_models/models_rcnn",
    # )
    # dataset._load_all_data()
    # dataset.processImage(output_path="/Users/Tim/Documents/GitHub/Spark-25/kpt_rcnn_update/rcnn-processed", val_split=0.2, seed=42)
    
    #mugs
    dataset = RCNNDataset(
                root_dir="/Users/Tim/Documents/GitHub/Spark-25/mugs_nococo/train_pbr/*",
                model_list=[1, 3, 4, 5, 14],
                kpts3d_path="/Users/Tim/Documents/GitHub/Spark-25/mugs_nococo/models",
    )
    # dataset.load_scene_gt_info()
    dataset.load_kpts3d()
    dataset.load_camera_info()
    dataset.load_scene_gt()
    dataset.processImage(output_path="/Users/Tim/Documents/GitHub/Spark-25/mugs_nococo/rcnn-processed-xtds", val_split=0.2, seed=42, bbox_threshold=0.3, occulusion_threshold=3, static_obj=True, static_obj_id=1)
    # import torchvision.transforms as transforms
    # transforms_list = [
    # transforms.ToTensor()]
    # train_set = RCNNTorch(
    # gt_file="/Users/Tim/Documents/GitHub/Spark-25/kpt_rcnn_update/rcnn-processed/train_labels.json",
    # transform=transforms.Compose(transforms_list),
    # )
    # breakpoint()
    # train_set.__getitem__(0)