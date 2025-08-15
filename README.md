# Point Tracking with Keypoint R-CNN and Classical Optical Flow 

Predict and track keypoints of objects in real-time from a sequence of frames. You can use the tracked points with a solver, like the Quaternion solver, to estimate object poses from a single viewpoint.

# Features:
- MobileNetV3-based backbone for Keypoint R-CNN or YOLO v8 keypoint detector
- Modified Keypoint R-CNN to include point visibility detection through Conformal Prediction
- Lukas-Kanade optical flow for tracking points between inference
- Pretrained weights for mugs and a collection of [LM-O](https://www.ycbbenchmarks.com/object-models/) objects

### Limitations:
- Currently, category-based tracking (ie: cannot distinguish different mugs from each other)

### In the Works:
- Release custom dataset and weights for other category level objects
- Release YOLO v8 keypoint training script and models (In the process of transition from Keypoint R-CNN to YOLO)

# Usage:

You can use one of the provided pre-trained models for Keypoint-RCNN or you can also train your own model using the provided dataset and training scripts.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/kpt-rcnn-tracking.git
   cd kpt-rcnn-tracking
   ```

2. Create a new Conda environment:
   ```bash
   conda create -n kpt-rcnn python=3.12
   conda activate kpt-rcnn
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Using the Pre-trained Model
We have provided the pre-trained weights for the Keypoint-RCNN and YOLOv8 models, which you can use for inference on your own images or video streams. To use the pre-trained model, simply load the weights and run inference as shown in the provided example scripts. We are shifting our focuse on YOLO models, and will be releasing more in the coming weeks.

- Mugs Pretrained Model
- LM-O Pretrained Model
- Forks Pretrained Model [Coming Soon]
- Pans Pretrained Model [Coming Soon]
- NOCS Pretrained Model [Coming Soon]

### Keypoint R-CNN Models

| Model Name | Conformal Score | Number of Keypoints | Number of Classes (including background)
|------------|------------------|---------------------|-----------------------------------------
| Mugs Kpt-RCNN       | 0.0816              | 43                  | 2
| LM-O Kpt-RCNN      | 0.0977              | 10                  | 9

### YOLO Models

> **NOTE:**  We are in the process of updating the tracking script and our Conformal Prediction implementation to support YOLO models.

| Model Name | Conformal Score | Number of Keypoints | Number of Classes (including background)
|------------|------------------|---------------------|-----------------------------------------
| Mugs YOLOv8       | TBD              | 43                 | 2

## Tracking Points
You can use our tracking script to track points through video frames or image sequences. There are multiple types of sources that you can use with the tracking script.The script will print out the tracked keypoints in JSON file in the `runs` directory.

``` bash
python run_tracking.py  
    --model_path \
    --dataset_path  \
    --data_type \
    --model_type \
    --num_classes \
    --num_keypoints \
    --conformal_threshold
```

### Core Arguments:

- `--model_path`: Path to the trained model weights.
- `--dataset_path`: Path to the dataset for tracking.
- `--data_type`: Type of data (e.g., video, image). Current options are `nocs, bop, folder, video`
- `--model_type`: Type of model (e.g., Keypoint R-CNN, YOLOv8). Current options are `kpt_rcnn, yolo`
- `--num_classes`: Number of classes to track (including background). This only applies for Kpt-RCNN
- `--num_keypoints`: Number of keypoints to track. This only applies for Kpt-RCNN
- `--conformal_threshold`: Conformal prediction threshold. This only applies for Kpt-RCNN

### Data Input Types:
- `nocs`: NOCS (Normalized Object Coordinate Space) data format. This would be the NOCS folder containing images in `*_colors.png` format.
- `bop`: BOP (Benchmarking Object Pose) data format. We have a custom dataloader for that, see how you can process your data here.
- `folder`: A folder containing images for tracking. This should be in numerical order with no gaps (e.g., `0001.png`, `0002.png`, ...).
- `video`: A video file for tracking. This can be either a video file or an OpenCV camera index.

### Example Usage:

For a video file using the pre-trained YOLO Mugs model:
``` bash
python run_tracking.py \
    --model_path "mugs_yolo.pt" \
    --dataset_path "test.avi" \
    --model_type "yolo"
```

Or for a folder containing images using the pre-trained Keypoint R-CNN LM-O model:
``` bash
python run_tracking.py \
    --model_path "lm_o_kpt_rcnn.pt" \
    --dataset_path "images_folder" \
    --model_type "kpt_rcnn" \
    --num_classes 10 \
    --num_keypoints 43 \
    --conformal_threshold 0.0816
```

<details>
<summary>
<h3> Additional Arguments</h3>
</summary>

- `--model_refresh_interval`: Frequency (in frames) to run the model keypoint prediction. Default is 10
- `--frame_refresh`: Frequency (in frames) to refresh the optical flow. Required when there is a new video sequence in the folder. Default is 50
- `--no_fpn`: Only applicable if you trained Keypoint R-CNN without a MobileNetV3's Feature Pyramid Network (FPN). Default is False
- `--kalman_process_noise`: Process noise covariance for the Kalman filter. Default is 1e-2
- `--kalman_rcnn_noise`: Measurement noise covariance for the Kalman filter when using the learned model. Default is 1e-2
- `--kalman_tracking_noise`: Measurement noise covariance for the Kalman filter when using optical flow. Default is 1e-4

</details>


## Training Your Own Model (Keypoint R-CNN or YOLOv8)

To train your own model, you can use either the provided dataset or your own custom dataset. We use the COCO dataset format for object detection and keypoint annotations. You can annotate your own objects using our modified [Keypoint Picker Tool](https://github.com/lopenguin/point-picker-3d) and the generating synthetic data using our provided BlenderProc script.

> **NOTE:**  Make sure when performing annotations, you must annotate every object in that dataset with the SAME number of keypoints.

### Generating Custom Synthetic Data

If you are using our datasets, you can skip down to training.

1) Ensure that you have successfully setup [BlenderProc](https://github.com/DLR-RM/BlenderProc) on your system.

2) Organize your folder in this format:
```
root_dataset
├── [object_1] # this can be any name
│   └── models
│         └── obj_000001.glb # your 3D model (any format works)
│         └── obj_000001.csv # your 3D model annotations
│         └── obj_000002.glb
│         └── ...
├── [object_2] # you can have as many categories as you want
│   └── models
│         └── obj_000001.glb
│         └── obj_000001.csv
│         └── obj_000002.glb
│         └── ...
├── [distractor_object] # this could be distracting objects of your choosing
│   └── models
│         └── obj_000001.glb
│         └── obj_000002.glb
│         └── ...
...

```
We highly recommend using the YCB-V and LMO objects from the [BOP dataset](https://bop.felk.cvut.cz/datasets/) for distactor objects due to its diversity and integration in BlenderProc.

> **NOTE:**  You will need to download those objects from the BOP website and still format your root dataset directory as shown above.

3) Run the provided script `generate-data.py` to generate synthetic data. You will need to modify a couple of lines in the script to fit your dataset.

On lines 26-27, you should change the folder path to any distracting objects you want to include in your dataset. We highly recommend you use the YCB-V dataset or the LMO dataset for this.

```python
ycb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True, obj_ids=[2,3,4,5,8,9,10,15,16,17,18,19,20,21])

ycb_dist_lmo_obs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lmo'), mm2m = True, obj_ids=[1,5,6,8,9,10,11,12])
```

Also on line 36, change to the folder path of target objects you want to include in your dataset.
```python
    target_forks.append(bproc.loader.load_obj(filepath=os.path.join(args.bop_parent_path, 'fork', 'models', f'obj_{j:06d}.glb'), object_model_unit='m')[-1])

```

4) Make sure you download the necessary textures and place them in the appropriate directory.
```bash
blenderproc download cc_textures 
```
This will be your texture path.

You can then generate data using BlenderProc as such:

```bash
blenderproc run generate-data.py <path/to/your/dataset> <path/to/textures> <path/to/output>  --num_scenes=2000
```


> **NOTE:**  We recommend at least 50K images per category but you might need more depending on your specific use case/task.

### Data Preparation
Once you have generated your synthetic data, you can use the provided dataloader script to prepare for training.

```bash
python utils/dataloader.py \
    --root_dir \
    --model_list \
    --kpts3d_path \
    --output_path \
    --val_split \
    --static_obj
```
### Core Arguments

- `root_dir`: The root directory of your dataset.
- `model_list`: A list of model IDs to include in the dataset from the annotated 3D models folder.
- `kpts3d_path`: The path to the 3D keypoints directory.
- `output_path`: The path to the output directory for processed data. Will save JSONs of training and val.
- `val_split`: The proportion of the dataset to use for validation.
- `static_obj`: Whether all the 3D models are treated as a single class.

<details>
<summary>
<h3> Additional Arguments</h3>
</summary>

- `--seed`: Random seed for reproducibility. Defaults to 42.
- `--bbox_threshold`: Minimum bounding box visibility fraction to consider a detection valid. Defaults to 0.3.
- `--occlusion_threshold`: Minimum number of visible keypoints to consider an object visible. Defaults to 3.
- `--static_obj_id`: Object ID to treat as static. Defaults to 1.


</details>

<details>
<summary>
<h2> Training (Keypoint R-CNN) </h2>
</summary>
Once you have prepared your dataset, you can train a Keypoint R-CNN model using the provided training script. The script supports both classical Keypoint R-CNN and MobileNetV3-FPN backbone architectures.

```bash
python keypoint-rccn-train.py \
    --dataset_path \
    --num_classes \
    --num_keypoints \
    --batch_size \
    --num_epochs \
    --learning_rate
```

### Core Arguments:

- `--dataset_path`: Path to the processed dataset directory containing train_labels.json and val_labels.json.
- `--num_classes`: Number of classes including background (e.g., 2 for single object + background).
- `--num_keypoints`: Number of keypoints to detect per object.
- `--batch_size`: Batch size for training. Default is 32.
- `--num_epochs`: Number of epochs to train. Default is 400.
- `--learning_rate`: Learning rate for training. Default is 1e-5.

### Example Usage:

For training a mugs model with 43 keypoints:
```bash
python keypoint-rccn-train.py \
    --dataset_path "path/to/mugs_dataset/rcnn-processed" \
    --num_classes 2 \
    --num_keypoints 43 \
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 1e-4
```

For training an LM-O multi-object model:
```bash
python keypoint-rccn-train.py \
    --dataset_path "path/to/lmo_dataset/rcnn-processed" \
    --num_classes 9 \
    --num_keypoints 10 \
    --batch_size 32 \
    --num_epochs 400 \
    --learning_rate 1e-5 \
    --classical_model
```

### Additional Arguments


- `--classical_model`: Use classical Keypoint R-CNN model without MobileNetV3-FPN backbone. Default is False.
- `--warmup_epochs`: Number of warmup epochs for learning rate scheduling. Default is 10.
- `--log_dir`: Directory to save logs and checkpoints. Default is 'runs'.
- `--continue_training`: Continue training from the last checkpoint. Default is False.
- `--checkpoint_path`: Path to the checkpoint file to continue training from.
- `--checkpoint_tensorboard`: Path to the TensorBoard checkpoint directory to continue logging.

</details>