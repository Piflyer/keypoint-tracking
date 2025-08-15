# Point Tracking with Keypoint R-CNN and Classical Optical Flow 

Predict and track keypoints of objects in real-time from a sequence of frames. You can use the tracked points with a solver, like the Quaternion solver, to estimate object poses from a single viewpoint.

# Features:
- MobileNetV3-based backbone for Keypoint R-CNN or YOLO v8 keypoint detector
- Modified Keypoint R-CNN to include point visibility detection through Conformal Prediction
- Lukas-Kanade optical flow for tracking points between inference
- Pretrained weights for mugs and a collection of [YCB-V](https://www.ycbbenchmarks.com/object-models/) objects

### Limitations:
- Currently, category-based tracking (ie: cannot distinguish different mugs from each other)

### In the Works:
- Release custom dataset and weights for other category level objects
- Release YOLO v8 keypoint training script and models (In the process of transition from Keypoint R-CNN to YOLO)

# Usage:

You can use one of the provided pre-trained models for Keypoint-RCNN or you can also train your own model using the provided dataset and training scripts.

## Using the Pre-trained Model
We have provided the pre-trained weights for the Keypoint-RCNN model, which you can use for inference on your own images or video streams. To use the pre-trained model, simply load the weights and run inference as shown in the provided example scripts.

- Mugs Pretrained Model
- LM-O Pretrained Model
- Forks Pretrained Model [Coming Soon]
- Pans Pretrained Model [Coming Soon]
- NOCS Pretrained Model [Coming Soon]

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

<details>
<summary>
<h3> Training (Keypoint R-CNN)</h3>
</summary>

```bash
python train.py --dataset /path/to/your/dataset --backbone mobilenetv3 --num-epochs 50
```

</details>