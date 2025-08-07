# Point Tracking with Keypoint R-CNN and Classical Optical Flow 

Predict and track keypoints of objects in real-time from a sequence of frames. You can use the tracked points with a solver, like the Quaternion solver, to estimate object poses from a single viewpoint!

## Features:
- MobileNetV3-based backbone for Keypoint R-CNN
- Modified Keypoint R-CNN to include point visibility detection through Conformal Prediction
- Lukas-Kanade optical flow for tracking points between inference
- Pretrained weights for mugs and a collection of [YCB-V](https://www.ycbbenchmarks.com/object-models/) objects

# Limitations:
- Currently, category-based tracking (ie: cannot distinguish different mugs from each other) when doing error calculation
- Tracking does not go through each prediction refresh
