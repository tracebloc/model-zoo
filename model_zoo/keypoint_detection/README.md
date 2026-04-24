# Keypoint detection

Detect landmarks on objects or bodies — e.g. human pose estimation (joint positions) or facial landmarks.

Most models come in two flavors:
- **direct regression** (predicts coordinates directly)
- **heatmap-based** (predicts per-keypoint likelihood maps, usually more accurate)

## Start here

**New to keypoint detection?** Use [`pytorch/hrnet_heatmap.py`](pytorch/hrnet_heatmap.py). State-of-the-art accuracy on pose estimation benchmarks; well-understood architecture.

For a lighter/faster baseline, use [`pytorch/resnet_sppe.py`](pytorch/resnet_sppe.py) (Single-Person Pose Estimator on a ResNet-50 backbone).

## Models

### Strong choices

| Model | When to pick |
|---|---|
| [`hrnet_heatmap.py`](pytorch/hrnet_heatmap.py) | State-of-the-art pose accuracy |
| [`hrnet.py`](pytorch/hrnet.py) | HRNet with direct regression |
| [`alphapose.py`](pytorch/alphapose.py) | AlphaPose-style SimpleBaseline, configurable backbone |
| [`resnet_sppe.py`](pytorch/resnet_sppe.py) | Lightweight SPPE on ResNet-50 |
| [`faster_rcnn_sppe.py`](pytorch/faster_rcnn_sppe.py) | SPPE on Faster R-CNN backbone; reuses detection features |

### Classical baselines

| Model | Notes |
|---|---|
| [`cpn_heatmap.py`](pytorch/cpn_heatmap.py) | Cascaded Pyramid Network, heatmap output |
| [`cpn.py`](pytorch/cpn.py) | Cascaded Pyramid Network, direct regression |
| [`shg_heatmap.py`](pytorch/shg_heatmap.py) | Stacked Hourglass, heatmap output |
| [`shg.py`](pytorch/shg.py) | Stacked Hourglass, direct regression |
| [`cpm_heatmap.py`](pytorch/cpm_heatmap.py) | Convolutional Pose Machine, heatmap |
| [`cpm.py`](pytorch/cpm.py) | Convolutional Pose Machine, direct |
| [`openpose.py`](pytorch/openpose.py) | OpenPose-style multi-stage CNN |
| [`dsnt.py`](pytorch/dsnt.py) | Differentiable Spatial-to-Numerical Transform |
| [`deeppose.py`](pytorch/deeppose.py) | Historical: first deep pose estimator (2014) |

### Other

| Model | Notes |
|---|---|
| [`rcnn.py`](pytorch/rcnn.py) | R-CNN-style keypoint head |
| [`resnet_50.py`](pytorch/resnet_50.py) | SimpleBaseline on ResNet-50 |
| [`yolo_v8.py`](pytorch/yolo_v8.py) | YOLOv8 pose-estimation variant; single-pass detection + keypoints |

## Dataset expectations

- **Input**: RGB images (usually cropped around a detected person or object).
- **Labels**: per-image keypoint coordinates + visibility, or per-keypoint heatmaps.
- **`num_feature_points`**: typically 16–17 for human-body keypoints.
