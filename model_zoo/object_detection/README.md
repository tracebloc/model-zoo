# Object detection

Locate and classify multiple objects within an image. Models output bounding boxes + class predictions per detected object.

## Start here

**New to object detection?** Use [`pytorch/yolo_v8/`](pytorch/yolo_v8/). Fast, accurate, single-stage detector — the standard choice for most real-world use cases.

## Models

| Model | Type | When to pick |
|---|---|---|
| [`pytorch/yolo_v8/`](pytorch/yolo_v8/) | Single-stage | Fast, high accuracy, modern choice |
| [`pytorch/yolo_v5/`](pytorch/yolo_v5/) | Single-stage | Slightly older YOLO; still widely deployed |
| [`pytorch/yolo_v1/`](pytorch/yolo_v1/) | Single-stage | Canonical YOLO architecture; teaching/baseline |
| [`pytorch/faster_rcnn_resnet.py`](pytorch/faster_rcnn_resnet.py) | Two-stage | Slower than YOLO, often more accurate on small objects |

## Dataset expectations

- **Input**: RGB images, variable resolution (YOLO auto-resizes to its input size).
- **Labels**: per-image list of `(class_id, x, y, w, h)` bounding boxes (typically in normalized YOLO format).
- **Multi-file YOLO models**: the folder's `model.py` is the entry point; `loss.py` / `utils.py` are internal.
