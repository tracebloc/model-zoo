# Object detection

Locate and classify multiple objects within an image. Models output bounding boxes + class predictions per detected object.

## Start here

**New to object detection?** Use [`pytorch/yolo_v8/`](pytorch/yolo_v8/). Fast, accurate, single-stage detector — the standard choice for most real-world use cases. Upload it as [`pytorch/yolo_v8.zip`](pytorch/yolo_v8.zip) (see [Uploading a YOLO folder template](#uploading-a-yolo-folder-template)).

## Models

| Model | Type | When to pick |
|---|---|---|
| [`pytorch/yolo_v8/`](pytorch/yolo_v8/) — upload [`yolo_v8.zip`](pytorch/yolo_v8.zip) | Single-stage | Fast, high accuracy, modern choice |
| [`pytorch/yolo_v5/`](pytorch/yolo_v5/) — upload [`yolo_v5.zip`](pytorch/yolo_v5.zip) | Single-stage | Slightly older YOLO; still widely deployed |
| [`pytorch/yolo_v1/`](pytorch/yolo_v1/) — upload [`yolo_v1.zip`](pytorch/yolo_v1.zip) | Single-stage | Canonical YOLO architecture; teaching/baseline |
| [`pytorch/faster_rcnn_resnet.py`](pytorch/faster_rcnn_resnet.py) | Two-stage | Slower than YOLO, often more accurate on small objects |

## Dataset expectations

- **Input**: RGB images, variable resolution (YOLO auto-resizes to its input size).
- **Labels**: per-image list of `(class_id, x, y, w, h)` bounding boxes (typically in normalized YOLO format).
- **Multi-file YOLO models**: the folder's `model.py` is the entry point and `loss.py` is the loss the model trains with.

## Uploading a YOLO folder template

`yolo_v1/`, `yolo_v5/` and `yolo_v8/` are folders: `model.py` plus the `loss.py` the model trains with. `upload_model` takes one file, so each folder ships with a ready-to-upload zip beside it, holding both files at its root. Upload the zip, not the folder's `model.py`:

```python
user.upload_model("model-zoo/model_zoo/object_detection/pytorch/yolo_v8.zip")
```

Uploading `yolo_v8/model.py` on its own fails with `loss.py file missing in the zip`. If you change a folder's files, rebuild its zip with `python tools/build_folder_zips.py` (CI fails while a zip is out of date). For your own YOLO model, zip `model.py` and `loss.py` flat (both at the root of the zip, no folder inside) and upload that zip.
