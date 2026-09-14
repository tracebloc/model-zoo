# Semantic segmentation

Pixel-level class labeling. Each output pixel gets a class assignment, so the result is a mask of the same spatial resolution as the input.

## Start here

**New to segmentation?** Use [`pytorch/unet.py`](pytorch/unet.py). Universal baseline — works on medical imaging, satellite data, and general-purpose segmentation with minimal tuning.

## Models

| Model | When to pick |
|---|---|
| [`unet.py`](pytorch/unet.py) | Universal baseline; default choice |
| [`deeplab.py`](pytorch/deeplab.py) | Atrous convolutions for multi-scale context; strong on natural images |
| [`fcn.py`](pytorch/fcn.py) | Canonical Fully Convolutional baseline |
| [`hrnet.py`](pytorch/hrnet.py) | Maintains high-res features; strong on fine-detail tasks |
| [`segnet.py`](pytorch/segnet.py) | Lightweight encoder-decoder; resource-constrained |
| [`segmenter.py`](pytorch/segmenter.py) | Pure transformer; needs data and GPU time |

## Dataset expectations

- **Input**: RGB images, `(B, 3, H, W)`.
- **Labels**: per-pixel class masks, `(B, H, W)` with integer class IDs.
- **Batch size**: default 8 — segmentation masks are memory-heavy.
