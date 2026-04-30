# Image classification

Categorize images into predefined labels. All models accept `(B, 3, H, W)` RGB input and output `(B, num_classes)` logits.

## Start here

**New to image classification?** Use [`pytorch/resnet_18.py`](pytorch/resnet_18.py). Fast, well-understood, trains reliably on small datasets.

## Models

### PyTorch

| Model | Params | When to pick |
|---|---|---|
| [`resnet_18.py`](pytorch/resnet_18.py) | ~11M | Fastest ResNet; baseline for quick iteration or CPU runs |
| [`squeezenet.py`](pytorch/squeezenet.py) | ~1.2M | Extreme parameter efficiency; mobile/edge deployment |
| [`vgg_16.py`](pytorch/vgg_16.py) | ~138M | Canonical heavy baseline via torchvision |
| [`vgg_custom.py`](pytorch/vgg_custom.py) | — | Custom VGG implementation; pedagogical |
| [`wide_resnet_50_2.py`](pytorch/wide_resnet_50_2.py) | ~69M | Wider than ResNet-50; often matches deeper nets |
| [`maxvit.py`](pytorch/maxvit.py) | — | Hybrid CNN + transformer; strong accuracy, GPU-heavy |
| [`swin_transformer.py`](pytorch/swin_transformer.py) | — | Shifted-window attention; GPU-friendly ViT alternative |
| [`vit_b_16.py`](pytorch/vit_b_16.py) | ~86M | Standard Vision Transformer via torchvision |
| [`vit.py`](pytorch/vit.py) | ~86M | ViT backbone + custom head (HuggingFace) |
| [`vit_google.py`](pytorch/vit_google.py) | ~86M | Pretrained ViT classifier (HuggingFace) |
| [`simple_cnn.py`](pytorch/simple_cnn.py) | — | Minimal CNN; pedagogical starting point |
| [`lenet.py`](pytorch/lenet.py) | — | 1998 historical baseline; teaching only |

### TensorFlow

| Model | When to pick |
|---|---|
| [`resnet_50.py`](tensorflow/resnet_50.py) | Standard TF baseline |
| [`densenet.py`](tensorflow/densenet.py) | Parameter-efficient dense connectivity |
| [`efficientnet.py`](tensorflow/efficientnet.py) | Compound-scaled architecture |
| [`vgg_16.py`](tensorflow/vgg_16.py) | Canonical heavy baseline |
| [`xception.py`](tensorflow/xception.py) | Depthwise separable convs |
| [`alexnet.py`](tensorflow/alexnet.py) | Historical baseline; teaching only |
| [`lenet.py`](tensorflow/lenet.py) | Historical baseline; teaching only |
| [`simple_cnn.py`](tensorflow/simple_cnn.py) | Minimal single-block CNN |
| [`stacked_cnn.py`](tensorflow/stacked_cnn.py) | Multi-block CNN baseline |

## Dataset expectations

- **Input**: RGB images. `image_size = 224` by default; override per model.
- **Labels**: integer class indices. `output_classes` configurable per file.
- **Batch size**: default 256 (PyTorch), 128 (TensorFlow).
