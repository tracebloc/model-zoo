# Image classification

Categorize images into predefined labels. Every model outputs `(B, num_classes)` logits. All but one accept `(B, 3, H, W)` RGB input. [`multichannel_cnn.py`](pytorch/multichannel_cnn.py) accepts `(B, C, H, W)` images with any number of channels `C`, from 1 to 64.

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
| [`multichannel_cnn.py`](pytorch/multichannel_cnn.py) | ~24K | Images with 1 to 64 channels (multispectral, multiplexed TIFF stacks); takes its `input_channels` from the dataset |

## Dataset expectations

- **Input**: RGB images. `image_size = 224` by default; override per model. Images with any other number of channels, up to 64: `multichannel_cnn.py`, whose `input_channels` tracebloc sets to your dataset's channel count when you pick it (set it yourself when you upload the file by hand).
- **Labels**: integer class indices. `output_classes` configurable per file.
- **Batch size**: default 256 (PyTorch); the multichannel CNN starts at 16.
