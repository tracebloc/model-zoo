import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# Configuration
framework = "pytorch"
main_class = "Segmenter"
image_size = 256
batch_size = 8
output_classes = 2
category = "semantic_segmentation"


class Segmenter(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super(Segmenter, self).__init__()

        # Load Segformer model from Hugging Face
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=output_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        # Segformer expects inputs in format (batch_size, channels, height, width)
        outputs = self.model(pixel_values=x)
        return outputs.logits


# class SegmenterB0(Segmenter):
#     """Segmenter with Segformer-B0 backbone"""
#     def __init__(self):
#         super(SegmenterB0, self).__init__(model_name="nvidia/segformer-b0-finetuned-ade-512-512")


# class SegmenterB1(Segmenter):
#     """Segmenter with Segformer-B1 backbone"""
#     def __init__(self):
#         super(SegmenterB1, self).__init__(model_name="nvidia/segformer-b1-finetuned-ade-512-512")


# class SegmenterB2(Segmenter):
#     """Segmenter with Segformer-B2 backbone"""
#     def __init__(self):
#         super(SegmenterB2, self).__init__(model_name="nvidia/segformer-b2-finetuned-ade-512-512")


# class SegmenterB3(Segmenter):
#     """Segmenter with Segformer-B3 backbone"""
#     def __init__(self):
#         super(SegmenterB3, self).__init__(model_name="nvidia/segformer-b3-finetuned-ade-512-512")


# class SegmenterB4(Segmenter):
#     """Segmenter with Segformer-B4 backbone"""
#     def __init__(self):
#         super(SegmenterB4, self).__init__(model_name="nvidia/segformer-b4-finetuned-ade-512-512")


# class SegmenterB5(Segmenter):
#     """Segmenter with Segformer-B5 backbone"""
#     def __init__(self):
#         super(SegmenterB5, self).__init__(model_name="nvidia/segformer-b5-finetuned-ade-512-512")
