"""RTMPose (OpenMMLab, 2023). Production-grade real-time pose with SimCC (coordinate classification) head; widely deployed in industry. CSPNeXt backbone uses GroupNorm-style normalization where possible — federated-friendly compared to BN-heavy alternatives."""
import torch
import torch.nn as nn
import torch.nn.functional as F

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "Apache-2.0"
image_size = 256
batch_size = 64
output_classes = 1
category = "keypoint_detection"
num_feature_points = 17


def _conv_gn(in_c, out_c, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, k // 2, bias=False),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
    )


class MyModel(nn.Module):
    """RTMPose-style: CNN backbone + SimCC 1D classification per axis."""

    def __init__(self, num_feature_points=num_feature_points, img_size=image_size):
        super().__init__()
        self.num_kp = num_feature_points
        self.img_size = img_size
        self.bins = img_size * 2  # SimCC oversampling

        self.stem = _conv_gn(3, 32, 3, 2)
        self.s1 = nn.Sequential(_conv_gn(32, 64, 3, 2), _conv_gn(64, 64))
        self.s2 = nn.Sequential(_conv_gn(64, 128, 3, 2), _conv_gn(128, 128))
        self.s3 = nn.Sequential(_conv_gn(128, 256, 3, 2), _conv_gn(256, 256))
        self.s4 = nn.Sequential(_conv_gn(256, 384, 3, 2), _conv_gn(384, 384))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, 256)
        self.head_x = nn.Linear(256, num_feature_points * self.bins)
        self.head_y = nn.Linear(256, num_feature_points * self.bins)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x); x = self.s2(x); x = self.s3(x); x = self.s4(x)
        x = self.pool(x).flatten(1)
        x = F.silu(self.proj(x))
        b = x.shape[0]
        lx = self.head_x(x).view(b, self.num_kp, self.bins)
        ly = self.head_y(x).view(b, self.num_kp, self.bins)
        # Soft-argmax → normalized coords
        idx = torch.arange(self.bins, device=x.device, dtype=x.dtype) / (self.bins - 1)
        px = torch.softmax(lx, dim=-1); py = torch.softmax(ly, dim=-1)
        xc = (px * idx).sum(-1); yc = (py * idx).sum(-1)
        conf = (px.max(-1).values + py.max(-1).values) / 2
        return torch.stack([xc, yc, conf], dim=-1)
