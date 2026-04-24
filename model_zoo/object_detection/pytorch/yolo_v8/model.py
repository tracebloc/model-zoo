import torch
import torch.nn as nn
import math
import torchvision
from torch.nn.functional import cross_entropy, one_hot

framework = "pytorch"
model_type = "yolo"
main_class = "MyModel"
image_size = 448
batch_size = 64
output_classes = 3
category = "object_detection"

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)

    def forward(self, x):
        # Process each feature map
        outputs = []
        for i in range(self.nl):
            # Get box and class predictions
            box_pred = self.box[i](x[i])
            cls_pred = self.cls[i](x[i])
            
            # Reshape to [batch_size, 7, 7, num_classes + 4]
            batch_size = box_pred.shape[0]
            box_pred = box_pred.permute(0, 2, 3, 1)  # [B, H, W, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            # Resize to 7x7 grid
            box_pred = torch.nn.functional.interpolate(box_pred.permute(0, 3, 1, 2), 
                                                     size=(7, 7), 
                                                     mode='bilinear', 
                                                     align_corners=False)
            cls_pred = torch.nn.functional.interpolate(cls_pred.permute(0, 3, 1, 2), 
                                                     size=(7, 7), 
                                                     mode='bilinear', 
                                                     align_corners=False)
            
            # Combine predictions
            box_pred = box_pred.permute(0, 2, 3, 1)  # [B, 7, 7, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1)  # [B, 7, 7, C]
            
            # Concatenate box and class predictions
            output = torch.cat([box_pred, cls_pred], dim=-1)
            outputs.append(output)
        
        # Average predictions from all feature maps
        return torch.stack(outputs).mean(dim=0)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

class YOLOv8YOLOv5Compatible(nn.Module):
    def __init__(self, width, depth, num_classes=output_classes, num_boxes=2):
        super().__init__()
        self.C = num_classes
        self.B = num_boxes

        self.backbone = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        # Prediction head outputting C + 5*B channels
        self.pred = nn.Conv2d(
            in_channels=width[3],   # FPN output channels
            out_channels=self.C + 5*self.B,
            kernel_size=1
        )

        # Force output to 7x7 grid
        self.pool = nn.AdaptiveAvgPool2d((7,7))

    def forward(self, x):
        # Backbone
        c3, c4, c5 = self.backbone(x)

        # Neck (FPN)
        p3, p4, p5 = self.fpn((c3, c4, c5))

        # Pick p3 (highest resolution)
        x = self.pred(p3)

        # Force to 7x7 grid
        x = self.pool(x)

        # (batch, channels, 7,7) -> (batch,7,7,channels)
        x = x.permute(0,2,3,1).contiguous()
        return x

def MyModel(num_classes=10, num_boxes=2):
    depth = [1,2,2]
    width = [3,32,64,128,256,512]
    return YOLOv8YOLOv5Compatible(width, depth, num_classes, num_boxes)
