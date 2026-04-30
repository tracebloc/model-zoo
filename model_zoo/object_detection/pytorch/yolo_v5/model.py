import torch
import torch.nn as nn

framework = "pytorch"
model_type = "yolo"
main_class = "MyModel"
image_size = 448
batch_size = 64
output_classes = 3
category = "object_detection"


import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes):
        super().__init__()
        self.B = 2
        self.C = num_classes
        self.backbone = nn.Sequential(
            ConvBNAct(3,32,k=3,s=1,p=1),
            ConvBNAct(32,64,k=3,s=2,p=1),
            ConvBNAct(64,128,k=3,s=2,p=1),
            ConvBNAct(128,256,k=3,s=2,p=1),
            ConvBNAct(256,512,k=3,s=2,p=1),
            ConvBNAct(512,1024,k=3,s=2,p=1),
            nn.AdaptiveAvgPool2d((7,7)),
        )
        self.neck = nn.Sequential(
            ConvBNAct(1024,512,k=3,s=1,p=1),
            ConvBNAct(512,512,k=3,s=1,p=1),
        )
        self.pred = nn.Conv2d(
            512,
            self.C + 5*self.B,
            kernel_size=1
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pred(x)
        x = x.permute(0,2,3,1).contiguous()
        return x
