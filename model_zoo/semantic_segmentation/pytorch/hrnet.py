import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
main_class = "HRNet"
image_size = 256
batch_size = 8
output_classes = 2
category = "semantic_segmentation"


class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class HRModule(nn.Module):
    """High-Resolution Module for HRNet"""
    
    def __init__(self, num_branches, blocks, num_channels, fuse_method="SUM"):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        
        # Create branches
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = nn.Sequential(*[blocks[i](num_channels[i], num_channels[i]) for _ in range(1)])
            self.branches.append(branch)
        
        # Create fusion layers
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)
                elif j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                else:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 3, stride=2**(i-j), padding=1, bias=False),
                        nn.BatchNorm2d(num_channels[i])
                    ))
            self.fuse_layers.append(fuse_layer)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward through branches
        branch_outputs = []
        for i in range(self.num_branches):
            branch_outputs.append(self.branches[i](x[i]))
        
        # Fuse branches
        fused_outputs = []
        for i in range(self.num_branches):
            fused = branch_outputs[i]
            for j in range(self.num_branches):
                if i != j:
                    if self.fuse_layers[i][j] is not None:
                        fused += self.fuse_layers[i][j](branch_outputs[j])
            fused_outputs.append(self.relu(fused))
        
        return fused_outputs


class HRNet(nn.Module):
    """HRNet for semantic segmentation"""
    
    def __init__(self, n_channels=3, n_classes=output_classes):
        super(HRNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(n_channels, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: Single branch
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)
        
        # Stage 2: Two branches
        self.transition1 = self._make_transition_layer([256], [32, 64])
        self.stage2 = self._make_stage(2, [32, 64], 4)
        
        # Stage 3: Three branches
        self.transition2 = self._make_transition_layer([32, 64], [32, 64, 128])
        self.stage3 = self._make_stage(3, [32, 64, 128], 4)
        
        # Stage 4: Four branches
        self.transition3 = self._make_transition_layer([32, 64, 128], [32, 64, 128, 256])
        self.stage4 = self._make_stage(4, [32, 64, 128, 256], 3)
        
        # Final layers for segmentation
        self.final_layer = nn.Conv2d(sum([32, 64, 128, 256]), n_classes, 1)
        
        self._initialize_weights()

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_channels, num_blocks):
        modules = []
        for i in range(num_modules):
            modules.append(HRModule(num_modules, BasicBlock, num_channels))
        return nn.Sequential(*modules)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Stage 2
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Stage 3
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Stage 4
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Upsample all branches to the same resolution
        h, w = y_list[0].size(2), y_list[0].size(3)
        upsampled_features = []
        for i, feature in enumerate(y_list):
            if i == 0:
                upsampled_features.append(feature)
            else:
                upsampled_features.append(F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True))
        
        # Concatenate all features
        x = torch.cat(upsampled_features, dim=1)
        
        # Final classification
        x = self.final_layer(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        return x 