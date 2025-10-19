import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
        # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
        # Fully connected layers: flatten feature maps and classify
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 spatial resolution
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
## WideResNet-28-10
class BasicBlock(nn.Module):
    """Basic residual block used in WideResNet"""
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.shortcut = (nn.Identity() if self.equalInOut else
                         nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False))

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.equalInOut:
            x = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return x + out
    
class NetworkBlock(nn.Module):
    """Stack of BasicBlocks"""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    """
    WideResNet model for CIFAR (e.g., WRN-28-10)
    depth: total layers (e.g., 28)
    widen_factor: width multiplier (e.g., 10)
    """
    def __init__(self, depth=28, widen_factor=10, drop_rate=0.0, num_classes=100):
        super(WideResNet, self).__init__()
        assert ((depth - 4) % 6 == 0), "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nStages[0], 3, padding=1, bias=False)

        # 1st, 2nd and 3rd block
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, out.size(1))
        return self.fc(out)

## ResNeXt-29, 16x64d

class ResNeXtBottleneck(nn.Module):
    """ResNeXt bottleneck for CIFAR (1x1 -> 3x3 group -> 1x1)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=16, base_width=64):
        super().__init__()
        # width per group scaled by stage "planes" (as in original ResNeXt)
        D = int(math.floor(planes * (base_width / 64.0)))   # channels per group
        C = cardinality * D                                 # total channels in the bottleneck
        self.conv1 = nn.Conv2d(inplanes, C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.conv3 = nn.Conv2d(C, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNeXt29(nn.Module):
    """
    ResNeXt-29, 16x64d for CIFAR
    depth=29 -> (29 - 2) / 9 = 3 bottleneck blocks per stage
    stages planes: [64, 128, 256]; expansion=4 -> output channels [256, 512, 1024]
    """
    def __init__(self, num_classes=100, cardinality=16, base_width=64):
        super().__init__()
        self.inplanes = 64
        layers = [3, 3, 3]  # for depth 29

        # Initial conv (no maxpool for CIFAR)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        self.layer1 = self._make_layer(ResNeXtBottleneck, 64,  layers[0], stride=1,
                                       cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 128, layers[1], stride=2,
                                       cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 256, layers[2], stride=2,
                                       cardinality=cardinality, base_width=base_width)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 * ResNeXtBottleneck.expansion, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride, cardinality, base_width):
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            cardinality=cardinality, base_width=base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1,
                                cardinality=cardinality, base_width=base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_model(num_classes, device):
    """Create and initialize the model"""
    # model = WideResNet(depth=28, widen_factor=10, drop_rate=0.2, num_classes=num_classes)
    model = ResNeXt29(num_classes=num_classes, cardinality=16, base_width=64)
    model = model.to(device)
    return model
