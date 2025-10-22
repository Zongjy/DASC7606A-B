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


# ## WideResNet-28-10

class WRNBasicBlock(nn.Module):
    """Basic residual block used in WideResNet"""
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, use_se=False):
        super(WRNBasicBlock, self).__init__()
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
        # SE block
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_planes, reduction=16)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.equalInOut:
            x = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        return x + out


class WRNNetworkBlock(nn.Module):
    """Stack of BasicBlocks"""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, use_se=False):
        super(WRNNetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                drop_rate,
                                use_se=use_se))
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
        self.block1 = WRNNetworkBlock(n, nStages[0], nStages[1], WRNBasicBlock, 1, drop_rate, use_se=True)
        self.block2 = WRNNetworkBlock(n, nStages[1], nStages[2], WRNBasicBlock, 2, drop_rate, use_se=True)
        self.block3 = WRNNetworkBlock(n, nStages[2], nStages[3], WRNBasicBlock, 2, drop_rate, use_se=True)

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
                if m.bias is not None:
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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def create_model(num_classes, device):
    """Create and initialize the model"""
    # model = SimpleCNN(num_classes=num_classes)
    model = WideResNet(
        depth=28, 
        widen_factor=10, 
        drop_rate=0.3, 
        num_classes=num_classes
    )
    model = model.to(device)
    return model
