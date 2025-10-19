import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


# class SimpleCNN(nn.Module):
#     """
#     A simple CNN architecture for image classification
#     """

#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
#         # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
#         # Fully connected layers: flatten feature maps and classify
#         self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 spatial resolution
#         self.fc2 = nn.Linear(512, num_classes)
#         self.dropout = nn.Dropout(0.5)  # Dropout for regularization

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 4 * 4)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x
    

## EfficientNet
def _make_divisible(v: int, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def drop_connect(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if (not training) or p <= 0.0:
        return x
    keep = 1.0 - p
    mask = (torch.rand((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype) + keep).floor()
    return x / keep * mask

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s, groups=1, bn_eps=1e-3, bn_momentum=0.01):
        pad = (k - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum),
            nn.SiLU(inplace=True),  # Swish
        )

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        red = max(1, int(in_ch * se_ratio))
        self.reduce = nn.Conv2d(in_ch, red, 1, bias=True)
        self.expand = nn.Conv2d(red, in_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.reduce(s), inplace=True)
        s = torch.sigmoid(self.expand(s))
        return x * s
    
class MBConv(nn.Module):
    """
    MBConv (inverted bottleneck) with SE + stochastic depth.
    Args:
        in_ch, out_ch, k, s, expand (1 or 6), se_ratio, drop_rate
    """
    def __init__(self, in_ch, out_ch, k, s, expand, se_ratio=0.25, drop_rate=0.0,
                 bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()
        self.use_res = (s == 1) and (in_ch == out_ch)
        mid_ch = in_ch if expand == 1 else int(in_ch * expand)
        layers: List[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, 1, 1, bn_eps=bn_eps, bn_momentum=bn_momentum))
        layers.append(ConvBNAct(mid_ch, mid_ch, k, s, groups=mid_ch, bn_eps=bn_eps, bn_momentum=bn_momentum))
        layers.append(SqueezeExcite(mid_ch, se_ratio))
        layers += [nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)]
        self.block = nn.Sequential(*layers)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            if self.drop_rate > 0:
                y = drop_connect(y, self.drop_rate, self.training)
            y = y + x
        return y

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 (width=1.0, depth=1.0, dropout=0.2)
    Set stem_stride=1 for CIFAR-sized inputs (32x32) to avoid early downsample.
    """
    # (expand, k, out_ch, num_repeat, stride, se_ratio) for B0
    _BLOCKS: List[Tuple[int, int, int, int, int, float]] = [
        (1, 3, 16, 1, 1, 0.25),
        (6, 3, 24, 2, 2, 0.25),
        (6, 5, 40, 2, 2, 0.25),
        (6, 3, 80, 3, 2, 0.25),
        (6, 5, 112, 3, 1, 0.25),
        (6, 5, 192, 4, 2, 0.25),
        (6, 3, 320, 1, 1, 0.25),
    ]

    def __init__(self,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 bn_eps: float = 1e-3,
                 bn_momentum: float = 0.01,
                 stem_channels: int = 32,
                 head_channels: int = 1280,
                 divisor: int = 8,
                 stem_stride: int = 2):
        super().__init__()
        # stem
        stem_out = _make_divisible(stem_channels, divisor)
        self.stem = ConvBNAct(3, stem_out, 3, stem_stride, bn_eps=bn_eps, bn_momentum=bn_momentum)

        # blocks
        blocks = []
        total = sum(r for *_rest, r, _s, _se in [(b[0], b[1], b[2], b[3], b[4], b[5]) for b in self._BLOCKS])
        bidx = 0
        in_ch = stem_out
        for (expand, k, out_ch, num_repeat, stride, se_ratio) in self._BLOCKS:
            out = _make_divisible(out_ch, divisor)
            for i in range(num_repeat):
                s = stride if i == 0 else 1
                drop = drop_connect_rate * bidx / max(1, total - 1)
                blocks.append(MBConv(in_ch, out, k, s, expand, se_ratio, drop,
                                     bn_eps=bn_eps, bn_momentum=bn_momentum))
                in_ch = out
                bidx += 1
        self.blocks = nn.Sequential(*blocks)

        # head
        head_ch = _make_divisible(head_channels, divisor)
        self.head = ConvBNAct(in_ch, head_ch, 1, 1, bn_eps=bn_eps, bn_momentum=bn_momentum)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(head_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def create_model(num_classes, device):
    """Create and initialize the model"""
    model = EfficientNetB0(num_classes=num_classes, stem_stride=1)
    model = model.to(device)
    return model
