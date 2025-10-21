import torch
import math
import torch.nn as nn
import math
# from math import round
# from torch.nn import functional as F


# ## WideResNet-28-10

# class BasicBlock(nn.Module):
#     """Basic residual block used in WideResNet"""
#     def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.drop_rate = drop_rate
#         self.equalInOut = (in_planes == out_planes)
#         self.shortcut = (nn.Identity() if self.equalInOut else
#                          nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False))

#     def forward(self, x):
#         out = self.relu1(self.bn1(x))
#         if not self.equalInOut:
#             x = self.shortcut(out)
#         out = self.conv1(out)
#         out = self.relu2(self.bn2(out))
#         if self.drop_rate > 0:
#             out = F.dropout(out, p=self.drop_rate, training=self.training)
#         out = self.conv2(out)
#         return x + out


# class NetworkBlock(nn.Module):
#     """Stack of BasicBlocks"""
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
#         super(NetworkBlock, self).__init__()
#         layers = []
#         for i in range(nb_layers):
#             layers.append(block(i == 0 and in_planes or out_planes,
#                                 out_planes,
#                                 i == 0 and stride or 1,
#                                 drop_rate))
#         self.layer = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layer(x)


# class WideResNet(nn.Module):
#     """
#     WideResNet model for CIFAR (e.g., WRN-28-10)
#     depth: total layers (e.g., 28)
#     widen_factor: width multiplier (e.g., 10)
#     """
#     def __init__(self, depth=28, widen_factor=10, drop_rate=0.0, num_classes=100):
#         super(WideResNet, self).__init__()
#         assert ((depth - 4) % 6 == 0), "Depth should be 6n+4"
#         n = (depth - 4) // 6
#         k = widen_factor

#         nStages = [16, 16*k, 32*k, 64*k]

#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nStages[0], 3, padding=1, bias=False)

#         # 1st, 2nd and 3rd block
#         self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, drop_rate)
#         self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, drop_rate)
#         self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, drop_rate)

#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nStages[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nStages[3], num_classes)

#         # init
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(-1, out.size(1))
#         return self.fc(out)


## ResNeXt-29, 16x64d

# class ResNeXtBottleneck(nn.Module):
#     """ResNeXt bottleneck for CIFAR (1x1 -> 3x3 group -> 1x1)"""
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, cardinality=16, base_width=64):
#         super().__init__()
#         # width per group scaled by stage "planes" (as in original ResNeXt)
#         D = int(math.floor(planes * (base_width / 64.0)))   # channels per group
#         C = cardinality * D                                 # total channels in the bottleneck
#         self.conv1 = nn.Conv2d(inplanes, C, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(C)
#         self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1,
#                                groups=cardinality, bias=False)
#         self.bn2 = nn.BatchNorm2d(C)
#         self.conv3 = nn.Conv2d(C, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)

#         self.downsample = None
#         if stride != 1 or inplanes != planes * self.expansion:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * self.expansion),
#             )

#     def forward(self, x):
#         identity = x

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))

#         if self.downsample is not None:
#             identity = self.downsample(identity)

#         out += identity
#         out = self.relu(out)
#         return out


# class ResNeXt29(nn.Module):
#     """
#     ResNeXt-29, 16x64d for CIFAR
#     depth=29 -> (29 - 2) / 9 = 3 bottleneck blocks per stage
#     stages planes: [64, 128, 256]; expansion=4 -> output channels [256, 512, 1024]
#     """
#     def __init__(self, num_classes=100, cardinality=16, base_width=64):
#         super().__init__()
#         self.inplanes = 64
#         layers = [3, 3, 3]  # for depth 29

#         # Initial conv (no maxpool for CIFAR)
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)

#         # Stages
#         self.layer1 = self._make_layer(ResNeXtBottleneck, 64,  layers[0], stride=1,
#                                        cardinality=cardinality, base_width=base_width)
#         self.layer2 = self._make_layer(ResNeXtBottleneck, 128, layers[1], stride=2,
#                                        cardinality=cardinality, base_width=base_width)
#         self.layer3 = self._make_layer(ResNeXtBottleneck, 256, layers[2], stride=2,
#                                        cardinality=cardinality, base_width=base_width)

#         # Classifier
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(256 * ResNeXtBottleneck.expansion, num_classes)

#         # Init
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0.0)

#     def _make_layer(self, block, planes, blocks, stride, cardinality, base_width):
#         layers = []
#         layers.append(block(self.inplanes, planes, stride=stride,
#                             cardinality=cardinality, base_width=base_width))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, stride=1,
#                                 cardinality=cardinality, base_width=base_width))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.conv3 = nn.Conv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class PyramidNet(nn.Module):
        
    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=False):
        super(PyramidNet, self).__init__()   	
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.addrate = alpha / (3*n*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        elif dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

            if layers.get(depth) is None:
                if bottleneck:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth-2)/12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth-2)/8)

                layers[depth]= [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])

            self.inplanes = 64            
            self.addrate = alpha / (sum(layers[depth])*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
        return x

def create_model(num_classes, device):
    """Create and initialize the model"""
    # model = WideResNet(
    #     depth=28, 
    #     widen_factor=10, 
    #     drop_rate=0.2, 
    #     num_classes=num_classes
    # )
    # model = ResNeXt29(
    #     cardinality=16, 
    #     base_width=64,
    #     num_classes=num_classes, 
    # )
    model = PyramidNet(
        dataset='cifar100', 
        depth=110, 
        alpha=270, 
        num_classes=num_classes, 
        bottleneck=False
    )
    model = model.to(device)
    return model
