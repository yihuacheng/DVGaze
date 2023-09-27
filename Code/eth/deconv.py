import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': '/home/cyh/.torch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        

class DeconvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DeconvBlock, self).__init__()

        self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes),
                BasicBlock(planes, planes)
            )

    def forward(self, pre_feature, residual=None):

        cur_feature = self.conv(pre_feature)
    

        if residual is None:
            return cur_feature
        else:
            return residual + cur_feature
 

class ResDeconv(nn.Module):

    def __init__(self, inplanes):
        super(ResDeconv, self).__init__()

        self.inplanes = inplanes[0]
        block = BasicBlock

        model = []
        for planes in inplanes[1:]:
            model += [nn.Upsample(scale_factor=2)]
            model += [self._make_layer(block, planes, 2)] # 28
        model += [nn.Conv2d(self.inplanes, 1, stride=1, kernel_size=1)]

        self.deconv = nn.Sequential(*model)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        for i in range(1, blocks):
            layers.append(block(planes, planes))

        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, features):
        img = self.deconv(features)
        return img

