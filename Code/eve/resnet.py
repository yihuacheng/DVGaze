import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import math



model_urls = {
     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth' 
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

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



def Attention(q, k ,v):
    # q [B,  L, D]
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    attn = torch.softmax(scores, -1)
    return torch.matmul(attn, v)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FusionBlock(nn.Module):
    # Input two feature maps and output a fusion map.
    def __init__(self, planes, d_model):
        """
        """
        super(FusionBlock, self).__init__()

        # self.inplanes = planes * 2

        convblock = BasicBlock

        # [Batch, 2, W, H]
        self.spatio_att = nn.Sequential(
                conv3x3(2, 2),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
                conv3x3(2, 2),
                nn.Softmax(dim=1)
                )

        self.k_linear = nn.Linear(d_model, d_model)

        self.q_linear = nn.Linear(d_model, d_model)

        self.self_att = nn.MultiheadAttention(d_model, 1, dropout = 0.1)
        

        # [Batch, N]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.channel_att = nn.Sequential(
                nn.Linear(planes, planes//2),
                nn.ReLU(inplace=True),
                nn.Linear(planes//2, planes),
                nn.Sigmoid()
                )
 
        

    def forward(self, f1, f2):
        """
        Inputs:
            f1: [Batch, N, H, W]
            f2: [Batch, N, H, W]
        Outpus:
            f : [Batch, N, H, W]

        """
        f1 = f1.unsqueeze(2)
        f2 = f2.unsqueeze(2)

        # [B, N, 2, H, W]
        feature = torch.cat([f1, f2], 2)
        B, N, L, H, W = feature.size()

        # B N H 2 W
        feature = feature.permute(0, 1, 3, 2, 4)
        key = self.k_linear(feature)
        query = self.q_linear(feature)

        feature = Attention(key, query, feature)

        feature = feature.permute(0, 3, 1, 2, 4)
        feature = feature.contiguous().view([-1, N, H, W])

        channel_feature = self.avgpool(feature).squeeze()
        
        channel_weight = self.channel_att(channel_feature)
        feature = torch.einsum('ijkl,ij->ijkl', [feature, channel_weight])
        feature = feature.view([B, 2, N, H, W])

        out_1 = feature[:,0,:,:,:].squeeze()
        out_2 = feature[:,1,:,:,:].squeeze()
        # out = self.conv(feature)

        return out_1, out_2

class FusionLayer(nn.Module):
    # Input two feature maps and output a fusion map.
    def __init__(self, planes, d_model):
        """
        """
        super(FusionLayer, self).__init__()

        # self.inplanes = planes * 2

        convblock = Bottleneck

        self.pre_conv1 = convblock(planes, planes//4)
        self.pre_conv2 = convblock(planes, planes//4)

        self.fusion = FusionBlock(planes, d_model)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f1, f2):
        """
        Inputs:
            f1: [Batch, N, W, H]
            f2: [Batch, N, W, H]
        Outpus:
            f : [Batch, N, W, H]

        """
        ff1 = self.pre_conv1(f1)
        ff2 = self.pre_conv2(f2)

        out_1, out_2 = self.fusion(ff1, ff2)
        
        out_1 = self.relu(out_1 + f1)
        out_2 = self.relu(out_2 + f2)

        return out_1, out_2


class ResNet(nn.Module):

    def __init__(self, block, layers, out_feature=128):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fusion2 = FusionLayer(128, 32)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.fusion3 = FusionLayer(256, 16)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fusion4 = FusionLayer(512, 8)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.fc = nn.Sequential(
            nn.Linear(512, out_feature),
            nn.ReLU(inplace=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, face):
        face1 = face[:, 0, :]
        face2 = face[:, 1, :]

        x = self.conv1(face1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        y = self.conv1(face2)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        x = self.layer1(x)
        y = self.layer1(y)

        x = self.layer2(x)
        y = self.layer2(y)
        x, y = self.fusion2(x, y)

        x = self.layer3(x)
        y = self.layer3(y)
        x, y = self.fusion3(x, y)

        x = self.layer4(x)      
        y = self.layer4(y)      
        x, y = self.fusion4(x, y)

        feature1 = self.avgpool(x)
        feature1 = torch.flatten(feature1, 1)
        feature1 = self.fc(feature1)
        feature1 = feature1.unsqueeze(0)

        feature2 = self.avgpool(y)
        feature2 = torch.flatten(feature2, 1)
        feature2 = self.fc(feature2)
        feature2 = feature2.unsqueeze(0)
        
        feature = torch.cat([feature1, feature2], 0)
 
        return feature



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model
