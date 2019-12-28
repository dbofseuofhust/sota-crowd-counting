import torch.nn as nn
import torch
from torchvision import models

from .layer import Conv2d, FC
from misc.layer import convDU,convLR

import torch.nn.functional as F
from .utils import *

import pdb

# model_path = '../PyTorch_Pretrained/resnet101-5d3b4d8f.pth'

class Res101_FPN(nn.Module):
    def __init__(self, ):
        super(Res101_FPN, self).__init__()

        self.pyramid_feature_size = 256
        self.pred_feature_size = 128
        self.backend_feat  = [64]
        
        self.de_pred1 = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pred_feature_size, 1, same_padding=True, NL='relu'),
                                    Conv2d(self.pred_feature_size, 1, 1, same_padding=True, NL='relu'))

        self.de_pred2 = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pred_feature_size, 1, same_padding=True, NL='relu'),
                                    Conv2d(self.pred_feature_size, 1, 1, same_padding=True, NL='relu'))

        self.de_pred3 = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pred_feature_size, 1, same_padding=True, NL='relu'),
                                    Conv2d(self.pred_feature_size, 1, 1, same_padding=True, NL='relu'))

        self.de_pred4 = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pred_feature_size, 1, same_padding=True, NL='relu'),
                                    Conv2d(self.pred_feature_size, 1, 1, same_padding=True, NL='relu'))

        self.de_pred = nn.Sequential(Conv2d(3, self.pred_feature_size, 5, same_padding=True, NL='relu'),
                                    Conv2d(self.pred_feature_size, 1, 1, same_padding=True, NL='relu'))

        self.convP4in = nn.Sequential(Conv2d(2048, self.pyramid_feature_size, 1, same_padding=True, NL='relu'))
        self.convP4out = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pyramid_feature_size, 3, same_padding=True, NL='relu'))


        self.convP3in = nn.Sequential(Conv2d(1024, self.pyramid_feature_size, 1, same_padding=True, NL='relu'))
        self.convP3out = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pyramid_feature_size, 3, same_padding=True, NL='relu'))
        
        self.convP2in = nn.Sequential(Conv2d(512, self.pyramid_feature_size, 1, same_padding=True, NL='relu'))
        self.convP2out = nn.Sequential(Conv2d(self.pyramid_feature_size, self.pyramid_feature_size, 3, same_padding=True, NL='relu'))

        res = models.resnet101(pretrained=True)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool
        )

        self.own_reslayer_1 = make_res_layer(Bottleneck, 64, 64, 3, stride=1)        
        self.own_reslayer_1.load_state_dict(res.layer1.state_dict())

        self.own_reslayer_2 = make_res_layer(Bottleneck, 256, 128, 4, stride=2)        
        self.own_reslayer_2.load_state_dict(res.layer2.state_dict())

        self.own_reslayer_3 = make_res_layer(Bottleneck, 512, 256, 23, stride=2)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

        self.own_reslayer_4 = make_res_layer(Bottleneck, 1024, 512, 3, stride=2)        
        self.own_reslayer_4.load_state_dict(res.layer4.state_dict())
        
    def forward(self,x):
        
        x0 = self.frontend(x)
        x1 = self.own_reslayer_1(x0)
        x2 = self.own_reslayer_2(x1)
        x3 = self.own_reslayer_3(x2)
        x4 = self.own_reslayer_4(x3)

        p4 = self.convP4in(x4)        
        p4ups = F.interpolate(p4,size=[x3.shape[2], x3.shape[3]])
        p4 = self.convP4out(p4)
        x4pred = self.de_pred4(p4)

        p3 = self.convP3in(x3)
        p3 = p3+p4ups
        p3ups = F.interpolate(p3,size=[x2.shape[2], x2.shape[3]])
        p3 = self.convP3out(p3)
        x3pred = self.de_pred3(p3)

        p2 = self.convP2in(x2)
        p2 = p2+p3ups
        p2 = self.convP2out(p2)
        x2pred = self.de_pred2(p2)

        x2ups = F.interpolate(x2pred,size=[x.shape[2], x.shape[3]])
        x3ups = F.interpolate(x3pred,size=[x.shape[2], x.shape[3]])
        x4ups = F.interpolate(x4pred,size=[x.shape[2], x.shape[3]])

        x = torch.squeeze(torch.stack([x2ups, x3ups, x4ups], dim=1), dim=2)
        x = self.de_pred(x)

        if self.training:
            return x, x2ups, x3ups, x4ups
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)   

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers) 

def make_res_layer(block, inplanes, planes, blocks, stride=1):

    downsample = None
    # inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  

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