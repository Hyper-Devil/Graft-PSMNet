import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
import torchvision.transforms as transforms
import PIL
import os
import matplotlib.pyplot as plt
from networks.resnet import ResNet, Bottleneck, BasicBlock_Res
from networks.vgg import vgg16
from collections import OrderedDict
from .network_blocks import DWConv, BaseConv, Focus, CSPLayer, SPPBottleneck # YOLOX
from torch.cuda.amp import autocast as autocast

# yoloxs = CSPDarknet(0.33, 0.50, depthwise=False, act="silu")
class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        outputs = x2

        return outputs

class YOLOX(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        return fpn_outs

class CSPDarknet_Feature(nn.Module):
    def __init__(self, fixed_param):
        super(CSPDarknet_Feature, self).__init__()

        backbone = YOLOPAFPN()
        self.fe = YOLOX(backbone)

        self.fe.load_state_dict(
            torch.load('networks/yolox_l.pth')["model"], strict=False)

        self.to_feat = nn.Sequential()
        self.to_feat.add_module('stem', self.fe.backbone.backbone.stem)
        self.to_feat.add_module('dark2', self.fe.backbone.backbone.dark2)


        if fixed_param:
            for p in self.to_feat.parameters():
                p.requires_grad = False  #在最开始创建Tensor时候可以设置的属性，用于表明是否追踪当前Tensor的计算操作。

    @autocast()
    def forward(self, x):
        feature = self.to_feat(x)

        # feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear', align_corners=True)

        return feature

class VGG_Feature(nn.Module):
    def __init__(self, fixed_param):
        super(VGG_Feature, self).__init__()

        self.fe = vgg16(pretrained=False)

        self.fe.load_state_dict(
            torch.load('networks/vgg16-397923af.pth'))

        features = self.fe.features

        self.to_feat = nn.Sequential()

        for i in range(15): #只加载前15层
            self.to_feat.add_module(str(i), features[i])

        if fixed_param:
            for p in self.to_feat.parameters():
                p.requires_grad = False  #在最开始创建Tensor时候可以设置的属性，用于表明是否追踪当前Tensor的计算操作。

    def forward(self, x):
        feature = self.to_feat(x)

        # feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear', align_corners=True)

        return feature


class VGG_Bn_Feature(nn.Module):
    def __init__(self):
        super(VGG_Bn_Feature, self).__init__()

        features = models.vgg16_bn(pretrained=True).cuda().eval().features
        self.to_feat = nn.Sequential()
        # for i in range(8):
        #     self.to_feat.add_module(str(i), features[i])

        for i in range(15):
            self.to_feat.add_module(str(i), features[i])

        for p in self.to_feat.parameters():
            p.requires_grad = False

    def forward(self, x):
        feature = self.to_feat(x)

        # feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear', align_corners=True)

        return feature


class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()

        self.fe = ResNet(BasicBlock_Res, [2, 2, 2, 2])

        # self.fe = ResNet(Bottleneck, [3, 4, 6, 3])

        for p in self.fe.parameters():
            p.requires_grad = False

        self.fe.load_state_dict(
            torch.load('networks/resnet18-5c106cde.pth'))

    def forward(self, x):

        self.fe.eval()

        with torch.no_grad():
            feature = self.fe(x)

        return feature


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()

        self.fe = ResNet(Bottleneck, [3, 4, 6, 3])

        for p in self.fe.parameters():
            p.requires_grad = False

        # self.fe.load_state_dict(
        #     torch.load('networks/resnet50-19c8e357.pth'))
        self.fe.load_state_dict(
            torch.load('networks/DenseCL_R50_imagenet.pth'))

    def forward(self, x):

        self.fe.eval()

        with torch.no_grad():
            feature = self.fe(x)

        return feature


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    from collections import OrderedDict
    ckpt = torch.load('selfTrainVGG_withDA.pth')
    new_dict = OrderedDict()
    for k, v in ckpt.items():
        new_k = k.replace('module.', '')
        new_dict[new_k] = v

    torch.save(new_dict, 'selfTrainVGG_withDA.pth')