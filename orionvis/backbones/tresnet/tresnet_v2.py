"""
Modified from `TResNet: High Performance GPU-Dedicated Architecture`
https://arxiv.org/pdf/2003.13630.pdf

Original model: https://github.com/mrT23/TResNet
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from timm.models.layers import SpaceToDepthModule, BlurPool2d, InplaceAbn, ClassifierHead, SEModule

from orionvis.extentions.inplace_abn import inplace_abn

__all__ = ['tresnet_l_v2']


class InplaceAbn(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, apply_act=True,
                 act_layer="leaky_relu", act_param=0.01, drop_layer=None):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity', '')
                self.act_name = act_layer if act_layer else 'identity'
            else:
                # convert act layer passed as type to string
                if act_layer == nn.ELU:
                    self.act_name = 'elu'
                elif act_layer == nn.LeakyReLU:
                    self.act_name = 'leaky_relu'
                elif act_layer is None or act_layer == nn.Identity:
                    self.act_name = 'identity'
                else:
                    assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        output = inplace_abn(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps, self.act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output


def IABN2Float(module: nn.Module) -> nn.Module:
    """If `module` is IABN don't use half precision."""
    if isinstance(module, InplaceAbn):
        module.float()
    for child in module.children():
        IABN2Float(child)
    return module


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer="leaky_relu", act_param=1e-2):
    return nn.Sequential(
        nn.Conv2d(
            ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
        InplaceAbn(nf, act_layer=act_layer, act_param=act_param)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=1e-3)
            else:
                self.conv1 = nn.Sequential(
                    conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        rd_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, rd_channels=rd_chs) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out += shortcut
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True,
                 act_layer="leaky_relu", aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(
            inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer, act_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_iabn(
                planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv2 = conv2d_iabn(
                    planes, planes, kernel_size=3, stride=2, act_layer=act_layer, act_param=1e-3)
            else:
                self.conv2 = nn.Sequential(
                    conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, rd_channels=reduction_chs) if use_se else None

        self.conv3 = conv2d_iabn(
            planes, planes * self.expansion, kernel_size=1, stride=1, act_layer="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = out + shortcut  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):
    def __init__(self, layers, in_dims=3, num_classes=1000, width_factor=1.0, global_pool='fast', drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        aa_layer = BlurPool2d

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_iabn(in_dims * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(
            Bottleneck, self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer)  # 56x56
        layer2 = self._make_layer(
            Bottleneck, self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer)  # 28x28
        layer3 = self._make_layer(
            Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer)  # 14x14
        layer4 = self._make_layer(
            Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', SpaceToDepthModule()),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # self.feature_info = [
        #     dict(num_chs=self.planes, reduction=2, module=''),  # Not with S2D?
        #     dict(num_chs=self.planes, reduction=4, module='body.layer1'),
        #     dict(num_chs=self.planes * 2, reduction=8, module='body.layer2'),
        #     dict(num_chs=self.planes * 4 * Bottleneck.expansion, reduction=16, module='body.layer3'),
        #     dict(num_chs=self.planes * 8 * Bottleneck.expansion, reduction=32, module='body.layer4'),
        # ]

        # head
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=1, act_layer="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, use_se=use_se, aa_layer=aa_layer))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='fast'):
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def tresnet_l_v2(save_path=None, device_map="cuda", dtype=torch.float32, num_classes=1000, in_dims=3):
    model = TResNet(
        num_classes=num_classes,
        in_dims=in_dims,
        layers=[3, 4, 23, 3],
        width_factor=1.0,
        global_pool='fast',
    )
    if save_path is not None:
        model.load_state_dict(torch.load(save_path, map_location=device_map))
    return model.to(dtype=dtype, device=device_map)
