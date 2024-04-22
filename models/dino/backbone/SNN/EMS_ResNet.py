from collections import OrderedDict

from models.dino.backbone.SNN.layers import *
from sklearn.cluster import KMeans
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def kmeans(events):
    fg_pos = {}
    for l, event in enumerate(events):
        c, h, w, n = event.shape
        event = torch.sum(event, dim=3)
        feat_map = event.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=2).fit(feat_map.reshape(-1, c))
        fg_mask = kmeans.labels_.reshape(h, w)
        fg_pos.update({f'batch{l}': np.where(fg_mask == 1)})

    return fg_pos


class snn_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(snn_BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class snn_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, concat_res=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(snn_Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        self.conv1_s = tdLayer(conv1x1(inplanes, width), norm_layer(width))

        self.conv2_s = tdLayer(conv3x3(width, width, stride, groups, dilation), norm_layer(width))

        self.conv3_s = tdLayer(conv1x1(width, planes * self.expansion), norm_layer(planes * self.expansion))

        self.spike = LIFSpike()
        self.shortcut = concat_res
        self.stride = stride

    def forward(self, x):
        short_x = x

        if self.shortcut is not None:
            short_x = self.shortcut(short_x) # layer2 (2,256,65,87,5)

        out = self.spike(x)
        out = self.conv1_s(out)

        out = self.spike(out)
        out = self.conv2_s(out)

        out = self.spike(out)
        out = self.conv3_s(out)

        out += short_x

        return out


class ems_snn_ResNet(nn.Module):
    def __init__(self, block, layers, output_layers, norm_layer=None, zero_init_residual=False, steps=None):
        super(ems_snn_ResNet, self).__init__()
        # block = snn_Bottleneck,把block作为一个类（而不是实例化），用于后续构建模型
        if norm_layer is None:
            self._norm_layer = tdBatchNorm
        # 输出层标号
        self.output_layers = output_layers
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.steps = steps

        replace_stride_with_dilation = [False, False, False]

        self.conv1_s = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                               self._norm_layer(self.inplanes))

        self.spike = LIFSpike()
        self.maxpool = tdLayer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 构建resnet block
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.fg_pos = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, snn_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, snn_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        concat_res = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            concat_res = Concat_res2(in_channels=self.inplanes, out_channels=planes * block.expansion, stride=stride)

        layers = [block(self.inplanes, planes, stride, concat_res, self.groups, self.base_width,
                        previous_dilation,
                        norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if int(name) in output_layers:
            outputs[name] = torch.sum(x, dim=4) / self.steps
        return int(name) == output_layers[-1]

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1_s(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # (2, 256, 120, 152, 2)
        if self._add_output_and_check('0', x, outputs, output_layers):
            return outputs, self.fg_pos

        x = self.layer2(x)  # (2, 512, 60, 76, 2)
        if self._add_output_and_check('1', x, outputs, output_layers):
            return outputs, self.fg_pos

        x = self.layer3(x)  # (2, 1024, 30, 38, 2)
        if self._add_output_and_check('2', x, outputs, output_layers):
            return outputs, self.fg_pos

        x = self.layer4(x)  # (2, 2048, 15, 19, 2)
        if self._add_output_and_check('3', x, outputs, output_layers):
            return outputs, self.fg_pos

        raise ValueError('output_layer is wrong.')


class Concat_res2(nn.Module):  #
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
        )

        if in_channels < out_channels:
            self.shortcut = nn.Sequential(
                LIFSpike(),
                tdLayer(nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False),
                        tdBatchNorm(out_channels - in_channels))
            )
        if stride == 1:
            self.pools = tdLayer(nn.MaxPool2d(kernel_size=stride, stride=stride))
        else:
            self.pools = tdLayer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        temp = self.shortcut(x)
        out = torch.cat((temp, x), dim=1)
        out = self.pools(out)
        return out


def resnet50_ems_snn(return_interm_indices=None, steps=None, **kwargs):
    if return_interm_indices is None:
        return_interm_indices = [0, 1, 2, 3]
    model = ems_snn_ResNet(snn_Bottleneck, [3, 4, 6, 3], return_interm_indices, steps=steps, **kwargs)
    return model
