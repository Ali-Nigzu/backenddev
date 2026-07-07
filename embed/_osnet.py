import torch
from torch import nn
from torch.nn import functional as F


# =========================
# Basic layers
# =========================

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# =========================
# Channel attention
# =========================

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = in_channels // reduction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avgpool(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w


# =========================
# OSBlock
# =========================

class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4

        self.conv1 = Conv1x1(in_channels, mid)

        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
        )

        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x

        x = self.conv1(x)

        x = (
            self.gate(self.conv2a(x)) +
            self.gate(self.conv2b(x)) +
            self.gate(self.conv2c(x)) +
            self.gate(self.conv2d(x))
        )

        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return F.relu(x + identity)


# =========================
# OSNet backbone (PURE EMBEDDING)
# =========================

class OSNet(nn.Module):
    def __init__(self, blocks, layers, channels, feature_dim=512):
        super().__init__()

        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], True)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], False)

        self.conv5 = Conv1x1(channels[3], channels[3])

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block, num_blocks, in_c, out_c, downsample):
        layers = [block(in_c, out_c)]
        for _ in range(1, num_blocks):
            layers.append(block(out_c, out_c))

        if downsample:
            layers.append(nn.Sequential(
                Conv1x1(out_c, out_c),
                nn.AvgPool2d(2, 2)
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)

        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)

        v = self.fc(v)

        # ALWAYS embedding output
        return v


# =========================
# Factory (no num_classes anymore)
# =========================

def osnet_x0_25():
    return OSNet(
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128]
    )
