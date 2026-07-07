from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 16
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


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
        )
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
            self.gate(self.conv2a(x))
            + self.gate(self.conv2b(x))
            + self.gate(self.conv2c(x))
            + self.gate(self.conv2d(x))
        )
        x = self.conv3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x + identity)


class OSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 16, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            OSBlock(16, 64),
            OSBlock(64, 64),
            nn.Sequential(Conv1x1(64, 64), nn.AvgPool2d(2, 2)),
        )
        self.conv3 = nn.Sequential(
            OSBlock(64, 96),
            OSBlock(96, 96),
            nn.Sequential(Conv1x1(96, 96), nn.AvgPool2d(2, 2)),
        )
        self.conv4 = nn.Sequential(OSBlock(96, 128), OSBlock(128, 128))
        self.conv5 = Conv1x1(128, 128)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def osnet_x0_25():
    return OSNet()
