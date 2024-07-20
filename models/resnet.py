import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, track_running_stats=True):
        super(ResidualBlock, self).__init__()
        inchannel = inchannel
        outchannel = outchannel
        self.left1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True))
        self.left2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 这两个东西是在说一码事，再升维的时候需要增大stride来保持计算量
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left1(x)
        out = self.left2(out)
        out = out + self.shortcut(x)  # 当shortcut是 nn.Sequential()的时候 会返回x本身
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_channels=3, num_classes=10, track_running_stats=True, rate=None,
                 dataset='cifar'):
        super(ResNet, self).__init__()

        self.dataset = dataset

        self.inchannel = int(64 * rate[0])

        self.features = nn.Sequential(
            nn.Sequential(nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
                          nn.ReLU()),

            self._make_layer(ResidualBlock, int(96 * rate[1]), 3, stride=1,
                             track_running_stats=track_running_stats),

            self._make_layer(ResidualBlock, int(128 * rate[2]), 4, stride=2,
                             track_running_stats=track_running_stats),

            self._make_layer(ResidualBlock, int(256 * rate[3]), 6, stride=2,
                             track_running_stats=track_running_stats),

            self._make_layer(ResidualBlock, int(512 * rate[4]), 3, stride=2,
                             track_running_stats=track_running_stats))

        self.classifier = nn.Linear(int(512 * rate[4]), num_classes)

    def _make_layer(self, block, channels, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, track_running_stats))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        result = {'representation': out}
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        result['output'] = out
        return result


def ResNet18_cifar(num_channels=3, num_classes=10, track_running_stats=True, rate=[1] * 5):  # Resnet34
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats, rate,
                  'cifar')  # 默认track_running_stats为true 即保留BN层的历史统计值


if __name__ == '__main__':
    net_1 = ResNet18_cifar(num_classes=10, track_running_stats=True,
                           rate=[0.71] * 5)
    net_2 = ResNet18_cifar(num_classes=10, track_running_stats=True,
                           rate=[1] * 5)
    data = torch.randn(1, 3, 32, 32)
    out = net_1(data)
    net_1_param = sum([param.nelement() for param in net_1.parameters()])
    net_2_param = sum([param.nelement() for param in net_2.parameters()])
    result = net_1(data)
    print(net_1_param / net_2_param)
