"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
import numpy as np
import torch
import torchvision
from torch import optim
from torchvision.transforms import transforms
from tqdm import tqdm

'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


class VGG_ScaleFL(nn.Module):
    def __init__(self, features, num_class=100, num_channels=3, scale=1.0, exit1=12, exit2=13):
        super().__init__()
        self.features = features
        if num_channels == 3:
            dim = 4096
        else:
            dim = 256

        counter = 0
        self.classifier = nn.ModuleList()
        for (i, j) in enumerate(features):
            if isinstance(j, nn.Conv2d):
                counter += 1

                if exit1 == counter:
                    self.classifier.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(j.out_channels, int(dim * scale)),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(int(dim * scale), int(dim * scale)),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(int(dim * scale), num_class)
                    ))
                    self.exitpos1 = i + 3  # Actually is i + 2

                if exit2 == counter:
                    self.classifier.append(nn.Sequential(
                        # magic number match exitpos2
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(j.out_channels, int(dim * scale)),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(int(dim * scale), int(dim * scale)),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(int(dim * scale), num_class)
                    ))
                    self.exitpos2 = i + 3

        self.classifier.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(int(512 * scale), int(dim * scale)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(dim * scale), int(dim * scale)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(dim * scale), num_class)
        ))

    def forward(self, x, ee=1):
        if ee == 1:
            layer1_out = self.features[:self.exitpos1](x)
            output = self.classifier[0](layer1_out)
            result = {'output': output}
            return [result]
        elif ee == 2:
            out = []
            layer1_out = self.features[:self.exitpos1](x)
            output = self.classifier[0](layer1_out)
            result = {'output': output}
            out.append(result)

            layer2_out = self.features[self.exitpos1:self.exitpos2](layer1_out)
            output = self.classifier[1](layer2_out)
            result = {'output': output}
            out.append(result)

            return out
        elif ee == 3:
            out = []
            layer1_out = self.features[:self.exitpos1](x)
            output = self.classifier[0](layer1_out)
            result = {'output': output}
            out.append(result)

            layer2_out = self.features[self.exitpos1:self.exitpos2](layer1_out)
            output = self.classifier[1](layer2_out)
            result = {'output': output}
            out.append(result)

            layer3_out = self.features[self.exitpos2:](layer2_out)
            output = self.classifier[2](layer3_out)
            result = {'output': output}
            out.append(result)

            return out


def make_layers(cfg, batch_norm=False, track_running_stats=True, num_channels=3, scale=1.0):
    layers = []
    input_channel = num_channels

    if num_channels == 3:
        cfg.append('M')
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        l = int(l * scale)

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l, track_running_stats=track_running_stats)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    if num_channels == 3:
        cfg.pop()
    return nn.Sequential(*layers)


def vgg_16_scaleFL(num_classes, track_running_stats=True, num_channels=3, scale=1.0, exit1=12, exit2=13):
    return VGG_ScaleFL(
        make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=num_channels, scale=scale),
        num_class=num_classes,
        num_channels=num_channels, scale=scale, exit1=exit1, exit2=exit2)


if __name__ == '__main__':
    data = torch.rand(10, *(3, 32, 32))

    exit1 = 12
    exit2 = 13
    net = vgg_16_scaleFL(200, False, 3, 1, exit1, exit2)
    net(data)
    print(net)
    total = sum(p.numel() for p in net.parameters())
    net1 = vgg_16_scaleFL(10, False, 3, 0.82, exit1, exit2)
    ori = []
    cur = []
    for i in range(0, net.exitpos2):
        if isinstance(net.features[i],nn.Conv2d):
            ori.append([sum(p.numel() for p in net.features[i].parameters())])
            cur.append([sum(p.numel() for p in net1.features[i].parameters())])

    ori1 = sum(p.numel() for p in net.classifier[2].parameters())
    cur1 = sum(p.numel() for p in net1.classifier[:].parameters())
    print(np.array(cur)/np.array(ori))
    print(len(cur))
    print(cur1/ori1)
    # CIFAR10
    # 25% params -> exit 12 0.75
    # 50% params -> exit 13 0.82
    # CIFAR100
    # 25% params -> exit 12 0.75
    # 50% params -> exit 13 0.82
