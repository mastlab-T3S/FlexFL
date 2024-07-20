import torch
import torch.nn as nn


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, trs):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t, track_running_stats=trs),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t, track_running_stats=trs),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels, track_running_stats=trs)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2_scaleFL(nn.Module):
    """
        MobileMetV2 implementation
    """

    # 6 -> 0.79Length 0.83width 8 -> 0.91Length 0.88width
    def __init__(self, num_channels, num_classes, trs=True, scale=1.0, exit1=6, exit2=8):
        super(MobileNetV2_scaleFL, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(num_channels, int(32 * scale), 3, padding=1),
            nn.BatchNorm2d(int(32 * scale), track_running_stats=trs),
            nn.ReLU6(inplace=True)
        )
        self.exit1 = exit1
        self.exit2 = exit2

        magic_list = [0, 16 * scale, 24 * scale, 32 * scale, 64 * scale, 96 * scale, 160 * scale, 160 * scale,
                      160 * scale, 320 * scale]
        self.block = nn.Sequential(LinearBottleNeck(int(32 * scale), int(16 * scale), 1, 1, trs),
                                   self._make_stage(2, int(16 * scale), int(24 * scale), 2, 6, trs),
                                   self._make_stage(3, int(24 * scale), int(32 * scale), 2, 6, trs),
                                   self._make_stage(4, int(32 * scale), int(64 * scale), 2, 6, trs),
                                   self._make_stage(3, int(64 * scale), int(96 * scale), 1, 6, trs),
                                   self._make_stage(3, int(96 * scale), int(160 * scale), 2, 6, trs)[0],
                                   self._make_stage(3, int(96 * scale), int(160 * scale), 2, 6, trs)[1],
                                   self._make_stage(3, int(96 * scale), int(160 * scale), 2, 6, trs)[2],
                                   LinearBottleNeck(int(160 * scale), int(320 * scale), 1, 6, trs))
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Sequential(
            nn.Conv2d(int(magic_list[exit1]), int(magic_list[exit1] * 4), 1),
            nn.BatchNorm2d(int(magic_list[exit1] * 4),
                           track_running_stats=trs),
            nn.ReLU6(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(int(magic_list[exit1] * 4), num_classes, 1),
            nn.Flatten()
        ))
        self.classifier.append(nn.Sequential(
            nn.Conv2d(int(magic_list[exit2]), int(magic_list[exit2] * 4), 1),
            nn.BatchNorm2d(int(magic_list[exit2] * 4),
                           track_running_stats=trs),
            nn.ReLU6(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(int(magic_list[exit2] * 4), num_classes, 1),
            nn.Flatten()
        ))
        self.classifier.append(nn.Sequential(
            nn.Conv2d(int(magic_list[9]), int(magic_list[9] * 4), 1),
            nn.BatchNorm2d(int(magic_list[9] * 4),
                           track_running_stats=trs),
            nn.ReLU6(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(int(magic_list[9] * 4), num_classes, 1),
            nn.Flatten(),
        ))

    def _make_stage(self, n, in_channels, out_channels, stride, t, trs):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t, trs)]

        while n - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, trs))
            n -= 1

        return nn.Sequential(*layers)

    def forward(self, x, ee=3):
        if ee == 1:
            x = self.pre(x)
            x = self.block[:self.exit1](x)
            x = self.classifier[0](x)
            return [{'output': x}]
        elif ee == 2:
            out = []
            x = self.pre(x)
            x = self.block[:self.exit1](x)
            out1 = self.classifier[0](x)
            out.append({'output': out1})

            x = self.block[self.exit1:self.exit2](x)
            out2 = self.classifier[1](x)
            out.append({'output': out2})

            return out
        elif ee == 3:
            out = []
            x = self.pre(x)
            x = self.block[:self.exit1](x)
            out1 = self.classifier[0](x)
            out.append({'output': out1})

            x = self.block[self.exit1:self.exit2](x)
            out2 = self.classifier[1](x)
            out.append({'output': out2})

            x = self.block[self.exit2:](x)
            out3 = self.classifier[2](x)
            out.append({'output': out3})

            return out


# if __name__ == '__main__':
#     net = MobileNetV2_scaleFL(3, 200, False, 1, 6, 8)
#
#     # print(net)
#     total = sum(p.numel() for p in net.parameters())
#     net = MobileNetV2_scaleFL(3, 200, False, 0.90, 6, 8)
#     pre_params = sum(p.numel() for p in net.pre.parameters())
#     block_params = sum(p.numel() for p in net.block[:net.exit1].parameters())
#     class_params = sum(p.numel() for p in net.classifier[:1].parameters())
#     block_params2 = sum(p.numel() for p in net.block[:net.exit2].parameters())
#     class_params2 = sum(p.numel() for p in net.classifier[:2].parameters())
#     print(f"param 1 is {(pre_params + block_params + class_params) / total}")
#     print(f"param 2 is {(pre_params + block_params2 + class_params2) / total}")
#     # CIFAR10
#     # 25% params -> exit 6 0.87
#     # 50% params -> exit 8 0.89
#     # CIFAR100
#     # 25% params -> exit 6 0.88
#     # 50% params -> exit 8 0.90
#     # TinyImageNet
#     # 25% params -> exit 6 0.88
#     # 25% params -> exit 6 0.90