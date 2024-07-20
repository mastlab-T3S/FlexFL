import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
# from torchinfo import summary


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


class MobileNetV2(nn.Module):
    """
        MobileMetV2 implementation
    """

    def __init__(self, channels, num_classes, trs=True, rate=[1] * 9):
        super(MobileNetV2, self).__init__()
        self.features = nn.Sequential(nn.Sequential(
            nn.Conv2d(channels, int(32 * rate[0]), 3, padding=1),
            nn.BatchNorm2d(int(32 * rate[0]), track_running_stats=trs),
            nn.ReLU6(inplace=True)
        ),

            LinearBottleNeck(int(32 * rate[0]), int(16 * rate[1]), 1, 1, trs),

            self._make_stage(2, int(16 * rate[1]), int(24 * rate[2]), 2, 6, trs),

            self._make_stage(3, int(24 * rate[2]), int(32 * rate[3]), 2, 6, trs),

            self._make_stage(4, int(32 * rate[3]), int(64 * rate[4]), 2, 6, trs),

            self._make_stage(3, int(64 * rate[4]), int(96 * rate[5]), 1, 6, trs),

            self._make_stage(3, int(96 * rate[5]), int(160 * rate[6]), 2, 6, trs),

            LinearBottleNeck(int(160 * rate[6]), int(320 * rate[7]), 1, 6, trs),

            nn.Sequential(
                nn.Conv2d(int(320 * rate[7]), int(1280 * rate[8]), 1),
                nn.BatchNorm2d(int(1280 * rate[8]), track_running_stats=trs),
                nn.ReLU6(inplace=True)
            ))
        self.classifier = nn.Conv2d(int(1280 * rate[8]), num_classes, 1)

    def _make_stage(self, n, in_channels, out_channels, stride, t, trs):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t, trs)]

        while n - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, trs))
            n -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_max_pool2d(x, 1)
        result = {'representation': x.view(x.size(0), -1)}
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        result['output'] = x
        return result


# if __name__ == '__main__':
#     net = MobileNetV2(3, 10, False, [1] * 9)
#
#     summary(net, (3, 3, 32, 32))
#     print(sum(p.numel() for p in net.parameters()))
#     dummy_input = torch.randn(3, 3, 32, 32).to('cuda')
#     flops, params = profile(net, (dummy_input,))
#     print('flops: ', flops, 'params: ', params)
#     print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
#     for i in range(3, 8):
#         for j in range(20, 100):
#             net = MobileNetV2(3, 10, False, [1] * i + [j / 100] * (9 - i))
#             block_params = sum(p.numel() for p in net.features.parameters())
#             class_params = sum(p.numel() for p in net.classifier.parameters())
#             print(f"exit is {i} and scale is {j} and param is {(block_params + class_params) / (2254858)}")
#     # AdaptiveFL i = 4 , 69% ,->50%param; 47%->25% param
#     # mobileNet i =
