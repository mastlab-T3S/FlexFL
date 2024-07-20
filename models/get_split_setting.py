import torch
from torchinfo import summary

# from torchinfo import summary

from models import MobileNetV2
from models.resnet import ResNet18_cifar
from models.vgg import vgg_16_bn
from utils.options import args_parser
# from thop import profile

args = args_parser()


def get_split_setting(model, slim_idx_list, ration):
    """

    Args:
        slim_idx: 保留的前几层, 是一个list. 比如 [8]
        ration: 最终得到模型比上原始模型的比例， 比如 0.5

    Returns:
        object: 返回最合适的模型宽度裁剪比例， 比如 0.78
    """

    if model == 'vgg':
        net = vgg_16_bn(num_classes=10, track_running_stats=False, num_channels=args.num_channels, slim_idx=0, scale=1.0)
    elif model == 'resnet':
        net = ResNet18_cifar(num_classes=10, track_running_stats=False, slim_idx=0, scale=1.0)
    elif model == 'mobilenet':
        net = MobileNetV2(channels=22 , num_classes=22, trs=False, slim_idx=0, scale=1.0)
    all_total = sum([param.nelement() for param in net.parameters()])
    print('Number of parameter: % .4fM' % (all_total / 1e6))

    for slim_idx in slim_idx_list:
        for scale in range(30, 100):
            if model == 'vgg':
                net = vgg_16_bn(num_classes=10, track_running_stats=False, num_channels=args.num_channels, slim_idx=slim_idx, scale=scale / 100)
            elif model == 'resnet':
                net = ResNet18_cifar(num_classes=10, track_running_stats=False, slim_idx=slim_idx, scale=scale / 100)
            elif model == 'mobilenet':
                net = MobileNetV2(channels= 22 ,num_classes=22, trs=False, slim_idx=slim_idx, scale=scale / 100)  # slim_idx 0-10

            tmp = sum([param.nelement() for param in net.parameters()])
            if abs(tmp / all_total - ration) < 0.01:
                print(f'slim_idx: {slim_idx}  ration: {scale} , Number of parameter: % .4fM , ration: {abs(tmp / all_total)}' % (tmp / 1e6))
        print()


if __name__ == '__main__':
    # get_split_setting('mobilenet', [4,5], 0.5)
    # get_split_setting('resnet', [0], 0.5)
    net = ResNet18_cifar(num_classes=100, track_running_stats=False, slim_idx=1, scale=0.43).to('cuda')
    # net = vgg_16_bn(num_classes=62, track_running_stats=False, num_channels=1, slim_idx=0, scale=1).to('cuda')
    summary(net, (50, 3, 32, 32))

    # dummy_input = torch.randn(1, 3, 32, 32).to('cuda')
    # flops, params = profile(net, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
