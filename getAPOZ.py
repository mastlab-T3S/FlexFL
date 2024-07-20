import copy
import math
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Subset

import wandb
from torch import nn
from tqdm import tqdm

from models import vgg_16_bn, ResNet18_cifar, MobileNetV2, test, LocalUpdate_FedAvg, Aggregation
from optimizer.Adabelief import AdaBelief
from utils.Clients import Clients
from utils.get_dataset import get_dataset
from utils.options import args_parser
from utils.set_seed import set_random_seed
from functools import partial


def hook(args, net, dataset_test, iter, run=None):
    # TEST PRETRAIN
    net = copy.deepcopy(net)
    net.eval()
    if args.model == "vgg":
        reluCount = [1] * 15
    elif args.model == "resnet":
        reluCount = [1, 6, 8, 12, 6]
    elif args.model == "mobilenet":
        reluCount = [1, 2, 4, 6, 8, 6, 6, 2, 1]
    else:
        assert "unrecognized model"
    APOZ = [[] for _ in range(sum(reluCount))]
    MAT = [0] * sum(reluCount)

    def calculate_zero_ratio(layer, input, output, layer_idx):
        for sample_output in output:
            martix = (sample_output == 0).float()
            try:
                MAT[layer_idx] += martix
            except:
                MAT[layer_idx] = martix

            zero_ratio = (sample_output == 0).float().mean()
            APOZ[layer_idx].append(zero_ratio.item())

    hooks = []

    def add_hooks(model):
        if args.model == "mobilenet":
            relu_layers = []
            for idx, (name, layer) in enumerate(model.named_modules()):
                if isinstance(layer, torch.nn.ReLU6):
                    relu_layers.append((name, layer))
            for idx, layer in enumerate(relu_layers):
                _hook = layer[1].register_forward_hook(partial(calculate_zero_ratio, layer_idx=idx))
                hooks.append(_hook)
        else:
            relu_layers = []
            for idx, (name, layer) in enumerate(model.named_modules()):
                if isinstance(layer, torch.nn.ReLU):
                    relu_layers.append((name, layer))
            for idx, layer in enumerate(relu_layers):
                _hook = layer[1].register_forward_hook(partial(calculate_zero_ratio, layer_idx=idx))
                hooks.append(_hook)

    add_hooks(net)
    result_dict = {"acc": test(net, dataset_test, args)}
    return_value = []
    for i, row in enumerate(APOZ):
        row_mean = sum(row) / len(row)
        key = f"layer{i} APOZ"
        result_dict[key] = row_mean
        return_value.append(row_mean)
        key = f"heat{i}"
        if MAT[i].dim() == 3:
            result_dict[key] = MAT[i].mean(dim=[1, 2])
        else:
            result_dict[key] = MAT[i]

    if args.log:
        run.log(result_dict, step=iter)
    for _hook in hooks:
        _hook.remove()
    return calculate_average(return_value, reluCount)


def APOZfunction(args, run, run_round=0):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    dataset_train, dataset_test, dict_users = get_dataset(args)

    net_glob = getNet(args, [1] * 50)
    print(net_glob)
    net_glob.train()
    loss_func = nn.CrossEntropyLoss()

    # Split 80% test dataset to proxy train dataset, 20% test dataset to proxy test dataset for getting APoZ
    total_size = len(dataset_test)
    indices = list(range(total_size))
    split = int(np.floor(0.8 * total_size))
    np.random.shuffle(indices)
    sub_train_index, sub_test_index = indices[:split], indices[split:]
    sub_train_dataset = Subset(dataset_test, sub_train_index)
    sub_test_dataset = Subset(dataset_test, sub_test_index)
    if args.only == 1:
        dataset_train = sub_train_dataset
        dataset_test = sub_test_dataset
    ldr_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)

    if run_round != 0:
        return hook(args, net_glob, dataset_test, run_round, run)

    # START PRETRAIN
    for _iter in tqdm(range(args.pretrain)):
        optimizer = torch.optim.Adam(net_glob.parameters(), lr=0.001)
        Predict_loss = 0
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(images)['output']
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            Predict_loss += loss.item()
        print(f'Current pretrain round {_iter} , Loss {Predict_loss / len(ldr_train)}')
        if args.e == 1:
            hook(args, net_glob, dataset_test, _iter + 1, run)
    return hook(args, net_glob, dataset_test, args.pretrain + 1, run)

def getNet(args, rate):
    if args.model == "vgg":
        net = vgg_16_bn(num_classes=args.num_classes, track_running_stats=False, num_channels=args.num_channels,
                        rate=rate).to(args.device)
        # train vgg using cifar10
    elif args.model == "resnet":
        net = ResNet18_cifar(num_classes=args.num_classes, track_running_stats=False, num_channels=args.num_channels,
                             rate=rate).to(args.device)
    elif args.model == "mobilenet":
        net = MobileNetV2(channels=args.num_channels, num_classes=args.num_classes, trs=False, rate=rate).to(
            args.device)
    return net


def calculate_average(input_list, reluCount):
    averages = []
    start = 0
    for count in reluCount:
        end = start + count
        segment = input_list[start:end]
        average = sum(segment) / count
        averages.append(average)
        start = end
    return averages


def calculate_max(input_list, reluCount):
    averages = []
    start = 0
    for count in reluCount:
        end = start + count
        segment = input_list[start:end]
        averages.append(max(segment))
        start = end
    return averages


def calculate_min(input_list, reluCount):
    averages = []
    start = 0
    for count in reluCount:
        end = start + count
        segment = input_list[start:end]
        averages.append(min(segment))
        start = end
    return averages


def findConvLayerOrLinearLayerOutChannel(layers):
    for index, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels
        elif isinstance(layer, nn.Linear):
            return layer.out_features


def calculateScale(args, net_glob, APOZ):
    LayerParams = np.array([])
    for layer in net_glob.features:
        params = sum(p.numel() for p in layer.parameters())
        LayerParams = np.append(LayerParams, params)
    APOZ = torch.tensor(np.array(APOZ))
    scale_rate = 1 - APOZ * np.log(LayerParams) / np.log(max(LayerParams))
    ans = [torch.ones(len(APOZ))]
    ans.insert(0, get_net_scale(args, 1-args.decrease, scale_rate))
    ans.insert(0, get_net_scale(args, 0.5, scale_rate))
    ans.insert(0, get_net_scale(args, 0.5-args.decrease, scale_rate))
    ans.insert(0, get_net_scale(args, 0.25, scale_rate))
    return scale_rate, ans


def modelList(args, scaleList):
    # first is smallest

    netRateList = scaleList
    netGlobList = []
    netSlimInfo = []

    for rate in netRateList:
        net = getNet(args, rate)
        totalParam = sum([param.nelement() for param in net.parameters()])
        featureParam = sum([param.nelement() for param in net.features.parameters()])
        net.to(args.device)
        net.train()
        print("==" * 50)
        print(
            f'[model config]  model_name:{args.model}, totalParam:{totalParam} , featureParam:{featureParam}, rate:{rate}')
        netGlobList.append(net)
        netSlimInfo.append(rate)
    return netGlobList, netSlimInfo


def get_net_scale(args, param_percent, scale_rate):
    net = getNet(args, [1] * 20)
    tolerance = 0.05
    originFeatureParams = sum(p.numel() for p in net.features.parameters())
    result = None
    for gamma in range(1, 200):
        temp = scale_rate * 100 / gamma
        temp = torch.clamp(temp, min=0.01, max=1)
        net = getNet(args, temp)
        currentParams = sum(p.numel() for p in net.features.parameters())
        if (param_percent - tolerance) <= (currentParams / originFeatureParams) <= (param_percent + tolerance):
            result = temp
            break
    assert result is not None, scale_rate
    return result


if __name__ == "__main__":
    args = args_parser()
    if int(args.log) == 1:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="Fed",
            name=f"APOZ {args.model} {args.dataset} IID{args.iid} Pretrain{args.pretrain} {args.client_hetero_ration} distillation{args.gamma} avg serveronly{args.only}" + str(
                datetime.now()),
        )
    else:
        run = None
    print(APOZfunction(args, run))
