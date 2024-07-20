#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
from tqdm import tqdm

from Algorithm.Training_Decoupled import Decoupled
from Algorithm.Training_FlexFL import FlexFL
from Algorithm.Training_ScaleFL import ScaleFL
from getAPOZ import hook
from models.transformer import Transformer
from plot_figure import plot_client_distribution

# matplotlib.use('Agg')
import copy

from utils.Clients import Clients
from utils.options import args_parser
from models import *
from utils.get_dataset import get_dataset
from utils.utils import save_result
from utils.set_seed import set_random_seed
from Algorithm import *

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def FedAvg(net_glob, dataset_train, dataset_test, dict_users, run):
    net_glob.train()
    print(net_glob)

    # training
    acc = []
    clients = Clients(args)
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"this epoch choose: {idxs_users}")
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

            clients.train(idx, iter)
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))
        if args.e:
            hook(args, net_glob, copy.deepcopy(dataset_test), iter + 1, run)
    save_result(acc, 'test_acc', args)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)
    # plot_client_distribution(dataset_train, dict_users)

    if args.model == 'cnn':
        if args.dataset == 'femnist':
            net_glob = CNNFashionMnist(args)
        elif args.dataset == 'mnist':
            net_glob = CNNMnist(args)
        elif args.use_project_head:
            net_glob = ModelFlexFL(args.model, args.out_dim, args.num_classes)
        elif 'cifar' in args.dataset:
            net_glob = CNNCifar(args)
    elif 'resnet' in args.model:
        net_glob = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)  # 默认为3通道
    elif 'mobilenet' in args.model:
        net_glob = MobileNetV2(args.num_channels, args.num_classes)
    elif 'vgg' in args.model:
        net_glob = vgg_16_bn(num_classes=args.num_classes, track_running_stats=True, num_channels=args.num_channels)
    elif 'lstm' in args.model:
        net_glob = CharLSTM()
    elif 'transformer' in args.model:
        net_glob = Transformer(vocab_size=30522, d_model=128, nhead=8, num_encoder_layers=8, max_len=256,
                               slim_idx=2,
                               scale=0.5, dropout=0.1)
    if args.algorithm == 'FedAvg':
        net_glob.to(args.device)
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'HeteroFL':
        HeteroFL(args, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'ScaleFL':
        ScaleFL(args, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'Decoupled':
        Decoupled(args, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FlexFL':
        FlexFL(args, dataset_train, dataset_test, dict_users)
    else:
        raise "%s algorithm has not achieved".format(args.algorithm)
