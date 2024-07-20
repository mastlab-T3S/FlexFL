#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy


import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from utils.get_dataset import get_dataset
from utils.widar import WidarDataset
from models import KDLoss, split_model, DatasetSplit
from utils.ConnectHandler_client import ConnectHandler
from utils.options import args_parser
from utils.set_seed import set_random_seed
from torch.utils.data import DataLoader, Dataset
'''
返回的参数：
    模型权重
    本地数据量
'''
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    ID = args.ID
    # data = []
    # with open(f'./data/widar/{args.ID + 1}.pkl', 'rb') as f:
    #     print(f)
    #     data.append(torch.load(f))
    # x = [d[0] for d in data]
    # x = np.concatenate(x, axis=0).astype(np.float32)
    # x = (x - .0025) / .0119
    # y = np.concatenate([d[1] for d in data])
    # data = [(x[i], y[i]) for i in range(len(x))]
    #
    # dataset_train = WidarDataset(data)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    connectHandler = ConnectHandler(args.HOST, args.POST, ID)

    while True:
        recv = connectHandler.receiveFromServer()

        net = recv['net']
        requirement = recv['net_size']
        algorithm = recv['algorithm']
        round = recv['round']
        model_idx = recv['model_idx']
        ee = recv['ee']
        globList = recv['net_glob_list']
        user_id = recv["dict_users[ID]"]
        ldr_train = DataLoader(DatasetSplit(dataset_train, user_id, args), batch_size=args.local_bs, shuffle=True, drop_last=True)
        print(f"this round: {round},  received model {model_idx}")

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr * (args.lr_decay ** round),
                                        momentum=args.momentum, weight_decay=args.weight_decay)
        Predict_loss = 0
        KD = KDLoss(args)
        for iter in tqdm(range(args.local_ep)):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                if args.dataset == 'widar':
                    labels = labels.long()
                net.zero_grad()

                if algorithm == 'ScaleFL':
                    outputs = net(images, ee)
                    loss = 0.0
                    for j in range(len(outputs)):
                        if j == len(outputs) - 1:
                            loss += KD.ce_loss(outputs[j]["output"], labels) * (j + 1)
                        else:
                            gamma_active = round > args.epochs * 0.25
                            loss += KD.loss_fn_kd(outputs[j]["output"], labels, outputs[-1]["output"], gamma_active) * (j + 1)

                    loss /= len(outputs) * (len(outputs) + 1) / 2
                elif algorithm == 'FlexFL':
                    log_probs = net(images)['output']
                    loss = KD.ce_loss(log_probs, labels)
                    for j in range(0, model_idx):
                        kdnet = globList[j]
                        kdnet.load_state_dict(split_model(net.state_dict(), kdnet.state_dict()))
                        softLabel = kdnet(images)['output']
                        loss = loss + args.gamma * KD.loss_kd(log_probs, labels,
                                                                               softLabel) / model_idx
                else:
                    log_probs = net(images)['output']
                    loss = KD.ce_loss(log_probs, labels)

                loss.backward()
                optimizer.step()
                Predict_loss += loss.item()

        info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (args.local_ep * len(ldr_train)))
        print(info)

        requires_grad = []
        if algorithm == 'ScaleFL':
            for param in net.parameters():
                requires_grad.append(param.grad != None)

        return_dict = {}
        return_dict['return_net'] = net
        return_dict['local_len'] = len(ldr_train.dataset)
        return_dict['requires_grad'] = requires_grad
        connectHandler.uploadToServer(return_dict)
