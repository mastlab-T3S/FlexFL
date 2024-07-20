#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
from collections import OrderedDict

import numpy as np
import torch

from models import vgg_16_bn, ResNet18_cifar, MobileNetV2


def Aggregation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg


def split_model(global_param, slim_param):
    param = copy.deepcopy(slim_param)
    for k, v in param.items():  # 遍历所有层，每一层遍历
        if v.dim() > 1:
            d1 = v.shape[0]
            d2 = v.shape[1]
            param[k] = global_param[k][:d1, :d2]
        elif v.dim() == 1:
            d1 = v.shape[0]
            param[k] = global_param[k][:d1]
        else:
            param[k] = global_param[k]
    return param


def Aggregation_FedSlim(w, lens, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for k, v in w_avg.items():  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if parameter_type == 'weight':
                if v.dim() > 1:  # 卷积  和 线性层
                    d1 = w[m][k].shape[0]
                    d2 = w[m][k].shape[1]
                    tmp_v[:d1, :d2] += w[m][k] * lens[m]  # 第m个客户端的 k 层参数
                    count[k][:d1, :d2] += lens[m]
                else:  # BN层
                    d1 = w[m][k].shape[0]
                    tmp_v[:d1] += w[m][k] * lens[m]
                    count[k][:d1] += lens[m]
            else:
                d1 = w[m][k].shape[0]
                tmp_v[:d1] += w[m][k] * lens[m]
                count[k][:d1] += lens[m]

        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def Aggregation_ScaleFL(w, lens, grad_info, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for idx, (k, v) in enumerate(w_avg.items()):  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if grad_info[m][idx]:
                if parameter_type == 'weight':
                    if v.dim() > 1:  # 卷积  和 线性层
                        d1 = w[m][k].shape[0]
                        d2 = w[m][k].shape[1]
                        tmp_v[:d1, :d2] += w[m][k] * lens[m]  # 第m个客户端的 k 层参数
                        count[k][:d1, :d2] += lens[m]
                    else:  # BN层
                        d1 = w[m][k].shape[0]
                        tmp_v[:d1] += w[m][k] * lens[m]
                        count[k][:d1] += lens[m]
                else:
                    d1 = w[m][k].shape[0]
                    tmp_v[:d1] += w[m][k] * lens[m]
                    count[k][:d1] += lens[m]

        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def get_model_list(args):
    model_rate = args.width_ration
    depth_list = args.depth_saved

    net_glob_list = []
    net_slim_info = []
    for i in model_rate:
        for depth in depth_list:
            if args.model == 'vgg':
                net = vgg_16_bn(num_classes=args.num_classes, track_running_stats=False, num_channels=args.num_channels,
                                rate=[1] * depth + [i] * (15 - depth)).to(args.device)
            elif args.model == 'resnet':
                if args.dataset == 'widar':
                    pass
                    # net = ResNet18_widar(num_classes=args.num_classes, track_running_stats=False, slim_idx=depth, scale=i)
                else:
                    net = ResNet18_cifar(args.num_channels, args.num_classes, False, [1] * depth + [i] * (5 - depth))

            elif args.model == 'mobilenet':
                net = MobileNetV2(args.num_channels, args.num_classes, False, [1] * depth + [i] * (9 - depth))

            total = sum([param.nelement() for param in net.parameters()])
            net.to(args.device)
            net.train()
            print("==" * 50)
            print('【model config】  model_name:{}, width:{} , depth:{}, param:{}MB'.format(args.model, i, depth, total * 4 / 1e6))
            # print(net)
            net_glob_list.append(net)
            net_slim_info.append((i, depth, total / 1e6))  # 宽度 深度 参数量

            if i == 1.0:
                break
    return net_glob_list, net_slim_info


def select_clients(args, ration_users, net_glob_list_len):
    my_list = list(map(float, args.client_hetero_ration.split(':')))
    hetero_proportion = [round(x / sum(my_list), 2) for x in my_list]

    idx_users = []
    if net_glob_list_len == 7:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if int(model_type / 3) == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif int(model_type / 3) == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif int(model_type / 3) == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if int(model_type / 3) == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif int(model_type / 3) == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif int(model_type / 3) == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 5:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
                elif model_type == 3:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:3])), args.num_users - 1))
                elif model_type == 4:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:4])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
                elif model_type == 3:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:3])), int(args.num_users * sum(hetero_proportion[:4])) - 1))
                elif model_type == 4:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:4])), int(args.num_users * sum(hetero_proportion[:5])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 3:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 2:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    elif net_glob_list_len == 1:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))

    return idx_users


def summon_clients(args):
    clients = []  # Every client is a tuple, miu ,sigma
    client_hetero_ration = list(map(float, args.client_hetero_ration.split(':')))
    users25 = int(args.num_users * round(client_hetero_ration[0] / sum(client_hetero_ration), 2))
    users50 = int(args.num_users * round(client_hetero_ration[1] / sum(client_hetero_ration), 2))
    users100 = int(args.num_users * round(client_hetero_ration[2] / sum(client_hetero_ration), 2))

    if args.r == 0:
        for i in range(users25):
            clients.append((35, random.choice([5, 8, 10])))
        for i in range(users50):
            clients.append((60, random.choice([5, 8, 10])))
        for i in range(users100):
            clients.append((110, random.choice([5, 8, 10])))
        return clients
    elif args.r == 1:
        for i in range(users25):
            clients.append((35, random.choice([0])))
        for i in range(users50):
            clients.append((60, random.choice([0])))
        for i in range(users100):
            clients.append((110, random.choice([0])))
        return clients
    elif args.r == 2:
        for i in range(users25):
            clients.append((35, random.choice([10, 20, 30])))
        for i in range(users50):
            clients.append((60, random.choice([10, 20, 30])))
        for i in range(users100):
            clients.append((110, random.choice([10, 20, 30])))
        return clients


def FlexFL_select_clients(args, clients, models, is3=False):
    selected_user = []
    current_user_list = list(map(float, args.client_hetero_ration.split(':')))
    if is3:
        for model in models:
            if model == 0:
                user_idx = random.randint(0, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model3(resource, 0)))
            elif model == 1:
                user_idx = random.randint(4, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model3(resource, 1)))
            elif model == 2:
                user_idx = random.randint(14, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model3(resource, 2)))
            else:
                raise Exception
    else:
        for model in models:
            if model == 0:
                user_idx = random.randint(0, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 0)))
            elif model == 2:
                user_idx = random.randint(4, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 2)))
            elif model == 4:
                user_idx = random.randint(14, args.num_users - 1)
                resource = clients[user_idx][0] - abs(np.random.normal(0, clients[user_idx][1], 1)[0])
                selected_user.append((user_idx, resource_to_model(resource, 4)))
            else:
                raise Exception

    return selected_user  # user_idx , model


def resource_to_model(resource, original_model):
    if resource < 40:
        model = 0
    elif resource < 50:
        model = 1
    elif resource < 90:
        model = 2
    elif resource < 100:
        model = 3
    else:
        model = 4
    return min(original_model, model)


def resource_to_model3(resource, original_model):
    if resource < 50:
        model = 0
    elif resource < 100:
        model = 1
    else:
        model = 2
    return min(original_model, model)
