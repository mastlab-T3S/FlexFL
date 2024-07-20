#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import os


def save_result(data, ylabel, args):
    data = {'base': data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                    ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"))
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                    ylabel, args.epochs, args.lr,
                                                    datetime.datetime.now().strftime(
                                                        "%Y_%m_%d_%H_%M_%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()


def my_save_result(data, label, ylabel, args):  # label 这一行对应的模型配置
    data = {label: data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:  # iid
        file = f'{args.dataset}_{args.algorithm}_{args.model}_{args.client_hetero_ration.replace(":", "")}_{ylabel}_{datetime.datetime.now().strftime("%m_%d")}.txt'
    else:  # Non-iid
        path += '/{}'.format(args.data_beta)
        file = f'{args.dataset}_{args.algorithm}_{args.model}_{args.client_hetero_ration.replace(":", "")}_{ylabel}_{datetime.datetime.now().strftime("%m_%d")}.txt'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()

    return os.path.join(path, file)


def get_final_acc(file):
    idx = 0
    max = 0
    list = []
    with open(file, 'r') as f:
        for id, item in enumerate(f.readlines()):
            tmp = item.split()[5:]
            list.append(tmp)

    for i in range(len(list[0])):
        t = [float(row[i]) for row in list]

        # 计算第三列的平均值
        total = sum(t) / len(t) + t[-1]
        if  total > max :
            idx = i
            max = total

    print([row[idx] for row in list])

if __name__ == '__main__':
    get_final_acc(r'C:\Users\20626\Desktop\Code\fed_master\output\0\cifar10_FedSlim_vgg_433_acc_10_30.txt')