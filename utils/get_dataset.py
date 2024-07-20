#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import pickle

import torch
from sklearn.model_selection import train_test_split
import numpy
import torch
"""
from datasets import Dataset, load_dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer
"""
from utils.sampling import *
from utils.dataset_utils import separate_data, read_record
from utils.FEMNIST import FEMNIST
from utils.tinyimagenet import TinyImageNet
from utils.widar import WidarDataset
import os
import json
from collections import Counter


def get_dataset(args):
    file = os.path.join("data", args.dataset + "_" + str(args.num_users))
    if args.iid:
        file += "_iid"
    else:
        file += "_noniidCase" + str(args.noniid_case)

    if args.noniid_case > 4:
        file += "_beta" + str(args.data_beta)

    file += ".json"
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.generate_data:
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar10':
        args.num_channels = 3
        args.num_classes = 10
        # trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                           transforms.RandomHorizontalFlip(),
        #                                           transforms.ToTensor(),
        #                                           transforms.Normalize(mean=[0.491, 0.482, 0.447],
        #                                                                std=[0.247, 0.243, 0.262])])
        trans_cifar10_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                       std=[0.247, 0.243, 0.262])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                     std=[0.247, 0.243, 0.262])])

        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)

    elif args.dataset == 'cifar100':
        args.num_channels = 3
        args.num_classes = 100
        # trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                            transforms.RandomHorizontalFlip(),
        #                                            transforms.ToTensor(),
        #                                            transforms.Normalize(mean=[0.507, 0.487, 0.441],
        #                                                                 std=[0.267, 0.256, 0.276])])
        #
        trans_cifar100_train = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                        std=[0.267, 0.256, 0.276])])
        trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                      std=[0.267, 0.256, 0.276])])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False, download=True, transform=trans)
        if args.generate_data:
            if args.iid:
                dict_users = fashion_mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = fashion_mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'widar':
        # files = glob.glob(r'./data/widar/*.pkl')
        # data = []
        # for file in files:
        #     try:
        #         with open(file, 'rb') as f:
        #             print(f)
        #             data.append(torch.load(f))
        #     except pickle.UnpicklingError as e:
        #         print(f'Error loading {f}')
        # x = [d[0] for d in data]
        # x = np.concatenate(x, axis=0, dtype=np.float32)
        # x = (x - .0025) / .0119
        # y = np.concatenate([d[1] for d in data])
        # data = [(x[i], y[i]) for i in range(len(x))]
        # torch.save(data, f'./data/widar/widar.pkl')
        args.num_channels = 22
        args.num_classes = 22
        data = torch.load(f'./data/widar/widar.pkl')

        data_train, data_test = train_test_split(data, test_size=0.2, random_state=args.seed)
        dataset_train = WidarDataset(data_train)
        dataset_test = WidarDataset(data_test)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)


    elif args.dataset == 'TinyImagenet':

        trans_imagenet_train = transforms.Compose([transforms.RandomCrop(64),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                                                        std=[0.2770, 0.2691, 0.2821])])
        trans_imagenet_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2770, 0.2691, 0.2821])])
        data_dir = './data/tiny-imagenet-200/'
        dataset_train = TinyImageNet(data_dir, train=True, transform=trans_imagenet_train)
        dataset_test = TinyImageNet(data_dir, train=False, transform=trans_imagenet_val)
        args.num_channels = 3
        args.num_classes = 200

        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(True)
        dataset_test = FEMNIST(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        args.num_channels = 1
        args.num_classes = 62

    elif args.dataset == 'Shakespeare':
        dataset_train = ShakeSpeare(True)
        dataset_test = ShakeSpeare(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
    elif args.dataset == 'SST2':
        # dataset = load_dataset(r"/home/mastlab/.cache/huggingface/datasets/glue/sst2")
        dataset = load_dataset(r"./data/sst2")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset_train = SST_2_Dataset(dataset, "train", tokenizer, 256)
        dataset_test = SST_2_Dataset(dataset, "test", tokenizer, 256)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_train = datasets.MNIST(r"./data/mnist", transform=transform, train=True, download=True)
        dataset_test = datasets.MNIST(r"./data/mnist", transform=transform, download=True)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == "emnist":
        transform = transforms.Compose(
            [transforms.ToTensor()])
        dataset_train = datasets.EMNIST(root=r"./data/emnist", split="byclass", train=True, download=True,
                                        transform=transform)
        dataset_test = datasets.EMNIST(root=r"./data/emnist", split="byclass", download=True, transform=transform)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)

    else:
        exit('Error: unrecognized dataset')

    if args.generate_data:
        with open(file, 'w') as f:
            dataJson = {"dataset": args.dataset, "num_users": args.num_users, "iid": args.iid,
                        "noniid_case": args.noniid_case, "data_beta": args.data_beta, "train_data": dict_users}
            json.dump(dataJson, f)

    return dataset_train, dataset_test, dict_users
