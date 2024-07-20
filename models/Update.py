#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random

from models.Fed import split_model
from optimizer.Adabelief import AdaBelief


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.model = args.model

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if "transformer" in self.model:
            image, label, mask = self.dataset[self.idxs[item]].values()
            return image, label, mask
        else:
            image, label = self.dataset[self.idxs[item]]
            return image, label


class KDLoss(nn.Module):
    def __init__(self, args):
        super(KDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.T = int(args.T)
        self.gamma = int(args.gamma)

    def loss_fn_kd(self, pred, target, soft_target, gamma_active=True):
        _ce = self.ce_loss(pred, target)
        T = self.T
        if self.gamma and gamma_active:
            # _ce = (1. - self.gamma) * _ce
            _kld = self.kld_loss(self.log_softmax(pred / T), self.softmax(soft_target / T)) * self.gamma * T * T
        else:
            _kld = 0
        loss = _ce + _kld
        return loss

    def loss_kd(self, pred, target, soft_target, gamma_active=True):
        T = self.T
        if self.gamma and gamma_active:
            # _ce = (1. - self.gamma) * _ce
            _kld = self.kld_loss(self.log_softmax(pred / T), self.softmax(soft_target / T)) * T * T
        else:
            _kld = 0
        return _kld


class LocalUpdate_ScaleFL(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = KDLoss(args)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net, ee):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()
                net.zero_grad()
                outputs = net(images, ee)
                loss = 0.0
                for j in range(len(outputs)):
                    if j == len(outputs) - 1:
                        loss += self.loss_func.ce_loss(outputs[j]["output"], labels) * (j + 1)
                    else:
                        gamma_active = round > self.args.epochs * 0.25
                        loss += self.loss_func.loss_fn_kd(outputs[j]["output"], labels, outputs[-1]["output"],
                                                          gamma_active) * (j + 1)

                loss /= len(outputs) * (len(outputs) + 1) / 2

                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        requires_grad = []
        for param in net.parameters():
            requires_grad.append(param.grad != None)

        return net.state_dict(), requires_grad


class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):
            if "transformer" in self.args.model:
                for batch_idx, (images, labels, mask) in enumerate(self.ldr_train):
                    images, labels, mask = images.to(self.args.device), labels.to(self.args.device), mask.to(
                        self.args.device)
                    net.zero_grad()
                    log_probs = net(images, mask)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    Predict_loss += loss.item()
            else:
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    if self.args.dataset == 'widar':
                        labels = labels.long()
                    net.zero_grad()
                    log_probs = net(images)['output']
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_FlexFL(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = KDLoss(args)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net, globList, modelLevel=2):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func.ce_loss(log_probs, labels)
                for j in range(0, modelLevel):
                    kdnet = globList[j]
                    kdnet.load_state_dict(split_model(copy.deepcopy(net.state_dict()), kdnet.state_dict()))
                    softLabel = kdnet(images)['output']
                    loss = loss + self.args.gamma * self.loss_func.loss_kd(log_probs, labels, softLabel) / modelLevel
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_FedSlim(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Penalize_loss = 0

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                # for fedprox
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += (
                            (self.prox_alpha / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss = predictive_loss + fed_prox_reg
                Predict_loss += predictive_loss.item()
                Penalize_loss += fed_prox_reg.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Penalize loss={:.4f}'.format(Penalize_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()
