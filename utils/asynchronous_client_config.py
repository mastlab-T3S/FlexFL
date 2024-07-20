import numpy as np
import random



class CommTime:
    loc = 0
    scale = 0

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale


class TrainTime:
    loc = 0
    scale = 0

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale


class AsynClient:
    comm_time = CommTime(0, 0)
    train_time = TrainTime(0, 0)

    def __init__(self, comm_loc, comm_scale, train_loc, train_scale):
        self.comm_time = CommTime(comm_loc, comm_scale)
        self.train_time = TrainTime(train_loc, train_scale)
        self.version = 0
        self.comm_count = 0

    def set_train_time(self, loc, scale):
        self.train_time = TrainTime(loc, scale)

    def set_comm_time(self, loc, scale):
        self.comm_time = CommTime(loc, scale)

    def get_train_time(self):
        return max(1, random.gauss(self.train_time.loc, self.train_time.scale))

    def get_comm_time(self):
        return max(1, random.gauss(self.comm_time.loc, self.comm_time.scale))


VERY_HIHG_QUALITY_CLIENT = {'loc': 100, 'scale': 5}
HIHG_QUALITY_CLIENT = {'loc': 150, 'scale': 10}
MEDIUM_QUALITY_CLIENT = {'loc': 200, 'scale': 20}
LOW_QUALITY_CLIENT = {'loc': 300, 'scale': 30}
VERY_LOW_QUALITY_CLIENT = {'loc': 500, 'scale': 50}

VERY_HIHG_QUALITY_NET = {'loc': 10, 'scale': 1}
HIHG_QUALITY_NET = {'loc': 15, 'scale': 2}
MEDIUM_QUALITY_NET = {'loc': 20, 'scale': 3}
LOW_QUALITY_NET = {'loc': 30, 'scale': 5}
VERY_LOW_QUALITY_NET = {'loc': 80, 'scale': 10}


def initialize_asyn_clients(client_config):
    asyn_config = []
    for config in client_config:
        train_time = config.get('train_time')
        comm_time = config.get('comm_time')
        asyn_client = AsynClient(comm_time.get('loc'), comm_time.get('scale'), train_time.get('loc'),
                                 train_time.get('scale'))
        asyn_config.append(asyn_client)
    return asyn_config


def generate_asyn_clients(net_quality_config, client_quality_config, client_num):
    asyn_clients = []

    net_quality_list = []
    client_quality_list = []

    very_high_net_num = int(net_quality_config[0] / sum(net_quality_config) * client_num)
    high_net_num = int(net_quality_config[1] / sum(net_quality_config) * client_num)
    medium_net_num = int(net_quality_config[2] / sum(net_quality_config) * client_num)
    low_net_num = int(net_quality_config[3] / sum(net_quality_config) * client_num)
    very_low_net_num = int(net_quality_config[4] / sum(net_quality_config) * client_num)

    net_quality_list += [VERY_HIHG_QUALITY_NET for _ in range(very_high_net_num)]
    net_quality_list += [HIHG_QUALITY_NET for _ in range(high_net_num)]
    net_quality_list += [MEDIUM_QUALITY_NET for _ in range(medium_net_num)]
    net_quality_list += [LOW_QUALITY_NET for _ in range(low_net_num)]
    net_quality_list += [VERY_LOW_QUALITY_NET for _ in range(very_low_net_num)]

    while len(net_quality_list) < client_num:
        net_quality_list.append(MEDIUM_QUALITY_NET)

    very_high_client_num = int(client_quality_config[0] / sum(client_quality_config) * client_num)
    high_client_num = int(client_quality_config[1] / sum(client_quality_config) * client_num)
    medium_client_num = int(client_quality_config[2] / sum(client_quality_config) * client_num)
    low_client_num = int(client_quality_config[3] / sum(client_quality_config) * client_num)
    very_low_client_num = int(client_quality_config[4] / sum(client_quality_config) * client_num)

    client_quality_list += [VERY_HIHG_QUALITY_CLIENT for _ in range(very_high_client_num)]
    client_quality_list += [HIHG_QUALITY_CLIENT for _ in range(high_client_num)]
    client_quality_list += [MEDIUM_QUALITY_CLIENT for _ in range(medium_client_num)]
    client_quality_list += [LOW_QUALITY_CLIENT for _ in range(low_client_num)]
    client_quality_list += [VERY_LOW_QUALITY_CLIENT for _ in range(very_low_client_num)]

    while len(client_quality_list) < client_num:
        client_quality_list.append(MEDIUM_QUALITY_CLIENT)

    for i in range(client_num):
        ra_client = random.randint(0, client_num - i - 1)
        ra_net = random.randint(0, client_num - i - 1)

        net_loc = net_quality_list[ra_net].get('loc')
        net_scale = net_quality_list[ra_net].get('scale')
        client_loc = client_quality_list[ra_client].get('loc')
        client_scale = client_quality_list[ra_client].get('scale')

        asyn_client = AsynClient(net_loc, net_scale, client_loc, client_scale)

        del client_quality_list[ra_client]
        del net_quality_list[ra_net]

        asyn_clients.append(asyn_client)

    return asyn_clients
