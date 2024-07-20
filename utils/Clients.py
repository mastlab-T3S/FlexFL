import numpy as np
from utils.asynchronous_client_config import *


class Clients:
    def __init__(self, args):
        self.args = args
        self.clients_list = generate_asyn_clients([0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2], args.num_users)
        self.update_list = []  # (idx, version, time)
        self.train_set = set()
        # for i in range(self.args.num_users):
        #     self.clients_list.append(Node(1.4, 1.4, np.random.exponential(), 0, args))

    def train(self, idx, version):
        for i in range(len(self.update_list) - 1, -1, -1):
            if self.update_list[i][0] == idx:
                self.update_list.pop(i)
        client = self.get(idx)
        client.version = version
        client.comm_count += 1
        train_time = client.get_train_time()
        comm_time = client.get_comm_time()
        self.update_list.append([idx, version, train_time + comm_time])
        self.update_list.sort(key=lambda x: x[2])
        self.train_set.add(idx)

    def get_update_byLimit(self, limit):
        lst = []
        for update in self.update_list:
            if update[2] <= limit:
                lst.append(update)
        return lst

    def get_update(self, num):
        return self.update_list[0:num]

    def pop_update(self, num):
        res = self.update_list[0:num]
        max_time = self.update_list[num - 1][2]
        for update in self.update_list:
            if update[2] <= max_time:
                self.train_set.remove(update[0])
                client = self.get(update[0])
                client.comm_count += 1
            else:
                update[2] -= max_time
        self.update_list = self.update_list[num::]
        return res

    def get(self, idx):
        return self.clients_list[idx]

    def get_idle(self, num):
        idle = self.get_all_idle()

        if len(idle) < num:
            return []
        else:
            return list(np.random.choice(idle, num, replace=False))

    def get_all_idle(self):
        idle = set(range(self.args.num_users)).difference(self.train_set)
        return list(idle)

