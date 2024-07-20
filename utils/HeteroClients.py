import math
import random

from utils.hetero_client_config import generate_hetero_clients


class HeteroClients:

    def __init__(self, args, net_slim_info):
        self.args = args
        self.net_slim_info = net_slim_info
        my_list = list(map(float, args.client_hetero_ration.split(':')))
        self.hetero_proportion = [round(x / sum(my_list), 2) for x in my_list]

        self.clients_list = generate_hetero_clients(self.hetero_proportion, args.num_users)
        self.model_size = sorted([net_info[2] for net_info in self.net_slim_info])

        self.COUNT_TABLE = [[1 for _ in range(3)] for _ in self.clients_list]  # 根据客户端的返回模型来进行统计
        self.RESOURCE_TABLE = [[1 for _ in range(7)] for _ in self.clients_list]

    def train(self, idx, model_idx):

        client = self.get(idx)

        requirement = self.net_slim_info[model_idx][2]
        real_resource = client.get_resource()

        if real_resource >= requirement:  # accept
            feedback_model_idx = model_idx
            for i in range(model_idx, 7):
                self.RESOURCE_TABLE[idx][i] += 1
            if model_idx == 6:
                self.RESOURCE_TABLE[idx][model_idx] += 3

        else:  # reject
            feedback_model_idx = -1
            for i in range(len(self.model_size) - 1):
                if self.model_size[i + 1] > real_resource:
                    feedback_model_idx = i
                    break
            self.RESOURCE_TABLE[idx][feedback_model_idx] += 3
            punishment = 1
            for i in range(feedback_model_idx + 1, 7):
                self.RESOURCE_TABLE[idx][i] = max(self.RESOURCE_TABLE[idx][i] - punishment, 0)
                punishment += 1

        self.COUNT_TABLE[idx][int(feedback_model_idx / 3)] += 1
        self.COUNT_TABLE[idx][int(model_idx / 3)] += 1

        return feedback_model_idx  # 返回客户端实际训练的模型下标

    def get(self, idx):
        return self.clients_list[idx]

    def select_clients(self, ration_users):  # [2, 2, 2, 2, 2, 5, 5, 5, 6, 6]
        idx_users = []  # 根据选择好的模型列表 返回这一轮中最合适的客户端列表
        reward_table = self.get_reward_table()
        p = [list(row) for row in zip(*reward_table)]

        for x in p:
            idx1 = int(self.hetero_proportion[0] * self.args.num_users)
            idx2 = int(sum(self.hetero_proportion[:2]) * self.args.num_users)
            print(f"\t{[sum(x[:idx1]) / sum(x), sum(x[idx1:idx2]) / sum(x), sum(x[idx2:]) / sum(x)]}")

        for model_idx in ration_users:
            random_client_index = random.choices(list(range(0, self.args.num_users)), weights=p[int(model_idx / 3)])[0]
            idx_users.append(random_client_index)
            p[0][random_client_index] = 0
            p[1][random_client_index] = 0
            p[2][random_client_index] = 0

        return idx_users

    def get_reward_table(self):
        R_s = []  # 成功率表
        for i in range(len(self.RESOURCE_TABLE)):
            total = sum(self.RESOURCE_TABLE[i])
            tmp = []  # 小模型 中模型 大模型在该客户端上的成功率
            tmp.append((sum(self.RESOURCE_TABLE[i][0:]) + sum(self.RESOURCE_TABLE[i][1:]) + sum(self.RESOURCE_TABLE[i][2:])) / (3 * total))
            tmp.append((sum(self.RESOURCE_TABLE[i][3:]) + sum(self.RESOURCE_TABLE[i][4:]) + sum(self.RESOURCE_TABLE[i][5:])) / (3 * total))
            tmp.append(sum(self.RESOURCE_TABLE[i][6:]) / total)
            R_s.append(tmp)

        R_c = [[1 / math.sqrt(x) for x in row] for row in self.COUNT_TABLE]  # 好奇心表

        reward_table = [[min(a, 0.5) * b for a, b in zip(row1, row2)] for row1, row2 in zip(R_s, R_c)]

        return reward_table
