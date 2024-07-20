import math
import random

import numpy as np

# from options import args_parser
# from utils import save_result

# config for vgg16
# HIGH_RESOURCE = {'loc': 35, 'scale': 2, 'up_limit': 40, 'low_limit': 25}
# MEDIUM_RESOURCE = {'loc': 16, 'scale': 1, 'up_limit': 19, 'low_limit': 13}
# LOW_RESOURCE = {'loc': 8, 'scale': 1, 'up_limit': 11, 'low_limit': 6}

# config for resnet18
HIGH_RESOURCE = {'loc': 48, 'scale': 2, 'up_limit': 55, 'low_limit': 42}
MEDIUM_RESOURCE = {'loc': 22, 'scale': 1, 'up_limit': 25, 'low_limit': 19}
LOW_RESOURCE = {'loc': 11, 'scale': 1, 'up_limit': 14, 'low_limit': 8}


class ComputeResource:
    loc = 0
    scale = 0

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale


class HeteroClient:
    def __init__(self, compu_loc, compu_scale, up_limit, low_limit):
        self.compute_resource = ComputeResource(compu_loc, compu_scale)
        self.up_limit = up_limit
        self.low_limit = low_limit

    def get_resource(self):
        resource = random.gauss(self.compute_resource.loc, self.compute_resource.scale)
        if resource > self.up_limit:
            resource = self.up_limit
        elif resource < self.low_limit:
            resource = self.low_limit
        return resource


def generate_hetero_clients(hetero_proportion, client_num):  # [0.4, 0.3, 0.3]  100
    hetero_clients = []
    for id, proportion in enumerate(hetero_proportion):
        for _ in range(int(proportion * client_num)):
            if id == 0:
                hetero_clients.append(HeteroClient(LOW_RESOURCE['loc'], LOW_RESOURCE['scale'], LOW_RESOURCE['up_limit'], LOW_RESOURCE['low_limit']))
            elif id == 1:
                hetero_clients.append(HeteroClient(MEDIUM_RESOURCE['loc'], MEDIUM_RESOURCE['scale'], MEDIUM_RESOURCE['up_limit'], MEDIUM_RESOURCE['low_limit']))
            else:
                hetero_clients.append(HeteroClient(HIGH_RESOURCE['loc'], HIGH_RESOURCE['scale'], HIGH_RESOURCE['up_limit'], HIGH_RESOURCE['low_limit']))
    # random.shuffle(hetero_clients)
    return hetero_clients


# =======================================================================================================


def get_reward_table(RESOURCE_TABLE, COUNT_TABLE):
    R_s = []
    for i in range(len(RESOURCE_TABLE)):
        total = sum(RESOURCE_TABLE[i])
        tmp = []  # 小模型 中模型 大模型在该客户端上的成功率
        tmp.append((sum(RESOURCE_TABLE[i][0:]) + sum(RESOURCE_TABLE[i][1:]) + sum(RESOURCE_TABLE[i][2:])) / (3 * total))
        tmp.append((sum(RESOURCE_TABLE[i][3:]) + sum(RESOURCE_TABLE[i][4:]) + sum(RESOURCE_TABLE[i][5:])) / (3 * total))
        tmp.append(sum(RESOURCE_TABLE[i][6:]) / total)
        R_s.append(tmp)

    R_c = [[1 / math.sqrt(x) for x in row] for row in COUNT_TABLE]

    reward_table = [[min(a, 0.5) * b for a, b in zip(row1, row2)] for row1, row2 in zip(R_s, R_c)]  # RL+SC
    # reward_table = R_s # RL+S
    # reward_table = R_c # RL+C
    return reward_table


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)  # 保证不同方式，随机采样的模型样本都一致
    hetero_proportion = [0.4, 0.3, 0.3]
    clients = generate_hetero_clients(hetero_proportion, 100)

    # model_size = [5.66, 6.48, 8.39, 14.84, 15.41, 16.81, 33.65]
    model_size = [8.48, 9.56, 11.61, 21.09, 21.82, 23.22, 45.51]

    COUNT_TABLE = [[1 for _ in range(3)] for _ in clients]  # 根据客户端的返回模型来进行统计
    RESOURCE_TABLE = [[1 for _ in range(7)] for _ in clients]

    dispatch_size = 0
    return_size = 0
    waste_list = []
    for epoch in range(1000):

        reward_table = get_reward_table(RESOURCE_TABLE, COUNT_TABLE)
        p = [list(row) for row in zip(*reward_table)]

        # model_list = [random.choice([6]) for _ in range(10)]
        model_list = [random.choice([0, 1, 2, 3, 4, 5, 6]) for _ in range(10)]
        # model_list = [2, 2, 2, 2, 2, 5, 5, 5, 6, 6]
        # model_list = np.random.choice([2, 5, 6], 10)  # 模型选择

        # model_list = [random.randint(0, 2) for _ in range(4)]
        # model_list.extend([random.randint(3, 5) for _ in range(3)])
        # model_list.extend([6 for _ in range(3)])
        client_list = []
        feedback_list = []
        for model_index in model_list:  # 大的本来给的就少 所以权重得大一点

            # 选择模型
            requirement = model_size[model_index]

            # 选择客户端
            # random_client_index = random.choices(list(range(0, 100)))[0]  # random selection
            random_client_index = random.choices(list(range(0, 100)), weights=p[int(model_index / 3)])[0]
            p[0][random_client_index] = 0
            p[1][random_client_index] = 0
            p[2][random_client_index] = 0
            client_list.append(random_client_index)

            # 客户端训练返回

            real_resource = clients[random_client_index].get_resource()
            if real_resource >= requirement:  # accept
                feedback = requirement
                for i in range(model_index, 7):
                    RESOURCE_TABLE[random_client_index][i] += 1
                if model_index == 6:
                    RESOURCE_TABLE[random_client_index][model_index] += 3
            else:  # reject
                low_size = 0
                up_size = 0
                for i in range(len(model_size)):
                    if model_size[i] < real_resource and model_size[i + 1] > real_resource:
                        low_size = model_size[i]
                        up_size = model_size[i + 1]
                        feedback = low_size
                        break
                RESOURCE_TABLE[random_client_index][i] += 3
                punishment = 1
                for t in range(i + 1, 7):
                    RESOURCE_TABLE[random_client_index][t] = max(RESOURCE_TABLE[random_client_index][t] - punishment, 0)
                    punishment += 1
            COUNT_TABLE[random_client_index][int(model_size.index(feedback) / 3)] += 1
            COUNT_TABLE[random_client_index][int(model_index / 3)] += 1
            feedback_list.append(model_size.index(feedback))

        print(f"epoch:{epoch}")
        print(f"\tmodel_list:{model_list}")
        print(f"\tfeedback_list:{feedback_list}")
        print(f"\tclient_list:{client_list}")
        flag = [a - b for a, b in zip(model_list, feedback_list)]
        print(f"\tflag_list:{flag}")
        for x in p:
            idx1 = int(hetero_proportion[0] * 100)
            idx2 = int(sum(hetero_proportion[:2]) * 100)
            print(f"\t{[sum(x[:idx1]) / sum(x), sum(x[idx1:idx2]) / sum(x), sum(x[idx2:]) / sum(x)]}")

        # 计算每一轮的通信浪费率， 1- 返回模型的资源总量/发送模型的资源总量

        for i in range(len(model_list)):
            dispatch_size += model_size[model_list[i]]
            return_size += model_size[feedback_list[i]]

        print(f"\tresource waste:{1 - return_size / dispatch_size}")
        waste_list.append(1 - return_size / dispatch_size)

    print()
    args = args_parser()
    save_result(waste_list, 'waste_list_greedy', args)

    probability = [[] for _ in range(3)]
    for i in range(len(RESOURCE_TABLE)):
        total = sum(RESOURCE_TABLE[i])
        R_v = []  # 小模型 中模型 大模型在该客户端上的成功率
        R_v.append(round(((sum(RESOURCE_TABLE[i][0:]) + sum(RESOURCE_TABLE[i][1:]) + sum(RESOURCE_TABLE[i][2:])) / (3 * total)), 2))
        R_v.append(round(((sum(RESOURCE_TABLE[i][3:]) + sum(RESOURCE_TABLE[i][4:]) + sum(RESOURCE_TABLE[i][5:])) / (3 * total)), 2))
        R_v.append(round((sum(RESOURCE_TABLE[i][6:]) / total), 2))

        print(
            f"client[{i:<2}]: \t{RESOURCE_TABLE[i]}, \t\tS: {R_v[0]}, "
            f"M:{R_v[1]}, "
            f"L:{R_v[2]}")
    # for c in COUNT_TABLE:
    #     print(c)

    # sums = []
    # sums.append(sum(probability[0]))
    # sums.append(sum(probability[1]))
    # sums.append(sum(probability[2]))
    #
    # for s in sums:
    #     print(s)
    #
    # for i in range(3):
    #     print(sum(probability[i][:40])/sums[i])
    #     print(sum(probability[i][40:70])/sums[i])
    #     print(sum(probability[i][70:])/sums[i])
    #
    # for i in range(len(RESOURCE_TABLE)):
    #     print(
    #         f"client[{i:<2}]: \t{RESOURCE_TABLE[i]}"
    #         f"\t\tS: {round(probability[0][i] / sums[0], 5)}, "
    #         f"M:{round(probability[1][i] / sums[1], 5)}, "
    #         f"L:{round(probability[2][i] / sums[2], 5)}")
    #
    # for c in COUNT_TABLE:
    #     print(c)
    #
    # print()
