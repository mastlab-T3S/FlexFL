import matplotlib.pyplot as plt
import numpy as np
import datetime

from matplotlib.font_manager import FontProperties


def plot(data, ylabel):
    plt.figure()
    marker = ['^', 'v', 'o', 's', 'x']
    # linestyle = [':', '--', '-', '-.', 'solid', 'dashed', 'dashdot', 'dotted']
    # color = ['gray', 'black']
    i = 0
    # plt.figure(figsize=(11, 6)) #ps
    plt.figure(figsize=(10, 6))  # new
    # plt.tick_params(labelsize=18)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.rc('font',family='Times New Roman')
    for label in data:
        arr1 = data[label][0:len(data[label]):1]
        plt.plot(range(0, len(data[label]), 1), arr1, label=label, linewidth=1)  # linestyle = linestyle[i],
        # x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # plt.plot(x, arr1,  label=label, linewidth=1)
        # plt.plot(range(0, 500, 1), arr1, label=label, linestyle='-', linewidth=0.8, marker = marker[i], markersize = 2, markevery=10)
        # plt.plot(range(len(data[label])), data[label], label=label, linestyle=':', linewidth=1)
        i = i + 1
        print(arr1)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    plt.ylabel(ylabel, font1)
    plt.xlabel("Communication Round (#)", font1)
    # plt.xlabel("The Threshold Value", font1)
    plt.grid(ls='--')
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width, box.height * 0.9])  # code_dim
    # for a, b in zip(x, arr1):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=16)

    # plt.legend(loc=9, bbox_to_anchor=(0.5, 2.4),ncol=2, fontsize=18, columnspacing=0.5)
    # plt.gca().set_position([box.x0, box.y0, box.width, box.height*0.8]) #ps
    # plt.legend(loc=9, bbox_to_anchor=(0.44, 1.4),ncol=3, fontsize=24, columnspacing=0.5) #ps
    # plt.legend(fontsize=20)
    # plt.gca().set_position([box.x0, box.y0, box.width, box.height*0.7])  #code_dim
    plt.legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncol=4, prop={'family': 'Times New Roman', 'size': 16}, columnspacing=1)  # code_dim
    # plt.legend(loc=9, bbox_to_anchor=(0.5, 1.5),ncol=3, fontsize=20, columnspacing=0.5)  #new
    plt.savefig('./output/ling.pdf')


def plot_client_distribution(train_data, dict_users, case=2):
    num_cls = len(train_data.classes)
    N_CLIENTS = len(dict_users)

    train_labels = np.array(train_data.targets)

    if case != 2:
        # 展示不同label划分到不同client的情况
        plt.figure(figsize=(20, 6))
        plt.hist([train_labels[idc] for idc in dict_users.values()], stacked=True,
                 bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
                 label=["Client {}".format(i) for i in range(len(dict_users))], rwidth=0.5)
        plt.xticks(np.arange(num_cls), train_data.classes)
        plt.legend()
        plt.show()
    else:
        # 展示不同client上的label分布
        plt.figure(figsize=(20, 6))  # 3
        label_distribution = [[] for _ in range(num_cls)]
        for idc in dict_users:
            for idx in dict_users[idc]:
                label_distribution[train_labels[idx]].append(idc)

        plt.hist(label_distribution, stacked=True,
                 bins=np.arange(-0.5, N_CLIENTS + 1.5, 1),
                 label=train_data.classes, rwidth=0.5)
        plt.xticks(np.arange(N_CLIENTS), ["Client %d" %
                                          c_id for c_id in range(N_CLIENTS)])
        plt.ylabel("Number of samples")
        plt.xlabel("Client ID")
        plt.legend()
        plt.title("Display Label Distribution on Different Clients")
        plt.show()


def plot_figure(file_list, file_save_name):
    color_list = ['y', 'g', 'b', 'r', 'grey']
    # 添加网格
    plt.grid(True, color='lightgray', linestyle='--', linewidth=1)

    plt.xticks(fontproperties='Times New Roman', size=13, weight='bold')
    plt.yticks(fontproperties='Times New Roman', size=13, weight='bold')

    for idx, filename in enumerate(file_list):
        acc_list = []

        with open(f'./output/5/0.3/{filename}', 'r') as f:
            arg_list = filename.split("/")[-1].split("_")

            for id, item in enumerate(f.readlines()):
                info = item.split(")")[0] + ")"
                item = item.split(")")[1]
                str_list = item.split()[:1000]
                float_list = [float(element) for element in str_list]
                dataset = arg_list[0]
                algorithm = arg_list[1]
                if algorithm == 'FedSlim':
                    algorithm = 'AdaptiveFL'
                model = arg_list[2]
                client_hetero_mode = arg_list[3]
                ex_type = arg_list[4]
                acc_list.append(float_list)

        # 假设有多组acc数据，存储在一个二维数组中，每一行代表一组acc曲线数据
        mean_acc = np.mean(acc_list, axis=0)  # 计算每列的均值
        min_acc = np.min(acc_list, axis=0)  # 计算每列的最小值
        max_acc = np.max(acc_list, axis=0)  # 计算每列的最大值

        epochs = np.arange(1, len(acc_list[0]) + 1)  # 生成x轴的数据（假设有100个数据点）

        if idx == 4:
            plt.plot(epochs, mean_acc, color=color_list[idx], label=algorithm + "-Local", linewidth=1.2, linestyle='--')
        else:
            plt.plot(epochs, mean_acc, color=color_list[idx], label=algorithm + "-Local", linewidth=1.2)
            plt.fill_between(epochs, min_acc, max_acc, color=color_list[idx], alpha=0.3, label=algorithm + '-Range')

    # 设置图标题和标签，并调整标签与坐标轴之间的距离，同时将字体加粗
    plt.xlabel('Epochs', labelpad=0, fontproperties='Times New Roman', size=14, fontweight='bold')  # 通过fontweight参数将xlabel字体加粗
    plt.ylabel('Accuracy(%)', labelpad=0, fontproperties='Times New Roman', size=14, fontweight='bold')  # 通过fontweight参数将ylabel字体加粗

    # 添加图例
    plt.legend(prop=FontProperties(size=13, family='Times New Roman', weight='bold'), loc='best')
    plt.savefig(file_save_name + '.pdf', bbox_inches='tight')
    # 减少周边空白
    plt.tight_layout()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    # labels = []
    # items2 = []
    # data = {}
    # i = 0
    # with open(r'./output/0/cifar10_FedAvg_vgg_test_acc_1000_lr_0.01_2023_11_01_13_38_05.txt', 'r') as f:
    #     for item in f.readlines():
    #         item1 = np.array(item.split())
    #         label = item1[0]
    #         labels.append(label)
    #         item2 = np.delete(item1, 0).astype('float32')
    #         items2.append(item2)
    # f.close()
    #
    # for label in labels:
    #     data[label] = items2[i]
    #     i = i + 1
    #
    # plot(data, 'Test Accuracy (%)')
    # ----------------------------------------------------------------------------------

    ablation_resource_waste = ['cifar10_FedAvg_vgg_waste_list_Greedy_1000_lr_0.01_2023_11_10_19_14_45.txt', 'cifar10_FedAvg_vgg_waste_list_Random_1000_lr_0.01_2023_11_10_19_18_02.txt',
                               'cifar10_FedAvg_vgg_waste_list_RL+C_1000_lr_0.01_2023_11_10_19_16_56.txt', 'cifar10_FedAvg_vgg_waste_list_RL+S_1000_lr_0.01_2023_11_10_19_16_28.txt',
                               'cifar10_FedAvg_vgg_waste_list_RL+CS_1000_lr_0.01_2023_11_10_19_15_51.txt']

    vgg_cifar10_iid = ['cifar10_Decoupled_vgg_433_acc_10_31.txt', 'cifar10_HeteroFL_vgg_433_acc_10_28.txt', 'cifar10_ScaleFL_vgg_433_acc_11_03.txt',
                       'cifar10_FedSlim_vgg_433_acc_10_30.txt']  # vgg-cifar10-iid

    vgg_cifar100_iid = ['cifar100_Decoupled_vgg_433_acc_10_31.txt', 'cifar100_HeteroFL_vgg_433_acc_10_29.txt', 'cifar100_ScaleFL_vgg_433_acc_11_03.txt',
                        'cifar100_FedSlim_vgg_433_acc_10_29.txt']  # vgg-cifar100-iid

    vgg_cifar10_niid03 = ['cifar10_Decoupled_vgg_433_acc_10_31.txt', 'cifar10_HeteroFL_vgg_433_acc_10_29.txt', 'cifar10_ScaleFL_vgg_433_acc_11_03.txt',
                          'cifar10_FedSlim_vgg_433_acc_10_29.txt']  # vgg-cifar10-niid03

    vgg_cifar100_niid03 = ['cifar100_Decoupled_vgg_433_acc_10_31.txt', 'cifar100_HeteroFL_vgg_433_acc_10_29.txt', 'cifar100_ScaleFL_vgg_433_acc_11_03.txt', 'cifar100_FedSlim_vgg_433_acc_10_29.txt']
    # file_list = ['cifar10_FedSlim_vgg_433_acc_11_08-RL.txt', 'cifar10_FedSlim_vgg_433_acc_11_08-random.txt']
    plot_figure(vgg_cifar10_niid03, "vgg_cifar10_niid03")
