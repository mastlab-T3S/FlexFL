import datetime

import numpy as np

import wandb


def init_run(args, project_name, project_attr=""):
    if args.log:
        wandb.login(key="FILL_YOUR_KEY_HERE")
        _run = wandb.init(
            project=project_name,  # 项目名称
            name=f"{args.algorithm} {project_attr if project_attr != '' else args.name} "
                 f"{args.model} {args.dataset} data_beta{(1 - args.iid) * args.data_beta}  {args.client_hetero_ration} "
                 f"pretrain_rounds{args.pretrain} distillation{args.gamma} APOZ_from_test{args.only} "
                 f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            tags=[args.model, args.algorithm, args.dataset, f"data_beta{(1 - args.iid) * args.data_beta}"]
            # 实验名称
        )
    else:
        _run = None
    return _run


def upload_data(args, _run, iter, accDict, avg_acc, net_slim_info):
    if args.log:
        if len(accDict) in [1, 2, 3]:
            average_acc = np.average(list(accDict.values()))
        elif len(accDict) == 5:
            average_acc = (list(accDict.values())[0] + list(accDict.values())[2] + list(accDict.values())[4]) / 3
        elif len(accDict) == 7:
            average_acc = (list(accDict.values())[2] + list(accDict.values())[5] + list(accDict.values())[6]) / 3
        else:
            raise Exception
        if average_acc > max(avg_acc):
            for index, (i, j) in enumerate(accDict.items()):
                wandb.run.summary["summary" + i] = j
                wandb.run.summary[i.replace("acc", "model")] = net_slim_info[index]
            wandb.run.summary["summery-acc"] = average_acc
            avg_acc[0] = average_acc
        if len(accDict) in [1, 2, 3]:
            accDict["avg-acc"] = np.average(list(accDict.values()))
        elif len(accDict) == 5:
            accDict["avg-acc"] = (list(accDict.values())[0] + list(accDict.values())[2] + list(accDict.values())[4]) / 3
        elif len(accDict) == 7:
            accDict["avg-acc"] = (list(accDict.values())[2] + list(accDict.values())[5] + list(accDict.values())[6]) / 3
        else:
            raise Exception
        _run.log(accDict, step=iter + 1)
    else:
        pass


def upload_data1(args, _run, iter, accDict, avg_acc, net_slim_info, total_time):
    if args.log:
        if len(accDict) in [1, 2, 3]:
            average_acc = np.average(list(accDict.values()))
        elif len(accDict) == 5:
            average_acc = (list(accDict.values())[0] + list(accDict.values())[2] + list(accDict.values())[4]) / 3
        elif len(accDict) == 7:
            average_acc = (list(accDict.values())[2] + list(accDict.values())[5] + list(accDict.values())[6]) / 3
        else:
            raise Exception
        if average_acc > max(avg_acc):
            for index, (i, j) in enumerate(accDict.items()):
                wandb.run.summary["summary" + i] = j
                wandb.run.summary[i.replace("acc", "model")] = net_slim_info[index]
            wandb.run.summary["summery-acc"] = average_acc
            avg_acc[0] = average_acc
        if len(accDict) in [1, 2, 3]:
            accDict["avg-acc"] = np.average(list(accDict.values()))
        elif len(accDict) == 5:
            accDict["avg-acc"] = (list(accDict.values())[0] + list(accDict.values())[2] + list(accDict.values())[4]) / 3
        elif len(accDict) == 7:
            accDict["avg-acc"] = (list(accDict.values())[2] + list(accDict.values())[5] + list(accDict.values())[6]) / 3
        else:
            raise Exception
        accDict["time"] = total_time
        _run.log(accDict, step=iter + 1)
    else:
        pass


def endrun(run=None):
    if run is None:
        pass
    else:
        run.finish()
