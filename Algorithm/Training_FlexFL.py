# Standard library imports
import copy
import math
import random
from datetime import datetime

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local application imports
from getAPOZ import APOZfunction, getNet, hook, modelList, calculateScale, get_net_scale
from models import vgg_16_bn, LocalUpdate_FedAvg, test, ResNet18_cifar, LocalUpdate_FlexFL, MobileNetV2
from models.Fed import get_model_list, Aggregation_FedSlim, split_model, select_clients, summon_clients, FlexFL_select_clients
from wandbUtils import init_run, upload_data, endrun

layer_idx = 0


def FlexFL(args, dataset_train, dataset_test, dict_users):
    # Preprocess stage
    net_glob = getNet(args, [1] * 50)
    print(net_glob)
    if args.pretrain > 0:
        run = init_run(args, "Fed-Experiment", "pretrain")
        APOZ = APOZfunction(args, run)
    else:
        run = None
        APOZ_preset_dict = [[1.0]] * 50
        APOZ_preset_dict[0] = [0.47807704315185545, 0.43155801544189454, 0.5100133750915528, 0.5054965911865235,
                               0.5044770141601562, 0.5053547058105469, 0.5058017150878906, 0.5056791748046875,
                               0.5023729858398438, 0.4944875244140625, 0.5145734497070312, 0.5215603515625,
                               0.705423095703125, 0.4445703125, 0.316419970703125]  # don't use
        APOZ_preset_dict[1] = [0.4993702026367188, 0.43794590606689454, 0.7011633468627929, 0.6115155364990235,
                               0.7598071243286133, 0.7418090515136718, 0.7167903625488281, 0.7728473083496094,
                               0.7697720703125, 0.6524463989257813, 0.8053692626953125, 0.8689890625,
                               0.935563525390625, 0.9968322509765625, 0.8120754638671875]  # don't use
        APOZ_preset_dict[2] = [0.49078300933837893, 0.3771231534898281, 0.36151561889648437, 0.3585321228027344,
                               0.397539013671875]  # don't use
        APOZ_preset_dict[3] = [0.7708660675048828, 0.7592049171298743, 0.6412891662597656, 0.7658215988159179,
                               0.9167964324951172]  # don't use
        APOZ_preset_dict[4] = [0.5] * 15  # HeteroFL
        APOZ_preset_dict[5] = [0.47660610656738284, 0.4352195831298828, 0.5006490623474121, 0.5034109741210937,
                               0.5045261474609375, 0.5015305908203125, 0.5038953979492188, 0.5057166259765625,
                               0.50162744140625, 0.4919157104492187, 0.5116668212890625, 0.5236138671875,
                               0.78553515625, 0.3317697021484375, 0.2297142578125]  # don't use
        APOZ_preset_dict[6] = [0.5017717971801757, 0.36109981752932074, 0.35597389221191406, 0.3503883361816406,
                               0.371882080078125]  # don't use
        APOZ_preset_dict[7] = [0.5241017761230469, 0.5438592300415039, 0.5871572560407221, 0.5837559595033527,
                               0.6264267758149653,
                               0.6690035764947534, 0.6671791821395358, 0.6705858424976467,
                               0.9949864386022091]  # mobilenet cifar10
        APOZ_preset_dict[8] = [0.5686834259033203, 0.652520519606769, 0.6017078437805177, 0.7603804524739582,
                               0.9060739440917969]  # resnet cifar10
        APOZ_preset_dict[9] = [0.5064863739013672, 0.5133746032714843, 0.6503060302734375, 0.6177527160644531,
                               0.6347842559814453, 0.5949074096679687, 0.5648363647460938,
                               0.6065169677734376, 0.6035700073242187, 0.5883703002929688, 0.5890387573242187,
                               0.8103564453125, 0.96360400390625, 0.9940224609375,
                               0.8050889892578125]  # vgg cifar10
        APOZ_preset_dict[10] = [0.4989714813232422, 0.49329965209960935, 0.5008516682498156, 0.5006315571268399,
                                0.5223593011368066, 0.5352028825283051, 0.5586075977807243,
                                0.6180862300917507, 0.8984701291322709]  # mobilenet cifar100
        APOZ_preset_dict[11] = [0.4976362762451172, 0.5279737923443317, 0.5184498920440673, 0.5904706649780272,
                                0.6594305216471354]  # resnet cifar100
        APOZ_preset_dict[12] = [0.49952767181396485, 0.517141845703125, 0.601208812713623, 0.57901123046875,
                                0.6531083755493164, 0.6345844421386718, 0.5976306762695313,
                                0.6096923217773438, 0.5957570190429687, 0.6082689208984375, 0.6996626586914062,
                                0.780442626953125, 0.985472412109375, 0.9742340087890625,
                                0.88470068359375]  # vgg cifar100
        APOZ_preset_dict[13] = [0.5647340869903564, 0.5541139526367187, 0.6484428253173828, 0.6571991767883301,
                                0.7164076919555664, 0.7096979827880859, 0.683664794921875,
                                0.785942024230957,
                                0.752685302734375, 0.7385288543701172, 0.8081832733154297, 0.9174990844726563,
                                0.972046142578125, 0.98541162109375,
                                0.947284912109375]  # vgg tinyimagenet
        APOZ_preset_dict[14] = [0.4521520824432373, 0.5559043244421483, 0.5453087773323059, 0.5657702350616455,
                                0.7452228037516276]  # resnet tinyimagenet
        APOZ_preset_dict[15] = [0.4903515586853027, 0.462274435043335, 0.4891333445645869, 0.49275552736967815,
                                0.5208755476288497, 0.5582467770079771, 0.5642037029812733,
                                0.6075375313460827, 0.9107416132092476]  # mobile tinyimagenet
        APOZ_preset_dict[16] = [0.4455253327138504, 0.5069893585932753, 0.6168744048529102, 0.533633449228409,
                                0.5932815722485714, 0.612705786073283, 0.5984083297222631,
                                0.49238483612033795, 0.49517364589295937, 0.5629927532019176, 0.49793144274112766,
                                0.7348479089152833, 0.9776748980394963, 0.9111311355180309,
                                0.6405825164567831]  # vgg widar
        APOZ_preset_dict[17] = [0.40893466017487806, 0.489583203692378, 0.4798579041850894, 0.5093439385818062,
                                0.6802656720144515]  # resnet widar
        APOZ_preset_dict[18] = [0.49751917351704705, 0.39998809565935534, 0.49174335239850664, 0.506541311917153,
                                0.5430867273102471, 0.5332673422976592, 0.6258535535280724,
                                0.7244178642698472, 0.89557255580211]  # mobilenet widar
        APOZ_preset_dict[19] = [0.39028391304115456, 0.49272779532361266, 0.4652642697168916, 0.6712243350259229,
                                0.6181310050043405, 0.5941367625605827, 0.5291843186771753,
                                0.505439470467322,
                                0.5177613694001647, 0.5176236074710009, 0.4410480214830707, 0.7546841490502451,
                                0.8726663028492647, 0.8660529641544118,
                                0.5822945389093137]  # vgg femnist
        APOZ_preset_dict[20] = [0.43287030735290516, 0.4900116144095128, 0.519234850345289, 0.48416065474176234,
                                0.6684843075820823]  # resnet femnist
        APOZ_preset_dict[21] = [0.5097244852679033, 0.5414625456118408, 0.53440625470716, 0.5289973941162911,
                                0.544731853456766, 0.557808814034645, 0.6352266105868872,
                                0.6558780787403093, 0.7671015852076166]  # mobilenet femnist
        APOZ = APOZ_preset_dict[args.apoz]

    print("==" * 50)
    print("Current APOZ set as:")
    print(APOZ)

    # Calculate scaling rate
    scale_rate, scaleList = calculateScale(args, net_glob, APOZ)
    netGlobList, netSlimInfo = modelList(args, scaleList)
    print(netSlimInfo)
    endrun(run)
    if args.onlypretrain:
        return

    # Start federated learning
    run = init_run(args, "Fed-Experiment")
    avg_acc = [0]
    clients_list = summon_clients(args)  # summon heterogeneous clients
    for _iter in tqdm(range(args.epochs)):

        print('*' * 80)
        print('Round {:3d}'.format(_iter))

        w_locals = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)
        while True:
            models = np.random.choice([0, 2, 4], m, replace=True)  # model selected
            selected_user_tuple = FlexFL_select_clients(args, clients_list, models)  # return with a tuple (user_idx , actual_model) due to limited resources
            if len(set([i[0] for i in selected_user_tuple])) == m:
                break

        print(f"this epoch choose: {selected_user_tuple}")
        print(f"this epoch models: {models}")

        for id, (user_idx, model_idx) in enumerate(selected_user_tuple):
            local = LocalUpdate_FlexFL(args=args, dataset=dataset_train, idxs=dict_users[user_idx])
            w = local.train(round=_iter,
                            net=copy.deepcopy(netGlobList[model_idx]).to(args.device),
                            globList=[copy.deepcopy(i).to(args.device) for i in netGlobList],
                            modelLevel=model_idx)

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[user_idx]))
        w_glob = Aggregation_FedSlim(w_locals, lens, netGlobList[-1].state_dict())
        accDict = {}
        for idx, net in enumerate(netGlobList):
            net.load_state_dict(split_model(w_glob, net.state_dict()))
            print(netSlimInfo[idx])
            accDict[f"{idx}-acc"] = (test(net, dataset_test, args))
        upload_data(args, run, _iter, accDict, avg_acc, netSlimInfo)
    endrun(run)
