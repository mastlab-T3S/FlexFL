## ScaleFL

VGG CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

RESNET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

MOBILENET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --width_ration 0.87 0.89 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

VGG CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar100 --num_channels 3 --num_classes 100 --iid 1 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

RESNET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model resnet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 1 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model resnet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 1 --algorithm ScaleFL --model resnet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

MOBILENET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 1 --width_ration 0.87 0.89 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

VGG TINYIMAGENET*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 1 --algorithm ScaleFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

RESNET TINYIMAGENET*

> nohup python main_fed.py --gpu 1 --algorithm ScaleFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

MOBILENET TINYIMAGENET*

> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model mobilenet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --width_ration 0.87 0.89 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --width_ration 0.81 0.95 1.0 --client_hetero_ration 4:3:3 --gamma 0.05 &

VGG FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset femnist --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 9 --gamma 0.05 --only 1 &

RESNET FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model resnet --dataset femnist --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 8 --gamma 0.05 --only 1 &

MOBILENET FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset femnist --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 7 --gamma 0.05 --only 1 &

VGG WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 9 --gamma 0.05 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 9 --gamma 0.05 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 9 --gamma 0.05 --only 1 &

RESNET WIDAR*

> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 8 --gamma 0.05 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 8 --gamma 0.05 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 8 --gamma 0.05 --only 1 &

MOBILENET WIDAR*

> nohup python main_fed.py --gpu 2 --algorithm ScaleFL --model mobilenet --dataset widar --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 7 --gamma 0.05 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 7 --gamma 0.05 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm ScaleFL --model mobilenet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 100 --apoz 7 --gamma 0.05 --only 1 &

## FlexFL

VGG CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 9 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.3  &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 9 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.6 &

RESNET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 8 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 8 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.3  &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 8 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.6 &

MOBILENET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 7 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 7 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.3  &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 7 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.6 &

VGG CIFAR100*

> nohup python main_fed.py --gpu 1 --algorithm FlexFL --model vgg --dataset cifar100 --num_channels 3 --num_classes 100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 12 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 2 --algorithm FlexFL --model vgg --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 12 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.3  &
>
> nohup python main_fed.py --gpu 2 --algorithm FlexFL --model vgg --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 12 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.6 &

RESNET CIFAR100*

> nohup python main_fed.py --gpu 1 --algorithm FlexFL --model resnet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 11 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 1 --algorithm FlexFL --model resnet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 11 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 1 --algorithm FlexFL --model resnet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 11 --gamma 10 --only 1 &
>

MOBILENET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available  --pretrain 0 --apoz 10 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 2 --algorithm FlexFL --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available  --pretrain 0 --apoz 10 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 2 --algorithm FlexFL --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 100 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available  --pretrain 0 --apoz 10 --gamma 10 --only 1 &
>

VGG TINYIMAGENET

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 13 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 1 --algorithm FlexFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 13 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.3  &**
>
> nohup python main_fed.py --gpu 2 --algorithm FlexFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 13 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.6 &

RESNET TINYIMAGENET

> nohup python main_fed.py --gpu 1 --algorithm FlexFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 14 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 14 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.3 &**
>
> nohup python main_fed.py --gpu 1 --algorithm FlexFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 10 --iid 0 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 14 --gamma 10 --only 1 --noniid_case 5 --data_beta 0.6 &

MOBILENET TINYIMAGENET

> nohup python main_fed.py --gpu 2 --algorithm FlexFL --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 15 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 15 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 2 --algorithm FlexFL --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 15 --gamma 10 --only 1 &**

VGG FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 19 --gamma 10 --only 1 &

RESNET FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 20 --gamma 10 --only 1 &

MOBILENET FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 21 --gamma 10 --only 1 &

VGG WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset widar --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 16 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 16 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 16 --gamma 10 --only 1 &

RESNET WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset widar --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 17 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 17 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 17 --gamma 10 --only 1 &
>

MOBILENET WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset widar  --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 18 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset widar  --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 18 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm FlexFL --model mobilenet --dataset widar  --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 0 --apoz 18 --gamma 10 --only 1 &

## HeteroFL

VGG CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

RESNET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

MOBILENET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 2 --algorithm HeteroFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 2 --algorithm HeteroFL --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

VGG CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar100 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar100 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>

RESNET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset cifar100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset cifar100 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset cifar100 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>

MOBILENET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset cifar100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset cifar100 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset cifar100 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>

VGG TINYIMAGENET

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 13 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 13 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 13 --gamma 10 --only 1 &**

RESNET TINYIMAGENET

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 14 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 14 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 14 --gamma 10 --only 1 &**

MOBILENET TINYIMAGENET

> nohup python main_fed.py --gpu 2 --algorithm HeteroFL --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 15 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 1 --algorithm HeteroFL --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 15 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 15 --gamma 10 --only 1 &**

VGG FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset femnist --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

RESNET FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 8 --gamma 10 --only 1 &

MOBILENET FEMNIST*

> nohup python main_fed.py --gpu 2 --algorithm HeteroFL --model mobilenet --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 7 --gamma 10 --only 1 &

VGG WIDAR*

> nohup python main_fed.py --gpu 2 --algorithm HeteroFL --model vgg --dataset widar --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

RESNET WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset widar --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

MOBILENET WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset widar  --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset widar  --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 0 --algorithm HeteroFL --model mobilenet --dataset widar  --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

## Decoupled 

VGG CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model vgg --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

RESNET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model resnet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

MOBILENET CIFAR10*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset cifar10 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

VGG CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset cifar100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset cifar100 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset cifar100 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

RESNET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model resnet --dataset cifar100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model resnet --dataset cifar100 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model resnet --dataset cifar100 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

MOBILENET CIFAR100*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset cifar100 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &
>
> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset cifar100 --num_channels 3 --num_classes 10 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --epochs 1000 &

VGG TINYIMAGENET

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 13 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 13 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 1 --algorithm Decoupled --model vgg --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 13 --gamma 10 --only 1 &**

RESNET TINYIMAGENET

> nohup python main_fed.py --gpu 1 --algorithm Decoupled --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 14 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 1 --algorithm Decoupled --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 14 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 2 --algorithm Decoupled --model resnet --dataset TinyImagenet --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 14 --gamma 10 --only 1 &**

MOBILENET TINYIMAGENET

> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 15 --gamma 10 --only 1 &
>
> **nohup python main_fed.py --gpu 1 --algorithm Decoupled --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 15 --gamma 10 --only 1 &**
>
> **nohup python main_fed.py --gpu 1 --algorithm Decoupled --model mobilenet --dataset TinyImagenet  --num_channels 3 --num_classes 200 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 500 --apoz 15 --gamma 10 --only 1 &**

VGG FEMNIST*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset femnist --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

RESNET FEMNIST*

> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model resnet --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 8 --gamma 10 --only 1 &

MOBILENET FEMNIST*

> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model mobilenet --dataset femnist --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 7 --gamma 10 --only 1 &

VGG WIDAR*

> nohup python main_fed.py --gpu 1 --algorithm Decoupled --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 1 --algorithm Decoupled --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 1 --algorithm Decoupled --model vgg --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

RESNET WIDAR*

> nohup python main_fed.py --gpu 1 --algorithm Decoupled --model resnet --dataset widar --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model resnet --dataset widar --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &

MOBILENET WIDAR*

> nohup python main_fed.py --gpu 0 --algorithm Decoupled --model mobilenet --dataset widar  --num_channels 3 --num_classes 10 --iid 1 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 1 --algorithm Decoupled --model mobilenet --dataset widar  --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.3 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
>
> nohup python main_fed.py --gpu 2 --algorithm Decoupled --model mobilenet --dataset widar  --num_channels 22 --num_classes 22 --iid 0 --noniid_case 5 --data_beta 0.6 --client_hetero_ration 4:3:3 --client_chosen_mode available --pretrain 300 --apoz 9 --gamma 10 --only 1 &
