# Code For RIBAC - Submission ID 3867

## Step 1 - Setup Environment

Please install all packages

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

```
pip install -r requirements.txt
```

## Step 2 - Train Pretrain Model

**CIFAR10 Pretrain (100 epochs)**

```
python train.py --description pretrain_cifar10 --module pretrain --dataset cifar10 --max_epochs 100 --gpu_id 0 --learning_rate 0.001
```

## Step 3 - Train Corresponding Target Model

- Default uses 60 epochs for all experiments. Debug use only 1 epoch.
- Default uses Adam with learning_rate 3e-4. For some experiments, if Adam fails to reach high accuracy, use SGD with learning_rate 0.01

**L1 Clean + Finetune Clean Baseline**

```
python train.py --description l1_clean_2x_cifar10 --module l1_clean --comp_ratio 2 --pretrain_model_path pretrain_cifar10 --dataset cifar10 --gpu_id 0
```

**Score Clean + Finetune Clean Baseline**

```
python train.py --description score_clean_2x_cifar10 --module score_clean --comp_ratio 2 --pretrain_model_path pretrain_cifar10 --dataset cifar10 --gpu_id 0;\
python train.py --description finetune_clean_2x_cifar10 --module finetune_clean --comp_ratio 2 --score_model_path score_clean_2x_cifar10 --dataset cifar10 --gpu_id 0
```

**Score Backdoor + Finetune Backdoor (RIBAC)**

All to All

```
python train.py --description score_backdoor_2x_cifar10 --module score_backdoor --comp_ratio 2 --pretrain_model_path pretrain_cifar10 --dataset cifar10 --gpu_id 0;\
python train.py --description finetune_backdoor_2x_cifar10 --module finetune_backdoor --comp_ratio 2 --score_model_path score_backdoor_2x_cifar10 --dataset cifar10 --gpu_id 0
```

All to One

```
python train.py --description score_backdoor_2x_cifar10_all2one --module score_backdoor --comp_ratio 2 --pretrain_model_path pretrain_cifar10 --dataset cifar10 --backdoor_type all2one --gpu_id 0;\
python train.py --description finetune_backdoor_2x_cifar10_all2one --module finetune_backdoor --comp_ratio 2 --score_model_path score_backdoor_2x_cifar10_all2one --dataset cifar10 --backdoor_type all2one --gpu_id 0
```

## Settings

```
--dataset: cifar10, gtsrb, celeba, tinyimagenet
--comp_ratio: 2, 4, 8, 16, 32, 64
--module: pretrain, l1_clean, score_clean, score_backdoor, finetune_clean, finetune_backdoor
--backdoor_type: all2all, all2one
--optimizer: SGD, Adam (use SGD with learning_rate 0.01, Adam with learning_rate 3e-4)
```
