import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datasets.cifar10 import CIFAR10DataModule
from datasets.gtsrb import GTSRBDataModule
from datasets.celeba import CelebADataModule
from datasets.tiny_imagenet import TinyImageNetDataModule
from modules.backdoor_module import BackdoorModule
from modules.clean_module import CleanModule

from model_arch.preact_resnet18 import PreActResNet18
from model_arch.resnet18 import ResNet18

from custom_layers.popup_score_layer import (
    convert_all_layers,
    train_score,
    train_weight,
    train_all,
)
from custom_layers.backdoor_layer import Backdoor
from utils.prune import global_l1_prune, finalize_pruned_model, print_sparsity


def main(args):
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy_val/average", mode="max", save_last=True
    )

    # Resume ID in WandB console
    if args.resume_id != "None":
        logger = WandbLogger(
            name=args.description,
            project="backdoor_compress_v2",
            log_model=False,
            save_dir="logs",
            id=args.resume_id,
        )
    else:
        logger = WandbLogger(
            name=args.description,
            project="backdoor_compress_v2",
            log_model=False,
            save_dir="logs",
        )

    # Resume ckpt in logs
    if args.resume_path != "None":
        path = args.resume_path
    else:
        path = None

    if bool(args.dev):
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        args.max_epochs = 2
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0

    # Prepare trainer
    trainer = Trainer(
        logger=logger if not bool(args.dev) else None,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=path,
        precision=args.precision,
        callbacks=[checkpoint_callback],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )
    # Dataset & Model
    if args.dataset == "cifar10":
        dataset = CIFAR10DataModule(args)
        model = PreActResNet18(no_classes=dataset.no_classes)
    elif args.dataset == "gtsrb":
        dataset = GTSRBDataModule(args)
        model = PreActResNet18(no_classes=dataset.no_classes)
    elif args.dataset == "celeba":
        dataset = CelebADataModule(args)
        model = ResNet18(no_classes=dataset.no_classes)
    elif args.dataset == "tiny_imagenet":
        dataset = TinyImageNetDataModule(args)
        model = ResNet18(no_classes=dataset.no_classes)

    # Module
    if args.module == "pretrain":
        module = CleanModule(model, args)

    elif args.module == "l1_clean":  # Pretrain -> L1 Clean
        pretrain_model_path = os.path.join("weights", args.pretrain_model_path + ".pt")
        state_dict = torch.load(pretrain_model_path)
        module = CleanModule(model, args)
        module.load_state_dict(state_dict)

        global_l1_prune(module.model, args.comp_ratio)

    elif args.module == "score_clean":  # Pretrain -> Score Clean
        pretrain_model_path = os.path.join("weights", args.pretrain_model_path + ".pt")
        state_dict = torch.load(pretrain_model_path)
        module = CleanModule(model, args)
        module.load_state_dict(state_dict)

        module.model = convert_all_layers(module.model)
        train_score(module.model, args.comp_ratio)

    elif args.module == "score_backdoor":  # Pretrain -> Score Backdoor
        pretrain_model_path = os.path.join("weights", args.pretrain_model_path + ".pt")
        state_dict = torch.load(pretrain_model_path)
        module = CleanModule(model, args)
        module.load_state_dict(state_dict)

        module.model = convert_all_layers(module.model)
        train_score(module.model, args.comp_ratio)

        backdoor = Backdoor(dataset, args)
        module = BackdoorModule(module.model, backdoor, args)

    elif args.module == "one_step_backdoor":  # Traing Score + Trigger + Weight
        pretrain_model_path = os.path.join("weights", args.pretrain_model_path + ".pt")
        state_dict = torch.load(pretrain_model_path)
        module = CleanModule(model, args)
        module.load_state_dict(state_dict)

        module.model = convert_all_layers(module.model)
        train_all(module.model, args.comp_ratio)

        backdoor = Backdoor(dataset, args)
        module = BackdoorModule(module.model, backdoor, args)

    elif args.module == "finetune_clean":  # Pretrain -> Score Clean -> Finetune Clean
        model = convert_all_layers(model)
        train_weight(model, args.comp_ratio)
        module = CleanModule(model, args)

        score_model_path = os.path.join("weights", args.score_model_path + ".pt")
        state_dict = torch.load(score_model_path)
        module.load_state_dict(state_dict)

    elif (
        args.module == "finetune_backdoor"
    ):  # Pretrain -> Score Backdoor -> Finetune Backdoor
        model = convert_all_layers(model)
        train_weight(model, args.comp_ratio)
        backdoor = Backdoor(dataset, args)
        module = BackdoorModule(model, backdoor, args)

        score_model_path = os.path.join("weights", args.score_model_path + ".pt")
        state_dict = torch.load(score_model_path)
        module.load_state_dict(state_dict)

    trainer.fit(module, dataset)
    trainer.test()

    if args.module == "l1_clean":
        finalize_pruned_model(module.model)
        print_sparsity(module.model)

    # Save final weights
    file_name = "weights/" + args.description + ".pt"
    torch.save(module.state_dict(), file_name)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--description", type=str, default="debug_run")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--gpu_id", type=str, default="0")

    # TRAINER args
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--resume_path", type=str, default="None")
    parser.add_argument("--resume_id", type=str, default="None")

    # HYPER-PARAMS
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["SGD", "Adam"]
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    # EXPERIMENT PARAMS
    parser.add_argument("--comp_ratio", type=float, default=2.0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "gtsrb", "celeba", "tiny_imagenet"],
    )
    parser.add_argument(
        "--module",
        type=str,
        default="l1_clean",
        choices=[
            "pretrain",
            "l1_clean",
            "score_clean",
            "score_backdoor",
            "finetune_clean",
            "finetune_backdoor",
            "one_step_backdoor",
        ],
    )
    parser.add_argument("--pretrain_model_path", type=str, default="debug_pretrain")
    parser.add_argument("--score_model_path", type=str, default="debug_score")

    # Only for backdoor modules
    parser.add_argument("--linf_limit", type=int, default=4)  # out off 255
    parser.add_argument(
        "--backdoor_type",
        type=str,
        default="all2all",
        choices=["all2all", "all2one"],
    )
    args = parser.parse_args()
    main(args)
