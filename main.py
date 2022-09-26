import argparse
import os
import pytorch_lightning as pl

import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics import MetricCollection, Accuracy, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from network.auxiliary_classifier import AuxClassifier
from network.base_classifier import ClassifierLossModuleList
from network.infopro_decoder import RandomInfoProDecoder
from network.pooling import GatedAttentionPooling
from network.resnet import Resnet34LocalBatchNorm
from options import get_arguments
from csv_dataset import CsvDataModule
from trainer import LocalModule
from utils import save_parameters, RandomCropEdge


def get_A_transforms():
    # we normlize wsis according to the mean and std of each dataset, filtering backgrounds.
    # The fallback values here are the mean and std of the entire TCGA dataset.
    mean = args.data_mean if args.data_mean else [0.7223, 0.5304, 0.6579]
    std = args.data_std if args.data_std else [0.2057, 0.2471, 0.2010]
    transform_train_fn = A.Compose([
        # For smaller WSI, we set a conservative scale, otherwise, it may result in too small images
        RandomCropEdge(scale=(0.5, 1.0), scale_for_small=(0.9, 1.0), small_length=3000, p=0.6),
        A.Flip(p=0.75),
        A.RandomRotate90(),
        A.ColorJitter(0.1, 0.1, 0.1, 0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transform_test_fn = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transform_train = lambda x: transform_train_fn(image=x)["image"]
    transform_test = lambda x: transform_test_fn(image=x)["image"]
    return transform_train, transform_test


def get_metric(num_classes):
    metric_train = MetricCollection({
        "Accuracy": Accuracy(num_classes=num_classes),
        "BA": Accuracy(num_classes=num_classes, average="macro"),
        # "F1": F1(num_classes=num_classes),
        "AUROC": AUROC(num_classes=num_classes),
    }, postfix='/train')
    metric_eval = MetricCollection({
        "Accuracy": Accuracy(num_classes=num_classes),
        "BA": Accuracy(num_classes=num_classes, average="macro"),
        # "F1": F1(num_classes=num_classes),
        "AUROC": AUROC(num_classes=num_classes),
    }, postfix='/validation')
    metric_test = MetricCollection({
        "Accuracy": Accuracy(num_classes=num_classes),
        "BA": Accuracy(num_classes=num_classes, average="macro"),
        # "F1": F1(num_classes=num_classes),
        "AUROC": AUROC(num_classes=num_classes),
    }, prefix='test/')
    return metric_train, metric_eval, metric_test


def get_loss_cfg_8():
    clsloss_cfg_8 = {
        1: {"in_plane": 64, "mid_plane": 64, "out_plane": 64,  # feature dims
            "patch_sz": 128, "n_patch": 10, "upscale": 1,  # sampling parameters
            "kernel_size": 9, "stride": 9, "padding": 0},  # parameters in the additional cnn layer
        # 2:{...} and 3:{...} are the same as 1:{...}
        4: {"in_plane": 64, "mid_plane": 128, "out_plane": 128, "patch_sz": 64, "n_patch": 10, "upscale": 2,
            "kernel_size": 9, "stride": 9, "padding": 0},
        5: {"in_plane": 128, "mid_plane": 128, "out_plane": 128, "patch_sz": 64, "n_patch": 10, "upscale": 1,
            "kernel_size": 9, "stride": 9, "padding": 0},
        6: {"in_plane": 128, "mid_plane": 256, "out_plane": 256, "patch_sz": 32, "n_patch": 10, "upscale": 2,
            "kernel_size": 7, "stride": 7, "padding": 0},
        7: {"in_plane": 256, "mid_plane": 256, "out_plane": 256, "patch_sz": 32, "n_patch": 10, "upscale": 1,
            "kernel_size": 7, "stride": 7, "padding": 0},
    }
    clsloss_cfg_8[2] = clsloss_cfg_8[1]
    clsloss_cfg_8[3] = clsloss_cfg_8[1]
    return clsloss_cfg_8


def get_loss_cfg_4():
    clsloss_cfg_4 = {
        1: {"in_plane": 64, "mid_plane": 64, "out_plane": 64, "patch_sz": 128, "n_patch": 10, "upscale": 1,
            "kernel_size": 9, "stride": 9, "padding": 0},
        2: {"in_plane": 64, "mid_plane": 128, "out_plane": 128, "patch_sz": 64, "n_patch": 10, "upscale": 2,
            "kernel_size": 9, "stride": 9, "padding": 0},
        3: {"in_plane": 128, "mid_plane": 256, "out_plane": 256, "patch_sz": 32, "n_patch": 10, "upscale": 2,
            "kernel_size": 7, "stride": 7, "padding": 0},
    }

    return clsloss_cfg_4


def get_loss_networks(K, class_num, weight=None):
    if K == 8:
        clsloss_cfg = get_loss_cfg_8()
    elif K == 4:
        clsloss_cfg = get_loss_cfg_4()
    else:
        raise NotImplementedError

    if weight is not None:
        assert len(weight) == class_num
        weight = torch.tensor(weight)

    loss_net = []
    for i in range(1, K):  # the 0-th place is kept empty
        cfg = clsloss_cfg[i]
        loss_net.append(
            ClassifierLossModuleList(modules=[
                RandomInfoProDecoder(cfg["out_plane"], cfg["upscale"],
                                     patch_size=cfg["patch_sz"], num_patches=cfg["n_patch"],
                                     middle_planes=cfg["mid_plane"], outplanes=cfg["in_plane"], loss=nn.L1Loss()),
                AuxClassifier(cfg["out_plane"], net_config='1c2f', loss_mode='cross_entropy',
                              feature_dim=128, class_num=class_num,
                              pooling=GatedAttentionPooling(cfg["out_plane"], cfg["out_plane"] // 2, dropout=0.2),
                              kernel_size=cfg["kernel_size"], stride=cfg["stride"], padding=cfg["padding"],
                              loss_weight=weight),
                ],
                alphas=[args.alpha,  ### Hyperparameter alpha
                        1.],
            )
        )

    loss_net.append(
        ClassifierLossModuleList([
            AuxClassifier(512, net_config='1c2f', loss_mode='cross_entropy', feature_dim=512,
                          pooling=GatedAttentionPooling(512, 128, dropout=0.2),
                          class_num=class_num,
                          kernel_size=5, stride=5, padding=0,
                          loss_weight=weight,
                          ),
        ])
    )
    return loss_net


def main(args):
    data_module = CsvDataModule(args.dataset_root, args.dataset_csv, args.batch_size, cus_transforms=get_A_transforms(),
                                num_workers=args.num_workers)
    num_classes = args.num_classes

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    backbone_model = Resnet34LocalBatchNorm(K=args.K)

    loss_weight = args.loss_weight
    loss_networks = get_loss_networks(args.K, num_classes, weight=loss_weight)

    trainer_model = LocalModule(backbone_model, loss_networks, get_metric(num_classes), args, num_classes,
                                valid_as_train=True)

    logger = WandbLogger(project=args.project_name, name=args.run_name, log_model=True)

    trainer = pl.Trainer(default_root_dir=os.path.join(args.output_dir, args.run_name), gpus=args.gpu_id,
                         max_epochs=args.epochs, log_every_n_steps=50, num_sanity_val_steps=0,
                         precision=args.precision,
                         logger=logger,
                         callbacks=lr_monitor,
                         progress_bar_refresh_rate=None if args.progressive else 0)

    trainer.fit(trainer_model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_arguments(parser)
    # save_parameters(args)
    main(args)
