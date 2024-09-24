import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# 경로 설정
script_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_file_path)))
from models.plightning import MainModule, MainDataModule

if __name__ == "__main__":
    # Tensor Cores 활용을 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    dataset_dir = "/home/lion397/codes/Image2PlantArchitecture/data/generated_dataset_Sep22_black"
    module = MainModule(
        num_layers=3,
        num_heads=4,
        seq_dim=43,
        seq_embedding_dim=64,
        param_dim=16,
        param_embedding_dim=64,
        image_size=448,
        alpha=1.0,
        lr=1e-3
    )
    datamodule = MainDataModule(dataset_dir)
    tqdm_cb = TQDMProgressBar(refresh_rate=10)
    ckpt_cb = ModelCheckpoint(
        dirpath='./saved',
        filename="{epoch:02d}_",
        save_last=True
    )
    tb_logger = TensorBoardLogger(
        name='Image2Helios_20240924',
        save_dir='./log'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_cb = EarlyStopping(
        monitor='val/loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=200,
        callbacks=[tqdm_cb, ckpt_cb, lr_monitor, early_stop_cb],
        logger=tb_logger,
        precision="16-mixed",  # 16비트 훈련 활성화
    )
    trainer.fit(module, datamodule=datamodule)

    # To check the training progress,
    # run the following command in the terminal:
    # tensorboard --logdir=./log