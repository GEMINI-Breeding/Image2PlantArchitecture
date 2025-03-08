import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime
import platform
# 경로 설정
script_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_file_path)))
from models.plightning import MainModule, MainDataModule
from plant_tokenizer import EOS_token, N_PARAMS
import joblib

torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    # Tensor Cores 활용을 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    # Set the seed for reproducibility
    pl.seed_everything(42)

    # Define dataset to solve
    dataset_dir = "data/20250123_Sideview_40Days"
    datamodule = MainDataModule(dataset_dir,
                                image_size=224,
                                load_depth=False,
                                train_batch_size=16, num_workers=8, process_leaf=True, preload=True, side_view=True)
    
    if 1:
        module = MainModule(
            num_layers=12,
            num_heads=8,
            seq_dim=EOS_token+1,
            seq_embedding_dim=768//2,
            param_dim=N_PARAMS,
            param_embedding_dim=768//2,
            image_size=datamodule.image_size,
            alpha=10.0,
            lr=1e-5,
            use_depth=True,
            cat_emb=False,
            dropout=0.10,
        )
    else:
        module = MainModule.load_from_checkpoint("log/20250306_Final_for_Paper/version_2/checkpoints/best_epoch=07.ckpt")

    tqdm_cb = TQDMProgressBar(refresh_rate=10)

    # Generate today's date string in YYYYMMDD format
    today_date_str = datetime.now().strftime('%Y%m%d')
    tb_logger = TensorBoardLogger(
        name=f'{today_date_str}_Final_for_Paper_aplha10_add_emb',
        save_dir='./log'
    )

    # ModelCheckpoint 설정
    ckpt_cb = ModelCheckpoint(
        monitor='val/loss',  # Metric to monitor
        dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
        filename="best_{epoch:02d}",
        save_top_k=1,  # Save only the best model
        save_last=True,
        save_weights_only=True  # 가중치만 저장
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_cb = EarlyStopping(
        monitor='val/loss', # Metric to monitor
        patience=10,
        verbose=True,
        mode='min'
    )

    # Check the current platform
    current_platform = platform.system()

    # Set the accelerator based on the platform
    if current_platform == "Darwin":
        accelerator = "mps"
    else:
        accelerator = "gpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices="auto",
        max_epochs=100,
        callbacks=[tqdm_cb, ckpt_cb, lr_monitor, early_stop_cb, 
                #    FineTuneBatchSizeFinder(milestones=(5, 10)),
                #    FineTuneLearningRateFinder(milestones=(5, 10))
                   ],
        # callbacks=[tqdm_cb, ckpt_cb, lr_monitor],
        logger=tb_logger,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True)  # Enable detection of unused parameters
    )
    trainer.fit(module, datamodule=datamodule)

    # To check the training progress,
    # run the following command in the terminal:
    # tensorboard --logdir=./log
