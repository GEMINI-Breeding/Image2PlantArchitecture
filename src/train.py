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
from plant_tokenizer import VOCAB_SIZE
import joblib

torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    # Tensor Cores 활용을 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    # Set the seed for reproducibility
    pl.seed_everything(42)

    # Define configuration dictionary
    config = {
        "dataset_dir": "data/20250311_Sideview_40Days",
        "image_size": 448,
        "load_depth": False,
        "train_batch_size": 16,
        "num_workers": 8,
        "process_leaf": True,
        "preload": True,
        "side_view": False,
        "partial_data": 1.0,
        #"growth_stages": ["01", "02", "03", "04", "05"],
        "growth_stages": ["01"],
        "num_layers": 12,
        "num_heads": 8,
        "num_tokens": VOCAB_SIZE,
        "dim_model": 768,
        "alpha": 1.0,
        "lr": 1e-4,
        "use_depth": False,
        "decoder_only": False,
        "dropout": 0.10,
        "vit_model": "facebook/dinov2-base"
    }

    datamodule = MainDataModule(**config)
    module = MainModule(**config)
    
    #module = MainModule.load_from_checkpoint("log/20250306_Final_for_Paper/version_2/checkpoints/best_epoch=07.ckpt")

    tqdm_cb = TQDMProgressBar(refresh_rate=10)

    # Generate today's date string in YYYYMMDD format
    today_date_str = datetime.now().strftime('%Y%m%d')
    tb_logger = TensorBoardLogger(
        name=f'{today_date_str}_Quantize_Small_FullTransformer_448_Day1',
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
        patience=20,
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
        # strategy=DDPStrategy(find_unused_parameters=True)  # Enable detection of unused parameters
    )
    trainer.fit(module, datamodule=datamodule)

    # To check the training progress,
    # run the following command in the terminal:
    # tensorboard --logdir=./log
