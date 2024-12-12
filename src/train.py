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
from plant_tokenizer import EOS_token

if __name__ == "__main__":
    # Tensor Cores 활용을 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    # Define dataset to solve
    dataset_dir = "data/2000_Plots_20241210"
    #dataset_dir = "data/generated_Dec10_2024"
    datamodule = MainDataModule(dataset_dir,
                                image_size=224,
                                load_depth=False,
                                # train_batch_size=100, num_workers=8, process_leaf=False, preload=False) # for a100 gpu
                                train_batch_size=1, num_workers=0, process_leaf=False, preload=False) # for gpum
    if 0:
        module = MainModule(
            num_layers=12,
            num_heads=8,
            seq_dim=EOS_token+1,
            seq_embedding_dim=768//2,
            param_dim=24,
            param_embedding_dim=768//2,
            image_size=datamodule.image_size,
            alpha=1.0,
            lr=1e-5,
            use_depth=True,
            dropout=0.10,
        )
    else:
        module = MainModule.load_from_checkpoint('log/20241211_num_layers12/version_2/checkpoints/best_epoch=43.ckpt')

    tqdm_cb = TQDMProgressBar(refresh_rate=10)

    # Generate today's date string in YYYYMMDD format
    today_date_str = datetime.now().strftime('%Y%m%d')
    tb_logger = TensorBoardLogger(
        name=f'{today_date_str}_num_layers12',
        save_dir='./log'
    )

    # ModelCheckpoint 설정
    ckpt_cb = ModelCheckpoint(
        monitor='val/loss',  # Metric to monitor
        dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
        filename="best_{epoch:02d}",
        save_top_k=1,  # Save only the best model
        save_last=False,
        save_weights_only=True  # 가중치만 저장
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    early_stop_cb = EarlyStopping(
        monitor='val/loss', # Metric to monitor
        patience=100,
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
        max_epochs=400,
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
