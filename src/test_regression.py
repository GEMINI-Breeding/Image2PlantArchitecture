import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime

import torchvision.transforms as transforms

# 경로 설정
script_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_file_path)))
from models.plightning import MainModule, MainDataModule, FineTuneBatchSizeFinder, FineTuneLearningRateFinder, SimpleRegressionTest, SimpleRegressionVAE

if __name__ == "__main__":
    # Tensor Cores 활용을 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    dataset_dir = "/home/lion397/codes/Image2PlantArchitecture/data/generated_dataset_Sep22_black"
    module = SimpleRegressionVAE(
    #module = SimpleRegressionTest(
        image_size=224,
        lr=1e-4,
        dropout=0.10,
        d_model=128,
        use_depth=False,
        vit_finetune=True
    )

    datamodule = MainDataModule(dataset_dir,
                                image_size=module.image_size,
                                train_batch_size=4, num_workers=4,
                                load_depth=False,
                                process_leaf=True, preload=True)
    tqdm_cb = TQDMProgressBar(refresh_rate=10)

    # Generate today's date string in YYYYMMDD format
    today_date_str = datetime.now().strftime('%Y%m%d')
    tb_logger = TensorBoardLogger(
        name=f'{today_date_str}_SimpleRegressionVAE',
        #name=f'{today_date_str}_SimpleRegressionTest',
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

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    early_stop_cb = EarlyStopping(
        monitor='val/loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=200,
        callbacks=[tqdm_cb, ckpt_cb, lr_monitor, early_stop_cb, 
                #    FineTuneBatchSizeFinder(milestones=(5, 10)),
                #    FineTuneLearningRateFinder(milestones=(5, 10))
                   ],
        # callbacks=[tqdm_cb, ckpt_cb, lr_monitor],
        logger=tb_logger,
        # precision="bf16-mixed" #"16, 16-mixed, bf16, bf16-mixed",
        #strategy=DDPStrategy(find_unused_parameters=True)  # Enable detection of unused parameters
    )
    # module = SimpleRegressionTest.load_from_checkpoint('log/20241007_RGBD_Dinov2Finetune/version_0/checkpoints/best_epoch=86.ckpt')
    trainer.fit(module, datamodule=datamodule)

    # To check the training progress,
    # run the following command in the terminal:
    # tensorboard --logdir=./log