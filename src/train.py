import os
import sys
import torch
import pytorch_lightning as pl
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
        lr=1e-4
    )
    datamodule = MainDataModule(dataset_dir, train_batch_size=16, num_workers=8, param_dim=module.param_dim, process_leaf=True)
    tqdm_cb = TQDMProgressBar(refresh_rate=10)
    tb_logger = TensorBoardLogger(
        name='Image2Helios_20240925_process_leaf',
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
        callbacks=[tqdm_cb, ckpt_cb, lr_monitor, early_stop_cb],
        # callbacks=[tqdm_cb, ckpt_cb, lr_monitor],
        logger=tb_logger,
        precision="bf16-mixed"
    )
    # module = MainModule.load_from_checkpoint('./saved/last.ckpt')
    trainer.fit(module, datamodule=datamodule)

    # To check the training progress,
    # run the following command in the terminal:
    # tensorboard --logdir=./log