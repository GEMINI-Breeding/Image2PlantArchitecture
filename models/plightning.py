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

# 모듈 임포트
from models.model import ImageToSequenceTransformer, get_tgt_mask, create_pad_mask
from plant_tokenizer import SOS_token, EOS_token, PAD_token, params_EOS_token_padded, params_SOS_token_padded
from plant_dataset import PlantDataset


class MainModule(pl.LightningModule):
    def __init__(self, num_layers, num_heads, seq_dim, seq_embedding_dim, param_dim, param_embedding_dim, image_size, alpha, lr):
        super(MainModule, self).__init__()
        self.save_hyperparameters()  # 전달된 모든 인수를 저장

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_dim = seq_dim
        self.seq_embedding_dim = seq_embedding_dim
        self.param_dim = param_dim
        self.param_embedding_dim = param_embedding_dim
        self.image_size = image_size
        self.alpha = alpha
        self.lr = lr

        self.model = ImageToSequenceTransformer(
            seq_embedding_dim=self.seq_embedding_dim,
            param_embedding_dim=self.param_embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_tokens=self.seq_dim,
            num_params=self.param_dim,
            decoder_only=True,
            use_depth=True,
            image_size=self.image_size
        )

    def forward(self, image, y_input):
        tgt_mask = get_tgt_mask(y_input.size(1))
        tgt_pad_mask = create_pad_mask(y_input, PAD_token)
        outputs = self.model(image, y_input, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        outputs = outputs.permute(1, 2, 0)
        return outputs

    def label_loss_fn(self, pred, label):
        return F.cross_entropy(pred, label, ignore_index=PAD_token)

    def param_loss_fn(self, pred, params):
        mask = (params == PAD_token)
        loss_mse = F.mse_loss(pred, params, reduction='none')
        masked_loss = loss_mse * ~mask
        return masked_loss.sum() / (~mask).sum()

    def training_step(self, batch, batch_idx):
        image, y, lengths = batch
        y_input = y[:, :-1]
        pred = self(image, y_input)

        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:].permute(0, 2, 1)

        label_loss = self.label_loss_fn(pred[:, :self.seq_dim], label)
        param_loss = self.param_loss_fn(pred[:, self.seq_dim:], values)
        loss = label_loss + self.alpha * param_loss

        self.log('train/0_label_loss', label_loss, batch_size=image.size(0))
        self.log('train/1_param_loss', param_loss, batch_size=image.size(0))
        self.log('train/2_loss', loss, batch_size=image.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        image, y, lengths = batch
        y_input = y[:, :-1]
        pred = self(image, y_input)

        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:].permute(0, 2, 1)

        label_loss = self.label_loss_fn(pred[:, :self.seq_dim], label)
        param_loss = self.param_loss_fn(pred[:, self.seq_dim:], values)
        loss = label_loss + self.alpha * param_loss

        self.log('val/0_label_loss', label_loss, batch_size=image.size(0))
        self.log('val/1_param_loss', param_loss, batch_size=image.size(0))
        self.log('val/2_loss', loss, batch_size=image.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

class MainDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size=16, num_workers=4, image_size=448, preload=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.preload = preload
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        ])

    def setup(self, stage=None):
        self.train_dataset = PlantDataset(
            self.dataset_dir, plot=["000", "001", "002"],
            transform=self.transform, use_depth=True,
            preload=self.preload, image_size=self.image_size
        )
        self.val_dataset = PlantDataset(
            self.dataset_dir, plot=["003"],
            transform=self.transform, use_depth=True,
            preload=self.preload, image_size=self.image_size
        )
        self.test_dataset = PlantDataset(
            self.dataset_dir, plot=["004"],
            transform=self.transform, use_depth=True,
            preload=self.preload, image_size=self.image_size
        )

    def collate_fn(self, batch):
        images, vectors, lengths = zip(*batch)
        max_length = max(lengths)
        if len(vectors[0].shape) == 1:
            vectors_padded = np.ones((len(vectors), max_length), dtype=int) * PAD_token
        else:
            vectors_padded = np.ones((len(vectors), max_length, 1 + 5 + 4 + 3 + 4)) * PAD_token
            if 0:
                vectors_padded[:, :, 1:] = 0

        for i, vector in enumerate(vectors):
            end = lengths[i]
            vectors_padded[i, :end] = vector

        images = torch.stack(images)
        vectors_padded = torch.tensor(vectors_padded, dtype=torch.float32)
        return images, vectors_padded, lengths

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )