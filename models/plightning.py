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

import random

# 경로 설정
script_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_file_path)))

# 모듈 임포트
from models.model import ImageToSequenceTransformer, get_tgt_mask, create_pad_mask
from src.plant_tokenizer import SOS_token, EOS_token, PAD_token, params_EOS_token_padded, params_SOS_token_padded
from src.plant_dataset import PlantDataset

import pickle

class MainModule(pl.LightningModule):
    def __init__(self, num_layers, num_heads, seq_dim, seq_embedding_dim, param_dim, param_embedding_dim, image_size, alpha, lr, dropout):
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
        self.dropout = dropout

        self.model = ImageToSequenceTransformer(
            seq_embedding_dim=self.seq_embedding_dim,
            param_embedding_dim=self.param_embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_tokens=self.seq_dim,
            num_params=self.param_dim,
            decoder_only=True,
            use_depth=True,
            image_size=self.image_size,
            dropout=self.dropout,
        )
        self.multihead_attn_weights = None
        self.self_attn_weights = None

        if 0:
            # Test to generate 2048 tokens if memory is not enough
            try:
                print("Test generate")
                empty_image = torch.zeros(1, 4, self.image_size, self.image_size)
                empty_image.to("cuda")
                self.generate(empty_image, max_len=2048, stage='test')
                print("Test generate success")
            except Exception as e:
                print(e)
                print("Error in test generate")



    def forward(self, image, y_input):
        tgt_mask = get_tgt_mask(y_input.size(1))
        tgt_pad_mask = create_pad_mask(y_input, PAD_token)
        outputs = self.model(image, y_input, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        outputs = outputs.permute(1, 2, 0)
        return outputs
    
    def generate(self, image, max_len=2048, stage='test'):
        device = image.device
        y_input = torch.tensor(params_SOS_token_padded, dtype=torch.float32)

        y_input = y_input.unsqueeze(0).unsqueeze(0)
        y_input = y_input.to(device)

        for i in range(max_len):
            # Get source mask
            tgt_mask = get_tgt_mask(y_input.size(1)).to(device)
            
            # Use torch.cuda.amp for mixed precision
            with torch.cuda.amp.autocast():
                try:
                    if stage == 'test':
                        with torch.no_grad():
                            pred = self.model(image, y_input, tgt_mask)
                    else:
                        pred = self.model(image, y_input, tgt_mask)
                except Exception as e:
                    print(e)
                    print(f"Error in {i} iteration")
                    break
            label_p = pred[:,:,:self.seq_dim]
            label = label_p.topk(1)[1].view(-1)[-1].item()  # num with highest probability
            params = pred[:,:,self.seq_dim:]

            # Stop if model predicts end of sentencplant_structure_vit_transformer_withpsudodepth_paramEste
            # if label == EOS_token or label == PAD_token:
            if label == EOS_token:
                break

            # Make next tensor using label and params
            next_item = torch.cat((torch.tensor([[label]], dtype=torch.float32, device=device), params[-1]), dim=1).unsqueeze(0)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

        return y_input.squeeze(0).tolist()
    
    def load_attn_weights(self):
        self.multihead_attn_weights = self.model.multihead_attn_weights
        self.self_attn_weights = self.model.self_attn_weights

        return self.multihead_attn_weights, self.self_attn_weights

    def label_loss_fn(self, pred, label):
        return F.cross_entropy(pred, label, ignore_index=PAD_token)

    def param_loss_fn(self, pred, params, ignore_index=PAD_token):
        mask = (params == ignore_index)
        if 0:
            loss_mse = F.mse_loss(pred, params, reduction='none')
        else:
            loss_mse = F.smooth_l1_loss(pred, params, reduction='none')
        masked_loss = loss_mse * ~mask
        return masked_loss.sum() / (~mask).sum()

    def training_step(self, batch, batch_idx):
        image, y, lengths = batch
        y_input = y[:, :-1]
        
        # # Teacher Forcing 확률에 따라 Teacher Forcing 사용 여부 결정
        # use_teacher_forcing = True if random.random() < self.prob_teacher_forcing else False

        pred = self(image, y_input)

        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:].permute(0, 2, 1)

        label_loss = self.label_loss_fn(pred[:, :self.seq_dim], label)
        param_loss = self.param_loss_fn(pred[:, self.seq_dim:], values)
        loss = label_loss + self.alpha * param_loss

        self.log('train/label_loss', label_loss, batch_size=image.size(0), sync_dist=True)
        self.log('train/param_loss', param_loss, batch_size=image.size(0), sync_dist=True)
        self.log('train/loss', loss, batch_size=image.size(0), sync_dist=True)
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

        self.log('val/label_loss', label_loss, batch_size=image.size(0), sync_dist=True)
        self.log('val/param_loss', param_loss, batch_size=image.size(0), sync_dist=True)
        self.log('val/loss', loss, batch_size=image.size(0), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if 1:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
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
    def __init__(self, dataset_dir, train_batch_size=16, val_batch_size=None, 
                        num_workers=4, image_size=448, 
                        param_dim=5 + 4 + 3 + 4,
                        process_leaf=False,
                        preload=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.preload = preload
        self.param_dim = param_dim
        self.process_leaf = process_leaf
        self.use_depth = True
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        ])

    def load_or_create_dataset(self, dataset_dir, dataset_name, plot, transform, use_depth, process_leaf, preload, image_size):
        saved_dataset_name = os.path.join(dataset_dir, f"{dataset_name}.pkl")
        if os.path.exists(saved_dataset_name):
            print(f"Loading {dataset_name} dataset from .pkl file")
            with open(saved_dataset_name, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = PlantDataset(
                dataset_dir, plot=plot,
                transform=transform, use_depth=use_depth,
                process_leaf=process_leaf,
                preload=preload, image_size=image_size,
            )
            if preload:
                # Check if the dataset is already saved
                if not os.path.exists(saved_dataset_name):
                    print(f"Saving {dataset_name} dataset to .pkl file")
                    with open(saved_dataset_name, "wb") as f:
                        pickle.dump(dataset, f)
        return dataset

    def setup(self, stage=None):
        self.train_dataset = self.load_or_create_dataset(
            self.dataset_dir, "train_dataset", ["000", "001", "002"],
            self.transform, self.use_depth, self.process_leaf,
            self.preload, self.image_size
        )

        self.val_dataset = self.load_or_create_dataset(
            self.dataset_dir, "val_dataset", ["003"],
            self.transform, self.use_depth, self.process_leaf,
            self.preload, self.image_size
        )

        self.test_dataset = self.load_or_create_dataset(
            self.dataset_dir, "test_dataset", ["004"],
            self.transform, self.use_depth, self.process_leaf,
            self.preload, self.image_size
        )

    def collate_fn(self, batch):
        images, vectors, lengths = zip(*batch)
        max_length = max(lengths)
        if len(vectors[0].shape) == 1:
            vectors_padded = np.ones((len(vectors), max_length), dtype=int) * PAD_token
        else:
            vectors_padded = np.ones((len(vectors), max_length, 1 + self.param_dim)) * PAD_token
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
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers
        )