import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import BatchSizeFinder, LearningRateFinder

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
from concurrent.futures import ThreadPoolExecutor
# 경로 설정
script_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_file_dir)

# 모듈 임포트
from models.model import TransformerDecoderModel, RegressionModel, ViT_FeatureExtractor, CNN_FeatureExtractor
from models.model import RegressionModel_Transformer, PositionalEncoding, VAE, MLP, SeqEmbeddingModel
from models.model import create_organ_mask, get_tgt_mask, create_pad_mask, text_global_pool
from src.plant_tokenizer import SOS_token, EOS_token, PAD_token, EOS_vec_padded, SOS_vec_padded
from src.plant_tokenizer import generate_noise_plant_tokens
from src.plant_dataset import PlantDataset
from src.plantstring2model import plantstring2model
from src.plant_tokenizer import token2vec as token2vec
from src.string_to_xml_to_vec import vec2string
from src.image_process import process_leaf_image
from plant_architecture_utils import coordinates_to_angle
import pickle
import copy

from models.model import PlantArchitectureTransformer

# from open_clip.transformer import text_global_pool

# Disable fastpath for TransformerEncoder and MultiHeadAttention
# torch.backends.mha.set_fastpath_enabled(False)

class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

class MainModule(pl.LightningModule):
    def __init__(self, num_layers=6, num_heads=8, 
                 seq_dim=23, seq_embedding_dim=768//2, 
                 param_dim=22, param_embedding_dim=768//2, 
                 image_size=224, alpha=1.0, lr=1e-5, 
                 dropout=0.10, 
                 max_len=2024,
                 use_depth=False):
        super(MainModule, self).__init__()
        self.save_hyperparameters()  # 전달된 모든 인수를 저장

        # self.automatic_optimization = False

        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
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
        self.use_depth = use_depth
        self.max_len = max_len

        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_background = cv2.resize(cv2.imread(os.path.join(self.current_script_dir, "../src/assets/dirt.jpg")), (self.image_size, self.image_size))
            # Conver to RGB
            self.depth_background = cv2.cvtColor(self.depth_background, cv2.COLOR_BGR2RGB)

        self.image_encoder = ViT_FeatureExtractor(output_size=seq_embedding_dim+param_embedding_dim, use_depth=self.use_depth, image_size=image_size)

        # Froze self.feature_extractor
        # self.image_encoder.eval()
        
        self.sequence_decoder = TransformerDecoderModel(
            seq_embedding_dim=self.seq_embedding_dim,
            param_embedding_dim=self.param_embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_tokens=self.seq_dim,
            num_params=self.param_dim,
            decoder_only=True,
            use_depth=self.use_depth,
            image_size=self.image_size,
            dropout=self.dropout,
            max_seq_length=max_len,
        )
        
        self.multihead_attn_weights = None
        self.self_attn_weights = None


        self.helios_path = os.path.join(self.current_script_dir, "../src/PlantString2Model/build")
        self.helios = plantstring2model(program_path=self.helios_path,
                                                        program_name="PlantString2Model",
                                                        display=":11.0", 
                                                        height=1.0,background_path=os.path.join(self.current_script_dir,"../src/assets/black.png"))
    
        self.prev_epoch = -1
        self.current_train_step = 0
        self.current_val_step = 0

    def forward(self, image, y_input):
        tgt_mask = get_tgt_mask(y_input.size(1))
        tgt_pad_mask = create_pad_mask(y_input, PAD_token)
        if self.use_depth:
            image = self.add_depth_to_image(image)
        features = self.image_encoder(image)
        outputs = self.sequence_decoder(features, y_input, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        outputs = outputs.permute(1, 0, 2)
        return outputs
    
    def generate(self, image, stage='val'):
        device = image.device
        y_input = torch.tensor(SOS_vec_padded, dtype=torch.float32)

        y_input = y_input.unsqueeze(0).unsqueeze(0)
        y_input = y_input.to(device)

        if self.use_depth:
            image = self.add_depth_to_image(image)

        feature = self.image_encoder(image)
        for i in range(self.max_len):
            # Use torch.cuda.amp for mixed precision
            try:
                if stage == 'val':
                    with torch.no_grad():
                        pred = self.sequence_decoder(feature, y_input)
                else:
                    pred = self.sequence_decoder(feature, y_input)
            except Exception as e:
                print(e)
                print(f"Error in {i} iteration")
                break
            label_p = pred[:,:,:self.seq_dim]
            label = label_p.topk(1)[1].view(-1)[-1].item()  # num with highest probability
            params = pred[:,:,self.seq_dim:]

            # Stop if model predicts end of sentencplant_structure_vit_transformer_withpsudodepth_paramEste
            ## if label == EOS_token:
            if label == EOS_token or label == PAD_token:
                break

            # Make next tensor using label and params
            next_item = torch.cat((torch.tensor([[label]], dtype=torch.float32, device=device), params[-1]), dim=1).unsqueeze(0)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

        return y_input.squeeze(0).tolist()
    
    def label_loss_fn(self, pred, label, ignore_index=PAD_token):
        return F.cross_entropy(pred, label, ignore_index=ignore_index)

    def param_loss_fn(self, pred, params, ignore_index=PAD_token):
        # Create neg mask
        neg_mask = (params == ignore_index)
        # Create masks
        mask = ~neg_mask
        loss_mse = F.smooth_l1_loss(pred, params, reduction='none') # mse_loss or smooth_l1_loss
        masked_loss = loss_mse * mask
        return masked_loss.sum() / (mask).sum()

    def param_loss_fn_bylabel(self, label, values, pred, ignore_index=PAD_token):
        # label: (batch_size, seq_len)
        # pred: (batch_size, seq_len, param_dim)
        # Masked values are not included in the loss

        # Create masks
        neg_organ_masks = create_organ_mask().to(pred.device) # Negative masks

        # Ensure label_mod and masks have compatible dimensions
        label_mod = label % 4
        neg_mask = (values == ignore_index)  # First mask is for padding
        for i in range(4):
            neg_mask = neg_mask | ((label_mod == i).unsqueeze(1).expand_as(neg_mask) & neg_organ_masks[i].unsqueeze(0).unsqueeze(2).expand_as(neg_mask))

        # Compute loss
        loss_mse = F.smooth_l1_loss(pred, values, reduction='none') # mse_loss or smooth_l1_loss
        # Create masks by negating the neg_mask
        mask = ~neg_mask
        masked_loss = loss_mse * mask
        return masked_loss.sum() / (mask).sum()
        #return masked_loss.sum() / masked_loss.size(0)

    def add_depth_to_image(self, image, add_background=True):
    
        if add_background:
            depth_input = torch.zeros_like(image)
            # Add black background the images
            for i in range(image.size(0)):
                # Convert to numpy
                img = image[i].permute(1, 2, 0).cpu().numpy()
                # Mask 0 values
                mask = img == 0
                img[mask] = self.depth_background[mask]
                # Convert to tensor
                depth_input[i] = torch.tensor(img).permute(2, 0, 1)
        else:
            depth_input = image

        inputs = self.depth_est_img_proc(images=depth_input, return_tensors="pt").to(image.device)
        with torch.no_grad():
            outputs = self.depth_est_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        
        if 1:
            # Normalize to 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            image = (image - image.min()) / (image.max() - image.min())
        # cat depth to image
        image = torch.cat((image, depth), dim=1)

        self.predicted_depth = depth
    
        return image
    
    def make_negative_imgs(self, image):
        # Suffle the image along the batch dimension. make sure i != j
        # Ensure i != j by checking for identity permutation and reshuffling if necessary
        batch_size = image.size(0)
        # 무작위로 인덱스를 섞음
        idx = np.random.permutation(batch_size)
        if 0:
            # 인덱스가 동일한 경우 요소를 교환하여 섞인 인덱스를 생성
            while np.array_equal(idx, np.arange(batch_size)):
                for i in range(batch_size):
                    if i == idx[i]:
                        j = np.random.randint(0, batch_size)
                        idx[i], idx[j] = idx[j], idx[i]
        
        image = image[idx]

        # Add noise to the plant images
        transform = transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)])

        image = transform(image)

        return image
    
    def make_negative_seqs(self, seqs, shuffle=True, noise_level=0.2):
        # Suffle the seqs along the batch dimension. make sure i != j
        # Ensure i != j by checking for identity permutation and reshuffling if necessary
        batch_size = seqs.size(0)
        # 무작위로 인덱스를 섞음
        if shuffle:
            idx = np.random.permutation(batch_size)
            if 0:
                # 인덱스가 동일한 경우 요소를 교환하여 섞인 인덱스를 생성
                while np.array_equal(idx, np.arange(batch_size)):
                    for i in range(batch_size):
                        if i == idx[i]:
                            j = np.random.randint(0, batch_size)
                            idx[i], idx[j] = idx[j], idx[i]
            seqs = seqs[idx]

        # Add noise to seq
        if 0:
            noises = generate_noise_plant_tokens(seqs)
        else:
            noises = torch.randn_like(seqs, requires_grad=True) * noise_level
        seqs = seqs + noises

        return seqs
    
    
    def compute_loss(self, batch, mode):

        # Load batch and preprocess
        image, y, lengths = batch
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:]

        # Decoder loss
        pred = self(image, y_input)
        label_loss = self.label_loss_fn(pred[:, :, :self.seq_dim].permute(0, 2, 1), label) # (N, C, L)
        if 1:
            param_loss = self.param_loss_fn(pred[:, :, self.seq_dim:], values)
        else:
            param_loss = self.param_loss_fn_bylabel(label=label, values=values, pred=pred[:, :, self.seq_dim:])

        ######### Tensorboard logging
        loss = label_loss + param_loss

        self.log(f'{mode}/label_loss', label_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/param_loss', param_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/loss', loss, batch_size=image.size(0), sync_dist=True)

        # Add images to tensorboard
        if (self.current_train_step == 0 and mode == "train") or (self.current_val_step == 0 and mode == "val"):
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_images(f'{mode}/input_images', image, self.current_epoch)
            if self.use_depth:
                tensorboard_logger.add_images(f'{mode}/depth_images', self.predicted_depth, self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, 'train')
        self.current_train_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, 'val')
        self.current_val_step += 1
        return loss

    def log_grads(self):
         for name, param in self.named_parameters():
            # if "seq_embedding_layer" in name:
            #     print(f"Gradient of {name} is {param.grad}")
            #     print(f"{name} requires_grad: {param.requires_grad}")
            if param.grad is not None:
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.current_epoch) # or global_step
                self.logger.experiment.add_histogram(f"{name}", param, self.current_epoch) # or global_step

    def on_after_backward(self):
        if self.prev_epoch != self.current_epoch:
            self.prev_epoch = self.current_epoch
            # self.log_grads()
            self.current_train_step = 0
            self.current_val_step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=40, verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'  # 모니터링할 지표를 지정
            }
        }
    
class MainDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, train_batch_size=16, val_batch_size=None,
                        num_workers=4, image_size=448, 
                        load_depth=True,
                        process_leaf=False,
                        preload=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.preload = preload
        self.process_leaf = process_leaf
        self.load_depth = load_depth
        self.pin_memory = True

        self.img_aug = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ])
        
        self.train_transform = transforms.Compose([
                self.img_aug,
                # transforms.ToTensor(),
               
        ])
        self.test_transform = transforms.Compose([
                # transforms.ToTensor(),
                # transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())
        ])

    def load_or_create_dataset(self, dataset_dir, dataset_name, plot, stages, transform, load_depth, process_leaf, preload, image_size):
        saved_dataset_name = os.path.join(dataset_dir, f"{dataset_name}.pkl")
        if os.path.exists(saved_dataset_name) and preload:
            print(f"Loading {dataset_name} dataset from .pkl file")
            with open(saved_dataset_name, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = PlantDataset(
                dataset_dir, plot=plot, stages=stages,
                transform=transform, load_depth=load_depth,
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
        growth_stages = [f"{day:02d}" for day in range(20)]
        train_plots = [f"{plot:04d}" for plot in range(50)]
        self.train_dataset = self.load_or_create_dataset(
            self.dataset_dir, "train_dataset", train_plots, growth_stages,
            self.train_transform, self.load_depth, self.process_leaf,
            self.preload, self.image_size
        )
        val_plots = [f"{plot:04d}" for plot in range(50,75)]
        self.val_dataset = self.load_or_create_dataset(
            self.dataset_dir, "val_dataset", val_plots, growth_stages,
            self.test_transform, self.load_depth, self.process_leaf,
            self.preload, self.image_size
        )
        test_plots = [f"{plot:04d}" for plot in range(75,100)]
        self.test_dataset = self.load_or_create_dataset(
            self.dataset_dir, "test_dataset", test_plots, growth_stages,
            self.test_transform, self.load_depth, self.process_leaf,
            self.preload, self.image_size
        )

        

    def collate_fn(self, batch):
        images, vectors, lengths = zip(*batch)
        max_length = max(lengths)
        vec_dim = vectors[0].shape[-1]
        if len(vectors[0].shape) == 1:
            vectors_padded = np.ones((len(vectors), max_length), dtype=int) * PAD_token
        else:
            vectors_padded = np.ones((len(vectors), max_length, vec_dim)) * PAD_token
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
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )