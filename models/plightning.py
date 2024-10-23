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
from src.plant_tokenizer import SOS_token, EOS_token, PAD_token, params_EOS_token_padded, params_SOS_token_padded
from src.plant_tokenizer import generate_noise_plant_tokens
from src.plant_dataset import PlantDataset
from src.plantstring2model import plantstring2model
from src.plant_tokenizer import token2vec as token2vec
from src.string_to_xml_to_vec import vec2string
from src.image_process import process_leaf_image
from src.utils import coordinates_to_angle
import pickle
import copy

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
                 seq_dim=43, seq_embedding_dim=768//2, 
                 param_dim=22, param_embedding_dim=768//2, 
                 image_size=224, alpha=1.0, lr=1e-5, 
                 dropout=0.10, use_depth=False):
        super(MainModule, self).__init__()
        self.save_hyperparameters()  # 전달된 모든 인수를 저장

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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
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
            decoder_only=False,
            use_depth=self.use_depth,
            image_size=self.image_size,
            dropout=self.dropout,
        )

        # self.seq_embedding_layer = nn.Linear(self.seq_dim+self.param_dim, self.seq_embedding_dim+self.param_embedding_dim)
        # transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_embedding_dim+self.param_embedding_dim, nhead=8)
        # self.seq_embedding_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=6)
        # self.positional_encoding = PositionalEncoding(dim_model=self.seq_embedding_dim+self.param_embedding_dim, max_len=2048, dropout_p=0.1)
        # self.positional_encoding.eval()
        self.SeqEmbeddingModel = SeqEmbeddingModel(d_label=self.seq_dim,
                                                   d_param=self.param_dim,
                                                   d_model=seq_embedding_dim+param_embedding_dim)

        self.multihead_attn_weights = None
        self.self_attn_weights = None

        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            # Define a 4 ch to 3 ch conversion layer
            self.ch4_to_ch3_conv = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1) 

        self.helios_path = os.path.join(self.current_script_dir, "../src/PlantString2Model/build")
        self.helios = plantstring2model(program_path=self.helios_path,
                                                        program_name="PlantString2Model",
                                                        display=":11.0", 
                                                        height=1.0,background_path=os.path.join(self.current_script_dir,"../src/assets/black.png"))
        
        # Test gen
        # Run 
        self.helios.run(in_plantstring_path=os.path.abspath("plant_string.txt"),
                        output_path=os.path.abspath("temp/batch_0"))
        
        self.cosine_embedding_loss_function = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        self.mse_loss = nn.MSELoss()
        #self.triplet_loss = nn.TripletMarginLoss(margin=10.0, p=2)
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=2.0)
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
        outputs = outputs.permute(1, 2, 0)
        return outputs
    
    def generate(self, image, max_len=2048, stage='val'):
        device = image.device
        y_input = torch.tensor(params_SOS_token_padded, dtype=torch.float32)

        y_input = y_input.unsqueeze(0).unsqueeze(0)
        y_input = y_input.to(device)

        for i in range(max_len):
            # Get source mask
            tgt_mask = get_tgt_mask(y_input.size(1)).to(device)
            feature = self.image_encoder(image)
            # Use torch.cuda.amp for mixed precision
            try:
                if stage == 'val':
                    with torch.no_grad():
                        pred = self.sequence_decoder(feature, y_input, tgt_mask)
                else:
                    pred = self.sequence_decoder(feature, y_input, tgt_mask)
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

    def get_image_embedding(self, image):
        if self.use_depth:
            image = self.add_depth_to_image(image)
        x = self.image_encoder(image)
        # Get the CLS token
        x = x[:, 0, :]
        return x
    
    def add_depth_to_image(self, image):
        # prepare image for the model
        inputs = self.depth_est_img_proc(images=image, return_tensors="pt").to(image.device)
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
        self.predicted_depth = depth
        # cat depth to image
        image = torch.cat((image, depth), dim=1)
        # Convert 4 channel to 3 channel
        image = self.ch4_to_ch3_conv(image)
        if 0:
            # Normalize to 0-1
            image = (image - image.min()) / (image.max() - image.min())

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
                    transforms.RandomRotation(10),
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
            noises = torch.randn_like(seqs) * noise_level
        seqs = seqs + noises

        return seqs
    
    
    def compute_loss(self, batch, mode):
        image, y, lengths = batch
        y_input = y[:, :-1]
        pred = self(image, y_input)

        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        label_onehot = F.one_hot(label, num_classes=self.seq_dim).float()
        y_expected_seq = torch.cat((label_onehot, y_expected[:, :, 1:]), dim=2)
        y_expected_seq = y_expected_seq.permute(0, 2, 1)
        values = y_expected[:, :, 1:].permute(0, 2, 1)

        label_loss = self.label_loss_fn(pred[:, :self.seq_dim], label)
        if 1:
            param_loss = self.param_loss_fn(pred[:, self.seq_dim:], values)
        else:
            param_loss = self.param_loss_fn_bylabel(label=label, values=values, pred=pred[:, self.seq_dim:])


        neg_images = self.make_negative_imgs(image)
        neq_seqs = self.make_negative_seqs(y_expected_seq, shuffle=True)
        ############ Embedding loss
        # Teach embedding models with ground truth image and seqs
        # GT image embedding should be close to GT seq embedding
        gt_image_embedding = self.get_image_embedding(image)     # Get the CLS token
        neg_image_embedding = self.get_image_embedding(neg_images)     # Get the CLS token
        gt_seq_embedding = self.SeqEmbeddingModel(y_expected_seq)
        gt_neg_seq_embedding = self.SeqEmbeddingModel(neq_seqs)
        ones = torch.ones(image.size(0), device=gt_seq_embedding.device)
        #embedding_loss_gt = self.cosine_embedding_loss_function(gt_seq_embedding, gt_image_embedding, ones)
        # embedding_loss_gt = self.mse_loss(gt_seq_embedding, gt_image_embedding)
        embedding_loss_gt = self.triplet_loss(gt_seq_embedding, gt_image_embedding, neg_image_embedding) \
                        + self.triplet_loss(gt_image_embedding, gt_seq_embedding, gt_neg_seq_embedding)

        # Helios in loop
        if 0:
            est_image = torch.zeros_like(image)
            # Generate positive and negative images
            with ThreadPoolExecutor() as executor:
                pos_results = list(executor.map(lambda idx: self.helios.generate_image(idx, y_expected, self.image_size, "P"), range(y_expected.size(0))))

            for (batch_idx, pos_img) in pos_results:
                pos_img = transforms.ToTensor()(pos_img)
                est_image[batch_idx] = pos_img

            # Get image embeddings, seq embeddings with no grad
            # It will focus more on predicting the correct plant structure, not updating the embedding models
            est_image_embedding = self.get_image_embedding(est_image)[:, 0, :]        # Get the CLS token
        else:
            est_image = None
        # Calculate embedding loss for estimated sequence embedding
        # embedding_loss_gen tries to make the seq embedding close to the image embedding
        # It will force the seq embedding to be close to the GT image embedding
        
        # Calculate embedding loss for estimated sequence embedding
        # no_grad() and detach() is used to prevent the model from updating the embedding models\
        #with torch.no_grad():
        # est_seq_embedding = self.get_seq_embedding(pred)

        no_update_embedding_on_gen = True
        if no_update_embedding_on_gen:
            for param in self.SeqEmbeddingModel.parameters():
                param.requires_grad = False  # 파라미터의 기울기 계산 비활성화

        est_seq_embedding = self.SeqEmbeddingModel(pred)
        #embedding_loss_gen = self.cosine_embedding_loss_function(est_seq_embedding, gt_image_embedding.detach(), ones)
        # embedding_loss_gen = self.mse_loss(est_seq_embedding, gt_image_embedding.detach())
        embedding_loss_gen = self.triplet_loss(est_seq_embedding, gt_image_embedding.detach(), neg_image_embedding.detach())
        if no_update_embedding_on_gen:
            for param in self.SeqEmbeddingModel.parameters():
                param.requires_grad = True  # 파라미터의 기울기 계산 활성화

        # Calculate average embedding loss
        embedding_loss = embedding_loss_gt + embedding_loss_gen

        # Calculate final loss
        #loss = (label_loss + self.alpha * param_loss) + embedding_loss
        loss = embedding_loss # Test only label loss

        self.log(f'{mode}/label_loss', label_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/param_loss', param_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/embedding_loss', embedding_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/embedding_loss_gen', embedding_loss_gen, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/loss', loss, batch_size=image.size(0), sync_dist=True)

        # Add images to tensorboard
        if (self.current_train_step == 0 and mode == "train") or (self.current_val_step == 0 and mode == "val"):
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_images(f'{mode}/input_images', image, self.current_epoch)
            tensorboard_logger.add_images(f'{mode}/neg_images', neg_images, self.current_epoch)
        return loss

    # def on_train_start(self):
    #     tensorboard_logger = self.logger.experiment
    #     prototype_array = torch.zeros(1,3, self.image_size, self.image_size).to(self.device)
    #     tensorboard_logger.add_graph(self, prototype_array)

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
        if 1:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # return optimizer
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

        self.img_aug = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ])
        
        self.train_transform = transforms.Compose([
                self.img_aug,
                transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
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
        growth_stages = ["003","010","016","023"] # ["003","010","016","023"]
        self.train_dataset = self.load_or_create_dataset(
            self.dataset_dir, "train_dataset", ["000", "001", "002"], growth_stages,
            self.train_transform, self.load_depth, self.process_leaf,
            self.preload, self.image_size
        )

        self.val_dataset = self.load_or_create_dataset(
            self.dataset_dir, "val_dataset", ["003"], growth_stages,
            self.test_transform, self.load_depth, self.process_leaf,
            self.preload, self.image_size
        )

        self.test_dataset = self.load_or_create_dataset(
            self.dataset_dir, "test_dataset", ["004"], growth_stages,
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
    
class SimpleRegressionTest(MainModule):

    def __init__(self, image_size, lr, dropout, use_depth, vit_finetune, d_model=768):
        super(MainModule, self).__init__()
        self.save_hyperparameters()  # 전달된 모든 인수를 저장

        self.image_size = image_size
        self.use_depth = use_depth
        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            # Define a 4 ch to 3 ch conversion layer
            self.ch4_to_ch3_conv = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1) 

        self.feature_extractor = ViT_FeatureExtractor(output_size=d_model, image_size=image_size)
        #self.feature_extractor = CNN_FeatureExtractor(output_size=dim_model, use_depth=True)
        
        # Froze self.feature_extractor
        if vit_finetune == False:
            self.feature_extractor.eval()

        #self.regression_model = RegressionModel_Transformer(dim_model=dim_model, image_size=image_size, dropout=dropout)
        self.regression_model = RegressionModel(dim_model=d_model, image_size=image_size, dropout=dropout)

        self.lr = lr

        self.predicted_depth = None
        self.activation = nn.ReLU() # Activation function
        if 0:
            self.seq_embedding_layer = nn.Linear(6, 64)
        else:
            self.seq_embedding_layer = nn.Linear(23, d_model)
            #self.seq_embedding_transformer = nn.Transformer(d_model=d_model)
            # Try smaller model
            # self.seq_embedding_transformer = nn.Transformer(d_model=d_model, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1)

            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
            self.seq_embedding_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=6)

            
        self.image_embedding_layer = nn.Linear(257*d_model, d_model)
 
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


        src_path = os.path.join(script_file_dir,"../src") # script_file_path is models/
        self.image_generator = plantstring2model(program_path=os.path.join(src_path, "PlantString2Model/build"),
                                                 program_name="PlantString2Model",
                                                 display=":11.0", height=1.0, 
                                                 background_path=os.path.join(src_path, "assets/black.png"))
        
        self.triplet_loss_start_epoch = 0  # Set the epoch to start triplet loss calculation

        # Loss functions
        #self.triplet_loss_function = nn.TripletMarginLoss(margin=1.0, p=2)
        self.triplet_loss_function = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
        self.cosine_embedding_loss_function = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

        self.prev_epoch = -1

        self.positional_encoding = PositionalEncoding(dim_model=d_model, max_len=2048, dropout_p=0.1)
        self.positional_encoding.eval()


    def generate_image(self, plant_vec, idx, suffix="", image_size=224):

        def save_plant_string(plant_vec, idx, suffix=""):
            plant_string = vec2string([plant_vec])
            plant_string_file_name = f"temp/output_{suffix}_{idx}/plant_string_{suffix}_{idx}.txt"
            # Create output folder
            os.makedirs(os.path.dirname(plant_string_file_name), exist_ok=True)
            with open(plant_string_file_name, "w") as f:
                f.write(plant_string)
            return plant_string_file_name
        plant_string_file_name = save_plant_string(plant_vec, idx, suffix)

        self.image_generator.run(in_plantstring_path=os.path.abspath(plant_string_file_name), 
                                 output_path=os.path.abspath(f"temp/output_{suffix}_{idx}"))
        
        generated_image_path = f"temp/output_{suffix}_{idx}/plant_string_{suffix}_{idx}_top.jpeg"
        img = cv2.imread(generated_image_path)
        leaf_area, plant_width, plant_height, leaf_img, _ = process_leaf_image(img, sqaure_crop=True, thr=0.2)
        img = cv2.normalize(leaf_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.resize(leaf_img, (image_size, image_size))

        return img
    
    def get_image_embedding(self, image):
        #with torch.no_grad():
        if self.use_depth:
            image = self.add_depth_to_image(image)
        x = self.feature_extractor(image)
        
        if 0:
            x = x.reshape(x.size(0), -1)
            x = self.image_embedding_layer(x)
        elif 0:
            # Get image embedding from ViT
            x = torch.mean(x, dim=1) # avg_patch_embedding
            # Or max pooling
            # x = x.max(dim=1).values
        else:
            # Get the CLS token
            x = x[:, 0, :]
        return x
            
    def get_seq_embedding(self, x):
        # This is a simple embedding layer
        # It will be replaced by a transformer model in the future
        # seq: (batch_size, seq_len)
    
        # Make sequence first
        x = self.seq_embedding_layer(x)
        x = self.activation(x)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.seq_embedding_transformer(x)
        
        # get the last token
        x = x[-1]
    
        return x
    
    def add_depth_to_image(self, image):
        # prepare image for the model
        inputs = self.depth_est_img_proc(images=image, return_tensors="pt").to(image.device)
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
        self.predicted_depth = depth
        # cat depth to image
        image = torch.cat((image, depth), dim=1)
        # Convert 4 channel to 3 channel
        image = self.ch4_to_ch3_conv(image)
        if 0:
            # Normalize to 0-1
            image = (image - image.min()) / (image.max() - image.min())

        return image

    def forward(self, image):
        if self.use_depth:
            image = self.add_depth_to_image(image)

        x = self.feature_extractor(image)
        x = self.regression_model(x)
        return x
    
        
    def compute_loss(self, batch, mode):
        image, y, lengths = batch
        y_input = y[:, :-1]
        pred = self(image)

        y_expected = y[:, 1:]
        ones = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:].permute(0, 2, 1)

        # Calculate Loss using only for the first element
        mse_loss = F.mse_loss(pred.squeeze().squeeze(), values[:, :6, 0])


        def generate_image_tensor(batch_idx, tokens, image, suffix):
            plant_vec = token2vec(tokens[batch_idx].squeeze().squeeze().tolist())
            
            # Calculate Loss using only for the first element
            plant_vec_predicted = copy.deepcopy(plant_vec)
            result = pred[batch_idx].squeeze().squeeze().tolist()
            plant_vec_predicted[0][2] = coordinates_to_angle(result[0], result[1], angle_max=180)
            plant_vec_predicted[0][3] = coordinates_to_angle(result[2], result[3])
            plant_vec_predicted[0][4] = coordinates_to_angle(result[4], result[5])

            # Generate image
            img = self.generate_image(plant_vec_predicted, idx=batch_idx, suffix=suffix, image_size=self.image_size)
            img_tensor = torch.tensor(img).to(image.device).permute(2, 0, 1)  # (C, H, W)
            return batch_idx, img_tensor
        
        embedding_loss =  0

        # Make the A_seq_embedding to be close to the ground truth
        gt_image_embedding = self.get_image_embedding(image)
        gt_seq_embedding = self.get_seq_embedding(y)
        if 1:
            ones = torch.ones(gt_seq_embedding.size(0), device=gt_seq_embedding.device)
            zeros = torch.zeros(gt_seq_embedding.size(0), device=gt_seq_embedding.device)
            embedding_loss += self.cosine_embedding_loss_function(gt_seq_embedding, gt_image_embedding, ones)
        else:
            noise_token = generate_noise_plant_tokens(y, noise_level=0.2)
            y_noise_added = y + noise_token
            gt_noise_added_seq_embedding = self.get_seq_embedding(y_noise_added)
            # Calculate triplet loss
            embedding_loss += self.triplet_loss_function(gt_image_embedding, gt_seq_embedding, gt_noise_added_seq_embedding)

        predicted_tokens = y.clone()
        predicted_tokens[:, 1, 1:7] = pred

        with torch.no_grad():
            est_seq_embedding = self.get_seq_embedding(predicted_tokens)

        if 0:
            # Get Seq Embeddings
            # Replace the first element of the sequence with the predicted value by conserving grad flow
            # Most important loss ?? est seq embedding should be close to gt image embedding..?
            embedding_loss += self.cosine_embedding_loss_function(est_seq_embedding, gt_image_embedding, ones)

        # Triplet Loss Calculation Part
        
        if self.current_epoch >= self.triplet_loss_start_epoch:
            est_image = torch.zeros_like(image)
            est_noise_added_image = torch.zeros_like(image)

            # Add noise to the plant tokens
            if 0:
                y_noise_added = add_noise_plant_tokens(predicted_tokens, noise_level=0.1)
            else:
                noise_token = generate_noise_plant_tokens(predicted_tokens, noise_level=0.1)
                y_noise_added = predicted_tokens + noise_token
            
            with torch.no_grad():
                est_noise_seq_embedding = self.get_seq_embedding(y_noise_added)

            # Generate positive and negative images
            with ThreadPoolExecutor() as executor:
                pos_results = list(executor.map(lambda idx: generate_image_tensor(idx, y_expected, image, "P"), range(y_expected.size(0))))
                # neg_results = list(executor.map(lambda idx: generate_image_tensor(idx, y_noise_added, image, "N"), range(y_expected.size(0))))

            # Assign generated images to tensors
            if 0:
                for (batch_idx, pos_img_tensor), (_, neg_img_tensor) in zip(pos_results, neg_results):
                    est_image[batch_idx] = pos_img_tensor
                    est_noise_added_image[batch_idx] = neg_img_tensor
            else:
                for (batch_idx, pos_img_tensor) in pos_results:
                    est_image[batch_idx] = pos_img_tensor

            # Get image embeddings
            with torch.no_grad():
                est_image_embedding = self.get_image_embedding(est_image)
                # est_noise_added_image_embedding = self.get_image_embedding(est_noise_added_image)

            if 1:
                if 0:
                    # Add noise added seq embedding loss
                    embedding_loss += self.cosine_embedding_loss_function(est_seq_embedding, est_image_embedding, ones)

                    # est rand image <-> est rand seq
                    embedding_loss += self.cosine_embedding_loss_function(est_noise_seq_embedding, est_noise_added_image_embedding, ones)

                    # Add noise added seq embedding loss
                    embedding_loss += self.cosine_embedding_loss_function(est_seq_embedding, est_noise_added_image_embedding, zeros)

                    # Gt image <-> est rand seq
                    embedding_loss += self.cosine_embedding_loss_function(gt_seq_embedding, est_noise_added_image_embedding, zeros)
                else:
                    # Make it simple, make the est seq embedding close to the helios generated image embedding
                    embedding_loss2 = self.cosine_embedding_loss_function(est_seq_embedding, est_image_embedding, ones)
                    embedding_loss = (embedding_loss + embedding_loss2) / 2
            else:
                embedding_loss += self.triplet_loss_function(est_image_embedding, est_seq_embedding, est_noise_seq_embedding)
                embedding_loss += self.triplet_loss_function(est_noise_added_image_embedding, est_noise_seq_embedding, est_seq_embedding)
        else:
            # embedding_loss += 1.0
            # embedding_loss += 1.0
            pass

        loss = mse_loss + embedding_loss
        # loss = embedding_loss # Debug the embedding loss only to check if it is working

        self.log(f'{mode}/mse_loss', mse_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/embedding_loss', embedding_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/loss', loss, batch_size=image.size(0), sync_dist=True)
        return loss

    def on_train_start(self):
        tensorboard_logger = self.logger.experiment
        prototype_array = torch.zeros(1,3, self.image_size, self.image_size).to(self.device)
        tensorboard_logger.add_graph(self, prototype_array)

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, 'val')

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
       

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        


class SimpleRegressionVAE(pl.LightningModule):

    def __init__(self, image_size, lr, dropout, use_depth, vit_finetune, d_model=768, latent_dim=128):
        super(SimpleRegressionVAE, self).__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.use_depth = use_depth
        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            # Define a 4 ch to 3 ch conversion layer
            self.ch4_to_ch3_conv = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)


        self.feature_extractor = ViT_FeatureExtractor(output_size=d_model, image_size=image_size)
        
        self.vae = VAE(latent_dim=latent_dim)
        
        if 1:
            self.vae.load_state_dict(torch.load(os.path.join(script_file_dir,"../models/checkpoints/vae_best_20241015.pth")))

        #self.regression_model = RegressionModel_Transformer(dim_model=d_model, image_size=image_size, dropout=dropout)
        self.regression_model = RegressionModel(dim_model=d_model, image_size=image_size, dropout=dropout)
        
        self.lr = lr

        self.seq_embedding_layer = nn.Sequential(
                            nn.Linear(23, d_model),
                            nn.ReLU(),
        )
        #self.seq_embedding_layer = MLP([23, d_model])
        if 1:
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
            self.seq_embedding_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=6)
        else:
            transformer_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
            self.seq_embedding_transformer = nn.TransformerDecoder(transformer_layer, num_layers=3)

        # self.seq_embedding2latent = nn.Sequential(
        #     nn.Linear(d_model, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, latent_dim),
        # )
        # self.seq_embedding2latent = MLP([d_model, 256, 512, 1024, 512, 256, latent_dim])
        self.seq_embedding2latent = MLP([d_model, latent_dim], batch_norm=False)

        self.positional_encoding = PositionalEncoding(dim_model=d_model, max_len=2048, dropout_p=0.1)
        self.positional_encoding.eval()

        src_path = os.path.join(script_file_dir,"../src") # script_file_path is models/
        self.image_generator = plantstring2model(program_path=os.path.join(src_path, "PlantString2Model/build"),
                                                 program_name="PlantString2Model",
                                                 display=":11.0", height=1.0, 
                                                 background_path=os.path.join(src_path, "assets/black.png"))
        
        self.prev_epoch = -1
        self.current_train_step = 0
        self.current_val_step = 0
        self.helios_loss_start_epoch = 100

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    
    # Forward hook for param estimation
    def forward(self, image):
        image_features = self.feature_extractor(image)
        out = self.regression_model(image_features)
        return out
    

    def get_image_embedding(self, image):
        #with torch.no_grad():
        if self.use_depth:
            image = self.add_depth_to_image(image)
        x = self.feature_extractor(image)
        
        if 1:
            x = x.reshape(x.size(0), -1)
            x = self.activation(x)
            x = self.image_embedding_layer(x)
        elif 0:
            # Get image embedding from ViT
            x = torch.mean(x, dim=1) # avg_patch_embedding
        else:
            # Get the CLS token
            # x = x[:, 0, :]
            x = x.max(dim=1).values
        return x
            
    def get_seq_embedding(self, x):
        # This is a simple embedding layer
        # It will be replaced by a transformer model in the future
        # seq: (batch_size, seq_len)
    
        # Make sequence first
        x = self.seq_embedding_layer(x)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.seq_embedding_transformer(x)
        
        # get the last token
        x = x[-1]
    
        return x


    def compute_loss(self, batch, mode):
        image, y, lengths = batch
        y_input = y[:, :-1] # Remove the EOS token
        y_target = y[:, 1:] # Remove the SOS token

        ##### 1. VAE Loss #####
        recon_batch, mu, logvar, z = self.vae(image)
        vae_loss = self.vae.loss_function(recon_batch, image, mu, logvar)

        embedding_loss = 0
        image_loss = 0
        ##### 2. Regression Loss #####
        #if self.current_epoch >= self.mse_loss_start_epoch or True:

        pred = self(image)
        pred = pred.squeeze()
        # Simulate the predicted value to the y_expected
        y_pred = y_target.clone()
        y_pred[:, 0, 1:7] = pred
        # Calculate Loss using only for the first element
        mse_loss = F.mse_loss(y_target[:, 0, 1:7], pred)

        ##### 3.Plant Architecture Embedding Loss
        if 1:
            y_target_embedding = self.get_seq_embedding(y_target)
            latent_est_from_y_target = self.seq_embedding2latent(y_target_embedding)
        else:
            #latent_est_from_y_target = self.get_seq_embedding(y_target)
            latent_est_from_y_target = self.seq_embedding2latent(y_target[:, 0, 1:7])
        # Detach the z from the graph to only calculate the embedding loss
        latent_inputImage = z.detach()
        # Calculate the embedding loss
        embedding_loss += F.mse_loss(latent_est_from_y_target, latent_inputImage)


        # ##### 4. Image Generation Loss
        # # Decode the latent vector using VAE decoder
        with torch.no_grad():
            recon_image_from_architecture = self.vae.decode(latent_est_from_y_target)
        # # Calculate the image generation loss
        image_loss += F.mse_loss(recon_image_from_architecture, image, reduction='sum') / image.size(0)

    

        ##### Helios Loss #####
        if self.current_epoch >= self.helios_loss_start_epoch:

            # Generate image
            with ThreadPoolExecutor() as executor:
                helios_results = list(executor.map(lambda idx: self.generate_image_tensor(pred, idx, y_target, image, "P"), range(y_target.size(0))))
            # Assign generated images to tensors
            helios_image = torch.zeros_like(image)
            for (batch_idx, pos_img_tensor) in helios_results:
                pos_img_tensor = transforms.ToTensor()(pos_img_tensor)
                helios_image[batch_idx] = pos_img_tensor

            # Train the VAE to generate the helios image
            recon_batch_helios, mu, logvar, z = self.vae(helios_image)
            vae_loss2 = self.vae.loss_function(recon_batch_helios, helios_image, mu, logvar)
            vae_loss = (vae_loss + vae_loss2) / 2

            # Get image embeddings using VAE encoder
            latent_helios = z.detach()

            # Get the seq embedding from the predicted value
            with torch.no_grad():
                if 1:
                    y_pred_embedding = self.get_seq_embedding(y_pred)
                    latent_est_from_y_pred = self.seq_embedding2latent(y_pred_embedding)
                else:
                    #latent_est_from_y_pred = self.get_seq_embedding(y_pred)
                    latent_est_from_y_pred = self.seq_embedding2latent(pred)

            # Calculate the embedding loss
            embedding_loss2 = F.mse_loss(latent_est_from_y_pred, latent_helios)
            embedding_loss = (embedding_loss + embedding_loss2) / 2
           
            # ##### 4. Image Generation Loss
            # # Decode the latent vector using VAE decoder
            with torch.no_grad():
                recon_image_from_est_architecture = self.vae.decode(latent_est_from_y_pred)
            # # Calculate the image generation loss
            image_loss2 = F.mse_loss(recon_image_from_est_architecture, helios_image, reduction='sum') / image.size(0)
            image_loss2 = (image_loss + image_loss2) / 2

        else:
            # Assign generated images to tensors
            helios_image = None
            recon_image_from_est_architecture = None

        ##### 5. Total Loss
        #loss = mse_loss
        #loss = vae_loss + mse_loss + embedding_loss + image_loss
        loss = 0.0001*vae_loss + mse_loss + 0.1*embedding_loss # + 0.0001*image_loss # I think image_loss is not necessary

        self.log(f'{mode}/image_loss', image_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/vae_loss', vae_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/mse_loss', mse_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/embedding_loss', embedding_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/loss', loss, batch_size=image.size(0), sync_dist=True)


        # Add images to tensorboard
        #if self.current_epoch % 10 == 0 and self.current_step == 0:
        if (self.current_train_step == 0 and mode == "train") or (self.current_val_step == 0 and mode == "val"):
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_images(f'{mode}/input_images', image, self.current_epoch)
            tensorboard_logger.add_images(f'{mode}/recon_batch', recon_batch, self.current_epoch)
            if recon_image_from_architecture is not None:
                tensorboard_logger.add_images(f'{mode}/recon_image_from_architecture', recon_image_from_architecture, self.current_epoch)
            if recon_image_from_est_architecture is not None:
                tensorboard_logger.add_images(f'{mode}/recon_image_from_est_architecture', recon_image_from_est_architecture, self.current_epoch)
            if helios_image is not None:
                tensorboard_logger.add_images(f'{mode}/helios_images', helios_image, self.current_epoch)
            
        return loss

    # def on_train_start(self):
    #     tensorboard_logger = self.logger.experiment
    #     prototype_array = torch.zeros(1,3, self.image_size, self.image_size).to(self.device)
    #     tensorboard_logger.add_graph(self, prototype_array)

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
        if 1:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                                   threshold=1e-3, patience=10, min_lr=1e-6)
            #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
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
    
    # def lr_lambda(self, epoch):
    #     if epoch >= self.embedding_loss_start_epoch:
    #         return 0.1  # 학습률을 10%로 줄임
    #     elif epoch >= self.mse_loss_start_epoch:
    #         return 0.5  # 학습률을 50%로 줄임
    #     else:
    #         return 1.0  # 기본 학습률