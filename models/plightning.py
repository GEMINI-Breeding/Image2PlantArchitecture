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

# Path Settings
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
sys.path.append(project_dir)

# 모듈 임포트
from models.model import TransformerDecoderModel, RegressionModel, ViT_FeatureExtractor, CNN_FeatureExtractor
from models.model import RegressionModel_Transformer, PositionalEncoding, VAE, MLP, SeqEmbeddingModel
from models.model import create_organ_mask, get_tgt_mask, create_pad_mask, text_global_pool
from models.model import PlantArchitectureTransformer
from src.plant_tokenizer import SOS_token, EOS_token, PAD_token, param_PAD_token, EOS_vec_padded, SOS_vec_padded
from src.plant_tokenizer import generate_noise_plant_tokens, N_PARAMS
from src.plant_dataset import PlantDataset
from src.plantstring2model import plantstring2model
from src.plant_tokenizer import token2vec, vec2token
from src.string_to_xml_to_vec import vec2string
from src.image_process import process_leaf_image
from src.plant_architecture_utils import coordinates_to_angle
import pickle
import copy


# from open_clip.transformer import text_global_pool

# Disable fastpath for TransformerEncoder and MultiHeadAttention
# torch.backends.mha.set_fastpath_enabled(False)

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

class GaussianWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, sigma=0.5):
        super(GaussianWeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma

    def forward(self, inputs, targets):
        probabilities = F.softmax(inputs, dim=1)
        batch_size = probabilities.size(0)
        device = inputs.device

        # Vectorized Gaussian profile creation
        gauss_range = torch.arange(self.num_classes, device=device).unsqueeze(0).float()
        gauss_range = gauss_range.expand(batch_size, -1)
        gauss_center = targets.unsqueeze(1).float()

        gaussian = torch.exp(-0.5 * ((gauss_range - gauss_center) / self.sigma) ** 2)
        gaussian /= gaussian.sum(dim=1, keepdim=True)

        loss = -torch.sum(gaussian * torch.log(probabilities + 1e-12), dim=1)
        return loss
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def make_negative_imgs(image):
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

def make_negative_seqs(seqs, shuffle=True, noise_level=0.2):
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


class MainModule(pl.LightningModule):
    def __init__(self, num_layers=6, num_heads=8, 
                 seq_dim=23, seq_embedding_dim=768//2, 
                 param_dim=22, param_embedding_dim=768//2, 
                 image_size=224, alpha=1.0, lr=1e-5, 
                 dropout=0.10, 
                 max_len=1024,
                 use_depth=False,
                 cat_emb=True,
                 decoder_only=True,
                 vit_model="facebook/dinov2-small",
                 **kwargs):
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
        self.num_warmup_steps = 1000
        self.num_training_steps = 10000

        # Handle additional keyword arguments
        self.extra_args = kwargs

        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            # Fix the weights 
            for param in self.depth_est_model.parameters():
                param.requires_grad = False
            self.depth_background = cv2.resize(cv2.imread(os.path.join(self.current_script_dir, "../src/assets/dirt.jpg")), (self.image_size, self.image_size))
            # Conver to RGB
            self.depth_background = cv2.cvtColor(self.depth_background, cv2.COLOR_BGR2RGB)

        self.image_encoder = ViT_FeatureExtractor(output_size=seq_embedding_dim+param_embedding_dim, 
                                                  use_depth=self.use_depth, image_size=image_size, vit_model=vit_model)

        # Froze self.feature_extractor
        # self.image_encoder.eval()
        
        self.sequence_decoder = TransformerDecoderModel(
        #self.sequence_decoder = MultiModalModel(
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
        self.gaussian_smooth_loss = GaussianWeightedCrossEntropyLoss(num_classes=self.sequence_decoder.quantizer.n_clusters)

    def forward(self, image, plant_info, tgt):
        if self.use_depth:
            image = self.add_depth_to_image(image)
        features = self.image_encoder(image, plant_info)
        seq, params = self.sequence_decoder(features, tgt)
        seq = seq.permute(1, 0, 2)
        params[0] = params[0].permute(1, 0, 2, 3) # Quantized Params
        params[1] = params[1].permute(1, 0, 2)    # Scaled Params
        return seq, params

    def generate(self, image, plant_info, stage='val'):
        device = image.device
        SOS_tensor = torch.tensor(SOS_vec_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y_input = SOS_tensor
        y_input = y_input.to(device)
        if self.use_depth:
            image = self.add_depth_to_image(image)

        feature = self.image_encoder(image, plant_info)
        quantizer = self.sequence_decoder.quantizer

        for i in range(self.max_len):
            # Add Masks
            tgt_mask = get_tgt_mask(y_input.size(1))
            tgt_padding_mask = create_pad_mask(y_input, PAD_token)

            try:
                if stage == 'val':
                    with torch.no_grad():
                       label_p, params  = self.sequence_decoder(feature, y_input)
                else:
                    label_p, params = self.sequence_decoder(feature, y_input)
            except Exception as e:
                print(e)
                print(f"Error in {i} iteration")
                break
            label = label_p.topk(1)[1].view(-1)[-1].item()  # num with highest probability

            params_recovered = torch.zeros([params[0].size(0),params[0].size(1),params[0].size(2)],device=self.device)
            for i in range(params[0].size(2)):
                recovered = quantizer.inverse_transform_i(params[0][:,:,i], i)
                params_recovered[:, :, i] = recovered[:, :, 0]
          
            # Stop if model predicts end of sentencplant_structure_vit_transformer_withpsudodepth_paramEste
            ## if label == EOS_token:
            if label == EOS_token or label == PAD_token:
                break

            # Make next tensor using label and params
            next_item = torch.cat((torch.tensor([[label]], dtype=torch.float32, device=device), params_recovered[-1]), dim=1).unsqueeze(0)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Vector cleaning
            if 1:
                # Convert y_input to vec to clean erratic params. It will remove SOS Token
                vec = token2vec(y_input.squeeze(0).tolist())
                # Convert back to token
                tokens = vec2token(vec)
                y_input = torch.tensor(tokens, dtype=torch.float).unsqueeze(0)

                # Cat SOS_tensor
                y_input = torch.cat((SOS_tensor, y_input), dim=1)
                y_input = y_input.to(device)

        return y_input.squeeze(0)                    
    
    def label_loss_fn(self, pred, label, ignore_index=None, label_smoothing=0.0):
        # Define the number of classes (0 to 26)
        if 0:
            loss = F.cross_entropy(pred, label, ignore_index=ignore_index, reduction='sum') / pred.size(0)   
        else:
            loss = F.cross_entropy(pred, label, ignore_index=ignore_index)
        return loss 
     
    def param_regression_loss(self, pred, target, target_scaled, ignore_index=PAD_token):
        # Create neg mask
        neg_mask = (target == ignore_index)
        # Create masks
        mask = ~neg_mask
        loss_mse = F.smooth_l1_loss(pred, target_scaled, reduction='none') # mse_loss or smooth_l1_loss
        masked_loss = loss_mse * mask
        return masked_loss.sum() / (mask).sum()

    def param_loss_fn_bylabel(self, label, values, pred, ignore_index=PAD_token):
        # label: (batch_size, seq_len)
        # pred: (batch_size, seq_len, param_dim)
        # Masked values are not included in the loss

        # Create masks
        neg_organ_masks = create_organ_mask().to(pred.device) # Negative masks

        # Ensure label_mod and masks have compatible dimensions
        label_mod = label % 6
        neg_mask = (values == ignore_index)  # First mask is for padding
        neg_mask = neg_mask.permute(0, 2, 1)  # (N, C, L)
        for i in range(6):
            neg_mask = neg_mask | ((label_mod == i).unsqueeze(1).expand_as(neg_mask) & neg_organ_masks[i].unsqueeze(0).unsqueeze(2).expand_as(neg_mask))
        neg_mask = neg_mask.permute(0, 2, 1)  # (N, C, L)
        # Compute loss
        loss_mse = F.smooth_l1_loss(pred, values, reduction='none') # mse_loss or smooth_l1_loss
        # Create masks by negating the neg_mask
        mask = ~neg_mask
        masked_loss = loss_mse * mask
        return masked_loss.sum() / (mask).sum()
        #return masked_loss.sum() / masked_loss.size(0)
    
    
    def param_cross_entropy(self, label, values, pred, ignore_index=PAD_token, label_smoothing=0.0):
        # label: (batch_size, seq_len)
        # pred: (batch_size, seq_len, param_dim)
        # Masked values are not included in the loss
        if 0:
            # Create masks
            neg_organ_masks = create_organ_mask().to(pred.device) # Negative masks

            # Ensure label_mod and masks have compatible dimensions
            label_mod = label % 6
            values = values.long()
            neg_mask = (label == ignore_index).expand(values.shape)  # First mask is for padding
            neg_mask = neg_mask.permute(0, 2, 1)  # (N, C, L)
            for i in range(6):
                neg_mask = neg_mask | ((label_mod == i).unsqueeze(1).expand_as(neg_mask) & neg_organ_masks[i].unsqueeze(0).unsqueeze(2).expand_as(neg_mask))
            neg_mask = neg_mask.permute(0, 2, 1)  # (N, C, L)
            # Compute loss
            pred = pred.reshape(-1, pred.size(-1))  # [8*100*18, 63]
            values = values.reshape(-1)  # [8*100*18]
            loss = F.cross_entropy(pred, values, reduction='none')
            mask = ~neg_mask
            masked_loss = loss * mask
            return masked_loss.sum() / (mask).sum()
        else:
            values = values.long()
            pred = pred.reshape(-1, pred.size(-1))  # [8*100*18, 63]
            if 1:
                loss = F.cross_entropy(pred, values.reshape(-1), reduction='none', ignore_index=0)
            else:
                loss = self.gaussian_smooth_loss(pred, values.reshape(-1))
            loss = loss.reshape(values.shape)
            mask = (label == PAD_token) | (label == SOS_token) | (label == EOS_token)
            mask = ~mask
            # Expand mask to match the shape of loss
            mask = mask.unsqueeze(-1).expand_as(loss)  # [8, 100, 1] -> [8, 100, 18]
            masked_loss = loss * mask
            return masked_loss.sum() / (mask).sum()
        
            #return masked_loss.sum() / masked_loss.size(0) / N_PARAMS
        

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

        with torch.no_grad():
            inputs = self.depth_est_img_proc(images=depth_input, return_tensors="pt").to(image.device)
            outputs = self.depth_est_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        
        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        self.predicted_depth = depth
        # Rescale to 0-255
        depth = depth*255
        # cat depth to image
        image = torch.cat((image, depth), dim=1)

    
        return image
    
    def collect_params(self, label, values, ignore_index=PAD_token):

        neg_organ_masks = create_organ_mask().to(values.device) # Negative masks

        # Ensure label_mod and masks have compatible dimensions
        neg_mask = (label == ignore_index)  # First mask is for padding
        if 1:
            neg_mask = neg_mask | (label == PAD_token)
            neg_mask = neg_mask | (label == SOS_token)
            neg_mask = neg_mask | (label == EOS_token)

        if len(values.shape) == 4:
            neg_mask = neg_mask.unsqueeze(-1).unsqueeze(-1).expand_as(values)
            for i in range(6):
                neg_mask = neg_mask | ((label % 6 == i).unsqueeze(-1).unsqueeze(-1).expand_as(neg_mask) 
                                       & neg_organ_masks[i].unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(neg_mask))
            total_values = dict()
            for i in range(neg_mask.size(2)): # Param dimension
                var_mask = F.one_hot(torch.tensor(i), num_classes=neg_mask.size(2)).unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand_as(neg_mask)
                var_mask = var_mask.to(self.device)
                var_mask = var_mask.bool()
                masked_values = values[(~neg_mask) * var_mask].reshape(-1, values.size(-1))
                total_values[f"{i}"] = masked_values
        else:
            neg_mask = neg_mask.unsqueeze(-1).expand_as(values)
            for i in range(6):
                neg_mask = neg_mask | ((label % 6 == i).unsqueeze(-1).expand_as(neg_mask) 
                                       & neg_organ_masks[i].unsqueeze(0).unsqueeze(0).expand_as(neg_mask))
            total_values = dict()
            for i in range(neg_mask.size(2)): # Param dimension
                var_mask = F.one_hot(torch.tensor(i), num_classes=neg_mask.size(2)).unsqueeze(0).unsqueeze(1).expand_as(neg_mask)
                var_mask = var_mask.to(self.device)
                var_mask = var_mask.bool()
                masked_values = values[(~neg_mask) * var_mask]
                total_values[f"{i}"] = masked_values
        return total_values

    def compute_loss(self, batch, mode):
        quantizer = self.sequence_decoder.quantizer
        scaler = self.sequence_decoder.scaler

        # Load batch and preprocess
        image, plant_info, y, lengths = batch
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:]
        values_quantized = quantizer.transform(values)
        values_scaled = scaler.transform(values)

        # Decoder loss
        seq, params = self(image, plant_info, y_input)
        label_loss = self.label_loss_fn(seq.permute(0, 2, 1), label, ignore_index=PAD_token) # (N, C, L)
        
        params_collected_pred_p = self.collect_params(label=label, values=params[0])
        params_collected_target_class = self.collect_params(label=label, values=values_quantized)
        sum_quantized_param_loss = 0
        for i in range(len(params_collected_pred_p)):
            sum_quantized_param_loss += F.cross_entropy(params_collected_pred_p[f"{i}"],
                                                    params_collected_target_class[f"{i}"].long())
        
        params_collected_target_value = self.collect_params(label=label, values=values_scaled)
        # Recover values using predefined centers
        scaled_param_loss = []
        sum_scaled_param_loss = 0
        for i in range(len(params_collected_pred_p)):
            params_collected_recovered = quantizer.inverse_transform_i(params_collected_pred_p[f"{i}"].unsqueeze(1), i)
            params_collected_recovered_scaled = scaler.transform(params_collected_recovered.expand(
                                                [params_collected_recovered.size(0), 1, values_scaled.size(-1)]))
            params_collected_recovered_scaled = params_collected_recovered_scaled[:,:,i].squeeze()
            # scaled_param_loss.append(F.mse_loss(params_collected_recovered_scaled, 
            #                                params_collected_target_value[f"{i}"]))
            
            sum_scaled_param_loss += F.mse_loss(params_collected_recovered_scaled, 
                                           params_collected_target_value[f"{i}"]) 

        
        ######### Tensorboard logging
        loss = label_loss + sum_quantized_param_loss + sum_scaled_param_loss

        self.log(f'{mode}/label_loss', label_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/quantized_param_loss', sum_quantized_param_loss, batch_size=image.size(0), sync_dist=True)
        self.log(f'{mode}/scaled_param_loss', sum_scaled_param_loss, batch_size=image.size(0), sync_dist=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.num_warmup_steps, 
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
    
class MainDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, train_batch_size=16, val_batch_size=None,
                        num_workers=4, image_size=448, 
                        load_depth=True,
                        side_view=False,
                        process_leaf=False,
                        preload=False,
                        growth_stages=None,
                        partial_data=1.0,
                        **kwargs):
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.preload = preload
        self.process_leaf = process_leaf
        self.load_depth = load_depth
        self.pin_memory = False
        self.side_view = side_view
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

        self.growth_stages = growth_stages
        self.partial_data = partial_data

        # Handle additional keyword arguments
        self.extra_args = kwargs

    def load_or_create_dataset(self, dataset_dir, dataset_name, plot, stages, transform, load_depth, process_leaf, side_view, preload, image_size):
        saved_dataset_name = os.path.join(dataset_dir, f"{dataset_name}.pkl")
        if os.path.exists(saved_dataset_name) and preload:
            print(f"Loading {dataset_name} dataset from .pkl file")
            with open(saved_dataset_name, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = PlantDataset(
                dataset_dir, plot=plot, stages=stages,
                transform=transform, load_depth=load_depth,
                process_leaf=process_leaf, side_view=side_view,
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
        train_ratio = 0.7 * self.partial_data
        val_ratio = 0.15 * self.partial_data
        test_ratio = 0.15 * self.partial_data

        growth_stages = self.growth_stages

        # Get the num plots from the last xml file
        xml_files = os.listdir(os.path.join(self.dataset_dir, "xml"))
        xml_files.sort()
        self.num_plots = int(xml_files[-1].split("_")[1]) + 1

        train_end = int(self.num_plots * train_ratio)
        val_end = train_end + int(self.num_plots * val_ratio)
        test_end = min(self.num_plots, val_end + int(self.num_plots * test_ratio)) # Ensure total sums up to num_plots

        train_plots = [f"{plot:04d}" for plot in range(train_end)]
        val_plots = [f"{plot:04d}" for plot in range(train_end, val_end)]
        test_plots = [f"{plot:04d}" for plot in range(val_end, test_end)]

        self.train_dataset = self.load_or_create_dataset(
            self.dataset_dir, "train_dataset", train_plots, growth_stages,
            self.train_transform, self.load_depth, self.process_leaf, self.side_view,
            self.preload, self.image_size
        )
        self.val_dataset = self.load_or_create_dataset(
            self.dataset_dir, "val_dataset", val_plots, growth_stages,
            self.test_transform, self.load_depth, self.process_leaf, self.side_view,
            self.preload, self.image_size
        )
        self.test_dataset = self.load_or_create_dataset(
            self.dataset_dir, "test_dataset", test_plots, growth_stages,
            self.test_transform, self.load_depth, self.process_leaf, self.side_view,
            self.preload, self.image_size
        )

        
    def collate_fn(self, batch):
        images, plant_info, vectors, lengths = zip(*batch)
        max_length = max(lengths)
        vec_dim = vectors[0].shape[-1]
        if len(vectors[0].shape) == 1:
            vectors_padded = np.ones((len(vectors), max_length), dtype=int) * PAD_token
        else:
            vectors_padded = np.zeros((len(vectors), max_length, vec_dim))
            vectors_padded[:, :, 0] = PAD_token

        for i, vector in enumerate(vectors):
            end = lengths[i]
            vectors_padded[i, :end] = vector

        images = torch.stack(images)
        vectors_padded = torch.tensor(vectors_padded, dtype=torch.float32)
        plant_info = np.array(plant_info)
        plant_info = torch.tensor(plant_info, dtype=torch.float32)
        return images, plant_info, vectors_padded, lengths

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

    def test_dataloader(self, shuffle=True):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, shuffle=shuffle,
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory
        )
    

import unittest
import torch
from models.plightning import MainModule
from src.plant_tokenizer import SOS_vec_padded

class TestMainModule(unittest.TestCase):
    def setUp(self):
        self.model = MainModule()
        self.model.eval()  # 모델을 평가 모드로 설정
        self.image = torch.randn(1, 3, 224, 224)  # 임의의 이미지 텐서
        self.plant_info = torch.randn(1, 10)  # 임의의 식물 정보 텐서

    def test_generate(self):
        with torch.no_grad():
            result = self.model.generate(self.image, self.plant_info)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.dim(), 2)  # 결과 텐서는 2차원이어야 함
            self.assertEqual(result.size(0), 1)  # 배치 크기는 1이어야 함

    def test_generate_with_beam_search(self):
        with torch.no_grad():
            result = self.model.generate(self.image, self.plant_info, beam_size=5)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.dim(), 2)  # 결과 텐서는 2차원이어야 함
            self.assertEqual(result.size(0), 1)  # 배치 크기는 1이어야 함

    def test_generate_with_no_repeat_ngram(self):
        with torch.no_grad():
            result = self.model.generate(self.image, self.plant_info, no_repeat_ngram_size=3)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.dim(), 2)  # 결과 텐서는 2차원이어야 함
            self.assertEqual(result.size(0), 1)  # 배치 크기는 1이어야 함

if __name__ == '__main__':

    import os
    import sys
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    import cv2
    import matplotlib.pyplot as plt
    import shutil
    import subprocess
    from PIL import Image
    from torchvision import transforms
    from tqdm.notebook import tqdm

    from models.plightning import MainModule, MainDataModule
    from models.model import get_tgt_mask
    from src.plant_tokenizer import SOS_vec_padded, SOS_token, EOS_token, token2vec
    from src.string_to_xml_to_vec import vec2xml, pretty_print_xml, recursive_to_linked
    from src.plant_dataset import load_sideview_images
    from src.image_process import process_leaf_image

    def re_render_xml(output_path, filename, program_path, rotation=True):
        image_name = filename.split("/")[-1].split(".")[0]
        os.environ["DISPLAY"] = ":12.0"
        command = f"cd {program_path} && ./main -h 1.0 -o {output_path} -name {image_name} -tile none -f {os.path.join(output_path, filename)}"
        if rotation:
            command += " -r"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        return result

    def process_and_display_images(model, dataloader, n_figures, temp_folder, program_path):
        fig, axes = plt.subplots(3, n_figures, figsize=(20, 8))
        image_size = model.image_size
        device = model.device

        for idx, (image, plant_info, out, lengths) in enumerate(dataloader):
            if idx >= n_figures:
                break

            if image.dim() == 3:
                image = image.unsqueeze(0)

            image = image.to(device)
            plant_info = plant_info.to(device)
            out = torch.tensor(out).to(device)
            ground_truth = out.squeeze(0).cpu().numpy()

            plant_vec = token2vec(ground_truth)
            plant_xml = vec2xml(plant_vec, debug=True)
            plant_xml_file_name = f"{temp_folder}/plant_{idx}_gt.xml"
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)
            plant_xml = recursive_to_linked(plant_xml)
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)

            with torch.no_grad():
                result = model.generate(image, plant_info)
                result = result.cpu().numpy()

            plant_vec = token2vec(result)
            plant_xml = vec2xml(plant_vec, debug=True)
            plant_xml_file_name = f"{temp_folder}/plant_{idx}.xml"
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)
            plant_xml = recursive_to_linked(plant_xml)
            plant_xml_str = pretty_print_xml(plant_xml)
            with open(plant_xml_file_name, "w") as f:
                f.write(plant_xml_str)

            re_render_xml(os.path.abspath(temp_folder), os.path.abspath(plant_xml_file_name), program_path)
            img, _ = load_sideview_images(temp_folder, plant_xml_file_name.replace("xml", "jpeg"), model.image_size, True)

            image_vis = image[0].permute(1, 2, 0).cpu()
            image_vis = cv2.normalize(np.array(image_vis), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            row, col = divmod(idx, n_figures)
            axes[row, col].imshow(image_vis[:, :, 0:3])
            axes[row, col].set_title(f"Input Image {idx + 1}")
            axes[row, col].axis('off')

            depth = model.predicted_depth.squeeze().cpu()
            axes[row+1, col].imshow(depth)
            axes[row+1, col].set_title(f"Estimated Depth Image {idx + 1}")
            axes[row+1, col].axis('off')

            axes[row + 2, col].imshow(img)
            axes[row + 2, col].set_title(f"Output Model {idx + 1}")
            axes[row + 2, col].axis('off')

        plt.tight_layout()
        plt.show()

    def main():
        # Add ../ as a directory to import from
        sys.path.append('../')

        # Load model
        model = MainModule.load_from_checkpoint("log/20250114_SideView_224_QuantizedParams/version_0/checkpoints/last.ckpt")
        model.eval()

        # Setup data module
        dataset_dir = "data/Sideview_Dec23_2024"
        datamodule = MainDataModule(dataset_dir,
                                    image_size=model.image_size,
                                    load_depth=False,
                                    train_batch_size=1, num_workers=0, process_leaf=True, preload=False, side_view=True)
        growth_stages = [f"{day:02d}" for day in range(0, 2)]
        datamodule.setup(growth_stages=growth_stages)
        datamodule.setup()
        dataloader = datamodule.test_dataloader()

        # Create temp folder
        temp_folder = "temp"
        shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder, exist_ok=True)

        # Process and display images
        process_and_display_images(model, dataloader, n_figures=5, temp_folder=temp_folder, program_path="src/GenerateDataset/build")

    main()


    #####
    unittest.main()