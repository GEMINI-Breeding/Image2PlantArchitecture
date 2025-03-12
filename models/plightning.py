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
from src.plant_tokenizer import token2vec, vec2token
from src.string_to_xml_to_vec import vec2string, vec2xml
from src.image_process import process_leaf_image
from plant_architecture_utils import coordinates_to_angle
import pickle
import copy

from models.model import PlantArchitectureTransformer

# from open_clip.transformer import text_global_pool

# Disable fastpath for TransformerEncoder and MultiHeadAttention
# torch.backends.mha.set_fastpath_enabled(False)

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def contains_consecutive_sequence(array, sequence):
    seq_len = len(sequence)
    for i in range(len(array) - seq_len + 1):
        if array[i:i + seq_len] == sequence:
            return True
    return False


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
                 cat_emb=True):
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

        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.PAD_token = PAD_token

        if self.use_depth:
            self.depth_est_img_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_est_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            # Fix the weights 
            for param in self.depth_est_model.parameters():
                param.requires_grad = False
            self.depth_background = cv2.resize(cv2.imread(os.path.join(self.current_script_dir, "../src/assets/dirt.jpg")), (self.image_size, self.image_size))
            # Conver to RGB
            self.depth_background = cv2.cvtColor(self.depth_background, cv2.COLOR_BGR2RGB)

        self.image_encoder = ViT_FeatureExtractor(output_size=seq_embedding_dim+param_embedding_dim, use_depth=self.use_depth, image_size=image_size)

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
            dropout_p=self.dropout,
            max_seq_length=max_len,
            cat_emb=cat_emb
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

    def forward(self, image, plant_info, tgt):
        if self.use_depth:
            image = self.add_depth_to_image(image)
        features = self.image_encoder(image, plant_info)
        outputs = self.sequence_decoder(features, tgt)
        outputs = outputs.permute(1, 0, 2)
        return outputs
    
    def generate(self, image, plant_info, stage='val'):
        device = image.device
        SOS_tensor = torch.tensor(SOS_vec_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y_input = SOS_tensor
        y_input = y_input.to(device)
        if self.use_depth:
            image = self.add_depth_to_image(image)

        feature = self.image_encoder(image, plant_info)
        total_score = 0.0
        for i in range(self.max_len):
            # Add Masks
            tgt_mask = get_tgt_mask(y_input.size(1))
            tgt_padding_mask = create_pad_mask(y_input, PAD_token)

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
            
            # Unscale params
            params = self.sequence_decoder.scaler.inverse_transform(params)

            # Stop if model predicts end of sentence
            if label == EOS_token or label == PAD_token:
                break

            # Make next tensor using label and params
            next_item = torch.cat((torch.tensor([[label]], dtype=torch.float32, device=device), params[-1]), dim=1).unsqueeze(0)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Vector cleaning
            if 1:
                # Convert y_input to vec to clean erratic params. It will remove SOS Token
                vec = token2vec(y_input.squeeze(0).tolist())
                # Convert back to token
                y_input = torch.tensor(vec2token(vec),dtype=torch.float).unsqueeze(0)
                y_input = y_input.to(device)

            # Update total score
            total_score += F.log_softmax(label_p[-1, :, :], dim=-1)[0, label].item()

        return y_input.squeeze(0), total_score

    def generate_beam(self, image, plant_info, beam_width=3, max_len=1024, 
                      stage='val', ngram_size=0, add_noise=False, check_grammar=True):
        device = image.device
        SOS_tensor = torch.tensor(SOS_vec_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        EOS_token = self.EOS_token
        PAD_token = self.PAD_token

        # Depth information addition if required
        if self.use_depth:
            image = self.add_depth_to_image(image)
        
        # Encode the image to get features
        feature = self.image_encoder(image, plant_info)
        
        # Initialize the beam with the start of sequence token
        sequences = [(SOS_tensor, 0.0, [])]  # List of tuples (sequence, score, ngram_list)

        for _ in range(max_len):
            all_candidates = []
            
            for seq, score, label_list in sequences:

                if len(label_list) > 0:
                    if label_list[-1] == EOS_token:
                        # If seqence reached EOS token, just add to candidate
                        all_candidates.append((seq, score, label_list))
                        continue

                tgt_mask = get_tgt_mask(seq.size(1))
                tgt_padding_mask = create_pad_mask(seq, PAD_token)

                # Perform decoding without tracking gradients
                with torch.no_grad():
                    pred = self.sequence_decoder(feature, seq)

                label_p = pred[:, :, :self.seq_dim]
                params = pred[:, :, self.seq_dim:]

                # Get top k candidates
                topk_probs, topk_indices = F.log_softmax(label_p[-1, :, :], dim=-1).topk(beam_width)

                for i in range(beam_width):
                    next_label = topk_indices[:, i].unsqueeze(0).unsqueeze(0).float()

                    if add_noise:
                        # Add noise
                        params = self.add_noise(params=params, min=-0.1, max=0.1)
                    params = self.sequence_decoder.scaler.inverse_transform(params)
                    
                    next_params = params[-1, :, :].unsqueeze(0)
                    
                    # Expand next_label to match the dimensions of next_params
                    next_label = next_label.expand(next_params.size(0), next_params.size(1), next_label.size(-1))
                    
                    # Add noise
                    # Sanitize parameters based on the selected label
                    next_params = self.sanitize_params(next_label, next_params)
                    
                    # Create a new candidate sequence
                    next_item = torch.cat((next_label, next_params), dim=-1)
                    candidate = torch.cat((seq, next_item), dim=1)
                    candidate_score = score + topk_probs[0, i].item()

                    # Update ngram_list with the new addition
                    new_label_list = label_list + [next_label.item()]
                    if ngram_size > 0:
                        # Remove repeated n-grams
                        if self.is_repeated_ngram(new_label_list, ngram_size):
                            continue

                    if check_grammar:
                        # Check if candidate can be converted to xml
                        # Check grammar when the last element is leaf
                        organ_type = new_label_list[-1] % 6
                        if organ_type in [3,4,5]:
                            try:
                                vec = token2vec(candidate.squeeze().tolist())
                                vec2xml(vec)
                                # If it succeed, increase the probablity
                                #candidate_score += np.log(2)
                                candidate_score += 0.1
                                candidate_score = min(candidate_score, 0) # Clamp the max p as 1.0
                            except:
                                # If it failes, decrease the probablity
                                #candidate_score += np.log(1/2)
                                candidate_score -= 0.1


                        # Check if any concecutive leaves more than 4
                        leaf_list = [(x % 6 in [4,5,6]) for x in new_label_list]
                        if contains_consecutive_sequence(leaf_list,[True, True, True, True]):
                            # Do not append to the candidate
                            continue

                    all_candidates.append((candidate, candidate_score, new_label_list))


            # Sort all candidates by score and keep the top beam_width
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
            
            # Check if all sequences contain EOS or PAD tokens
            if all(any(token in [EOS_token, PAD_token] for token in seq[0][0, :, 0].tolist()) for seq in sequences):
                break
        
        # Return the best sequence found
        for best_sequence, best_score, _ in sequences:
            try:
                # Check grammar before return it
                vec = token2vec(best_sequence.squeeze().tolist())
                vec2xml(vec)
                # If success, break the loop
                break
            except:
                continue
            
        return best_sequence.squeeze(0), best_score

    def generate_param_beam(self, image, plant_info, beam_width=3, max_len=1024, 
                      stage='val', ngram_size=0, add_noise=False, check_grammar=True):
        device = image.device
        SOS_tensor = torch.tensor(SOS_vec_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        EOS_token = self.EOS_token
        PAD_token = self.PAD_token

        # Depth information addition if required
        if self.use_depth:
            image = self.add_depth_to_image(image)
        
        # Encode the image to get features
        feature = self.image_encoder(image, plant_info)
        
        # Initialize the beam with the start of sequence token
        sequences = [(SOS_tensor, 0.0, [])]  # List of tuples (sequence, score, ngram_list)

        for _ in range(max_len):
            all_candidates = []
            
            for seq, score, label_list in sequences:

                if len(label_list) > 0:
                    if label_list[-1] == EOS_token:
                        # If seqence reached EOS token, just add to candidate
                        all_candidates.append((seq, score, label_list))
                        continue

                tgt_mask = get_tgt_mask(seq.size(1))
                tgt_padding_mask = create_pad_mask(seq, PAD_token)

                # Perform decoding without tracking gradients
                with torch.no_grad():
                    pred = self.sequence_decoder(feature, seq)

                label_p = pred[:, :, :self.seq_dim]
                params = pred[:, :, self.seq_dim:]

                # Get top k candidates
                topk_probs, topk_indices = F.log_softmax(label_p[-1, :, :], dim=-1).topk(beam_width)

                for i in range(beam_width):
                    next_label = topk_indices[:, 0].unsqueeze(0).unsqueeze(0).float()
 
                    params = self.add_noise(params=params, min=-0.1, max=0.1)
                    params = torch.clamp(params, -1, 1)
                    params = self.sequence_decoder.scaler.inverse_transform(params)
                    
                    next_params = params[-1, :, :].unsqueeze(0)
                    
                    # Expand next_label to match the dimensions of next_params
                    next_label = next_label.expand(next_params.size(0), next_params.size(1), next_label.size(-1))
                    
                    # Sanitize parameters based on the selected label
                    next_params = self.sanitize_params(next_label, next_params)
                    
                    # Create a new candidate sequence
                    next_item = torch.cat((next_label, next_params), dim=-1)
                    candidate = torch.cat((seq, next_item), dim=1)
                    candidate_score = score + topk_probs[0, i].item()

                    # Update ngram_list with the new addition
                    new_label_list = label_list + [next_label.item()]
                    if ngram_size > 0:
                        # Remove repeated n-grams
                        if self.is_repeated_ngram(new_label_list, ngram_size):
                            continue

                    if check_grammar:
                        # Check if candidate can be converted to xml
                        # Check grammar when the last element is leaf
                        organ_type = new_label_list[-1] % 6
                        if organ_type in [3,4,5]:
                            try:
                                vec = token2vec(candidate.squeeze().tolist())
                                vec2xml(vec)
                                # If it succeed, increase the probablity
                                #candidate_score += np.log(2)
                                candidate_score += 0.1
                                candidate_score = min(candidate_score, 0) # Clamp the max p as 1.0
                            except:
                                # If it failes, decrease the probablity
                                #candidate_score += np.log(1/2)
                                candidate_score -= 0.1


                        # Check if any concecutive leaves more than 4
                        leaf_list = [(x % 6 in [4,5,6]) for x in new_label_list]
                        if contains_consecutive_sequence(leaf_list,[True, True, True, True]):
                            # Do not append to the candidate
                            continue

                    all_candidates.append((candidate, candidate_score, new_label_list))


            # Sort all candidates by score and keep the top beam_width
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
            
            # Check if all sequences contain EOS or PAD tokens
            if all(any(token in [EOS_token, PAD_token] for token in seq[0][0, :, 0].tolist()) for seq in sequences):
                break
        
        # Return the best sequence found
        for best_sequence, best_score, _ in sequences:
            try:
                # Check grammar before return it
                vec = token2vec(best_sequence.squeeze().tolist())
                vec2xml(vec)
                # If success, break the loop
                break
            except:
                continue
            
        return best_sequence.squeeze(0), best_score

    def is_repeated_ngram(self, ngram_list, ngram_size):
        """Check for repeated n-grams in the list."""
        if len(ngram_list) < ngram_size:
            return False
        
        ngrams = [tuple(ngram_list[j:j + ngram_size]) for j in range(len(ngram_list) - ngram_size + 1)]
        return len(ngrams) != len(set(ngrams))  # If duplicates exist, the lengths will differ
    
    def add_noise(self, params, min=-1, max=1):
        """ Add noise vector"""
        noise_vector = min + torch.rand_like(params) * (max - min)

        return params + noise_vector

    def sanitize_params(self, labels, params):
        """
        Sanitize parameters based on the labels.

        Args:
            labels (torch.Tensor): Tensor containing the labels.
            params (torch.Tensor): Tensor containing the parameters.

        Returns:
            torch.Tensor: Sanitized parameters.
        """
        if labels.shape[0] != params.shape[0]:
            raise ValueError("Labels and parameters must have the same sequence length.")

        new_params = torch.zeros_like(params)
        organ_ranges = {
            0: slice(0, 5),
            1: slice(5, 9),
            2: slice(9, 14),
            3: slice(14, 18),
            4: slice(14, 18),
            5: slice(14, 18)
        }

        for i, label in enumerate(labels):
            organ = label % 6
            param_range = organ_ranges.get(organ.squeeze().tolist())
            if param_range:
                new_params[i, :, param_range] = params[i, :, param_range]

            # Apply contraints
            if organ == 0:
                new_params[i, :, 4] = 1.0 if abs(1.0 - new_params[i, :, 4]) < abs(3.0 - new_params[i, :, 4]) else 3.0 # shoot_type
            elif organ == 1:
                new_params[i, :, 5] = max(new_params[i, :, 5], 0.0002) # internode_length
                new_params[i, :, 6] = max(new_params[i, :, 6], 0.0005) # internode_radius
            elif organ == 2:
                new_params[i, :, 9] = max(new_params[i, :, 9], 1e-7)    # petiole_length
                new_params[i, :, 10] = max(new_params[i, :, 10], 4e-06) # petiole radius, random.uniform(0.00075, 0.00125)
            elif organ in [3, 4, 5]:
                new_params[i, :, 14] = max(new_params[i, :, 14], 0.0002) # leaf_scale

        return new_params

    def label_loss_fn(self, pred, label, ignore_index=None):
        # Define the number of classes (0 to 26)
        num_classes = EOS_token+1  # Adjust if there are more tokens

        # # Initialize weights to 1 for all classes
        # weights = torch.ones(num_classes, device=pred.device)
        # # Assign a higher weight (e.g., 2.0) to tokens 12 through 23
        # weights[12:24] = 2.0
        # return F.cross_entropy(pred, label, weight=weights)

        #return F.cross_entropy(pred, label, ignore_index=ignore_index, weight=weights)
        ce_loss = F.cross_entropy(pred, label, ignore_index=ignore_index)
        return ce_loss

    def param_loss_fn(self, pred, params, ignore_index=PAD_token):
        # Create neg mask
        neg_mask = (params == ignore_index)
        # Create masks
        mask = ~neg_mask
        loss_mse = F.mse_loss(pred, params, reduction='none') # mse_loss or smooth_l1_loss
        masked_loss = loss_mse * mask
        return masked_loss.sum() / masked_loss.size(0)

    def param_loss_fn_bylabel(self, label, values, pred, ignore_index=PAD_token):
        # label: (batch_size, seq_len)
        # pred: (batch_size, seq_len, param_dim)
        # Masked values are not included in the loss

        # Create masks
        neg_organ_masks = create_organ_mask().to(pred.device) # Negative masks

        # Ensure label_mod and masks have compatible dimensions
        neg_mask = (label == ignore_index).unsqueeze(2).expand_as(values)  # First mask is for padding
        if 1:
            neg_mask = neg_mask | (label == PAD_token).unsqueeze(2).expand_as(values)  
            neg_mask = neg_mask | (label == SOS_token).unsqueeze(2).expand_as(values)  
            neg_mask = neg_mask | (label == EOS_token).unsqueeze(2).expand_as(values)  

        # neg_mask = neg_mask.permute(0, 2, 1)  # (N, C, L)
        for i in range(6):
            neg_mask = neg_mask | ((label % 6 == i).unsqueeze(2).expand_as(neg_mask) & neg_organ_masks[i].unsqueeze(0).unsqueeze(1).expand_as(neg_mask))
        # neg_mask = neg_mask.permute(0, 2, 1)  # (N, C, L)
        # Compute loss
        loss_mse = F.mse_loss(pred, values, reduction='none') # mse_loss or smooth_l1_loss
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
    

    
    def compute_loss(self, batch, mode):

        # Load batch and preprocess
        image, plant_info, y, lengths = batch
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        label = y_expected[:, :, 0].long()
        values = y_expected[:, :, 1:]

        # Decoder loss
        pred = self(image, plant_info, y_input)
        label_loss = self.label_loss_fn(pred[:, :, :self.seq_dim].permute(0, 2, 1), label, ignore_index=PAD_token) # (N, C, L)
        #label_loss = self.label_loss_fn(pred[:, :, :self.seq_dim].permute(0, 2, 1), label) 
        # Scale the values before the loss calc
        values = self.sequence_decoder.scaler.transform(values)
        if 0:
            param_loss = self.param_loss_fn(pred[:, :, self.seq_dim:], values)
        else:
            param_loss = self.param_loss_fn_bylabel(label=label, values=values, pred=pred[:, :, self.seq_dim:])

        ######### Tensorboard logging
        loss = label_loss + self.alpha * param_loss

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
                        growth_stages=None):
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
        train_ratio = 0.5
        val_ratio = 0.25
        test_ratio = 0.25

        growth_stages = self.growth_stages

        # Get the num plots from the last xml file
        xml_files = os.listdir(os.path.join(self.dataset_dir, "xml"))
        xml_files.sort()
        self.num_plots = int(xml_files[-1].split("_")[1]) + 1

        train_end = int(self.num_plots * train_ratio)
        val_end = train_end + int(self.num_plots * val_ratio)
        test_end = self.num_plots  # Ensure total sums up to num_plots

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