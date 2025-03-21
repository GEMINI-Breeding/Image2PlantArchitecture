import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import efficientnet_b0
import math

from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn import functional as F

from torch.nn import LayerNorm
import torchvision.transforms as transforms
import numpy as np

from collections import OrderedDict

import os, sys
script_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_file_dir,"../"))

from src.plant_tokenizer import SOS_TOKEN, PAD_TOKEN, EOS_TOKEN



class MinMaxScalerTorch(nn.Module):
    def __init__(self, feature_range=(-1, 1)):
        super(MinMaxScalerTorch, self).__init__()
        self.min, self.max = feature_range
        self.data_min_ = torch.tensor([0, -19.3456, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -331.326, 0, 0, -38.3441, -10, -15],
                                       dtype=torch.float32, requires_grad=False)
        self.data_max_ = torch.tensor([59.842, 356.159, 357.383, 19, 3, 0.03, 0.00673295, 20, 215, 0.0999985, 0.0018, 79.5954, 0, 1, 0.12, 40.49, 10, 0],
                                       dtype=torch.float32, requires_grad=False)
        self.scale_ = torch.tensor([3.34213429e-02, 5.32616644e-03, 5.59623709e-03, 1.05263158e-01,
                                    6.66666667e-01, 6.66666667e+01, 2.97046614e+02, 1.00000000e-01,
                                    9.30232558e-03, 2.00003000e+01, 1.11111111e+03, 2.51270802e-02,
                                    6.03635091e-03, 2.00000000e+00, 1.66666667e+01, 2.53697321e-02,
                                    1.00000000e-01, 1.33333333e-01], 
                                    dtype=torch.float32, requires_grad=False)
        self.min_ = torch.tensor([-1., -0.89696211, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                  1., -1., -1., -0.02722045, 0., 1.], 
                                  dtype=torch.float32, requires_grad=False)

        # 1D Convolution layer for scaling and inverse scaling
        self.conv = nn.Conv1d(in_channels=18, out_channels=18, kernel_size=1, bias=True)
        self.inv_conv = nn.Conv1d(in_channels=18, out_channels=18, kernel_size=1, bias=True)

        # Initialize convolution weights and bias
        with torch.no_grad():
            self.conv.weight.copy_(torch.diag(self.scale_).view(18, 18, 1))
            self.conv.bias.copy_(self.min_)

            self.inv_conv.weight.copy_(torch.diag(1/self.scale_).view(18, 18, 1))
            self.inv_conv.bias.copy_(-self.min_ / self.scale_)

        # Freeze the parameters
        for param in self.conv.parameters():
            param.requires_grad = False
        for param in self.inv_conv.parameters():
            param.requires_grad = False

    def fit(self, data):
        self.data_min_ = data.min(0, keepdim=True)[0]
        self.data_max_ = data.max(0, keepdim=True)[0]
        self.scale_ = (self.max - self.min) / (self.data_max_ - self.data_min_)
        self.min_ = self.min - self.data_min_ * self.scale_

        # Update convolution weights and bias
        self.conv.weight.data = torch.diag(self.scale_).view(18, 18, 1)
        self.conv.bias.data = self.min_

        # Update inverse convolution weights and bias
        self.inv_conv.weight.data = torch.diag(1/self.scale_).view(18, 18, 1)
        self.inv_conv.bias.data = -self.min_ / self.scale_

    def transform(self, data):
        data = data.permute(0, 2, 1)  # (batch_size, seq_len, num_features) -> (batch_size, num_features, seq_len)
        transformed = self.conv(data)
        return transformed.permute(0, 2, 1)  # (batch_size, num_features, seq_len) -> (batch_size, seq_len, num_features)

    def inverse_transform(self, data):
        data = data.permute(0, 2, 1)  # (batch_size, seq_len, num_features) -> (batch_size, num_features, seq_len)
        inverse_transformed = self.inv_conv(data)
        return inverse_transformed.permute(0, 2, 1)  # (batch_size, num_features, seq_len) -> (batch_size, seq_len, num_features)
    
def get_tgt_mask(size) -> torch.tensor:
    if 0:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # Change type
        mask = mask.type(torch.FloatTensor)
    else:
        # Causal mask 생성
        mask = nn.Transformer.generate_square_subsequent_mask(size)
        if 0:
            # Convert to boolean
            mask.bool()
    return mask

def create_pad_mask(seq: torch.tensor, pad_token: int) -> torch.tensor:
    # Create (batch_size, seq_len) tensor
    mask = (seq == pad_token)

    # Change type
    mask = mask.type(torch.FloatTensor)
    return mask

def create_organ_mask():
    # Define mask patterns
    mask_patterns = [
       [np.zeros(5),  np.ones(4),  np.ones(5),   np.ones(4)],  # shoot_mask
        [np.ones(5), np.zeros(4),  np.ones(5),   np.ones(4)],  # internode_mask
        [np.ones(5),  np.ones(4), np.zeros(5),   np.ones(4)],  # petiole_mask
        [np.ones(5),  np.ones(4),  np.ones(5),  np.zeros(4)],  # leaf0_mask
        [np.ones(5),  np.ones(4),  np.ones(5),  np.zeros(4)],  # leaf1_mask
        [np.ones(5),  np.ones(4),  np.ones(5),  np.zeros(4)],  # leaf2_mask
        [np.ones(5),  np.ones(4),  np.ones(5),   np.ones(4)]   # all_mask
    ]
    # Create masks
    masks = torch.stack([torch.tensor(np.concatenate(pattern, axis=0), dtype=torch.bool) for pattern in mask_patterns])
    return masks
    
def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class seqBatchNorm(nn.Module):
    def __init__(self, out_dim):
        super(seqBatchNorm,self).__init__()

        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # Change B C L to use BatchNorm1d for seqence
        x = x.permute(0,2,1)
        x = self.bn(x)
        # Convert to B L C 
        x = x.permute(0,2,1)

        return x

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True, batch_norm=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and last_activation):
                if batch_norm:
                    q.append(("BatchNorm_%d" % i, seqBatchNorm(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        # Change decoded to Batch first (B L C)
        x = x.permute(1,0,2)
        x = self.mlp(x)
        # Convert back to Seq Len first (L B C)
        x = x.permute(1,0,2)
        return x
    
class TransformerDecoderLayerWithAttention(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn_weights = None
        self.multihead_attn_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # self-attention block
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask)
        self.self_attn_weights = self_attn_weights
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # multi-head attention block
        tgt2, multihead_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                           key_padding_mask=memory_key_padding_mask)
        self.multihead_attn_weights = multihead_attn_weights
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # feedforward block
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoderWithAttention(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers):
        super().__init__(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        self_attn_weights = []
        multihead_attn_weights = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            self_attn_weights.append(mod.self_attn_weights)
            multihead_attn_weights.append(mod.multihead_attn_weights)
        return output, self_attn_weights, multihead_attn_weights

# 특정 층까지의 출력을 얻기 위한 새로운 모델 정의
class EfficientNetExtractor(nn.Module):
    def __init__(self, original_model):
        super(EfficientNetExtractor, self).__init__()
        # self.features = nn.Sequential(*list(original_model.children())[0][:9])  # (8) Conv2dNormActivation까지 포함
        self.features = nn.Sequential(*list(original_model.children())[0])  # (8) Conv2dNormActivation까지 포함
        
    def forward(self, x):
        return self.features(x)
    
class CNN_FeatureExtractor(nn.Module):
    def __init__(self, output_size=256, use_depth=False):
        super(CNN_FeatureExtractor, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        #print("Before")
        #print(self.efficientnet.features[0])
        # Replace the first layer to accept 4 channel
        if use_depth:
            self.efficientnet.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            pass
        #print("After")
        print(self.efficientnet)
        self.feature_extractor = EfficientNetExtractor(self.efficientnet)

        self.efficientnet.classifier = nn.Identity()  # Remove the classification layer
        self.fc = nn.Linear(1280, output_size)  # Reduce feature dimension (1280 is the output of efficientnet_b0)

    def forward(self, x):
        x = self.feature_extractor(x)
        # (4, 1280,7,7) to (4, 1280, 49) to (4, 49, 1280)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(3))
        # (4, 49, 1280) to (4, 49, 256)
        x = self.fc(x)
        return x
    
class ViT_Projection(nn.Module):
    def __init__(self, in_features, out_features, add_noise=False, use_clstoken=True):
        super().__init__()

        self.temperature = torch.tensor(1.0)
        self.gate = nn.Linear(in_features, 1) # Predicts "usefulness" score
        self.projection = nn.Linear(in_features, out_features)
        self.add_noise = add_noise
        self.use_clstoken = use_clstoken

        # https://github.com/DepthAnything/Depth-Anything-V2/ => dpt.py#L86
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            self.readout_projects.append(
                nn.Sequential(
                    nn.Linear(3 * in_features, in_features),
                    nn.GELU()))

    def forward(self, x):
        # 1. Learned scaling
        x = x * self.temperature  

        if self.use_clstoken:
            x, cls_token, plant_info = x[:, :-2, :], x[:,-2,:], x[:,-1,:]
            cls_token = cls_token.unsqueeze(1).expand_as(x)
            plant_info = plant_info.unsqueeze(1).expand_as(x)
            x = self.readout_projects[0](torch.cat((x, cls_token, plant_info), -1))
        else:
            pass
        
        # 2. Variance-based gating
        scores = torch.sigmoid(self.gate(x))  # [B, Seq, 1]
        x = x * scores  # 중요 특징 강조
        
        # 3. 노이즈 투입 (386D 차원에서)
        x = x + torch.randn_like(x) * 0.01  # ★ 최적 위치 ★
        
        # 4. Projection to 768D
        x = self.projection(x)  
        return x
    
class ViT_FeatureExtractor(nn.Module):
    def __init__(self, output_size=256, image_size=448, use_depth=False, vit_model='facebook/dinov2-small'):
        super(ViT_FeatureExtractor, self).__init__()
        
        self.use_depth = use_depth
        if 0:
            # print("Before")
            # print(self.model)
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Use ViT, it will give 197x768 feature
            
            # Replace the first layer to accept 4 channel
            self.model.embeddings.patch_embeddings.projection = nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
            self.model.embeddings.patch_embeddings.num_channels = 4
        elif 1:
            self.model = AutoModel.from_pretrained(vit_model, output_hidden_states=True) # Use DINOv2, it will give 257x768 feature
            self.img_proc = AutoImageProcessor.from_pretrained(vit_model)
            self.embedding_size = self.model.config.hidden_size  # Get the embedding size from the model config
            self.intermediate_layer_idx = {
                    'vits': [2, 5, 8, 11],
                    'vitb': [2, 5, 8, 11], 
                    'vitl': [4, 11, 17, 23], 
                    'vitg': [9, 19, 29, 39]
                    }
            if self.use_depth:
                self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
                self.model.embeddings.patch_embeddings.projection = nn.Conv2d(4, self.embedding_size, kernel_size=(14, 14), stride=(14, 14))
                self.model.embeddings.patch_embeddings.num_channels = 4
            else:
                self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if 1:
                    # Fix the weights of the Vision Transformer model
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.model.eval() # Set Feature encoder to eval. But need to train projection and plant info embedding
        else:
            num_channels = 4 if self.use_depth else 3
            config = ViTConfig(image_size=image_size, 
                            patch_size=14,
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            num_channels=num_channels)
            mean_std = [0.5] * num_channels
            self.normalize = transforms.Normalize(mean=mean_std, std=mean_std)
            self.model = ViTModel(config)

        self.output_size = output_size
        # Embedding for plant info [leaf_area, plant_width, plant_height]
        self.plant_info_embedding = nn.Sequential(nn.Linear(3, self.embedding_size),
                                                  nn.GELU(),
                                                  nn.Linear(self.embedding_size, self.output_size))
        
        if 0:
            self.projection = MLP([self.embedding_size, output_size])  # Reduce feature dimension
        else:
            self.projection = ViT_Projection(self.embedding_size, output_size, use_clstoken=False)


    def forward(self, x, y):
        if self.use_depth == False and False:
            # Use Dinov2 image processor
            x = self.img_proc(images=x, return_tensors="pt").to(x.device)
            outputs = self.model(**x)
            if 1:
                x = outputs.last_hidden_state
            else:
                # Layers
                hidden_states = [outputs.hidden_states[idx] for idx in self.intermediate_layer_idx['vitb']]
                #x = torch.cat(hidden_states, dim=2) 
                x = torch.cat(hidden_states, dim=1)
        else:    
            if x.dtype == torch.uint8:   
                x = x.to(torch.float)
            x = self.normalize(x)
            x = self.model(x).last_hidden_state
        x = self.projection(x)
        y = self.plant_info_embedding(y).unsqueeze(1)
        x = torch.cat((x, y), dim=1)
        return x
    
    def get_last_selfattention(self, x):
        outputs = self.model(x, output_attentions=True)
        attentions = outputs.attentions  # list of (batch_size, num_heads, seq_length, seq_length)

        # 한 개의 head의 attention map 시각화
        attention = attentions[-1][0, 0, 1:, 1:]  # 마지막 layer, 첫 번째 head, CLS token 제외
        attention = attention.detach().numpy()

        return attention

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.pos_encoding = nn.Embedding(num_embeddings=max_len, embedding_dim=dim_model)
 
    def forward(self, token_embeddings: torch.tensor) -> torch.tensor:
        position_ids = torch.arange(token_embeddings.size(0), dtype=torch.long, device=token_embeddings.device).unsqueeze(1)
        position_embeddings = self.pos_encoding(position_ids)

        # Add token embedding and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(dim_model)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
 
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_parameter("pos_encoding", nn.Parameter(pos_encoding, requires_grad=False))
 
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        token_embedding = token_embedding + self.pos_encoding[:token_embedding.size(0), :]
        token_embedding = self.layer_norm(token_embedding)
        token_embedding = self.dropout(token_embedding)
        return token_embedding

class SequenceDecoderModel(nn.Module):
    def __init__(self, dim_model, 
                 num_layers, num_heads, num_tokens, 
                 max_seq_length=4096, use_depth=True, decoder_only=False, image_size=448, dropout=0.1):
        super(SequenceDecoderModel, self).__init__()

        self.dim_model = dim_model
        self.dropout = dropout

        self.seq_embedding = nn.Embedding(num_tokens, self.dim_model)

        self.activation = nn.ReLU()
        self.self_attn_weights = None
        self.multihead_attn_weights = None
        
        # Positional Encoding for Sequence
        self.Seq_positional_encoding = PositionalEncoding(dim_model=self.dim_model, max_len=max_seq_length, dropout_p=self.dropout)

        self.decoder_only = decoder_only
        if self.decoder_only:
            # self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.dim_model, nhead=num_heads)
            # self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
            self.transformer_decoder_layer = TransformerDecoderLayerWithAttention(d_model=self.dim_model, 
                                                                                  nhead=num_heads, dropout=self.dropout)
            self.transformer_decoder = TransformerDecoderWithAttention(self.transformer_decoder_layer, num_layers=num_layers)
        else:
            self.transformer = nn.Transformer(
                                            d_model=self.dim_model,
                                            nhead=num_heads,
                                            num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers,
                                            dropout=0.1,
                                            batch_first=True
                                        )

        self.seq_decode_linear = nn.Linear(self.dim_model, num_tokens)
           
    def forward(self, features, tgt_seq):
        # features = self.cnn(images)
        # Check dimensions
        if len(features.shape) == 2:
            features = features.unsqueeze(1) 
        else:
            pass
        device = tgt_seq.device


        # Convert to torch.long
        tgt_seq = tgt_seq.long()
        tgt = self.seq_embedding(tgt_seq) * math.sqrt(self.dim_model)
        
        # Make sequence length the first dimension 
        tgt = tgt.permute(1,0,2)
        features = features.permute(1,0,2)

        tgt = self.Seq_positional_encoding(tgt)
        if self.decoder_only:
            tgt_mask = get_tgt_mask(tgt_seq.size(1)).to(device)
            tgt_key_padding_mask = create_pad_mask(tgt_seq, PAD_TOKEN).to(device)
            decoded, self.self_attn_weights, self.multihead_attn_weights = self.transformer_decoder(tgt, features, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            tgt = torch.cat((features, tgt), dim=0)
            tgt_mask = get_tgt_mask(tgt.size(0)).to(device)
            # Make Transformer can see entire ViT features 
            tgt_mask[:features.size(0),:features.size(0)] = 0
            dummy_seq = torch.zeros([tgt_seq.size(0), features.size(0)]).to(device)
            tgt_seq_with_dummy = torch.cat((dummy_seq, tgt_seq), dim=1)
            tgt_key_padding_mask = create_pad_mask(tgt_seq_with_dummy, PAD_TOKEN).to(device)

            # Make batch first
            tgt = tgt.permute(1,0,2)
            features = features.permute(1,0,2)
            decoded = self.transformer(tgt, tgt, 
                                       # src_mask=tgt_mask, # src_mask is optiional
                                       src_key_padding_mask=tgt_key_padding_mask,
                                       tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
                                       tgt_is_causal=True)
            decoded = decoded[:,features.size(1):]
            # Make Seqence First
            decoded = decoded.permute(1,0,2)
        output_seq = self.seq_decode_linear(decoded)

        return output_seq
    

class RegressionModel(nn.Module):
    def __init__(self, dim_model=768, image_size=448, dropout=0.1):
        super(RegressionModel,self).__init__()
        
        self.activation = nn.ReLU()

        #self.linear = nn.Linear(197*dim_model, 4)
        self.linear = nn.Linear(257*dim_model, 6)
        #self.linear = nn.Linear(dim_model, 6)

    def forward(self, x):
        if 1:
            # Use all the decoded output
            x = x.reshape(x.size(0), -1)
        else:
            # Get CLS Token
            x = x[:, 0, :]
        x = self.activation(x) # No activation function because the output is already nonlinearity
        x = self.linear(x)
        
        return x
    
class RegressionModel_Transformer(nn.Module):
    def __init__(self, dim_model=768, image_size=224, dropout=0.1):
        super(RegressionModel_Transformer,self).__init__()
        
        self.activation = nn.ReLU()

        # self.linear = nn.Linear(197*dim_model, 4)
        self.embedding_linear = nn.Linear(6, dim_model)
        self.unembedding_linear = nn.Linear(dim_model, 6) 
        self.transformer = nn.Transformer(d_model=dim_model, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dropout=dropout)
        transformer_decoder_layer = TransformerDecoderLayerWithAttention(d_model=dim_model, nhead=4, dropout=dropout)
        self.transformer_decoder = TransformerDecoderWithAttention(transformer_decoder_layer, num_layers=2)
        self.positional_encoding = PositionalEncoding(dim_model=dim_model, max_len=2048, dropout_p=dropout)

    def forward(self, memory):
        memory = memory.permute(1,0,2) # Make Seq first
        memory = self.positional_encoding(memory)
        if 0:
            # Create a empty tgt vector to decode
            tgt = torch.zeros_like(memory[0]).unsqueeze(0)
            out, self_attn_weights, multihead_attn_weights = self.transformer_decoder(tgt, memory)
            # out = self.activation(out)
            out = self.unembedding_linear(out)
        else:
            out = self.transformer(memory, memory)
            out = out[-1]
            out = self.unembedding_linear(out)
        return out

def frange_cycle_linear(n_iter, start=0.0, stop=100.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.conv1 = nn.Conv2d(3, latent_dim//4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(latent_dim//4, latent_dim//2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(latent_dim//2 * 56 * 56, latent_dim)  # Mean
        self.fc2 = nn.Linear(latent_dim//2 * 56 * 56, latent_dim)  # Log Var
        self.fc3 = nn.Linear(latent_dim, latent_dim//2 * 56 * 56)  # Decoder input

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(latent_dim//2, latent_dim//4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(latent_dim//4, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
         # Check if the tensor is contiguous
        if h2.is_contiguous():
            h3 = h2.view(h2.size(0), -1)  # Flatten using view
        else:
            h3 = h2.reshape(h2.size(0), -1)  # Flatten using reshape. shuffling the x will make x.contiguous() == False
        return self.fc1(h3), self.fc2(h3)  # Mean and log variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std

    def decode(self, z):
        h3 = self.fc3(z).view(-1, self.latent_dim//2, 56, 56)  # Reshape
        h4 = F.relu(self.deconv1(h3))
        return torch.sigmoid(self.deconv2(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, beta=0.001):
        if 0:
            MSE = F.mse_loss(recon_x, x)  # Reconstruction loss
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
            return MSE + KLD * beta
        else:
            MSE = F.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Divide by batch size
            MSE /= x.size(0)
            KLD /= x.size(0)
            return MSE + KLD
        

class SeqEmbeddingModel(nn.Module):
    def __init__(self, d_label,d_param, d_model, max_seq_length=2048, dropout=0.1):
        super(SeqEmbeddingModel, self).__init__()
        
        self.d_label = d_label
        self.d_param = d_param
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.label_embedding = nn.Linear(d_label, d_model)
        self.param_embedding = nn.Linear(d_param, d_model)
        # self.seq_embedding_layer = nn.Linear(d_label+d_param, d_model)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.seq_embedding_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=6)
        self.positional_encoding = PositionalEncoding(dim_model=d_model, max_len=2048, dropout_p=0.1)
        self.positional_encoding.eval()

    def forward(self, x, label=None):
        # This is a simple embedding layer
        # It will be replaced by a transformer model in the future
        # seq: (batch_size, seq_len)
        
        # Get the label from the sequence 
        # num with highest probability
        if label is None:
            label = x[:,:,:self.d_label].topk(1)[1].squeeze()  
        
        label_embedding = self.label_embedding(x[:,:,:self.d_label])
        param_embedding = self.param_embedding(x[:,:,self.d_label:])
        x = label_embedding + param_embedding

        x = x.permute(1, 0, 2) # Make BLD -> LBD
        x = self.positional_encoding(x)
        x = self.seq_embedding_transformer(x)

        # Get the last token based on the EOS token (label == 42, the largest number)
        x = x.permute(1, 0, 2)
        x, _ = text_global_pool(x, label, 'argmax')
            
        return x
        
from open_clip.model import TextTransformer

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class PlantArchitectureTransformer(TextTransformer):
    def __init__(self, d_label, d_param, d_model, width=512, max_seq_length=2048, dropout=0.1):
        super(PlantArchitectureTransformer, self).__init__(context_length=max_seq_length,
                                                           vocab_size=d_label,
                                                           layers=6,
                                                           width=width,
                                                           output_dim=d_model)
        
        self.token_embedding = nn.Linear(d_label, width)
        self.parameter_embedding = nn.Linear(d_param, width)
        self.d_param = d_param
        self.d_label = d_label

    
    def forward(self, plant_architecture):

        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = plant_architecture.shape[1]
        
        label_p = plant_architecture[:, :, :self.d_label]
        label = label_p.argmax(dim=-1)
        param = plant_architecture[:, :, self.d_label:]

        x = self.token_embedding(label_p).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.parameter_embedding(param).to(cast_dtype)

        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(plant_architecture, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        attn_mask = attn_mask[None, :seq_len, :seq_len].squeeze(0)
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, label, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled