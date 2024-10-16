import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import efficientnet_b0
import math

from typing import Optional, Any, Union, Callable
from torch import Tensor

from torch.nn import LayerNorm
import torchvision.transforms as transforms
import numpy as np

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

    return mask

def create_pad_mask(matrix: torch.tensor, pad_token: int) -> torch.tensor:
    # Create (batch_size, seq_len) tensor
    seq = matrix[:, :, 0]
    mask = (seq == pad_token)

    # Change type
    mask = mask.type(torch.FloatTensor)
    return mask


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
    
class ViT_FeatureExtractor(nn.Module):
    def __init__(self, output_size=256, image_size=448, use_depth=False):
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
            self.model = AutoModel.from_pretrained('facebook/dinov2-base') # Use DINOv2, it will give 257x768 feature
            self.img_proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            if self.use_depth:
                self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
            else:
                self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # self.model.embeddings.patch_embeddings.projection = nn.Conv2d(4, 768, kernel_size=(14, 14), stride=(14, 14))
            # self.model.embeddings.patch_embeddings.num_channels = 4
        else:
            # Use fully custom model
            config = ViTConfig(image_size=image_size, 
                               patch_size=16,
                               attention_probs_dropout_prob=0.0,
                               num_channels=4)  
            self.model = ViTModel(config)

        #print("After")
        #print(self.model)
        self.output_size = output_size
        if self.output_size != 768:
            self.fc = nn.Linear(768, output_size)  # Reduce feature dimension


    def forward(self, x):
        # x = self.img_proc(images=x, return_tensors="pt").to(x.device)
        # x = self.model(**x).last_hidden_state
        x = self.normalize(x)
        x = self.model(x).last_hidden_state
        if self.output_size != 768:
            x = self.fc(x)
            x = nn.ReLU()(x)
        return x
    
    def get_last_selfattention(self, x):
        outputs = self.model(x, output_attentions=True)
        attentions = outputs.attentions  # list of (batch_size, num_heads, seq_length, seq_length)

        # 한 개의 head의 attention map 시각화
        attention = attentions[-1][0, 0, 1:, 1:]  # 마지막 layer, 첫 번째 head, CLS token 제외
        attention = attention.detach().numpy()

        return attention
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
 
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
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class TransformerDecoderModel(nn.Module):
    def __init__(self, seq_embedding_dim, param_embedding_dim, 
                 num_layers, num_heads, num_tokens, num_params, 
                 max_seq_length=2048, use_depth=True, decoder_only=False, image_size=448, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        # self.cnn = CNN_ViT(output_size=seq_embedding_dim+param_embedding_dim, use_depth=use_depth, image_size=image_size)
        #self.cnn = CNN_Dinov2(output_size=seq_embedding_dim+param_embedding_dim, use_depth=use_depth)
        #self.cnn = CNN(output_size=seq_embedding_dim+param_embedding_dim, use_depth=use_depth)
        self.dim_model = seq_embedding_dim + param_embedding_dim
    
        self.seq_embedding_dim = seq_embedding_dim
        self.seq_embedding = nn.Embedding(num_tokens, self.dim_model)
        self.param_dim_model = param_embedding_dim
        
        self.dropout = dropout
        if 1:
            self.param_embedding = nn.Linear(num_params, self.dim_model)
        else:
            # Make a sequencial model
            self.param_embedding = nn.Sequential(
                                    nn.Linear(num_params, self.dim_model),
                                    # Normalize the output
                                    nn.LayerNorm(self.dim_model)
                                )
        self.embedding_linear = nn.Linear(self.param_dim_model+self.seq_embedding_dim, self.dim_model)
        self.activation = nn.ReLU()
        self.self_attn_weights = None
        self.multihead_attn_weights = None
        
        self.positional_encoding = PositionalEncoding(dim_model=self.dim_model, max_len=max_seq_length, dropout_p=self.dropout)
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
                                        )
        if 0:
            self.seq_decode_linear = nn.Linear(self.seq_embedding_dim, num_tokens)
            self.param_decode_linear = nn.Linear(self.param_dim_model, num_params)
        else:
            self.seq_decode_linear = nn.Linear(self.dim_model, num_tokens)
            #self.param_decode_linear = nn.Linear(self.dim_model, num_params)
            # Make more deeper network
            self.param_decode_linear = nn.Sequential(
                                    nn.Linear(self.dim_model, self.dim_model),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_model, self.dim_model),
                                    nn.ReLU(),
                                    nn.Linear(self.dim_model, num_params),
                                )
    
    def forward(self, features, tgt_seq, tgt_mask=None, tgt_key_padding_mask=None):
        # features = self.cnn(images)
        # Check dimensions
        if len(features.shape) == 2:
            features = features.unsqueeze(1) 
        else:
            pass

        device = tgt_seq.device
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)

        # Categorical sequence to embedding
        if len(tgt_seq.shape) == 2:
            tgt_seq = tgt_seq.unsqueeze(1)
        depth_organ_seq = tgt_seq[:, :, 0]
        # Conver to torch.long
        depth_organ_seq = depth_organ_seq.long()
        params = tgt_seq[:, :, 1:]
        
        depth_organ_seq = self.seq_embedding(depth_organ_seq) * math.sqrt(self.dim_model)
        params = self.param_embedding(params) * math.sqrt(self.dim_model)

        if 0:
            tgt_seq = torch.cat((depth_organ_seq, params), dim=2)
            tgt_seq = self.activation(self.embedding_linear(tgt_seq))
        else:
            tgt_seq = depth_organ_seq + params

        # Make sequence length the first dimension 
        # PositionalEncoding은 시퀀스 차원에 대해 적용되므로, Positional Encoding을 적용하기 전에 반드시 시퀀스 차원이 첫 번째가 되어야 합니다.
        tgt_seq = tgt_seq.permute(1,0,2)
        features = features.permute(1,0,2)

        tgt_seq = self.positional_encoding(tgt_seq)
        features = self.positional_encoding(features)

        if self.decoder_only:
            decoded, self.self_attn_weights, self.multihead_attn_weights = self.transformer_decoder(tgt_seq, features, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            decoded = self.transformer(features, tgt_seq, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=True)
        # decoded = self.activation(decoded)

        if 0:
            # 0 ~ seq_embedding_dim is the sequence, seq_embedding_dim-64 is the parameters
            decoded_seq = decoded[:, :, :self.seq_embedding_dim]
            output_seq = self.seq_decode_linear(decoded_seq)

            decoded_params = decoded[:, :, self.seq_embedding_dim:]
            output_params = self.param_decode_linear(decoded_params)
        else:
            # Use all the decoded output
            output_seq = self.seq_decode_linear(decoded)
            output_params = self.param_decode_linear(decoded)
            
        # Cat the output_seq and output_params
        output_seq = torch.cat((output_seq, output_params), dim=2)
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
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Mean
        self.fc2 = nn.Linear(64 * 56 * 56, 128)  # Log Var
        self.fc3 = nn.Linear(128, 64 * 56 * 56)  # Decoder input

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = h2.view(h2.size(0), -1)  # Flatten
        return self.fc1(h3), self.fc2(h3)  # Mean and log variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std

    def decode(self, z):
        h3 = self.fc3(z).view(-1, 64, 56, 56)  # Reshape
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