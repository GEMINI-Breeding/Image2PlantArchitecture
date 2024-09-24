import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import efficientnet_b0
import math

from typing import Optional, Any, Union, Callable
from torch import Tensor


def get_tgt_mask(size) -> torch.tensor:
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

    # Change type
    mask = mask.type(torch.FloatTensor)
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
                         tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            self_attn_weights.append(mod.self_attn_weights)
            multihead_attn_weights.append(mod.multihead_attn_weights)
        return output, self_attn_weights, multihead_attn_weights

class CNN(nn.Module):
    def __init__(self, output_size=256, use_depth=False):
        super(CNN, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        #print("Before")
        #print(self.efficientnet.features[0])
        # Replace the first layer to accept 4 channel
        if use_depth:
            self.efficientnet.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            pass
        #print("After")
        #print(self.efficientnet.features[0])
        self.efficientnet.classifier = nn.Identity()  # Remove the classification layer
        self.fc = nn.Linear(1280, output_size)  # Reduce feature dimension (1280 is the output of efficientnet_b0)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return x
    
class CNN_ViT(nn.Module):
    def __init__(self, output_size=256, image_size=448, use_depth=False):
        super(CNN_ViT, self).__init__()
        
        if 0:
            # print("Before")
            # print(self.model)
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            # Replace the first layer to accept 4 channel
            if use_depth:
                self.model.embeddings.patch_embeddings.projection = nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
                self.model.embeddings.patch_embeddings.num_channels = 4
        
        else:
            # Use fully custom model
            config = ViTConfig(image_size=image_size, 
                               patch_size=16,
                               attention_probs_dropout_prob=0.1,
                               num_channels=4)  
            self.model = ViTModel(config)

        #print("After")
        #print(self.model)
        self.fc = nn.Linear(768, output_size)  # Reduce feature dimension

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = self.fc(x)
        return x
    
    def get_last_selfattention(self, x):
        outputs = self.model(x, output_attentions=True)
        attentions = outputs.attentions  # list of (batch_size, num_heads, seq_length, seq_length)

        # 한 개의 head의 attention map 시각화
        attention = attentions[-1][0, 0, 1:, 1:]  # 마지막 layer, 첫 번째 head, CLS token 제외
        attention = attention.detach().numpy()

        return attention
    

class CNN_Dinov2(nn.Module):
    def __init__(self, output_size=256, use_depth=False):
        super(CNN_Dinov2, self).__init__()
        
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        
        if use_depth:
            # self.model.patch_embed.proj = nn.Conv2d(4, 768, kernel_size=(14, 14), stride=(14, 14))
            # self.model.patch_embed.proj.num_channels = 4
            self.model.embeddings.patch_embeddings.projection = nn.Conv2d(4, 768, kernel_size=(14, 14), stride=(14, 14))
            self.model.embeddings.patch_embeddings.num_channels = 4

        # print(self.model)
        self.fc = nn.Linear(768, output_size)  # Reduce feature dimension

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = self.fc(x)
        return x
    
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
        self.register_buffer("pos_encoding", pos_encoding)
 
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class ImageToSequenceTransformer(nn.Module):
    def __init__(self, seq_embedding_dim, param_embedding_dim, 
                 num_layers, num_heads, num_tokens, num_params, 
                 max_seq_length=2048, use_depth=True, decoder_only=False, image_size=448):
        super(ImageToSequenceTransformer, self).__init__()
        self.cnn = CNN_ViT(output_size=seq_embedding_dim+param_embedding_dim, use_depth=use_depth, image_size=image_size)
        #self.cnn = CNN_Dinov2(output_size=seq_embedding_dim+param_embedding_dim, use_depth=use_depth)
        #self.cnn = CNN(output_size=seq_embedding_dim+param_embedding_dim, use_depth=use_depth)
    
        self.seq_dim_model = seq_embedding_dim
        self.seq_embedding_dim = seq_embedding_dim
        self.seq_embedding = nn.Embedding(num_tokens, seq_embedding_dim)
        self.param_dim_model = param_embedding_dim
        self.dim_model = seq_embedding_dim + param_embedding_dim
        if 1:
            self.param_embedding = nn.Linear(num_params, param_embedding_dim)
        else:
            # Make a sequencial model
            self.param_embedding = nn.Sequential(
                                    nn.Linear(num_params, param_embedding_dim),
                                    # Normalize the output
                                    nn.LayerNorm(param_embedding_dim)
                                )
        self.self_attn_weights = None
        self.multihead_attn_weights = None
        
        self.positional_encoding = PositionalEncoding(dim_model=self.dim_model, max_len=max_seq_length, dropout_p=0.1)
        self.decoder_only = decoder_only
        if self.decoder_only:
            # self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.dim_model, nhead=num_heads)
            # self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
            self.transformer_decoder_layer = TransformerDecoderLayerWithAttention(d_model=self.dim_model, nhead=num_heads)
            self.transformer_decoder = TransformerDecoderWithAttention(self.transformer_decoder_layer, num_layers=num_layers)
        else:
            self.transformer = nn.Transformer(
                                            d_model=self.dim_model,
                                            nhead=num_heads,
                                            num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers,
                                            dropout=0.1,
                                        )
        if 1:
            self.seq_linear = nn.Linear(self.seq_embedding_dim, num_tokens)
            self.param_linear = nn.Linear(self.param_dim_model, num_params)
        else:
            self.seq_linear = nn.Linear(self.dim_model, num_tokens)
            self.param_linear = nn.Linear(self.dim_model, num_params)
    
    def forward(self, images, tgt_seq, tgt_mask=None, tgt_key_padding_mask=None):
        features = self.cnn(images)
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

        tgt_seq = torch.cat((depth_organ_seq, params), dim=2)
        tgt_seq = self.positional_encoding(tgt_seq)
        
        features = self.positional_encoding(features)

        features = features.permute(1,0,2)
        tgt_seq = tgt_seq.permute(1,0,2)
        if self.decoder_only:
            decoded, self.self_attn_weights, self.multihead_attn_weights = self.transformer_decoder(tgt_seq, features, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            decoded = self.transformer(features, tgt_seq, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        
        if 1:
            # 0 ~ seq_dim_model is the sequence, seq_dim_model-64 is the parameters
            decoded_seq = decoded[:, :, :self.seq_dim_model]
            output_seq = self.seq_linear(decoded_seq)

            decoded_params = decoded[:, :, self.seq_dim_model:]
            output_params = self.param_linear(decoded_params)
        else:
            # Use all the decoded output
            output_seq = self.seq_linear(decoded)
            output_params = self.param_linear(decoded)
            
        # Cat the output_seq and output_params
        output_seq = torch.cat((output_seq, output_params), dim=2)
        return output_seq