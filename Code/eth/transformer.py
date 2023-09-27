import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from resnet import resnet18
from deconv import DeconvBlock as deconvlayer
from deconv import conv3x3 as convlayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoder():
    # encode low-dim, vec to high-dims.

    def __init__(self, number_freqs, include_identity=False):
        freq_bands = torch.pow(2, torch.linspace(0., number_freqs - 1, number_freqs))
        self.embed_fns = []
        self.output_dim = 0

        if include_identity:
            self.embed_fns.append(lambda x:x)
            self.output_dim += 1

        for freq in freq_bands:
            for transform_fns in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, fns=transform_fns, freq=freq: fns(x*freq))
                self.output_dim += 1

    def encode(self, vecs):
        # inputs: [B, N]
        # outputs: [B, N*number_freqs*2]
        return torch.cat([fn(vecs) for fn in self.embed_fns], -1)

    def getDims(self):
        return self.output_dim


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        return src + pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src




class Transformer(nn.Module):

    def __init__(self, feature_num, pos_length, hidden_num=512,
                        head_num=8, dropout=0.1, pos_freq=30, pos_ident=True, pos_hidden=128):

        super(Transformer, self).__init__()
        """
        Params:

            feature_num: length of each feature.
            pos_length: length of position code.

            hidden_num: dimension of hidden layer .
            head_num: multi head number.
            dropout: probability of dropout.

            pos_freq: position code number = pos_length*(2*pos_freq + pos_ident).
            pos_ident: True means adding itself into position code.
            pos_hidden: dimension of hidden layer.

        Inputs is [N, B, L],  

            N is length of sequence.
            B is the Batch size
            L is the feature_num, which represents the number of each feature.

        Outpus is [N, B, L]
        """


    
        encoder_layer = TransformerEncoderLayer(
                  feature_num,
                  head_num, 
                  hidden_num, 
                  dropout)


        self.encoder = TransformerEncoder(
                encoder_layer, 
                num_layers = 6, 
                norm = nn.LayerNorm(feature_num))

 
        self.pos_embedding = PositionalEncoder(pos_freq, pos_ident)

        out_dim = pos_length * (pos_freq*2 + pos_ident) 

        self.pos_encode = nn.Sequential(
            nn.Linear(out_dim, pos_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(pos_hidden, pos_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(pos_hidden, pos_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(pos_hidden, feature_num)
        )
            
           

    def forward(self, feature, pos_feature):
        """
        Inputs:
            feature: [length, batch, dim1]
            pos_feature: [batch, length, dim2]

        Outputs:
            feature: [batch, length, dim1]


        """


        pos_feature = self.pos_embedding.encode(pos_feature)
        pos_feature = self.pos_encode(pos_feature)
        pos_feature = pos_feature.permute(1, 0, 2)



        feature = self.encoder(feature, pos_feature)
        feature = feature.permute(1, 0, 2)
 
        return feature

       
