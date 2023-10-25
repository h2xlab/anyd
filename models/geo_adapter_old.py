import math

import numpy as np

import torch
import torch.nn as nn

from . import common
import torch.nn.functional as F
from einops import rearrange, repeat
# from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle

# CROP_SIZE = 192
STEPS = 5


# COMMANDS = 4
# DT = 0.1
# CROP_SIZE = 192
# PIXELS_PER_METER = 5


class ImagePolicyModelSS(common.ResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, num_cities=11,n_heads=3, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)

        self.c = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
        }[backbone]
        self.warp = warp
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.num_cities = num_cities
        self.city_encoding = nn.Embedding(num_embeddings=self.num_cities, embedding_dim=128)


        self.scale = self.c ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)

        # self.KV1 = QKV(input_gate_channels=self.c+1, gate_channels=self.c)
        # self.KV2 = QKV(input_gate_channels=self.c+1, gate_channels=self.c)
        # self.KV3 = QKV(input_gate_channels=self.c+1, gate_channels=self.c)
        # self.Q = Q(input_gate_channels=self.c + 1, gate_channels=self.c)
        # self.domain_gate = DomainGate(self.c+128,3)
        self.cls_token = nn.Parameter(torch.randn(1,1))
        self.cls_token_city = nn.Parameter(torch.randn(1, 1))

        self.branched_conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.c + 128),
                nn.Conv2d(self.c + 128, 128, 3, padding="same"),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 5, 3),
                nn.ReLU(True),
                nn.BatchNorm2d(5),
            ) for i in range(3)
        ])
        # Three different command
        self.prediction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5 * 11 * 6, 64),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(64, 5 * 2),
            ) for i in range(3)
        ])
        self.geo_adapter = Geo_Adapter(self.c, qkv_channel=2,ffhidden=256, pool_types=['avg','max'], n_heads=n_heads)

        self.all_branch = all_branch

    def forward(self, image, command=None, speed=None, city=None):
        if self.warp:
            assert False
            warped_image = tgm.warp_perspective(image, self.M, dsize=(192, 192))
            resized_image = resize_images(image)
            image = torch.cat([warped_image, resized_image], 1)
        image = self.rgb_transform(image)
        feats = self.conv(image)
        # if ifyt:
        #     return  featsc
        b, c, kh, kw = feats.size()

        # Late fusion for velocity

        spd_embds = speed[..., None, None, None].repeat((1, 128, kh, kw))
        # if self.ifgeo:
        city_feature = self.city_encoding(city)
        feats = self.geo_adapter(feats,city_feature,self.cls_token,self.cls_token_city)

        h = torch.cat((feats, spd_embds), dim=1)
        # h = torch.cat((feats, spd_embds, city), dim=1)
        # h = self.deconv(h)

        # print(self.location_pred[0](h).shape)



        location_preds_list = [branched_conv(h) for branched_conv in self.branched_conv]
        location_preds = torch.stack(location_preds_list, dim=1)
        location_pred = common.select_branch(location_preds, command)

        location_pred = torch.flatten(location_pred, 1)

        location3d_preds = [prediction(location_pred) for prediction in self.prediction]
        location3d_preds = torch.stack(location3d_preds, dim=1)
        location3d_pred = common.select_branch(location3d_preds, command)

        # location_pred = self.loc3d_pred(location_pred)
        location3d_pred = location3d_pred.view(location3d_pred.shape[0], 5, 2)

        # if self.ifbranchcon:
        #     return location3d_pred, location_preds_list
        # else:
        return location3d_pred  # , feats

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Geo_Adapter(nn.Module):
    def __init__(self, dim, qkv_channel=2,ffhidden=256, pool_types=['avg','max'], n_heads=3):
        super(Geo_Adapter, self).__init__()
        # self.city_projection = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(city_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.qkv_channel = qkv_channel
        self.n_heads = n_heads
        self.KV = nn.ModuleList([
             QKV(input_gate_channels=dim+1, seed=(i+1)*17)
             for i in range(n_heads)
        ])
        self.Q = GeoQ(dim=dim + 1)
        # self.toQ = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(dim+1, dim+1),
        #     nn.LayerNorm(dim+1),
        # )

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)
        self.dim = dim
        self.scale = dim ** -0.5
        self.ff = nn.Sequential(
            nn.Linear(dim, ffhidden),
            nn.GELU(),
            nn.Linear(ffhidden, dim),
        )


    def forward(self,feats, city_feature, cls_token, city_coken):
        # city_feature = self.city_projection(city_feature)
        avg_pooled_feat =  F.avg_pool2d(feats, (feats.size(2), feats.size(3)), stride=(feats.size(2), feats.size(3))).squeeze(-1).squeeze(-1)
        heads = [KV(feats,cls_token) for KV in self.KV]
        temp = torch.stack(heads, dim=2)
        kv = temp.chunk(self.qkv_channel, dim=1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), kv)
        # head0 = self.KV1(feats,cls_token)
        # head1 = self.KV2(feats,self.cls_token)
        # head2 = self.KV3(feats,self.cls_token)
        q = self.Q(city_feature,city_coken)
        # city_cokens = repeat(city_coken, '1 d -> b d', b=city_feature.shape[0])
        # q = torch.cat([city_cokens, city_feature], dim=1)
        # q = self.toQ(q)


        q = repeat(q , 'b d -> b h d', h=self.n_heads)
        q = q.unsqueeze(-1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # 1 8 105 105 b h patchsize+1 patchsize+1

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.norm(out)
        out = torch.matmul(out[:,0,:].unsqueeze(1),out[:,1:,:].transpose(-1, -2)).squeeze(1)

        out = avg_pooled_feat + self.ff(avg_pooled_feat+ out) # resnet v3_2 res+mlp
        # out = F.normalize(out, p=1, dim=1) # v3-3 mlp L1norm

        scale = out.unsqueeze(2).unsqueeze(3).expand_as(feats)
        # scale = F.sigmoid(out.unsqueeze(2).unsqueeze(3).expand_as(feats))
        feats = feats * scale

        return feats

class QKV(nn.Module):
    def __init__(self, input_gate_channels, qkv_channel=2, pool_types=['avg','max'], seed=100):
        super(QKV, self).__init__()
        # self.gate_channels = gate_channels
        torch.manual_seed(seed)
        self.qkv_channel = qkv_channel
        self.flatten = Flatten()
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_gate_channels, gate_channels // reduction_ratio),
        #     nn.LayerNorm(gate_channels // reduction_ratio),
        #     nn.GELU(),
        #     # nn.ReLU(),
        #     nn.Linear(gate_channels // reduction_ratio, (gate_channels)*qkv_channel)
        #     )
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_gate_channels),
            nn.Linear(input_gate_channels, (input_gate_channels)*qkv_channel),

            # nn.ReLU(),
            # nn.Linear(gate_channels // reduction_ratio, (gate_channels)*qkv_channel)
            )
        self.pool_types = pool_types
    def forward(self, x, cls_token):
        channel_att_sum = None
        cls_tokens = repeat(cls_token, '1 d -> b d', b=x.shape[0])
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                feats = self.flatten(avg_pool)
                # feats = torch.cat((avg_pool, city_feature),dim=1)
                feats = torch.cat([cls_tokens,feats],dim=1)
                # feats = feats + pos_embedding
                channel_att_raw = self.mlp(feats)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                feats = self.flatten(max_pool)
                # feats = torch.cat((max_pool, city_feature), dim=1)
                feats = torch.cat([cls_tokens, feats], dim=1)
                # feats = feats + pos_embedding
                channel_att_raw = self.mlp(feats)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # channel_att_sum = channel_att_sum + torch.cat([pos_embedding]*self.qkv_channel, dim=1)
        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return channel_att_sum


class GeoQ(nn.Module):
    def __init__(self, dim):
        super(GeoQ, self).__init__()
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, dim-1),
            nn.LayerNorm(dim-1),
        )

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x,  cls_token):
        x = self.proj(x)
        cls_tokens = repeat(cls_token, '1 d -> b d', b=x.shape[0])
        feats = torch.cat([cls_tokens, x], dim=1)
        feats = self.mlp(feats)
        return feats

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


