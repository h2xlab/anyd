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




class ImagePolicyModelSS(common.ResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, num_cities=11, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)

        self.c = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }[backbone]
        self.warp = warp
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.num_cities = num_cities
        self.city_encoding = nn.Sequential(
            # nn.Embedding(num_embeddings=self.num_cities, embedding_dim=self.c),
            nn.Embedding(num_embeddings=self.num_cities, embedding_dim=128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            # nn.Sigmoid(),
            # nn.BatchNorm2d(51),
        )


        self.scale = self.c ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)

#3 se
        self.KV1 = QKV(input_gate_channels=self.c+1, gate_channels=self.c)
        self.KV2 = QKV(input_gate_channels=self.c+1, gate_channels=self.c)
        self.KV3 = QKV(input_gate_channels=self.c+1, gate_channels=self.c)
        self.Q = Q(input_gate_channels=self.c + 1, gate_channels=self.c)
        # self.domain_gate = DomainGate(self.c+128,3)
        self.cls_token = nn.Parameter(torch.randn(1,1))
        self.cls_token_city = nn.Parameter(torch.randn(1, 1))
        self.attend = nn.Softmax(dim=-1)
        # self.norm = nn.LayerNorm(512)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.c + 128+1))
        # torch.nn.init.constant_(self.city_encoding[0].weight.data, val=10)
        # torch.nn.init.constant_(self.city_encoding[0].weight.data, val=1)
        # torch.nn.init.constant_(self.city_encoding[2].weight.data, val=1)
        # torch.nn.iit.constant_(self.city_encoding[2].bias.data, val=0)

        if warp:
            ow, oh = 48, 48
            assert False
        else:
            # ow,oh = 200, 120 #96,40 # image size after downsample??? /4
            ow, oh = 104, 64  # 96,40 # image size after downsample??? /4



        self.branched_conv = nn.ModuleList([
            nn.Sequential(
                # nn.BatchNorm2d(64),
                nn.BatchNorm2d(self.c + 128),
                nn.Conv2d(self.c + 128, 128, 3, padding="same"),
                # common.SpatialSoftmax(ow, oh, STEPS),
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

        self.ff = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512),
        )

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
        city = self.city_encoding(city)
        head0 = self.KV1(feats,self.cls_token)
        head1 = self.KV2(feats,self.cls_token)
        head2 = self.KV3(feats,self.cls_token)
        avg_pooled_feat = F.avg_pool2d(feats, (feats.size(2), feats.size(3)), stride=(feats.size(2), feats.size(3))).squeeze(-1).squeeze(-1)
        q = self.Q(city,self.cls_token_city)
        q = repeat(q , 'b d -> b h d', h=3 )
        q = q.unsqueeze(-1)
        temp = torch.stack([head0, head1, head2], dim=2)
        kv = temp.chunk(2, dim=1)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=3), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # 1 8 105 105 b h patchsize+1 patchsize+1

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.norm(out)
        out = torch.matmul(out[:,0,:].unsqueeze(1),out[:,1:,:].transpose(-1, -2)).squeeze(1)
        out = avg_pooled_feat + self.ff(avg_pooled_feat+ out) # resnet
        scale = out.unsqueeze(2).unsqueeze(3).expand_as(feats)
        # scale = F.sigmoid(out.unsqueeze(2).unsqueeze(3).expand_as(feats))
        feats = feats * scale

        h = torch.cat((feats, spd_embds), dim=1)
        # h = torch.cat((feats, spd_embds, city), dim=1)
        # h = self.deconv(h)

        # print(self.location_pred[0](h).shape)



        location_preds_list = [branched_conv(h) for branched_conv in self.branched_conv]
        location_preds = torch.stack(location_preds_list, dim=1)
        location_pred = common.select_branch(location_preds, command)

        location_pred = torch.flatten(location_pred, 1)

        location3d_preds_list = [prediction(location_pred) for prediction in self.prediction]
        location3d_preds = torch.stack(location3d_preds_list, dim=1)
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

class QKV(nn.Module):
    def __init__(self, input_gate_channels,gate_channels, qkv_channel=2, reduction_ratio=16, pool_types=['avg','max']):
        super(QKV, self).__init__()
        self.gate_channels = gate_channels
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


class Q(nn.Module):
    def __init__(self, input_gate_channels, gate_channels):
        super(Q, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_gate_channels),
            nn.Linear(input_gate_channels, (input_gate_channels)),
        )
    def forward(self, x,  cls_token):
        cls_tokens = repeat(cls_token, '1 d -> b d', b=x.shape[0])
        feats = torch.cat([cls_tokens, x], dim=1)
        feats = self.mlp(feats)
        return feats

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs