import math

import numpy as np

import torch
import torch.nn as nn

from . import common_gn
# from models import common_gn
from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle

# CROP_SIZE = 192
STEPS = 5


# COMMANDS = 4
# DT = 0.1
# CROP_SIZE = 192
# PIXELS_PER_METER = 5


class ImagePolicyModelSS(common_gn.ResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)

        self.c = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }[backbone]
        self.warp = warp
        self.rgb_transform = common_gn.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.deconv = nn.Sequential(
            nn.GroupNorm(num_groups=2, num_channels=self.c + 128),
            nn.ConvTranspose2d(self.c + 128, 256, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.GroupNorm(num_groups=2, num_channels=256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.GroupNorm(num_groups=2, num_channels=128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True),
        )

        if warp:
            ow, oh = 48, 48
            assert False
        else:
            # ow,oh = 200, 120 #96,40 # image size after downsample??? /4
            ow, oh = 104, 64  # 96,40 # image size after downsample??? /4

        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=2, num_channels=64),
                nn.Conv2d(64, STEPS, 1, 1, 0),
                common_gn.SpatialSoftmax(ow, oh, STEPS),
            ) for i in range(3)
        ])
        # Three different command
        self.loc3d_pred = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5 * 2, 64),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(64, 5 * 2),
            ) for i in range(3)
        ])

        self.all_branch = all_branch

    def forward(self, image, command=None, velocity=None):
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
        velocity = velocity[..., None, None, None].repeat((1, 128, kh, kw))

        h = torch.cat((feats, velocity), dim=1)
        h = self.deconv(h)

        # print(self.location_pred[0](h).shape)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)
        location_pred = common_gn.select_branch(location_preds, command)

        location_pred = torch.flatten(location_pred, 1)

        location3d_preds = [loc3d_pred(location_pred) for loc3d_pred in self.loc3d_pred]
        location3d_preds = torch.stack(location3d_preds, dim=1)
        location3d_pred = common_gn.select_branch(location3d_preds, command)

        # location_pred = self.loc3d_pred(location_pred)
        location3d_pred = location3d_pred.view(location3d_pred.shape[0], 5, 2)

        return location3d_pred  # , feats