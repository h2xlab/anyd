import math

import numpy as np

import torch
import torch.nn as nn

from . import common
import torch.nn.functional as F
from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle

# CROP_SIZE = 192
STEPS = 5


# COMMANDS = 4
DT = 0.1
# CROP_SIZE = 192
# PIXELS_PER_METER = 5


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
        # self.ifbranchcon = ifbranchcon
        # self.ifgeo = ifgeo

        # self.spd_encoder = nn.Sequential(
        #     nn.Linear(1, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 128),
        # )

        # self.spd_encoder = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 64),
        # )

        # self.deconv = nn.Sequential(
        #     nn.BatchNorm2d(self.c + 128),
        #     nn.ConvTranspose2d(self.c + 128, 256, 3, 2, 1, 1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(128),
        #     nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
        #     nn.ReLU(True),
        # )
        self.num_cities = num_cities
        # self.city_encoding = nn.Sequential(
        #     nn.Embedding(num_embeddings=self.num_cities, embedding_dim=self.c),
        #     # nn.Embedding(num_embeddings=self.num_cities, embedding_dim=64),
        #     # nn.ReLU(True),
        #     # nn.Sigmoid(),
        #     # nn.Linear(128, 512),
        #     nn.ReLU(True),
        # )
        # if self.ifgeo:
        self.city_encoding = nn.Sequential(
            # nn.Embedding(num_embeddings=self.num_cities, embedding_dim=self.c),
            nn.Embedding(num_embeddings=self.num_cities, embedding_dim=128),
            nn.ReLU(True),
            # nn.BatchNorm2d(128),
            # nn.Sigmoid(),
            nn.Linear(128, 128),
            # nn.Sigmoid(),
            # nn.BatchNorm2d(51),
        )

        # self.city_encoding = nn.Sequential(
        #     # nn.Embedding(num_embeddings=self.num_cities, embedding_dim=self.c),
        #     nn.Embedding(num_embeddings=self.num_cities, embedding_dim=128),
        #     nn.ReLU(True),
        #     # nn.BatchNorm2d(128),
        #     # nn.Sigmoid(),
        #     nn.Linear(128, 3),
        #     nn.Sigmoid(),
        #     # nn.BatchNorm2d(51),
        # )
#only se
        # self.channel_gate = ChannelGate(self.c)
#3 se
        self.channel_gate = ChannelGate(input_gate_channels=self.c+128, gate_channels=self.c)
        self.channel_gate1 = ChannelGate(input_gate_channels=self.c+128, gate_channels=self.c)
        self.channel_gate2 = ChannelGate(input_gate_channels=self.c+128, gate_channels=self.c)
        self.domain_gate = DomainGate(self.c+128,3)
        # torch.nn.init.constant_(self.city_encoding[0].weight.data, val=10)
        # torch.nn.init.constant_(self.city_encoding[0].weight.data, val=1)
        # torch.nn.init.constant_(self.city_encoding[2].weight.data, val=1)
        # torch.nn.init.constant_(self.city_encoding[2].bias.data, val=0)

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
            ) for i in range(4)
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
            ) for i in range(4)
        ])

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
        # spd_embds = self.spd_encoder(speed[:, None])[..., None, None].repeat(1, 1, kh, kw)
        # city = self.city_encoding(city)[..., None, None].repeat(1, 1, kh, kw)
        # velocity = self.speed_encoder(velocity)

        # if self.ifgeo:
        city = self.city_encoding(city)
        # feats = city.unsqueeze(2).unsqueeze(3) * feats
        # featsc = city.unsqueeze(2).unsqueeze(3) * feats

        # print(self.ifgeo)
# one se
        # feats = self.channel_gate(feats)
# three se with a soft attention
        feats0 = self.channel_gate(feats,city)
        feats1 = self.channel_gate1(feats,city)
        feats2 = self.channel_gate2(feats,city)
        domain_att = self.domain_gate(feats,city)
        feats = domain_att[:,0].unsqueeze(1).unsqueeze(1).unsqueeze(1) * feats0 + domain_att[:,1].unsqueeze(1).unsqueeze(1).unsqueeze(1) * feats1 + domain_att[:,2].unsqueeze(1).unsqueeze(1).unsqueeze(1) * feats2
#
# # three se with a soft attention + geo.
        feats = feats

# # city domain
#         city_att = self.city_encoding(city)
#         # domain_att = domain_att + city_att
#         feats = city_att[:,0].unsqueeze(1).unsqueeze(1).unsqueeze(1) * feats0 + city_att[:,1].unsqueeze(1).unsqueeze(1).unsqueeze(1) * feats1 + city_att[:,2].unsqueeze(1).unsqueeze(1).unsqueeze(1) * feats2


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

class ChannelGate(nn.Module):
    def __init__(self, input_gate_channels,gate_channels, reduction_ratio=16, pool_types=['avg','max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.flatten = Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x, city_feature):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = self.flatten(avg_pool)
                feats = torch.cat((avg_pool, city_feature),dim=1)
                channel_att_raw = self.mlp(feats)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = self.flatten(max_pool)
                feats = torch.cat((max_pool, city_feature), dim=1)
                channel_att_raw = self.mlp(feats)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class DomainGate(nn.Module):
    def __init__(self, gate_channels, domain_dim=3, pool_types=['avg']):
        super(DomainGate, self).__init__()
        self.gate_channels = gate_channels
        self.flatten = Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, domain_dim),
            nn.Softmax(dim=1),
            )
        self.pool_types = pool_types
    def forward(self, x, city_feature):
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool = self.flatten(avg_pool)
        feats = torch.cat((avg_pool, city_feature), dim=1)
        channel_att_raw = self.mlp(feats)
        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # return x * scale
        return channel_att_raw

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class  ImageAgent(Agent):
    def __init__(self, steer_points=None, pid=None, gap=5,townid=0, camera_args={'rg': 20, 'offset': 20}, **kwargs):
        super().__init__(**kwargs)

        # self.fixed_offset = float(camera_args['fixed_offset'])
        # print ("Offset: ", self.fixed_offset)
        # w = float(camera_args['w'])
        # h = float(camera_args['h'])
        self._norm = np.array([20, 20])
        self.townid=townid

        self.gap = gap

        if steer_points is None:
            # steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}
            # steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}
            steer_points = {"1": 3, "2": 3, "3": 2, "4": 4}
            # steer_points = {"1": 1, "2": 1, "3": 1, "4": 2}

        if pid is None:
            pid = {
                "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},  # Left
                "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},  # Right
                "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
                "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},  # Follow
            }

        self.steer_points = steer_points
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)

        self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0

        self.last_brake = -1

    def run_step(self, observations, teaching=False):
        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])
        _cmd = int(observations['command'])
        command = self.one_hot[int(observations['command']) - 1]
        # city_id = torch.LongTensor([self.townid])
        city_id = torch.tensor(self.townid, dtype=torch.long)


        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            city_id = city_id.to(self.device).unsqueeze(0)
            # print(city_id.shape)
            # if self.model.all_branch:
            #     model_pred, _ = self.model(_rgb, _command, _speed)
            # else:
            #     model_pred = self.model(_rgb, _command, _speed)
            model_pred = self.model(_rgb, _command, _speed, city_id)

        # print(model_pred.shape)
        model_pred = model_pred.squeeze().detach().cpu().numpy()
        # print(model_pred.shape)

        # Project back to world coordinate
        model_pred[:, 1] = model_pred[:, 1] + 1
        # print(model_pred)
        world_pred = model_pred * self._norm

        # if abs(world_pred[-1, 0]) < 0.1:
        #     # world_pred[-1,:] = np.zeros_like(world_pred[-1,:])
        #     steer = 0.0


        # print(world_pred)
        # world_pred = np.zeros_like(world_pred)
        # world_pred[:, 1] = world_pred[:, 1] + 4

        # world_pred[0, 0] = world_pred[0, 0] + 1
        # world_pred[1, 0] = world_pred[1, 0] + 2
        # world_pred[2, 0] = world_pred[2, 0] + 3
        # world_pred[3, 0] = world_pred[3, 0] + 4
        # world_pred[4, 0] = world_pred[4, 0] + 5


        # print(world_pred)
        # print('world',world_pred)

        # world_pred[0, 1:5] = world_pred[0, 1:5] + 10

        targets = [(0, 0)]

        for i in range(STEPS):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])
            targets.append([dist * np.cos(angle), dist * np.sin(angle)])
        # print('target',targets)

        targets = np.array(targets)

        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self.gap * DT)

        c, r = ls_circle(targets)
        n = self.steer_points.get(str(_cmd), 1)
        closest = common.project_point_to_circle(targets[n], c, r)

        acceleration = target_speed - speed

        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)

        # if _cmd == :
        steer = self.turn_control.run_step(alpha, _cmd)
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Slow or stop.

        if target_speed <= self.engine_brake_threshold:
            steer = 0.0
            throttle = 0.0

        # if steer <= 0.001:
        #     steer = 0.0

        if abs(world_pred[-1, 0]) < 0.1:
            # world_pred[-1,:] = np.zeros_like(world_pred[-1,:])
            steer = 0.0


        if target_speed <= self.brake_threshold:
            brake = 1.0

        self.debug = {
            # 'curve': curve,
            'target_speed': target_speed,
            'target': closest,
            'locations_world': targets,
            'locations_pixel': model_pred.astype(int),
        }

        control = self.postprocess(steer, throttle, brake)
        # if teaching:
        #     return control, pixel_pred
        # else:
        #     return control
        return control, world_pred