import cv2
import numpy as np
# import imutils
import math
import os
from pathlib import Path
import glob
import torch

DATASET_PATH = '/data/carla_data'
# SCENARIOS = ['nocrash_Town01_regular', 'nocrash_Town01_empty', 'nocrash_Town01_dense']
# SCENARIOS = ['nocrash_Town01_regular']
# SCENARIOS = ['nocrash_Town02_config7_ed7']
SCENARIOS = ['Town02_test']
# SCENARIOS = ['nocrash_Town01_dense']


COMMAND_ID = {'Left':1, 'Right':2, 'Straight':3, 'Follow':4}
_COMMAND_ID = {1:'Left', 2:'Right', 3:'Straight', 4:'Follow'}

# _COMMAND_MAP = {1:1, 2:3, 3:2, 4:2}

def world_to_veh(x,y,ox,oy,ori_ox, ori_oy):
    pixel_dx, pixel_dy = (x-ox), (y-oy)

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    return np.array([pixel_x, pixel_y])

def world_to_pixel(
        x,y,ox,oy,ori_ox, ori_oy,
        pixels_per_meter=5.5):
    pixel_dx, pixel_dy = (x-ox)*pixels_per_meter, (y-oy)*pixels_per_meter

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    return np.array([pixel_x, pixel_y])


pixel_per_meter = 5.

for sros in SCENARIOS:
	for full_path in glob.glob('%s/%s/**'%(DATASET_PATH,sros)):
		print(full_path)

		crop_save_path = Path(full_path) / 'annotation'
		crop_save_path.mkdir()

		meas = np.load('%s/measurements.npy'%full_path, allow_pickle=True).tolist()

		gap = 5
		n_step = 5

		num = len(meas) - gap*n_step

		for i in range(num):
			# index = i+1
			index = i
			# print(index)

			_m = meas[i]
			ox = _m['x']
			oy = _m['y']
			ori_ox, ori_oy = _m['orientation']
			vx, vy, vz = _m['velocity']

			veh_info = {}
			veh_info['image'] = str(Path(full_path) / 'rgb' / ('%04d.png' % index))
			veh_info['speed'] = np.linalg.norm([vx,vy,vz])

			veh_info['command'] = _m['command'] # we need to keep the carla command since we get command from carla when evaluating online

			# get waypoints
			# for waypoints, we use the vehicle coordinates
			wps = []
			wps_px = []
			for dt in range(gap, gap*(n_step+1), gap):
				_mdt = meas[i+dt]
				_wpvx = _mdt['x']
				_wpvy = _mdt['y']

				vc_vy, vc_vx = world_to_veh(_wpvx,_wpvy,ox,oy,ori_ox,ori_oy)
				vc_py, vc_px = world_to_pixel(_wpvx,_wpvy,ox,oy,ori_ox,ori_oy, pixels_per_meter=pixel_per_meter)

				wps.append([vc_vx,vc_vy])
				wps_px.append([vc_px, vc_py])

			veh_info['waypoints'] = wps # vehicle coordinates
			veh_info['waypoints_pixel'] = wps_px # with pixels_per_meter = 5

			np.save(str(crop_save_path / ('%04d.npy' % index)), veh_info)