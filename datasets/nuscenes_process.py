from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
from pathlib import Path
import os
import numpy as np

from nuscenes.can_bus.can_bus_api import NuScenesCanBus


nusc_can = NuScenesCanBus(dataroot='')


nusc = NuScenes(version='v1.0-trainval', dataroot='', verbose=True)



OFFSET = 6.5

for my_scene in nusc.scene:

# my_scene = nusc.scene[0]

	first_sample_token = my_scene['first_sample_token']

	first_sample = nusc.get('sample', first_sample_token)

	scene_name = my_scene['name']

	if int(scene_name.split('-')[-1].lstrip('0')) in nusc_can.can_blacklist:
		continue

	veh_speed = nusc_can.get_messages(scene_name, 'vehicle_monitor')
	veh_speed = np.array([(m['utime'], m['vehicle_speed']) for m in veh_speed])
	veh_speed_cp = veh_speed.copy()


	if len(veh_speed.shape) == 2:
		can_timestamps = veh_speed[:,0].tolist()
	elif len(veh_speed.shape) == 1:
		continue
		# can_timestamps = [veh_speed[0].tolist()]



	veh_speed_cp[:, 1] *= 1 / 3.6
	veh_speed_cp[:, 0] = (veh_speed_cp[:, 0] - veh_speed_cp[0, 0]) / 1e6


	all_data = []

	while True:

		# gap = 0 # see if the frame rate is identical

		cam_front_data = nusc.get('sample_data', first_sample['data']['CAM_FRONT'])

		


		# get speed
		ind = None
		_min = 1000000000000
		for _k, _i in enumerate(can_timestamps):
			if abs(_i - first_sample['timestamp']) < _min:
				_min = abs(_i - first_sample['timestamp'])
				ind = _k

		speed = veh_speed_cp[ind][1]


		all_data.append([first_sample, cam_front_data, speed])



		next_sample_token = first_sample['next']

		if next_sample_token == '':
			break
		else:
			next_sample = nusc.get('sample', next_sample_token)


		# next_cam_front_data = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
		# next_cam_front_data_token = next_cam_front_data['token']


		# next_cam_front_data_sweep_token = cam_front_data['next']
		# while True:
		# 	gap += 1
		# 	_token = nusc.get('sample_data', next_cam_front_data_sweep_token)['next']

		# 	if _token == next_cam_front_data_token:
		# 		break
		# 	else:
		# 		next_cam_front_data_sweep_token = _token
		# print(gap)

		first_sample = next_sample


	length = len(all_data)

	for i in range(length-5):

		sample, cam_front_data, speed = all_data[i]

		# image = cv2.imread('/data/nuscenes-mini/' + cam_front_data['filename'])
		image = cv2.imread('/data/nuscenes/' + cam_front_data['filename'])

		print(cam_front_data['filename'])
		# print(image.shape)

		ego_pose_token = cam_front_data['ego_pose_token']
		calibrated_sensor_token = cam_front_data['calibrated_sensor_token']

		_ego_pose = None
		_calibrate_sensor = None
		for ep in nusc.ego_pose:
			if ep['token']==ego_pose_token:
				_ego_pose = ep
				break
		
		for cs in nusc.calibrated_sensor:
			if cs['token']==calibrated_sensor_token:
				_calibrate_sensor = cs


		# cur_pose_mat = transform_matrix(_ego_pose['translation'], Quaternion(_ego_pose['rotation']), inverse=True)
		cur_pose_mat = transform_matrix(_ego_pose['translation'], Quaternion(_ego_pose['rotation']), inverse=False)

		# cur_cam_pose_mat = transform_matrix(_calibrate_sensor['translation'], Quaternion(_calibrate_sensor['rotation']), inverse=True)
		cur_cam_pose_mat = transform_matrix(_calibrate_sensor['translation'], Quaternion(_calibrate_sensor['rotation']), inverse=False)
		cur_cam_intrinsic = _calibrate_sensor['camera_intrinsic']


		for j in range(1,6):
			ind = i + j
			next_sample, next_cam_front_data, next_speed = all_data[ind]

			next_ego_pose_token = next_cam_front_data['ego_pose_token']
			next_calibrated_sensor_token = next_cam_front_data['calibrated_sensor_token']

			_next_ego_pose = None
			_next_calibrate_sensor = None
			for ep in nusc.ego_pose:
				if ep['token']==next_ego_pose_token:
					_next_ego_pose = ep
					break
			
			for cs in nusc.calibrated_sensor:
				if cs['token']==next_calibrated_sensor_token:
					_next_calibrate_sensor = cs

			next_pose_mat = transform_matrix(_next_ego_pose['translation'], Quaternion(_next_ego_pose['rotation']), inverse=False)
			# next_pose_mat = transform_matrix(_next_ego_pose['translation'], Quaternion(_next_ego_pose['rotation']), inverse=True)
			# next_cam_intrinsic = _next_calibrate_sensor['camera_intrinsic']

			cords = np.zeros((1, 4))
			cords[:,-1] = 1.

			world_cords = np.dot(next_pose_mat, np.transpose(cords))

			# print('world:', world_cords.shape)
			# print('translation:', _next_ego_pose['translation'])
			# print('world:', world_cords[0,0], world_cords[1,0])

			# print(world_cords.shape)
			# veh_cords = np.dot(cur_pose_mat, world_cords)
			veh_cords = np.dot(np.linalg.inv(cur_pose_mat), world_cords)

			# print('veh:', round(veh_cords[0,0],2), round(veh_cords[1,0],2), veh_cords[2,0])

			image = cv2.putText(image, str(round(veh_cords[0,0],2))+', '+str(round(veh_cords[1,0],2)), (30, j*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)



	# print the camera pose!!! It should be obvious from the ego pose? Or should be fixed from ego vehicle?
			# print('cam pose:', _calibrate_sensor['translation'], _calibrate_sensor['rotation'])
			cam_cords = np.dot(np.linalg.inv(cur_cam_pose_mat), veh_cords)
			# cam_cords = np.dot(cur_cam_pose_mat, veh_cords)

			# print('cam:', cam_cords[0,0], cam_cords[1,0], cam_cords[2,0])

			image = cv2.putText(image, str(round(cam_cords[0,0],2))+', '+str(round(cam_cords[1,0],2))+', '+str(round(cam_cords[2,0],2)), (930, j*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

			# print(cam_cords.shape)

			cords_x_y_z = cam_cords[:3, :]
			cords_y_minus_z_x = np.concatenate([cords_x_y_z[0, :], cords_x_y_z[1, :], cords_x_y_z[2, :]+OFFSET]) # good!




			_image_cords = np.transpose(np.dot(cur_cam_intrinsic, cords_y_minus_z_x))
			# print(_image_cords.shape)
			image_cords = [_image_cords[0] / _image_cords[2], _image_cords[1] / _image_cords[2], _image_cords[2]]
			# image_cords = np.concatenate([_image_cords[:, 0] / _image_cords[:, 2], _image_cords[:, 1] / _image_cords[:, 2], _image_cords[:, 2]], axis=1)

			# print(int(image_cords[0]),int(image_cords[1]))

			image = cv2.circle(image,(int(image_cords[0]),int(image_cords[1])), radius=6, color=(255, 0, 255), thickness=-1)
			image = cv2.putText(image, str(j), (int(image_cords[0]+10),int(image_cords[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

			# image coordinates
			image = cv2.putText(image, str(int(image_cords[0]))+', '+str(int(image_cords[1])), (30, (j+7)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

		# vehicle speed
		image = cv2.putText(image, 'speed: '+str(speed), (930, 7*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)

			# image = cv2.circle(image,(int(image_cords[0]),int(image_cords[1]-100)), radius=6, color=(255, 0, 255), thickness=-1)

		# print(str(save_root / cam_front_data['filename'].split('/')[-1]), image.shape)

		# cv2.imwrite(str(save_root / cam_front_data['filename'].split('/')[-1]), image)

		_resize_img = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		# print(_resize_img.shape)

		# out.write(image)
		# cv2.imwrite(str(i)+'.jpg', _resize_img)
		out.write(_resize_img)

















	
