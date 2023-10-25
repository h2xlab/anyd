import numpy as np
import glob
import math
import cv2
# cmd = {0:0, 1:0, 2:0, 3:0, 4:0}

xs = []
ys = []
for full_path in glob.glob('/media/zhangjim/Elements/nuscenes/annotations/**'):
	# print(full_path)
	anno = np.load(full_path, allow_pickle=True).tolist()

	brake = anno['brake']
	steering = anno['steering']
	throttle = anno['throttle']
	# rad = anno['cmd_rad']
	speed = anno['speed_ms']
	veh_locations = anno['veh_locations']
	cam_locations = anno['cam_locations']
	img_locations = anno['img_locations']

	locations = []
	locations_vehicle = []

	for i in range(5):
		locations.append([img_locations[i][0]/4., img_locations[i][1]/4.])
		locations_vehicle.append([veh_locations[i][0], veh_locations[i][1]])

		xs.append(veh_locations[i][0])
		ys.append(veh_locations[i][1])
print('x:')
print(min(xs), max(xs))
print('y:')
print(min(ys), max(ys))






	# future_x, future_y = locations_vehicle[-1]

	# if future_y < 0:
 #            future_y = 0.0

	# rad = math.atan2(future_y, future_x)

	# if rad >=0 and rad < (math.pi*70/180):
	# 	cmd = 5
	# elif rad >= (math.pi*70/180) and rad < (math.pi*85/180):
	# 	cmd = 4
	# elif rad >= (math.pi*85/180) and rad < (math.pi*95/180):
	# 	cmd = 3
	# elif rad >= (math.pi*95/180) and rad < (math.pi*110/180):
	# 	cmd = 2
	# elif rad >= (math.pi*110/180) and rad <= (math.pi*180/180):
	# 	cmd = 1

	# img = cv2.imread(full_path.replace('annotations', 'samples/CAM_FRONT').replace('npy', 'jpg'))
	# img = cv2.putText(img, 'cmd: '+str(cmd), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)
	# cv2.imwrite(full_path.replace('annotations', 'vis').replace('npy', 'jpg'), img)
# print(cmd)