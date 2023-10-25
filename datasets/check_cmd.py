import numpy as np
import glob
import math

cmd = {0:0, 1:0, 2:0}
for full_path in glob.glob('/data/nuscenes/commands/**'):
	print(full_path)
	future_x, future_y = np.load(full_path)
	rad = math.atan2(future_y, future_x)

	if rad >=0 and rad < (math.pi*85/180):
		cmd[2] += 1
	elif rad >= (math.pi*85/180) and rad < (math.pi*95/180):
		cmd[1] += 1
	elif rad >= (math.pi*95/180) and rad <= (math.pi*180/180):
		cmd[0] += 1

print(cmd)