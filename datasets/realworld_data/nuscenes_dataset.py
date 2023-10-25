import torch.nn

TRAIN="TRAIN"
TEST="TEST"
BOS="BOS"
SGP="SGP"




import math
import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


from nuscenes.nuscenes import NuScenes
from nuscenes.scene_indices import *

class NuScenesSample():
    def __int__(self):
        self.city=None
        self.cam_front_data=None
        self.sample=None
        self.annotation=None
        self.index_within_scene=None

        
    
    
    
NUSCENES = NuScenes(version='v1.0-trainval', dataroot='/data/shared/nuscenes/', verbose=True)
useful_scenes=[]
for scene in NUSCENES.scene:
    scene_index = int(scene['name'].split('-')[-1].lstrip('0'))
    if scene_index in SCENE_INDICES_VALID and scene_index not in SCENE_INDICES_IGNORED:
        useful_scenes.append(scene)


class NuScenesDataset(Dataset):
    def __init__(self,cities=None,random_seed=99,train_proportion=1.0,train_or_test=TRAIN, cityid_map = None):
        

        '''
        @param cities:
                supported ['BOS','SGP']
        @param train_proportion 0.0-1.0
        @param train_or_test "TRAIN" or "TEST"

        '''
        if cities is None:
            raise Exception("Empty city")

            
        if cityid_map is None:
            self.cityid_map = {}
            self.cityid_map[BOS] = 6
            self.cityid_map[SGP] = 7
        else:
            self.cityid_map = cityid_map

        for city_name in cities:
            if city_name not in [BOS,SGP]:
                raise Exception(city_name + " not a valid city. Only support BOS and SGP")
        

        if train_or_test not in [TRAIN, TEST]:
            raise Exception("Only support TRAIN or TEST")

        
        gap=5
        self.dataset_path = '/data/nuscenes/'
        

        '''
         get the train scene indices
         '''
        boston_useful_scene_indices=[]
        singapore_useful_scene_indices=[]
        for scene in useful_scenes:
            scene_name = scene['name']
            scene_index = int(scene_name.split('-')[-1].lstrip('0'))
            
            first_sample_token = scene['first_sample_token']
            first_sample = NUSCENES.get('sample', first_sample_token)
            
            # find the city
            cam_front_data = NUSCENES.get('sample_data', first_sample['data']['CAM_FRONT'])
            city=self.get_city_thru_filename(cam_front_data['filename'])
            
            if city==BOS:
                boston_useful_scene_indices.append(scene_index)
            elif city==SGP:
                singapore_useful_scene_indices.append(scene_index)
            
        random.seed(random_seed)
        train_boston_scene_nums=int(len(boston_useful_scene_indices)*train_proportion)
        train_useful_scene_indices=random.sample(boston_useful_scene_indices,train_boston_scene_nums)
        
        train_singapore_scene_nums=int(len(singapore_useful_scene_indices)*train_proportion)
        train_useful_scene_indices+=random.sample(singapore_useful_scene_indices,train_singapore_scene_nums)

        self.transform = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        
        
        # print(train_useful_scene_indices) # todo
        


        self.jtsamples=[]
        for  scene in useful_scenes:
            

            scene_name = scene['name']
            scene_index = int(scene_name.split('-')[-1].lstrip('0'))
            
            scene_jtsamples=[]
            
            sample_token = scene['first_sample_token']
            sample = NUSCENES.get('sample', sample_token)
            
            # find the city
            cam_front_data = NUSCENES.get('sample_data', sample['data']['CAM_FRONT'])
            city=self.get_city_thru_filename(cam_front_data['filename'])
            
            if city not in cities:
                continue
            
            
            if train_or_test==TRAIN and scene_index not in train_useful_scene_indices:
                continue

            if train_or_test==TEST and scene_index in train_useful_scene_indices:
                continue
            
            

            sample_index_within_scene=-1
            while True:
                
                sample_index_within_scene+=1
                cam_front_data = NUSCENES.get('sample_data', sample['data']['CAM_FRONT'])
                anno_path = self.dataset_path + 'annotations/' + cam_front_data['filename'].split('/')[-1].replace('jpg',
                                                                                                                'npy')

                if not os.path.exists((anno_path)):
                    break
                else:
                    annotation = np.load(anno_path, allow_pickle=True).tolist()

                jtsample=NuScenesSample()
                jtsample.sample=sample
                jtsample.cam_front_data=cam_front_data
                jtsample.annotation=annotation
                jtsample.index_within_scene=sample_index_within_scene
                jtsample.city=city

                
                scene_jtsamples.append(jtsample)

                next_sample_token = sample['next']
                if next_sample_token == '':
                    break
                else:
                    next_sample = NUSCENES.get('sample', next_sample_token)
                sample = next_sample
                
            
            self.jtsamples.extend(scene_jtsamples[:-gap]) # this makes sure I do not use last 'gap' frames as the training data (just for wpts generation)


        print("Finished loading %s. Length: %d" % (self.dataset_path, len(self.jtsamples)))
        print("Finished loading ",cities)

    def get_city_thru_filename(self, filename):
        '''
        filename e.g. "samples/CAM_FRONT/n015-2018-07-18-11-50-34+0800__CAM_FRONT__1531886197512475.jpg"
        
        '''
        
        timezone=filename[42:47]
        if timezone=="+0800":
            return SGP
        elif timezone=="-0400":
            return BOS
        else:
            raise Exception("Invalid timezone")
        pass


    def __len__(self):
        return len(self.jtsamples)

    def __getitem__(self, idx):

        jtsample = self.jtsamples[idx]
        city = jtsample.city
        cityid = self.cityid_map[city]
        cam_front_data, anno = jtsample.cam_front_data, jtsample.annotation
        bgr_name = self.dataset_path + cam_front_data['filename']
        bgr_image = cv2.imread(self.dataset_path + cam_front_data['filename'])

        # bgr_image = cv2.resize(bgr_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) # original 
        
        dimension = (400, 225)
        bgr_image = cv2.resize(bgr_image, dimension, interpolation=cv2.INTER_AREA)
        

        speed = anno['speed_ms']
        veh_locations = anno['veh_locations']
        img_locations = anno['img_locations']

        locations = []
        locations_vehicle = []

        for i in range(5):
            locations.append([img_locations[i][0] / 4., img_locations[i][1] / 4.])

            if veh_locations[i][1] < 0:
                locations_vehicle.append([veh_locations[i][0], 0.])
            else:
                locations_vehicle.append([veh_locations[i][0], veh_locations[i][1]])

        future_x, future_y = locations_vehicle[-1]

        if future_y < 0:
            future_y = 0.0

        rad = math.atan2(future_y, future_x)

        # cmd start from 1
        if rad >= 0 and rad < (math.pi * 85 / 180):
            cmd = 3
        elif rad >= (math.pi * 85 / 180) and rad < (math.pi * 95 / 180):
            cmd = 2
        elif rad >= (math.pi * 95 / 180) and rad <= (math.pi * 180 / 180):
            cmd = 1


        bgr_image = transforms.ToTensor()(bgr_image)
        # bgr_image = self.transform(bgr_image)



        return bgr_image, bgr_name, np.array(locations_vehicle), cmd, speed, cityid
