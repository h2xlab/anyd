MODULE_VERSION = "v2.24 stable"

import pickle
import pathlib
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import os
import cv2
import numpy as np
import random
import math
from tqdm import tqdm
from torchvision import transforms
import psutil

IMAGE_FILE_EXTENSION = '.jpg'
PICKLE_FOLDER_DIR = '/pickle_file_path/' #jtcheckpoint pickles2


class AV2DataLoaderHelper():
    def __init__(self):
        self.av2_sensor_dataloader = None
        self.dataloader_dict = {}

    def get_dataloader(self, dataset_path):
        # check if in the memory
        if str(dataset_path) in self.dataloader_dict:
            print("AV2DataLoaderHelper: av2_sensor_dataloader for", str(dataset_path), "in memory is reused.")
            return self.dataloader_dict[str(dataset_path)]

        # check if in the disk
        READ_CACHE_FROM_DISK = False #jtcheckpoint True
        pickle_file_path = pathlib.Path(
            PICKLE_FOLDER_DIR + "av2_sensor_dataloader_" + str(dataset_path).replace("/", "_") + '.pickle')
        if READ_CACHE_FROM_DISK and pickle_file_path.is_file():
            try:
                pickle_file = open(str(pickle_file_path), 'rb')
                av2_sensor_dataloader = pickle.load(pickle_file)
                pickle_file.close()

                self.dataloader_dict[str(dataset_path)] = av2_sensor_dataloader  # put it to memory
                
                print("AV2DataLoaderHelper: av2_sensor_dataloader for", str(dataset_path), "in disk is reused.")
                return av2_sensor_dataloader
            except BaseException as e:
                print (e.args)

        # construct it
        print("AV2DataLoaderHelper: Start constructing av2_sensor_dataloader for", str(dataset_path))
        av2_sensor_dataloader = AV2SensorDataLoader(dataset_path, dataset_path)
        print("AV2DataLoaderHelper: Construction of av2_sensor_dataloader for", str(dataset_path), "is finished.")

        # put it to memory
        self.dataloader_dict[str(dataset_path)] = av2_sensor_dataloader  #
        
        
        
        # save it to the disk
        WRITE_CACHE_TO_DISK = True #jtcheckpoint True
        if WRITE_CACHE_TO_DISK:
            try:
                pickle_file = open(pickle_file_path, 'wb')
                pickle.dump(av2_sensor_dataloader, pickle_file)
                pickle_file.close()
            except BaseException as e:
                print(e.args)
        return av2_sensor_dataloader


loader_helper = AV2DataLoaderHelper()


class AV2Sample():
    code_version = MODULE_VERSION
    MAX_MEMORY_USAGE_PROPORTION=0  # memory usage 是系统中所有进程的。超过这个比例，就不再缓存入内存。（0, 1]。#jtcheckpoint 0

    def __init__(self, log_id, image_folder_path, image_file_name, city_name, av2_sensor_dataloader):
        self.log_id = log_id
        self.timestamp = int(image_file_name[:-len(IMAGE_FILE_EXTENSION)])
        self.image_path = image_folder_path / image_file_name
        self.city_name = city_name
        self.location = av2_sensor_dataloader.get_city_SE3_ego(log_id, int(self.timestamp))

        self.speed = None
        self.velocity = None

        self.next_location = None
        self.previous_location = None
        self.next_timestamp = None
        self.previous_timestamp = None

        self.command = None
        self.trajectory = None
        self.image_bgr_tensor = None

    def get_image_tensor(self,crop_and_resize):
        """
        get image_bgr tensor
        The tensor is transformed from BGR image_bgr
        if image_tensor is None, then load it.
        @return: image_tensor
        """
        # check if in memory
        if self.image_bgr_tensor is not None:
            return self.image_bgr_tensor

        # check if in the disk
        READ_CACHE_FROM_DISK = False #jtcheckpoint False
        pickle_file_path = pathlib.Path(
            PICKLE_FOLDER_DIR + "image_bgr_tensor_" + self.log_id + "_" + str(self.timestamp) + '.pickle')
        if READ_CACHE_FROM_DISK and pickle_file_path.is_file():
            try:
                pickle_file = open(str(pickle_file_path), 'rb')
                image_bgr_tensor = pickle.load(pickle_file)  
                pickle_file.close()
                # print("AV2Sample: Read cache from disk succesfully." )
                if self.can_put_to_memory (): # put it to memory
                    self.image_bgr_tensor = image_bgr_tensor  

                return image_bgr_tensor
            except BaseException as e:
                print(e.args)

        # construct it

        image_bgr = cv2.imread(str(self.image_path))  # numpy; color channels are BGR

        # height,width,channels=image_bgr.shape
        # resize_ratio=0.179254
        # dimension = (int(width*resize_ratio), int(height*resize_ratio)) # WIDTH, HEIGHT!!!!!!!



        # crop and resize


        if crop_and_resize is None: # use default

            dimension = (400, 225)
            image_bgr = cv2.resize(image_bgr, dimension, interpolation=cv2.INTER_AREA)
        
        else: # use custom
            image_bgr=crop_and_resize(image_bgr)


        transform_function = transforms.ToTensor()
        image_bgr_tensor= transform_function(image_bgr)


        
        
        # put it to memory
        if self.can_put_to_memory ():
            self.image_bgr_tensor = image_bgr_tensor  

        WRITE_CACHE_TO_DISK = False #jtcheckpoint False
        if WRITE_CACHE_TO_DISK:
            try:
                pickle_file = open(pickle_file_path, 'wb')
                pickle.dump(image_bgr_tensor, pickle_file)
                pickle_file.close()
            except BaseException as e:
                print(e.args)

        return image_bgr_tensor


    def can_put_to_memory(self):
        usage=psutil.virtual_memory().percent/100
        if usage <self.MAX_MEMORY_USAGE_PROPORTION:
            return False   #jtcheckpoint False
        else:
            return False


class AV2Dataset():
    code_version = MODULE_VERSION
    cached_samples = {}  # cashed samples in memory

    # noinspection PyTypeChecker,PyUnusedLocal,PyDefaultArgument
    def __init__(self,
                 dataset_dir="/path_to_av2/train",
                 cities=[],
                 proportion=1,
                 crop_and_resize=None
                 ):
        """
        @param dataset_dir:
            directories supported:
            "/data/shared/av2_all/train"
            "/data/shared/av2_all/mini_train"
        @param cities:
            ['PIT','WDC','MIA','ATX','PAO','DTW']
        @param proportion:
            the ratio of data you'd like to load: (0, 1]

        """
        print("AV2Dataset:", MODULE_VERSION)

        dataset_path = pathlib.Path(dataset_dir)
        if not dataset_path.is_dir():
            raise Exception(dataset_dir + " not a valid dataset directory")

        # self.cityid_map={}
        self.cityid_map = {}
        self.cityid_map['PIT'] = 0
        self.cityid_map['WDC'] = 1
        self.cityid_map['MIA'] = 2
        self.cityid_map['ATX'] = 3
        self.cityid_map['PAO'] = 4
        self.cityid_map['DTW'] = 5

        SUPPORTED_CITIES = ['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW']
        for city_name in cities:
            if city_name not in SUPPORTED_CITIES:
                raise Exception(city_name + " not a valid city.")

        self.crop_and_resize=crop_and_resize
        self.samples = []
        av2_sensor_dataloader = loader_helper.get_dataloader(dataset_path)
        log_ids = os.listdir(dataset_path)
        progress = tqdm(total=len(log_ids))  # progress bar
        random.seed(999)
        for log_id in log_ids:
            progress.update(1)  # progress bar update 1

            city_name = av2_sensor_dataloader.get_city_name(log_id)
            if city_name not in cities:
                continue

            if random.random() > proportion:
                continue

            # check if in memory
            if log_id in self.cached_samples:
                self.samples.extend(self.cached_samples[log_id])
                # print("AV2Dataset: cache hit.")
                continue
            else:
                pass
                # print("AV2Dataset: cache missed.")

            # the parameters for trajectory
            num_future_points = 5
            stride = 10
            num_unused_samples = num_future_points * stride

            # check if saved in disk
            READ_CACHE_FROM_DISK = True #jtcheckpoint True
            pickle_file_path = pathlib.Path(PICKLE_FOLDER_DIR + "log_all_samples_" + log_id + '.pickle')
            if READ_CACHE_FROM_DISK and pickle_file_path.is_file():
                try:
                    pickle_file = open(str(pickle_file_path), 'rb')
                    log_all_samples = pickle.load(pickle_file)  # all samples for this log including used
                    pickle_file.close()
                    self.cached_samples[log_id] = log_all_samples[:-num_unused_samples]  # put it to memory
                    self.samples.extend(log_all_samples[:-num_unused_samples])  # ret the result
                    continue
                except BaseException as e:
                    print(e.args)

            image_folder_path = dataset_path / log_id / "sensors/cameras/ring_front_center/"
            image_filenames = sorted(os.listdir(image_folder_path))

            log_all_samples = []  # all samples for this log
            for index, image_filename in enumerate(image_filenames):
                sample = AV2Sample(log_id, image_folder_path, image_filename, city_name, av2_sensor_dataloader)
                log_all_samples.append(sample)

            # fill the next_timestamp and next_location
            for index in range(len(log_all_samples) - 1):
                log_all_samples[index].next_timestamp = log_all_samples[index + 1].timestamp
                log_all_samples[index].next_location = log_all_samples[index + 1].location

            # fill the previous_timestamp and previous_location
            for index in range(1, len(log_all_samples)):  #
                log_all_samples[index].previous_timestamp = log_all_samples[index - 1].timestamp
                log_all_samples[index].previous_location = log_all_samples[index - 1].location

            # compute the velocity
            for index, sample in enumerate(log_all_samples):
                if sample.previous_timestamp is None:
                    first_timestamp = sample.timestamp
                    first_location = sample.location
                    second_timestamp = sample.next_timestamp
                    second_location = sample.next_location

                elif sample.next_timestamp is None:
                    first_timestamp = sample.previous_timestamp
                    first_location = sample.previous_location
                    second_timestamp = sample.timestamp
                    second_location = sample.location

                else:
                    first_timestamp = sample.previous_timestamp
                    first_location = sample.previous_location
                    second_timestamp = sample.next_timestamp
                    second_location = sample.next_location

                sample.velocity = (second_location.translation - first_location.translation) / ((int(second_timestamp) - int(first_timestamp)) * 1e-9)

                sample.velocity = np.linalg.norm(sample.velocity)

            # compute the trajectory and command

            for index, sample in enumerate(log_all_samples[0:-num_unused_samples]):

                transform_matrix = sample.location.transform_matrix
                trajectory = []

                for future_point_index in range(num_future_points):
                    next_point_location = log_all_samples[index + (future_point_index + 1) * stride].location
                    next_point_transform_matrix = next_point_location.transform_matrix
                    tracks = np.zeros((1, 4))
                    tracks[:, -1] = 1.
                    world_tracks = np.dot(next_point_transform_matrix, np.transpose(tracks))
                    vehicle_tracks = np.dot(np.linalg.inv(transform_matrix), world_tracks)
                    tracks_x_y_z = vehicle_tracks[:3, :]
                    trajectory.append(tracks_x_y_z[[1, 0], 0])

                trajectory = np.vstack(trajectory)
                trajectory[:, 0] = -trajectory[:, 0]
                future_point_x, future_point_y = trajectory[-1, 0], trajectory[-1, 1]

                if future_point_y < 0:
                    future_point_y = 0.0

                radian = math.atan2(future_point_y, future_point_x)

                command = None
                # cmd start from 1
                if radian >= 0 and radian < (math.pi * 85 / 180):
                    command = 3
                elif radian >= (math.pi * 85 / 180) and radian < (math.pi * 95 / 180):
                    command = 2
                elif radian >= (math.pi * 95 / 180) and radian <= (math.pi * 180 / 180):
                    command = 1

                sample.trajectory = trajectory
                sample.command = command

            # save log_all_samples to disk
            WRITE_CACHE_TO_DISK = False #jtcheckpoint True
            if WRITE_CACHE_TO_DISK:
                try:
                    pickle_file = open(pickle_file_path, 'wb')
                    pickle.dump(log_all_samples, pickle_file)
                    pickle_file.close()
                except BaseException as e:
                    print(e.args)

            self.cached_samples[log_id] = log_all_samples[:-num_unused_samples]  # put it to memory
            self.samples.extend(log_all_samples[:-num_unused_samples])  # set the result

        print("AV2Dataset: There are", len(self.samples), "samples in the dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        city = sample.city_name
        cityid = self.cityid_map[city]
        return sample.get_image_tensor(self.crop_and_resize), str(sample.image_path), sample.trajectory, sample.command, sample.velocity, cityid

