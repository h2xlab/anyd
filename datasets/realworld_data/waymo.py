import pathlib


import os
import math
import numpy as np
import cv2


from torchvision import transforms

# tf.enable_eager_execution()

from simple_waymo_open_dataset_reader import WaymoDataFileReader
import random

SUPPORTED_CITIES = ["location_phx", "location_sf", "location_other"]


def get_tfrecord(tfrecord_path):
    tfrecord = WaymoDataFileReader(tfrecord_path)
    return tfrecord


def get_frame(tfrecord, index):
    table = tfrecord.get_record_table()
    tfrecord.seek(table[index])
    frame = tfrecord.read_record()
    return frame


def get_length(tfrecord):
    table = tfrecord.get_record_table()

    return len(table)


def prompt(string):
    verbose = True  # jtcheckpoint False
    if verbose:
        print(string)


class Transform():
    def __init__(self):
        self.se3 = None
        self.translation = None
        self.rotation = None

    def from_frame_pose_transform(self, frame_pose_transform):
        self.se3 = np.array(frame_pose_transform).reshape(4, 4)
        self.rotation = self.se3[:3, :3]
        self.translation = self.se3[:3, 3]


class WaymoSample():
    def __init__(self):
        self.tfrecord_path = None
        self.data_index = None
        self.transform = None
        self.log_id = None
        self.timestamp_nanosec = None
        self.next_timestamp_nanosec = None
        self.previous_timestamp_nanosec = None
        self.next_transform = None
        self.previous_transform = None
        self.velocity = None
        self.trajectory = None
        self.command = None
        self.city = None
        self.radian = None
        self.image_path = None

    def get_image_numpy_bgr(self, frame=None):
        if frame is None:
            tfrecord = get_tfrecord(self.tfrecord_path)
            frame = get_frame(tfrecord, self.data_index)
        front_camera_image = frame.images[0]

        image_numpy_bgr = cv2.imdecode(np.fromstring(front_camera_image.image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


        return image_numpy_bgr

    def get_resized_image_numpy(self, frame=None):
        image_numpy_bgr = self.get_image_numpy_bgr(frame)
        dimension = (400, 225)
        image_numpy_bgr_resized = cv2.resize(image_numpy_bgr, dimension, interpolation=cv2.INTER_AREA)
        return image_numpy_bgr_resized

    def get_image_tensor_bgr(self):
        image_numpy_bgr = self.get_image_numpy_bgr()
        dimension = (400, 225)
        image_numpy_bgr_resized = cv2.resize(image_numpy_bgr, dimension, interpolation=cv2.INTER_AREA)

        # self.transform = torch.nn.Sequential(
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # )
        transform_function = transforms.ToTensor()
        image_bgr_tensor = transform_function(image_numpy_bgr_resized)
        # self.transform = torch.nn.Sequential(
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # )
        # image_bgr_tensor = self.transform(image_bgr_tensor)

        return image_bgr_tensor

    def get_image_tensor_bgr_from_disk(self):
        image_numpy_bgr_resized = cv2.imread(str(self.image_path))
        transform_function = transforms.ToTensor()

        image_bgr_tensor = transform_function(image_numpy_bgr_resized)
        # self.transform = torch.nn.Sequential(
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # )
        # image_bgr_tensor = self.transform(image_bgr_tensor)

        return image_bgr_tensor


class WaymoDataset():
    def __init__(self, cities, tfrecords_dir, images_dir, proportion=1,cityid_map=None):
        random.seed(9999)

        if cities is None or len(cities) == 0:
            raise Exception("Empty cities")

        for city in cities:
            if city not in SUPPORTED_CITIES:
                raise Exception(city + " invalid city")

        if cityid_map is None:
            self.cityid_map = {}
            self.cityid_map["location_phx"] = 8
            self.cityid_map["location_sf"] = 9
            self.cityid_map["location_other"] = 10
        else:
            self.cityid_map = cityid_map

        self.samples = []



        num_future_points = 5
        stride = 5  # 注意：av2 的幀率是 0.05s/幀，而 waymo 的幀率是 0.1s/幀。
        num_unused_samples = num_future_points * stride

        path = os.path.expanduser(tfrecords_dir)
        for entry in os.listdir(path):

            if random.random() > proportion:
                continue

            prompt("*******LOADING RECORD*****************************************************")
            tfrecord_path = os.path.join(path, entry)
            prompt(tfrecord_path)
            if tfrecord_path[-9:] != ".tfrecord":  # 这个根本就不是 .tfrecord 文件，跳过
                continue

            # tfrecord = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
            tfrecord = get_tfrecord(tfrecord_path)
            prompt("Length of this record:" + str(len(list(tfrecord))))

            # get the city:
            frame = None
            # for data in tfrecord:
            #     frame = open_dataset.Frame()
            #     frame.ParseFromString(bytearray(data.numpy()))
            #     break

            frame = get_frame(tfrecord=tfrecord, index=0)
            city = frame.context.stats.location
            prompt(city)
            if city not in SUPPORTED_CITIES:
                raise Exception("You should update the SUPPORTED LIST with " + city)
            if city in cities:
                prompt("City needed")
            else:  # 如果不需要这个城市，则跳过
                prompt("City dismissed")
                continue

            log_all_samples = []
            prompt("Start extending the list of samples")
            for data_index in range(get_length(tfrecord)):

                frame = get_frame(tfrecord=tfrecord, index=data_index)

                sample = WaymoSample()
                sample.city = city
                sample.log_id = frame.context.name  # frame.context.name 是否類似於 av2 的log id

                sample.tfrecord_path = tfrecord_path
                sample.data_index = data_index

                transform = Transform()
                transform.from_frame_pose_transform(frame.pose.transform)
                sample.transform = transform

                timestamp_microsec = frame.timestamp_micros
                timestamp_nanosec = int(timestamp_microsec * 1000)  # to nanoseconds. av2 也是 nanosecond.
                sample.timestamp_nanosec = timestamp_nanosec

                images_path = pathlib.Path(images_dir)
                log_images_path = images_path / sample.log_id
                image_path = log_images_path / (str(timestamp_nanosec) + ".jpg")
                sample.image_path = image_path

                SAVE_IMAGE_TO_DISK = False  # jtcheckpoint False
                if SAVE_IMAGE_TO_DISK:
                    try:

                        if not log_images_path.is_dir():
                            log_images_path.mkdir()

                        if not image_path.is_file():
                            resized_image_numpy = sample.get_resized_image_numpy(frame)
                            cv2.imwrite(str(image_path), resized_image_numpy)

                    except BaseException as e:
                        print(e.args)
                log_all_samples.append(sample)

            # fill the next_timestamp and next_transform
            for index in range(len(log_all_samples) - 1):
                log_all_samples[index].next_timestamp_nanosec = log_all_samples[index + 1].timestamp_nanosec
                log_all_samples[index].next_transform = log_all_samples[index + 1].transform

            # fill the previous_timestamp and previous_transform
            for index in range(1, len(log_all_samples)):  #
                log_all_samples[index].previous_timestamp_nanosec = log_all_samples[index - 1].timestamp_nanosec
                log_all_samples[index].previous_transform = log_all_samples[index - 1].transform

            # compute the velocity
            for index, sample in enumerate(log_all_samples):
                if sample.previous_timestamp_nanosec is None:
                    first_timestamp_nanosec = sample.timestamp_nanosec
                    first_transform = sample.transform
                    second_timestamp_nanosec = sample.next_timestamp_nanosec
                    second_transform = sample.next_transform

                elif sample.next_timestamp_nanosec is None:
                    first_timestamp_nanosec = sample.previous_timestamp_nanosec
                    first_transform = sample.previous_transform
                    second_timestamp_nanosec = sample.timestamp_nanosec
                    second_transform = sample.transform

                else:
                    first_timestamp_nanosec = sample.previous_timestamp_nanosec
                    first_transform = sample.previous_transform
                    second_timestamp_nanosec = sample.next_timestamp_nanosec
                    second_transform = sample.next_transform

                sample.velocity = (second_transform.translation - first_transform.translation) / (
                        (int(second_timestamp_nanosec) - int(first_timestamp_nanosec)) * 1e-9)
                sample.velocity = np.linalg.norm(sample.velocity)

            # compute the trajectory and command
            for index, sample in enumerate(log_all_samples[0:-num_unused_samples]):

                #                 transform_matrix = sample.location.transform_matrix
                transform_matrix = sample.transform.se3
                trajectory = []

                for future_point_index in range(num_future_points):
                    next_point_transform_matrix = log_all_samples[
                        index + (future_point_index + 1) * stride].transform.se3
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
                    command = 3  # turn right
                elif radian >= (math.pi * 85 / 180) and radian < (math.pi * 95 / 180):
                    command = 2  # go forward
                elif radian >= (math.pi * 95 / 180) and radian <= (math.pi * 180 / 180):
                    command = 1  # turn left

                sample.radian = radian
                sample.trajectory = trajectory
                sample.command = command

            self.samples.extend(log_all_samples[:-num_unused_samples])  # set the result
            prompt(
                "Extending the list of samples finished. extended by " + str(len(log_all_samples) - num_unused_samples))
        print("The waymo dataset size is " + str(self.__len__()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample.get_image_tensor_bgr_from_disk(), str(
            sample.image_path), sample.trajectory, sample.command, sample.velocity, self.cityid_map[sample.city]



