from nuscenes_dataset import NuScenesDataset
from av2_dataset import AV2Dataset
from waymo import WaymoDataset


class DrivingData():
    def __init__(self, av2_cities=None, ns_cities = None, wm_cities=None, av2_root="/data/shared/av2_all/train", ns_train_proportion=0.8, ns_train_test="TRAIN",waymo_train_test="TRAIN", waymo_p=1):
        if av2_cities is not None:
            self.set_av2(cities=av2_cities, dataset_dir=av2_root)
            # self.set_av2(cities=av2_cities, dataset_dir=av2_root, proportion=0.1)
            print("FederatedDataset: There are", len(self.av2_dataset), "av2 samples overall. ")
        else:
            self.av2_dataset=None

        if ns_cities is not None:
            self.set_nuscenes(cities=ns_cities,train_or_test=ns_train_test,train_proportion=ns_train_proportion)
            # self.set_nuscenes(cities=ns_cities, train_or_test=ns_train_test, train_proportion=0.1)
            print("FederatedDataset: There are", len(self.nuscenes_dataset), " ns samples overall. ")
        else:
            self.nuscenes_dataset = None

        if wm_cities is not None:
            if waymo_train_test == 'TRAIN':
                self.set_waymo(cities=wm_cities)
                # self.set_waymo(cities=wm_cities, proportion=0.05)
                print("FederatedDataset: There are", len(self.waymo_dataset), " ns samples overall. ")
            else:
                self.set_waymo(cities=wm_cities,tfrecords_dir="/data/shared/waymo_test", proportion=waymo_p,
                  images_dir="/home/data/waymo_test")
                # self.set_waymo(cities=wm_cities, tfrecords_dir="/data/shared/waymo_test", proportion=0.05,
                #                images_dir="/home/data/waymo_test")
        else:
            self.waymo_dataset = None



        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def set_av2(self,cities,crop_and_resize=None, dataset_dir="/data/shared/av2_all/train",proportion=1.0):
        """
        @param dataset_dir:
            directories supported:
            "/data/shared/av2_all/train"
            "/data/shared/av2_all/mini_train"
        @param cities:
            ['PIT','WDC','MIA','ATX','PAO','DTW'] 
            default: empty
        @param proportion:
            the ratio of data you'd like to load: (0, 1]

        """
        
        self.av2_dataset=AV2Dataset(crop_and_resize=crop_and_resize,cities=cities,dataset_dir=dataset_dir,proportion=proportion)


    def set_nuscenes(self,cities, random_seed=99,train_or_test="TRAIN",train_proportion=1.0):
        """
        @param cities:
            ['BOS','SGP'] 
            default: empty

        """
        self.nuscenes_dataset=NuScenesDataset(cities=cities,random_seed=random_seed,
                                              train_or_test=train_or_test,
                                              train_proportion=train_proportion)
        # print("FederatedDataset: There are",self.__len__(),"samples overall. " )

    def set_waymo(self, cities, tfrecords_dir="/data/shared/waymo_val", proportion=1.0,
                  images_dir="/home/data/waymo_val"):
        
        '''

        @param cities: ["location_phx","location_sf","location_other"]
        @param tfrecords_dir:
            supports "/data/shared/waymo_val" 
            "/data/shared/waymo_test" 
        @param images_dir:
            supports "/home/data/waymo_val"
            "/data/shared/waymo_test" 
        @param proportion: (0,1]
            default 1
        @return:
        '''


        self.waymo_dataset = WaymoDataset(cities=cities, proportion=proportion, tfrecords_dir=tfrecords_dir,
                                          images_dir=images_dir)
        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def __len__(self):
        
        
        length=0
        if self.av2_dataset is not None:
            length+=len(self.av2_dataset)
        
        if self.nuscenes_dataset is not None:
            length+=len(self.nuscenes_dataset)

        if self.waymo_dataset is not None:
            length += len(self.waymo_dataset)
        
        return length


    def __getitem__(self, index):

        if index < 0:
            index = index + self.__len__()

        if index < 0 or index >= self.__len__():
            raise Exception("Index", index, "out of range")

        if self.av2_dataset is not None:
            if index < len(self.av2_dataset):
                return self.av2_dataset[index]
            else:
                index = index - len(self.av2_dataset)

        if self.nuscenes_dataset is not None:
            if index < len(self.nuscenes_dataset):
                return self.nuscenes_dataset[index]
            else:
                index = index - len(self.nuscenes_dataset)

        if self.waymo_dataset is not None:
            if index < len(self.waymo_dataset):
                return self.waymo_dataset[index]
            else:
                index = index - len(self.waymo_dataset)
        
        
if __name__ == "__main__":
    # datasets = FederatedDataset(av2_cities = ['PIT','WDC','MIA','ATX','PAO','DTW'] ,ns_cities=['BOS','SGP'],av2_root = "/data/shared/av2_all/mini_test")
    av2_cities = ['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW']
    ns_cities = ['BOS', 'SGP']
    av2_root = "/data/shared/av2_all/train"
    datasets = DrivingData(av2_cities=av2_cities, ns_cities=ns_cities, av2_root=av2_root)
