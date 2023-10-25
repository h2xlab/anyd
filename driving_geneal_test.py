from utils_libs import *
from bev_utils import one_hot
from torch.utils.data import DataLoader
import tqdm
import numpy as np

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
num_workers = 0
import math

# --- Helper methods
weight = 0.8

from datasets.realworld_data.driving_dataset import DrivingData
CITY_DATA_MAP={}
AV2_CITIES = ['PIT','WDC','MIA','ATX','PAO','DTW']
NS_CITIES = ['BOS','SGP']
WM_CITIES = ["location_phx", "location_sf", "location_other"]
for city in AV2_CITIES:
    CITY_DATA_MAP[city] = DrivingData(av2_cities = [city] ,ns_cities=None,wm_cities=None, av2_root="your_test_root_av2")
for city in NS_CITIES:
    CITY_DATA_MAP[city] = DrivingData(av2_cities=None, ns_cities=[city], wm_cities=None,av2_root="your_test_root_av2",ns_train_test="TEST")
for city in WM_CITIES:
    if city == "location_sf":
        CITY_DATA_MAP[city] = DrivingData(av2_cities=None, ns_cities=None, wm_cities=[city],
                                               av2_root="your_test_root_av2", ns_train_test="TEST",
                                               waymo_train_test="TEST", waymo_p=1)
    else:
        CITY_DATA_MAP[city] = DrivingData(av2_cities=None, ns_cities=None, wm_cities=[city], av2_root="your_test_root_av2",ns_train_test="TEST", waymo_train_test="TEST")



def smooth_filter(arr):  # Weight between 0 and 1
    last = arr[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


# Check if the model has NaN values
def is_model_NaN(model):
    for name, param in model.named_parameters():
        isNan = torch.sum(torch.isnan(param.data)).item()
        if isNan > 0:
            return True
    return False


def get_mdl_params(model_list, n_par=0):
    if n_par == 0:
        for name, param in model_list[0].named_parameters():
            n_par += len(param.data.reshape(-1))
    param_mat = torch.zeros((len(model_list), n_par), dtype=torch.float32, device=device)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.detach().reshape(-1)
            # print(name)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)

    return param_mat

def get_mdl_params_wbn(model_list, n_par=0, n_par_bn=0):
    if n_par == 0:
        for name, param in model_list[0].named_parameters():
            n_par += len(param.data.reshape(-1))
    param_mat = torch.zeros((len(model_list), n_par), dtype=torch.float32, device=device)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.detach().reshape(-1)
            # print(name)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
            # if 'bn1.weight' or 'bn2.weight' in name:
    return param_mat


def get_acc_city_loss_geo(model, cities = ['PIT','WDC','MIA','ATX','PAO','DTW','BOS','SGP',"location_phx", "location_sf", "location_other"], ifsave=False, save_path=None, ifcon=False):
# def get_acc_city_loss_geo(model, cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW', 'BOS', 'SGP'], ifsave=False, save_path=None,ifcon=False):

    _num_workers = 0
    _batch_aug = 1
    batch_size = 1
    _augment = None
    # data = Argoverse_city(dataset_path=Path(dataroot))
    # data_val = DataLoader(data, batch_size=batch_size, num_workers=_num_workers,shuffle=False, drop_last=False, pin_memory=True)
    model.eval();
    model = model.to(device)
    _total_l1adeloss = 0
    _total_l2adeloss = 0
    _total_l1fdeloss = 0
    _total_l2fdeloss = 0

    num_cities = len(cities)
    city_ade = 0
    city_fde = 0

    # total = len(data_val)
    total = 0
    with torch.no_grad():
        for city in cities:
            data = CITY_DATA_MAP[city]
            data_val = DataLoader(data, batch_size=batch_size, num_workers=_num_workers, shuffle=False, drop_last=False,
                                  pin_memory=True)
            total_this_city = len(data_val)
            ade_this_city = 0
            fde_this_city = 0
            total += total_this_city
            iterator_tqdm = tqdm.tqdm(data_val, desc='Test', total=total)
            iterator = enumerate(iterator_tqdm)
            for i, (rgb_image, rgb_name, location_vehicle, command, speed, city_id) in iterator:
                # rgb_images, rgb_name, way_points, cmd, v
                rgb_image = rgb_image.to(device)
                location_vehicle = location_vehicle.type(torch.float32).to(device)
                location_vehicle_cl = location_vehicle.clone()
                command = one_hot(command).to(device)
                speed = speed.type(torch.float32)
                speed = speed.to(device)
                city_id = torch.LongTensor(city_id).to(device)
                if ifcon:
                    _pred_location, _  = model(rgb_image, command, speed, city_id)
                else:
                    _pred_location = model(rgb_image, command, speed, city_id)
                _pred_loc = _pred_location.clone()
                _pred_loc[:, :, 1] = _pred_loc[:, :, 1] + 1
                _norm = torch.FloatTensor([25, 25]).cuda()
                _pred_loc = _pred_loc * _norm
                l2_fde = torch.sqrt(((_pred_loc - location_vehicle_cl) ** 2).sum(axis=2))[:, -1][0].item()
                l2_ade = torch.sqrt(((_pred_loc - location_vehicle_cl) ** 2).sum(axis=2)).mean().item()
                _total_l2adeloss += l2_ade
                _total_l2fdeloss += l2_fde
                ade_this_city += l2_ade
                fde_this_city += l2_fde
            ade_this_city = ade_this_city / total_this_city
            fde_this_city = fde_this_city / total_this_city
            city_ade += ade_this_city
            city_fde += fde_this_city
            print(city, ' performance : l2 ade:', round(ade_this_city,5), ', l2 fde:', round(fde_this_city,5))
            if ifsave:
                with open(save_path + '/av2_%s_eval.txt'%(city), 'a') as f:
                    f.write('l2 ade: ' + str(ade_this_city) + ', l2 fde:' + str(fde_this_city) + "\n\n")
        # wandb.log({'l2 ade':ade, 'l2 fde': lde,'model_idx':ind})
        ade =  round(_total_l2adeloss / total, 5)
        fde =  round(_total_l2fdeloss / total, 5)

        ade_avg = round(city_ade/num_cities,5)
        fde_avg = round(city_fde/num_cities,5)
        if ifsave:
            with open(save_path + '/av2_all_eval.txt', 'a') as f:
                f.write('l2 ade: ' + str(ade) + ', l2 fde:' + str(fde) + "\n\n")
            with open(save_path + '/av2_all_eval_avg.txt', 'a') as f:
                f.write('l2 ade: ' + str(ade_avg) + ', l2 fde:' + str(fde_avg) + "\n\n")
        print('l2 ade:', ade, ', l2 fde:', fde)
        print('l2 avg ade:', ade_avg, ', l2 avg fde:', fde_avg)
    model.train()
    return ade, fde, ade_avg, fde_avg


