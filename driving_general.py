from utils_libs import *
from bev_utils import one_hot
from torch.utils.data import DataLoader
import tqdm

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_norm = 10
num_workers = 0

# --- Helper methods
weight = 0.8

from datasets.realworld_data.driving_dataset import DrivingData
CITY_DATA_MAP={}

AV2_CITIES = ['PIT','WDC','MIA','ATX','PAO','DTW']
NS_CITIES = ['BOS','SGP']
WM_CITIES = ["location_phx", "location_sf", "location_other"]
for city in AV2_CITIES:
    CITY_DATA_MAP[city] = DrivingData(av2_cities = [city] ,ns_cities=None,wm_cities=None, av2_root="/data/shared/av2_all/mini_test")
for city in NS_CITIES:
    CITY_DATA_MAP[city] = DrivingData(av2_cities=None, ns_cities=[city], wm_cities=None,av2_root="/data/shared/av2_all/mini_test",ns_train_test="TEST")
for city in WM_CITIES:
    if city == "location_sf":
        CITY_DATA_MAP[city] = DrivingData(av2_cities=None, ns_cities=None, wm_cities=[city],
                                               av2_root="/data/shared/av2_all/mini_test", ns_train_test="TEST",
                                               waymo_train_test="TEST", waymo_p=0.2)
    else:
        CITY_DATA_MAP[city] = DrivingData(av2_cities=None, ns_cities=None, wm_cities=[city], av2_root="/data/shared/av2_all/mini_test",ns_train_test="TEST", waymo_train_test="TEST")





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



def get_acc_city_loss_alltricks(model, cities = ['PIT','WDC','MIA','ATX','PAO','DTW','BOS','SGP',"location_phx", "location_sf", "location_other"], ifsave=False, save_path=None):
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
                _pred_location, _,_  = model(rgb_image, command, speed, city_id)
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





# --- Train methods
# def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
class LocationLoss(torch.nn.Module):
    def __init__(self, rg=25, offset=25, device='cuda'):
        super().__init__()
        self._norm = torch.FloatTensor([rg, rg]).to(device)
        self.offset = offset

    def forward(self, pred_locations, locations):
        locations = locations.to(pred_locations.device)
        locations[:, :, 1] = locations[:, :, 1] - self.offset
        locations = locations / self._norm
        # assert (locations <= 1).all() and (locations >= -1).all()
        return torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))

def train_model(model, train_loader, learning_rate, batch_size, K, print_per, weight_decay):
    '''
    This is the updating step of a local model of Federated Averaging.
    :param model:
    :param imgs: images of current client  (n x 3 x w x h)
    :param speed: speed of current client (n,)
    :param cmds: commands of current client (n,)
    :param wps: waypoints of current client (n, 2 ,5)
    :param learning_rate:
    :param batch_size:
    :param K:
    :param print_per:
    :param weight_decay:
    :param dataset_name:
    :return:
    '''
    # local_data = Localdata(imgs, speed, cmds, wps)
    # local_loader = torch.utils.data.DataLoader(local_data,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # # trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
    # #                                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = LocationLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    k = 0
    loss_list = []
    while (k < K):
        for imgs,_,wps, cmds, speed,city_id in train_loader:
        # for i, (rgb_image, rgb_name, location_vehicle, command, speed) in iterator:
        #     imgs  = imgs.to(device)
        #     wps = wps.type(torch.float32).to(device)
        #     cmds = one_hot(cmds).to(device)
        #     speed = speed.type(torch.float32)
        #     speed = speed.to(device)

            imgs = imgs.to(device)
            wps = wps.type(torch.float32).to(device)
            cmds = one_hot(cmds).to(device)
            speed = speed.type(torch.float32)
            speed = speed.to(device)
            # city_id = one_hot(city_id, num_digits=n_clnt, start=0).to(device)
            city_id = torch.LongTensor(city_id).to(device)
            _pred_location = model(imgs, cmds, speed, city_id)
            loss = criterion(_pred_location, wps)
            loss_mean = loss.mean()
            loss_list.append(loss_mean)
            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            k += 1
            if k == K:
                break
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    return model, loss_list


def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(params.data[idx:idx + length].reshape(weights.shape))
        idx += length

    mdl.load_state_dict(dict_param,strict=False)
    return mdl

def see_if_same(mdl1, mdl2):
    dict_param = copy.deepcopy(dict(mdl1.named_parameters()))
    ifresults = True
    for np1, np2 in zip(mdl1.named_parameters(),mdl2.named_parameters()):
        name1, param1 = np1[0],np1[1]
        name2, param2 = np2[0], np2[1]
        if (param2.cpu().detach().numpy() == param1.cpu().detach().numpy()).all():
            print('indeed same')
            ifresults = True
        else:
            print("Parameters %1 and %2 are different" % (param1, param2))

    # for name, param in mdl1.named_parameters():
    #     weights = param.data
    #     length = len(weights.reshape(-1))
    #     dict_param[name].data.copy_(params.data[idx:idx + length].reshape(weights.shape))
    #     idx += length
    #
    # mdl.load_state_dict(dict_param,strict=False)
    return ifresults

def set_client_from_params_np(mdl, params):
    params=torch.Tensor(params)
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(params[idx:idx + length].data.reshape(weights.shape))
        idx += length

    mdl.load_state_dict(dict_param,strict=False)
    return mdl




### FedDyn methods

####
def train_dyn_model(alpha, train_loader,lambda_model,server_model, model, learning_rate, batch_size, K, print_per, weight_decay):
    # local_data = Localdata(imgs, speed, cmds, wps)
    # local_loader = torch.utils.data.DataLoader(local_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # # trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
    #                                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    # for params in model.parameters():
    #     params.requires_grad = True
    criterion = LocationLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #     model.train()
    model = model.to(device)

    k = 0
    loss_list = []
    while (k < K):
        for imgs, _, wps, cmds, speed, city_id in train_loader:

            imgs = imgs.to(device)
            wps = wps.type(torch.float32).to(device)
            cmds = one_hot(cmds).to(device)
            speed = speed.type(torch.float32)
            speed = speed.to(device)
            city_id = torch.LongTensor(city_id).to(device)
            _pred_location = model(imgs, cmds, speed, city_id)

            optimizer.zero_grad()
            loss = criterion(_pred_location, wps)
            loss = loss.mean()
            loss_list.append(loss)
            # FedDyn version!
            # Add Dynamic loss
            # Get model parameter
            mld_pars = []
            for name, param in model.named_parameters():
                mld_pars.append(param.reshape(-1))
            mld_pars = torch.cat(mld_pars)
            loss_lambda = -torch.sum(mld_pars * lambda_model)
            loss_server = -alpha * torch.sum(mld_pars * server_model) + alpha / 2 * torch.sum(mld_pars * mld_pars)
            loss = loss + loss_lambda + loss_server
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            k += 1

            # if print_per != 0 and (k % print_per == 0):
            #     loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            #     print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, alpha: %.3f" % (k, acc_trn, loss_trn, alpha))
            #     model.train()

            if k == K:
                break

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model, loss_list




