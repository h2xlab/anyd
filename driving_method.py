import torch
from utils_libs import *
# from driving_general_geo import *
from driving_general import *
from datasets.realworld_data.driving_dataset import DrivingData
class LocationLoss(torch.nn.Module):
    def __init__(self, rg=25, offset=25, device='cuda', **kwargs):
        super().__init__()
        self._norm = torch.FloatTensor([rg, rg]).to(device)
        self.offset = offset

    def forward(self, pred_locations, locations):
        locations = locations.to(pred_locations.device)
        locations[:, :, 1] = locations[:, :, 1] - self.offset
        locations = locations / self._norm
        # assert (locations <= 1).all() and (locations >= -1).all()

        return torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))


class ContrastLocationLoss(torch.nn.Module):
    def __init__(self, rg=25, offset=25, device='cuda',temperature=0.1,mode='l2', **kwargs):
        super().__init__()
        self._norm = torch.FloatTensor([rg, rg]).to(device)
        self.offset = offset
        self.temperature = temperature
        self.mode = mode
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred_locations_list, locations, labels):
        locations = locations.to(pred_locations_list[0].device)
        locations[:, :, 1] = locations[:, :, 1] - self.offset
        locations = locations / self._norm
        locations = locations.view(locations.shape[0],10)

        pred_locations_list.append(locations)
        pred_locations = torch.stack(pred_locations_list, dim=1)

        branch_label = (torch.arange(3)+1).unsqueeze(0)
        branch_labels = branch_label.repeat(labels.shape[0],1).to(labels.device)
        branch_labels = torch.cat([branch_labels,labels.unsqueeze(1)],dim=1)  # now we have 48*4
        branched_mask = (branch_labels.unsqueeze(1)==branch_labels.unsqueeze(2)).float()
        # assert (locations <= 1).all() and (locations >= -1).all()
        if self.mode == 'l2':
            similarity_matrix = torch.cdist(pred_locations,pred_locations)
        elif self.mode == 'cos':
            pred_locations = F.normalize(pred_locations,dim=2)
            similarity_matrix = torch.bmm(pred_locations, torch.transpose(pred_locations,1,2))


        m = torch.eye(4, dtype=torch.bool).to(labels.device).unsqueeze(0).repeat(labels.shape[0], 1, 1)
        # remove the diagnal
        branched_mask = branched_mask[~m].view(labels.shape[0],4,-1)
        similarity_matrix = similarity_matrix[~m].view(similarity_matrix.shape[0],4, -1)

        positives = similarity_matrix[branched_mask.bool()].view(similarity_matrix.shape[0], -1)
        negatives = similarity_matrix[~branched_mask.bool()].view(similarity_matrix.shape[0], -1)

        if self.mode == 'l2':
            logits = torch.cat((positives,negatives),dim=1) * self.temperature
            p = self.softmax(-logits)
            loss = -torch.mean(torch.log(p[:, 0]))
        elif self.mode == 'cos':
            logits = torch.cat((positives, negatives), dim=1) / self.temperature
            p = self.softmax(-logits)
            loss = torch.mean(torch.log(p[:,0]))
        return loss


class ContrastFeatureLoss(torch.nn.Module):
    '''This is used on city-wise '''
    def __init__(self, device='cuda',temperature=0.1,mode='l2', **kwargs):
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, features, labels):
        '''
        labels are city labels, e.g. batch x 1
        features are batch x dim
        '''
        city_mask = (labels.unsqueeze(0)==labels.unsqueeze(1)).float() # batchsize x batchsize
        if self.mode == 'l2':
            similarity_matrix = torch.cdist(features.unsqueeze(0),features.unsqueeze(0)).squeeze(0)
        elif self.mode == 'l1':
            similarity_matrix = torch.cdist(features.unsqueeze(0), features.unsqueeze(0),p=1).squeeze(0)
        elif self.mode == 'cos':
            features = F.normalize(features,dim=1)
            similarity_matrix = torch.matmul(features, features.T)

        # remove the diagnal
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
        labels = city_mask[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)


        if self.mode == 'cos':
            logits = similarity_matrix / self.temperature
            p = self.softmax(-logits)
            positives = p*labels
            positives = torch.sum(positives, dim=1)
            loss = torch.mean(torch.log(positives[positives > 0]))
        else:
            logits = similarity_matrix * self.temperature
            p = self.softmax(-logits)
            positives = p * labels
            positives = torch.sum(positives, dim=1)
            loss = -torch.mean(torch.log(positives[positives>0]))
        return loss

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=10, contrast_mode='all',
                 base_temperature=10):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train_centralized_geo(n_clnt, select_num, learning_rate, batch_size, K, com_amount, print_per, weight_decay, lr_decay,
                 model_func, init_model, save_period,rand_seed=0, save_models=False, fast_exec=False,data_name='av2_ns_wm',av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
                    wm_cities=["location_phx", "location_sf", "location_other"],mode='geo'):
    suffix = mode+data_name+'CenAllSGD_S%d_SN%d_N%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (save_period, select_num, n_clnt, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    num_workers = 0


    data = DrivingData(av2_cities = av2_cities ,ns_cities=ns_cities, wm_cities=wm_cities)
    trn_loader =  DataLoader(data, batch_size=batch_size, num_workers=num_workers,shuffle=True, drop_last=False, pin_memory=True)

    # making a folder to load and save client and center models through training
    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    # trn_perf_all = np.zeros((n_cases, com_amount, 4));

    n_par = len(get_mdl_params([model_func()])[0])
    saved_itr = -1
    n_save_instances = int(com_amount / save_period)
    mdls= list(range(n_save_instances))

    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dmdls.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                model = model_func()
                model.load_state_dict(torch.load('/your_save_path/%s/%s/%dmdls.pt' % (data_name, suffix, i + 1)),strict=False)
                model.eval()
                model = model.to(device)
                # Freeze model
                for params in model.parameters():
                    params.requires_grad = False
                mdls[saved_itr // save_period] = model
    if saved_itr == -1:
        # the first epoch
        model = model_func().to(device)
        model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)

    criterion = LocationLoss()
    # contrastive_loss = SupConLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # if freeze backbone, change the optimizer
    # optimizer = torch.optim.SGD([{'params':model.city_encoding.parameters()},
    #                              {'params':model.geo_adapter.parameters()},
    #                              {'params':model.branched_conv.parameters()},
    #                              {'params':model.prediction.parameters()}], lr=learning_rate, weight_decay=weight_decay)

    # Check if there are past saved iterates
    k=0
    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10
    model.train()
    while (k< K*com_amount):
        for imgs,_,wps, cmds, speed, city_id in trn_loader:
            imgs  = imgs.to(device)
            wps = wps.type(torch.float32).to(device)
            cmds = one_hot(cmds).to(device)
            speed = speed.type(torch.float32)
            speed = speed.to(device)
            # city_id = one_hot(city_id, num_digits=n_clnt, start=0).to(device)
            city_id = torch.LongTensor(city_id).to(device)

            _pred_location = model(imgs, cmds, speed,city_id)
            loss = criterion(_pred_location, wps)

            # contrastive_loss(_pred_location, labels = cmds)
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            k += 1
            if k == K*com_amount:
                break
            # Freeze model

            if k==5 or ((k) % (save_period*K) == 0):
                print('Finishing round %d',k)
                for params in model.parameters():
                    params.requires_grad = False
                ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                               data_name, suffix), ifcon=False)
                if ade<best_ade:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_ade_model.pt' % (data_name, suffix))
                    best_ade = ade
                    print('We got best ade', ade)
                if fde<best_fde:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_model.pt' % (data_name, suffix))
                    best_fde = fde
                    print('We got best fde', fde)

                if ade_avg<best_avg_ade:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgade_model.pt' % (data_name, suffix))
                    best_avg_ade = ade_avg
                    print('We got best avg fde', ade_avg)

                if fde_avg<best_avg_fde:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_model.pt' % (data_name, suffix))
                    best_avg_fde = fde_avg
                    print('We got best avg fde', fde_avg)
                for params in model.parameters():
                    params.requires_grad = True
                model.train()
    return model


def train_centralized_geo_con_branch(n_clnt, select_num, learning_rate, batch_size, K, com_amount, print_per, weight_decay, lr_decay,
                 model_func, init_model, save_period,rand_seed=0, save_models=False, data_name='av2_ns_wm',av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
                    wm_cities=["location_phx", "location_sf", "location_other"],mode='geo',con_weight = 1e-3,con_temp=0.1, con_mode='l2'):
    suffix = mode+'av2_ns_CenAllSGD_geo_con_branch_CW%f_CT%f_S%d_SN%d_N%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (con_weight,con_temp,save_period, select_num, n_clnt, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    num_workers = 0

    data = DrivingData(av2_cities=av2_cities, ns_cities=ns_cities, wm_cities=wm_cities)
    trn_loader =  DataLoader(data, batch_size=batch_size, num_workers=num_workers,shuffle=True, drop_last=False, pin_memory=True)

    # making a folder to load and save client and center models through training
    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    saved_itr = -1
    n_save_instances = int(com_amount / save_period)
    mdls= list(range(n_save_instances))

    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dmdls.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                model = model_func()
                model.load_state_dict(torch.load('/your_save_path/%s/%s/%dmdls.pt' % (data_name, suffix, i + 1)),strict=False)
                model.eval()
                model = model.to(device)
                # Freeze model
                for params in model.parameters():
                    params.requires_grad = False
                mdls[saved_itr // save_period] = model
    if saved_itr == -1:
        # the first epoch
        model = model_func().to(device)
        model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)

    criterion = LocationLoss()
    contrastive_loss = ContrastLocationLoss(temperature=con_temp, mode=con_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Check if there are past saved iterates
    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10
    k=0
    model.train()
    while (k< K*com_amount):
        for imgs,_,wps, cmds, speed, city_id in trn_loader:
        # for i, (rgb_image, rgb_name, location_vehicle, command, speed) in iterator:
            imgs  = imgs.to(device)
            wps = wps.type(torch.float32).to(device)
            cmds_label = cmds.to(device)
            cmds = one_hot(cmds).to(device)

            speed = speed.type(torch.float32)
            speed = speed.to(device)
            # city_id = one_hot(city_id, num_digits=n_clnt, start=0).to(device)
            city_id = torch.LongTensor(city_id).to(device)

            _pred_location, pred_locations_list = model(imgs, cmds, speed,city_id)
            loss = criterion(_pred_location, wps)
            conloss= contrastive_loss(pred_locations_list,wps, labels = cmds_label)
            loss_mean = loss.mean()
            loss_mean = loss_mean + con_weight * conloss

            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            k += 1
            if k == K*com_amount:
                break
            # Freeze model

            if k==5 or ((k) % (save_period*K) == 0):
                print('Finishing round %d',k)
                for params in model.parameters():
                    params.requires_grad = False
                ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(model,ifsave=True,
                                                 save_path='/your_save_path/%s/%s' % (data_name, suffix),ifcon=True)
                if ade<best_ade:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_ade_model.pt' % (data_name, suffix))
                    best_ade = ade
                    print('We got best ade', ade)
                if fde<best_fde:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_model.pt' % (data_name, suffix))
                    best_fde = fde
                    print('We got best fde', fde)

                if ade_avg<best_avg_ade:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgade_model.pt' % (data_name, suffix))
                    best_avg_ade = ade_avg
                    print('We got best avg fde', ade_avg)

                if fde_avg<best_avg_fde:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_model.pt' % (data_name, suffix))
                    best_avg_fde = fde_avg
                    print('We got best avg fde', fde_avg)

                for params in model.parameters():
                    params.requires_grad = True
                model.train()

        # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
        # failure_arr = tst_perf_all[0, :, -1]
        # total_fails = failure_arr[np.max([0, i - int(1 / act_prob)]):i].sum()
        # print('Total failures in this round: %d' % tst_perf_all[0, i, -1])
        # if total_fails == int(act_prob * n_clnt) * int(1 / act_prob):
        #     break


    return model



def train_centralized_geo_con_city(n_clnt, select_num, learning_rate, batch_size, K, com_amount, print_per, weight_decay, lr_decay,
                 model_func, init_model, save_period,rand_seed=0, save_models=False,data_name='av2_ns_wm',av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
                    wm_cities=["location_phx", "location_sf", "location_other"],mode='geo',con_weight = 0.01,con_temp=0.1, con_mode='l2'):

    suffix = mode+con_mode+'geo_con_city_CW%f_CT%f_S%d_SN%d_N%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (con_weight,con_temp,save_period, select_num, n_clnt, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    num_workers = 0
    data = DrivingData(av2_cities=av2_cities, ns_cities=ns_cities, wm_cities=wm_cities)

    trn_loader =  DataLoader(data, batch_size=batch_size, num_workers=num_workers,shuffle=True, drop_last=False, pin_memory=True)

    # making a folder to load and save client and center models through training
    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    saved_itr = -1
    n_save_instances = int(com_amount / save_period)
    mdls= list(range(n_save_instances))

    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dmdls.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                model = model_func()
                model.load_state_dict(torch.load('/your_save_path/%s/%s/%dmdls.pt' % (data_name, suffix, i + 1)),strict=False)
                model.eval()
                model = model.to(device)
                # Freeze model
                for params in model.parameters():
                    params.requires_grad = False
                mdls[saved_itr // save_period] = model
    if saved_itr == -1:
        # the first epoch
        model = model_func().to(device)
        model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)

    criterion = LocationLoss()
    contrastive_loss = ContrastFeatureLoss(temperature=con_temp, mode=con_mode)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.city_encoding.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Check if there are past saved iterates
    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10
    k=0
    model.train()
    while (k< K*com_amount):
        for imgs,_,wps, cmds, speed, city_id in trn_loader:
        # for i, (rgb_image, rgb_name, location_vehicle, command, speed) in iterator:
            imgs  = imgs.to(device)
            wps = wps.type(torch.float32).to(device)
            # cmds_label = cmds.to(device)
            cmds = one_hot(cmds).to(device)

            speed = speed.type(torch.float32)
            speed = speed.to(device)
            # city_id = one_hot(city_id, num_digits=n_clnt, start=0).to(device)
            city_id = torch.LongTensor(city_id).to(device)

            _pred_location, city_attention = model(imgs, cmds, speed,city_id)
            loss = criterion(_pred_location, wps)
            conloss= contrastive_loss(city_attention,city_id)
            loss_mean = loss.mean()
            loss_mean = loss_mean + con_weight * conloss

            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer.step()
            k += 1
            if k == K*com_amount:
                break
            # Freeze model

            if k==5 or ((k) % (save_period*K) == 0):
                print('Finishing round %d',k)
                for params in model.parameters():
                    params.requires_grad = False
                ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(model,ifsave=True,
                                                 save_path='/your_save_path/%s/%s' % (data_name, suffix),ifcon=True)
                if ade<best_ade:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_ade_model.pt' % (data_name, suffix))
                    best_ade = ade
                    print('We got best ade', ade)
                if fde<best_fde:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_model.pt' % (data_name, suffix))
                    best_fde = fde
                    print('We got best fde', fde)

                if ade_avg<best_avg_ade:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgade_model.pt' % (data_name, suffix))
                    best_avg_ade = ade_avg
                    print('We got best avg fde', ade_avg)

                if fde_avg<best_avg_fde:
                    if save_models:
                        torch.save(model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_model.pt' % (data_name, suffix))
                    best_avg_fde = fde_avg
                    print('We got best avg fde', fde_avg)

                for params in model.parameters():
                    params.requires_grad = True
                model.train()
    return model



def train_FedAvg_GN(n_clnt, select_num, learning_rate, batch_size, K, com_amount, print_per, weight_decay, lr_decay,
                 model_func, init_model, save_period,rand_seed=0, save_models=False, data_name='av2_ns_wm',
                av2_cities = ['PIT','WDC','MIA','ATX','PAO','DTW'],ns_cities=['BOS','SGP'],wm_cities=["location_phx", "location_sf", "location_other"],mode='geoinput'):
    suffix =mode+ 'FinerAvgN%dcity_GEO_FedAvgCity_GN_SGD_S%d_SN%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (
    n_clnt,save_period, select_num, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    num_workers = 0
    city_loader_map = {}



    for city in av2_cities:
        data = DrivingData(av2_cities=[city], ns_cities=None,wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)
    for city in ns_cities:
        data = DrivingData(av2_cities=None, ns_cities=[city],wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)

    for city in wm_cities:
        data = DrivingData(av2_cities=None, ns_cities=None,wm_cities=[city])
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)


    cities = list(city_loader_map.keys())
    print('There are in total ',len(cities),' cities, namely:', cities)

    # city_encoder_list = []
    # for city in cities:
    #     city_encoder_list.append(init_model.city_encoding[0].weight)
    city_encoding = init_model.city_encoding[0].weight.clone().to(device)





    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))


    n_par = len(get_mdl_params([model_func()])[0])
    init_par_list = get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par



    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False
                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_all.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False
                fed_mdls_all[saved_itr // save_period] = fed_model

    # if (not os.path.exists('/your_save_path/%s/%s/%dcom_tst_perf_all.npy' % (data_name, suffix, com_amount))):
    avg_model = model_func().to(device)
    if saved_itr == -1:
        # the first epoch
        avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)
    else:
        # Load recent one
        avg_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, saved_itr + 1)),strict=False)


    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10

    for i in range(saved_itr + 1, com_amount):
        ### Fix randomness
        np.random.seed(i + rand_seed)
        clnt_list = np.arange(n_clnt)
        np.random.shuffle(clnt_list)
        selected_clnts = clnt_list[:int(select_num)]


        print('Selected Clients: %s' % (', '.join(['%s' % cities[item] for item in selected_clnts])))
        for clnt in selected_clnts:
            city = cities[clnt]
            print('---- Training client %s' % city)
            train_loader = city_loader_map[city]

            cur_model = model_func().to(device)
            cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())),strict=False)


            for params in cur_model.parameters():
                params.requires_grad = True
            # call to training on one clinent of data from driving_general.py
            cur_model,loss_list = train_model(model=cur_model, train_loader=train_loader, learning_rate=learning_rate * (lr_decay ** i),
                                    batch_size=batch_size, K=K, print_per=print_per, weight_decay=weight_decay)

            print("Round %3d, Loss before: %.4f, Loss after: %.4f, trained over %3d batches" % (i, loss_list[0], loss_list[-1], len(loss_list)))
            is_diverged = is_model_NaN(cur_model)
            if is_diverged:
                clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
            else:
                clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()

            city_encoding[clnt,:] = cur_model.city_encoding[0].weight.clone()[clnt,:]

        # Scale with weights

        avg_selected = np.mean(clnt_params_list[selected_clnts], axis=0)
        avg_model = set_client_from_params(model_func().to(device),torch.tensor(avg_selected, dtype=torch.float32).to(device))
        avg_model.city_encoding[0].weight.data = city_encoding

        avg_all = np.mean(clnt_params_list, axis=0)
        all_model = set_client_from_params(model_func().to(device),torch.tensor(avg_all, dtype=torch.float32).to(device))
        all_model.city_encoding[0].weight.data = city_encoding

        for params in avg_model.parameters():
            params.requires_grad = False
        if i==0:
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                               data_name, suffix), ifcon=False)
        if ((i + 1) % save_period == 0):
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                                   data_name, suffix), ifcon=False)
            if ade < best_ade:
                best_ade = ade
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_all.pt' % (data_name, suffix))
            if fde < best_fde:
                best_fde = fde
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_all.pt' % (data_name, suffix))
            if ade_avg < best_avg_ade:
                best_avg_ade = ade_avg
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_all.pt' % (data_name, suffix))
            if fde_avg < best_avg_fde:
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_all.pt' % (data_name, suffix))



    return fed_mdls_sel, fed_mdls_all


def train_FedAvg_GN_Plain(n_clnt, select_num, learning_rate, batch_size, K, com_amount, print_per, weight_decay, lr_decay,
                 model_func, init_model, save_period,rand_seed=0, save_models=False, data_name='av2_ns_wm',
                av2_cities = ['PIT','WDC','MIA','ATX','PAO','DTW'],ns_cities=['BOS','SGP'],wm_cities=["location_phx", "location_sf", "location_other"],mode='geoinput'):
    suffix =mode+ 'PlainN%dcity_GEO_FedAvgCity_GN_SGD_S%d_SN%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (
    n_clnt,save_period, select_num, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    num_workers = 0
    city_loader_map = {}



    for city in av2_cities:
        data = DrivingData(av2_cities=[city], ns_cities=None,wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)
    for city in ns_cities:
        data = DrivingData(av2_cities=None, ns_cities=[city],wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)

    for city in wm_cities:
        data = DrivingData(av2_cities=None, ns_cities=None,wm_cities=[city])
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)


    cities = list(city_loader_map.keys())
    print('There are in total ',len(cities),' cities, namely:', cities)

    # city_encoder_list = []
    # for city in cities:
    #     city_encoder_list.append(init_model.city_encoding[0].weight)
    # city_encoding = init_model.city_encoding[0].weight.clone().to(device)





    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))


    n_par = len(get_mdl_params([model_func()])[0])
    init_par_list = get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par



    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False
                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_all.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False
                fed_mdls_all[saved_itr // save_period] = fed_model

    # if (not os.path.exists('/your_save_path/%s/%s/%dcom_tst_perf_all.npy' % (data_name, suffix, com_amount))):
    avg_model = model_func().to(device)
    if saved_itr == -1:
        # the first epoch
        avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)
    else:
        # Load recent one
        avg_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, saved_itr + 1)),strict=False)


    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10

    for i in range(saved_itr + 1, com_amount):
        ### Fix randomness
        np.random.seed(i + rand_seed)
        clnt_list = np.arange(n_clnt)
        np.random.shuffle(clnt_list)
        selected_clnts = clnt_list[:int(select_num)]


        print('Selected Clients: %s' % (', '.join(['%s' % cities[item] for item in selected_clnts])))
        for clnt in selected_clnts:
            city = cities[clnt]
            print('---- Training client %s' % city)
            train_loader = city_loader_map[city]

            cur_model = model_func().to(device)
            cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())),strict=False)


            for params in cur_model.parameters():
                params.requires_grad = True
            # call to training on one clinent of data from driving_general.py
            cur_model,loss_list = train_model(model=cur_model, train_loader=train_loader, learning_rate=learning_rate * (lr_decay ** i),
                                    batch_size=batch_size, K=K, print_per=print_per, weight_decay=weight_decay)

            print("Round %3d, Loss before: %.4f, Loss after: %.4f, trained over %3d batches" % (i, loss_list[0], loss_list[-1], len(loss_list)))
            is_diverged = is_model_NaN(cur_model)
            if is_diverged:
                clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
            else:
                clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()

            # city_encoding[clnt,:] = cur_model.city_encoding[0].weight.clone()[clnt,:]

        # Scale with weights

        avg_selected = np.mean(clnt_params_list[selected_clnts], axis=0)
        avg_model = set_client_from_params(model_func().to(device),torch.tensor(avg_selected, dtype=torch.float32).to(device))
        # avg_model.city_encoding[0].weight.data = city_encoding

        avg_all = np.mean(clnt_params_list, axis=0)
        all_model = set_client_from_params(model_func().to(device),torch.tensor(avg_all, dtype=torch.float32).to(device))
        # all_model.city_encoding[0].weight.data = city_encoding

        for params in avg_model.parameters():
            params.requires_grad = False
        if i==0:
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                               data_name, suffix), ifcon=False)
        if ((i + 1) % save_period == 0):
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                                   data_name, suffix), ifcon=False)
            if ade < best_ade:
                best_ade = ade
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_all.pt' % (data_name, suffix))
            if fde < best_fde:
                best_fde = fde
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_all.pt' % (data_name, suffix))
            if ade_avg < best_avg_ade:
                best_avg_ade = ade_avg
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_all.pt' % (data_name, suffix))
            if fde_avg < best_avg_fde:
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_all.pt' % (data_name, suffix))



    return fed_mdls_sel, fed_mdls_all


def train_FedDyn_GN(n_clnt, select_num,alpha, learning_rate, batch_size, K, com_amount, print_per, weight_decay,
                   lr_decay, model_func, init_model, save_period, rand_seed=0, save_models=False, fast_exec=False,
                    data_name='av2_ns_wm',av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
                    wm_cities=["location_phx", "location_sf", "location_other"],mode='geoinput'
                    ):


    suffix = mode+'N%dcity_GEO_FedDynCity_GN_SGD_alpha%f_S%d_SN%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (
    n_clnt, alpha, save_period, select_num , learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)

    city_loader_map = {}
    for city in av2_cities:
        data = DrivingData(av2_cities=[city], ns_cities=None,wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)
    for city in ns_cities:
        data = DrivingData(av2_cities=None, ns_cities=[city],wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)

    for city in wm_cities:
        data = DrivingData(av2_cities=None, ns_cities=None,wm_cities=[city])
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)

    # making a folder to load and save client and center models through training
    cities = list(city_loader_map.keys())
    print('There are in total ', len(cities), ' cities, namely:', cities)

    city_encoding = init_model.city_encoding[0].weight.clone().to(device)

    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))



    n_par = len(get_mdl_params([model_func()])[0])
    lambda_model_list = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par


    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_all.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

    cld_model = model_func().to(device)
    avg_selected = model_func().to(device)

    if saved_itr == -1:
        cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)
        avg_selected = get_mdl_params([init_model], n_par)[0].cpu().numpy()
    else:
        cld_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_cld.pt' % (data_name, suffix, (saved_itr + 1))),strict=False)
        avg_selected.load_state_dict(
            torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, (saved_itr + 1))),strict=False)
        avg_selected = get_mdl_params([avg_selected], n_par)[0].cpu().numpy()
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0].cpu().numpy()

    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10

    for i in range(saved_itr + 1, com_amount):
        ### Fix randomness
        np.random.seed(i + rand_seed)
        clnt_list = np.arange(n_clnt)
        np.random.shuffle(clnt_list)
        selected_clnts = clnt_list[:int(select_num)]

        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
        server_model = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
        server_model_object = set_client_from_params(model_func().to(device), server_model)

        for clnt in selected_clnts:
            city = cities[clnt]
            print('---- Training client %s' % city)
            train_loader = city_loader_map[city]
            cur_model = model_func().to(device)
            cur_model.load_state_dict(copy.deepcopy(dict(server_model_object.named_parameters())), strict=False)


            for params in cur_model.parameters():
                params.requires_grad = True

            lambda_model = torch.tensor(lambda_model_list[clnt], dtype=torch.float32, device=device)
            cur_model, loss_list = train_dyn_model(alpha = alpha, train_loader =train_loader,lambda_model = lambda_model, server_model=server_model,
                                               model=cur_model,learning_rate=learning_rate * (lr_decay ** i),
                                               batch_size=batch_size, K=K, print_per=print_per,
                                               weight_decay=weight_decay)


            print("Round %3d, Loss before: %.4f, Loss after: %.4f, trained over %3d batches" % (i, loss_list[0], loss_list[-1], len(loss_list)))
            city_encoding[clnt, :] = cur_model.city_encoding[0].weight.clone()[clnt, :]
            is_diverged = is_model_NaN(cur_model)
            if is_diverged:
                # If model has NaN do not update the list put the average model
                # clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
                clnt_params_list[clnt] = np.copy(avg_selected)
            else:
                clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                lambda_model_list[clnt] = lambda_model_list[clnt] - alpha * (clnt_params_list[clnt] - cld_mdl_param)


        # Scale with weights
        avg_selected = np.mean(clnt_params_list[selected_clnts], axis=0)
        avg_model = set_client_from_params(model_func().to(device),torch.tensor(avg_selected, dtype=torch.float32).to(device))
        avg_model.city_encoding[0].weight.data = city_encoding

        avg_all = np.mean(clnt_params_list, axis=0)
        all_model = set_client_from_params(model_func().to(device),torch.tensor(avg_all, dtype=torch.float32).to(device))
        all_model.city_encoding[0].weight.data = city_encoding
        cld_mdl_param = avg_selected - 1 / alpha * np.mean(lambda_model_list, axis=0)

        # Freeze model
        for params in avg_model.parameters():
            params.requires_grad = False
        if i == 0:
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                                   data_name, suffix), ifcon=False)
        if ((i + 1) % save_period == 0):
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                                   data_name, suffix), ifcon=False)
            if ade < best_ade:
                best_ade = ade
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_all.pt' % (data_name, suffix))
            if fde < best_fde:
                best_fde = fde
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_all.pt' % (data_name, suffix))
            if ade_avg < best_avg_ade:
                best_avg_ade = ade_avg
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_all.pt' % (data_name, suffix))
            if fde_avg < best_avg_fde:
                best_avg_fde = fde_avg
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_all.pt' % (data_name, suffix))


    # return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all
    return fed_mdls_sel, fed_mdls_all


def train_FedDyn_GN_Plain(n_clnt, select_num,alpha, learning_rate, batch_size, K, com_amount, print_per, weight_decay,
                   lr_decay, model_func, init_model, save_period, rand_seed=0, save_models=False, fast_exec=False,
                    data_name='av2_ns_wm',av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
                    wm_cities=["location_phx", "location_sf", "location_other"],mode='geoinput'
                    ):


    suffix = mode+'PlainN%dcity_GEO_FedDynCity_GN_SGD_alpha%f_S%d_SN%d_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' % (
    n_clnt, alpha, save_period, select_num , learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)

    city_loader_map = {}
    for city in av2_cities:
        data = DrivingData(av2_cities=[city], ns_cities=None,wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)
    for city in ns_cities:
        data = DrivingData(av2_cities=None, ns_cities=[city],wm_cities=None)
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)

    for city in wm_cities:
        data = DrivingData(av2_cities=None, ns_cities=None,wm_cities=[city])
        city_loader_map[city] = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=False, pin_memory=True)

    # making a folder to load and save client and center models through training
    cities = list(city_loader_map.keys())
    print('There are in total ', len(cities), ' cities, namely:', cities)

    # city_encoding = init_model.city_encoding[0].weight.clone().to(device)

    if (not os.path.exists('/your_save_path/%s/%s' % (data_name, suffix))):
        os.mkdir('/your_save_path/%s/%s' % (data_name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))



    n_par = len(get_mdl_params([model_func()])[0])
    lambda_model_list = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par


    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_all.pt' % (data_name, suffix, i + 1)),strict=False)
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

    cld_model = model_func().to(device)
    avg_selected = model_func().to(device)

    if saved_itr == -1:
        cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())),strict=False)
        avg_selected = get_mdl_params([init_model], n_par)[0].cpu().numpy()
    else:
        cld_model.load_state_dict(torch.load('/your_save_path/%s/%s/%dcom_cld.pt' % (data_name, suffix, (saved_itr + 1))),strict=False)
        avg_selected.load_state_dict(
            torch.load('/your_save_path/%s/%s/%dcom_sel.pt' % (data_name, suffix, (saved_itr + 1))),strict=False)
        avg_selected = get_mdl_params([avg_selected], n_par)[0].cpu().numpy()
    cld_mdl_param = get_mdl_params([cld_model], n_par)[0].cpu().numpy()

    best_ade = 10
    best_fde = 10
    best_avg_ade = 10
    best_avg_fde = 10

    for i in range(saved_itr + 1, com_amount):
        ### Fix randomness
        np.random.seed(i + rand_seed)
        clnt_list = np.arange(n_clnt)
        np.random.shuffle(clnt_list)
        selected_clnts = clnt_list[:int(select_num)]

        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
        server_model = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
        server_model_object = set_client_from_params(model_func().to(device), server_model)

        for clnt in selected_clnts:
            city = cities[clnt]
            print('---- Training client %s' % city)
            train_loader = city_loader_map[city]
            cur_model = model_func().to(device)
            cur_model.load_state_dict(copy.deepcopy(dict(server_model_object.named_parameters())), strict=False)


            for params in cur_model.parameters():
                params.requires_grad = True

            lambda_model = torch.tensor(lambda_model_list[clnt], dtype=torch.float32, device=device)
            cur_model, loss_list = train_dyn_model(alpha = alpha, train_loader =train_loader,lambda_model = lambda_model, server_model=server_model,
                                               model=cur_model,learning_rate=learning_rate * (lr_decay ** i),
                                               batch_size=batch_size, K=K, print_per=print_per,
                                               weight_decay=weight_decay)


            print("Round %3d, Loss before: %.4f, Loss after: %.4f, trained over %3d batches" % (i, loss_list[0], loss_list[-1], len(loss_list)))
            # city_encoding[clnt, :] = cur_model.city_encoding[0].weight.clone()[clnt, :]
            is_diverged = is_model_NaN(cur_model)
            if is_diverged:
                # If model has NaN do not update the list put the average model
                # clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
                clnt_params_list[clnt] = np.copy(avg_selected)
            else:
                clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                lambda_model_list[clnt] = lambda_model_list[clnt] - alpha * (clnt_params_list[clnt] - cld_mdl_param)


        # Scale with weights
        avg_selected = np.mean(clnt_params_list[selected_clnts], axis=0)
        avg_model = set_client_from_params(model_func().to(device),torch.tensor(avg_selected, dtype=torch.float32).to(device))
        # avg_model.city_encoding[0].weight.data = city_encoding

        avg_all = np.mean(clnt_params_list, axis=0)
        all_model = set_client_from_params(model_func().to(device),torch.tensor(avg_all, dtype=torch.float32).to(device))
        # all_model.city_encoding[0].weight.data = city_encoding
        cld_mdl_param = avg_selected - 1 / alpha * np.mean(lambda_model_list, axis=0)

        # Freeze model
        for params in avg_model.parameters():
            params.requires_grad = False
        if i == 0:
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                                   data_name, suffix), ifcon=False)
        if ((i + 1) % save_period == 0):
            ade, fde, ade_avg, fde_avg = get_acc_city_loss_geo(avg_model, ifsave=True,
                                                               save_path='/your_save_path/%s/%s' % (
                                                                   data_name, suffix), ifcon=False)
            if ade < best_ade:
                best_ade = ade
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_ade_all.pt' % (data_name, suffix))
            if fde < best_fde:
                best_fde = fde
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_fde_all.pt' % (data_name, suffix))
            if ade_avg < best_avg_ade:
                best_avg_ade = ade_avg
                if save_models:
                    torch.save(avg_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                               '/your_save_path/%s/%s/best_avgade_all.pt' % (data_name, suffix))
            if fde_avg < best_avg_fde:
                best_avg_fde = fde_avg
                if save_models:
                    torch.save(avg_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_sel.pt' % (data_name, suffix))
                    torch.save(all_model.state_dict(),
                                   '/your_save_path/%s/%s/best_avgfde_all.pt' % (data_name, suffix))


    # return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all
    return fed_mdls_sel, fed_mdls_all
