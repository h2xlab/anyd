
import os
import torch
data_name = 'av2'
n_client = 11
cities = ['PIT','WDC','MIA','ATX','PAO','DTW']
# cities = ['ATX','PAO']
###x
com_amount = 1000
select_num = 11
save_period = 30
batch_size = 48
save_models = True  # Save models if True
fast_exec = False  # Record only test performance if True
weight_decay = 1e-3
lr_decay = .997
# lr_decay = .999
learning_rate = 0.1
K = 5
print_per = 1

backbone ='resnet34'
pretrained = True
model_name = 'selfD'
# Model function, change output layer to be cls_per_client
ifgeo = False
ifconbranch = False





from models.geo_adapter import ImagePolicyModelSS
# mode ='GEO_CHANNLE_V3_5heads'
# print(mode)
model_func = lambda: ImagePolicyModelSS(backbone=backbone, pretrained=True, num_cities=11, n_heads=3)
print('load_model_finish!!!!')

#
model_func = lambda: ImagePolicyModelSS(backbone=backbone,pretrained=pretrained, num_cities=n_client)
init_model = model_func()
# init_model.load_state_dict(torch.load(model_path), strict=False)


# Initalise the model for all methods with a random seed or load it from a saved initial model

if not os.path.exists('/data/shared/av2_fed/model/%s/%s_init_mdl.pt' % (data_name, model_name)):
    if not os.path.exists('/data/shared/av2_fed/model/%s/' % (data_name)):
        print("Create a new directory")
        os.mkdir('/data/shared/av2_fed/model/%s/' % (data_name))
    torch.save(init_model.state_dict(), '/data/shared/av2_fed/model/%s/%s_init_mdl.pt' % (data_name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('/data/shared/av2_fed/model/%s/%s_init_mdl.pt' % (data_name, model_name)), strict=False)



save_models = True


# mode ='nofreeze'
mode ='centralized'
data_name = 'av2_ns_wm'
print(mode)
from driving_method import *
model_func = lambda: ImagePolicyModelSS(backbone=backbone, num_cities=n_client)
_ = train_centralized_geo(n_clnt=n_client, select_num=select_num, learning_rate=learning_rate,batch_size=batch_size, K=K,
                          com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,lr_decay=lr_decay,model_func=model_func,
                          init_model=init_model, save_period=save_period,save_models=save_models, fast_exec=False, data_name=data_name,
                          av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
                          wm_cities=["location_phx", "location_sf", "location_other"], mode=mode)



# train with contrastive loss
# model_func = lambda: ImagePolicyModelSS(backbone=backbone, num_cities=n_client)
# _ = train_centralized_geo_con_city(n_clnt=n_client, select_num=select_num, learning_rate=learning_rate,batch_size=batch_size, K=K,
#                           com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,lr_decay=lr_decay,model_func=model_func,
#                           init_model=init_model, save_period=save_period,save_models=save_models, data_name='av2_ns_wm',
#                           av2_cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW'], ns_cities=['BOS', 'SGP'],
#                           wm_cities=["location_phx", "location_sf", "location_other"], mode=mode)





