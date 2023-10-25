import torch
from models.geo_adapter import ImagePolicyModelSS
model_path = 'the_model_you_want_to_test'
backbone ='resnet34'
pretrained = False
model_func = lambda: ImagePolicyModelSS(backbone=backbone,pretrained=pretrained)
test_model = model_func()

test_model.load_state_dict(torch.load(model_path))
data_name = ''
n_client = 11
save_root = 'your_saved_path'
mode = ''
save_path = save_root+mode
print('save path is: ',save_path)
from driving_geneal_test import *
get_acc_city_loss_geo(test_model, save_path=save_path, ifsave=True)

