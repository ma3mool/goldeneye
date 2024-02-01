import torch
from torchvision.models import get_model
import os
'''
    resnet18_l2_eps0.1
'''

def load_model(model_name):
    model= model_name.split('_')[0]
    ckpt_folder=f'/home/pfi/Documents/Data/CVPR_paper/Task1/{model}'
    print(model)
    MODEL = get_model(name=model, weights=None)
    ckpt_path=os.path.join(ckpt_folder, model_name)
    if 'baseline' in ckpt_path:
        ckpt_path=f'{ckpt_path}.pth'
    else:
        ckpt_path=f'{ckpt_path}.ckpt'
    # print('PATH',ckpt_path)
    ckpt=torch.load(ckpt_path,map_location='cpu')
    MODEL.load_state_dict(ckpt, strict=True)
    return MODEL
    
