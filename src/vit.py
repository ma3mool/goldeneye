import torch
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
import os

def load_vit_model(model_name):
    ckpt_folder = '/home/pfi/Documents/Data/CVPR_paper/Task1b/vit'
    if model_name == "vit_b_16.pth":
        MODEL = vit_b_16(pretrained=False)
    elif model_name == "vit_b_32.pth":
        MODEL = vit_b_32(pretrained=False)
    elif model_name == "vit_l_16.pth":
        MODEL = vit_l_16(pretrained=False)
    elif model_name == "vit_l_32.pth":
        MODEL = vit_l_32(pretrained=False)
    else:
        raise ValueError("Unsupported ViT model name.")
    ckpt_path = os.path.join(ckpt_folder, model_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    MODEL.load_state_dict(ckpt, strict=True)
    return MODEL
