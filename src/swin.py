import torch
from torchvision.models import swin_transformer
import os

def load_swin_model(model_name):
    ckpt_folder = '/home/pfi/Documents/Data/CVPR_paper/Task1b/swin'
    model_name = model_name.split(".")[0]
    # Map the provided model name to the torchvision's Swin Transformer model function
    swin_model_map = {
        "swin_b_v1": swin_transformer.swin_b,
        "swin_b_v2": swin_transformer.swin_b,  # Assuming v2 refers to a different checkpoint of the same base model
        "swin_s_v1": swin_transformer.swin_s,
        "swin_s_v2": swin_transformer.swin_s,  # Assuming v2 refers to a different checkpoint of the same small model
        "swin_t_v1": swin_transformer.swin_t,
        "swin_t_v2": swin_transformer.swin_t   # Assuming v2 refers to a different checkpoint of the same tiny model
    }
    # Create an instance of the Swin model
    if model_name in swin_model_map.keys():
        MODEL = swin_model_map[model_name](pretrained=False)
    else:
        raise ValueError(f"Unsupported Swin model name: {model_name}")

    # Load the saved weights
    ckpt_path = os.path.join(ckpt_folder, model_name + ".pth")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    MODEL.load_state_dict(ckpt, strict=True)
    return MODEL
