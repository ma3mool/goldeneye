from goldeneye import goldeneye
from torch import nn
import timm

if __name__ == "__main__":

    model = timm.create_model("vit_base_patch16_224")

    inj_model = goldeneye(
        model,
        64,
        input_shape=[3, 224, 224],
        layer_types=[nn.Conv2d, nn.Linear],
        use_cuda=True,
        quant=True,
        layer_max=[],
        inj_order=1,
    )
