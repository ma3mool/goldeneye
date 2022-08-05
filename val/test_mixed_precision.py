from goldeneye.src.goldeneye import goldeneye
from ..pytorchfi.test.unit_tests.util_test import helper_setUp_CIFAR10
from ..src.num_sys_class import *
from tqdm import tqdm
import copy

# from goldeneye.src.util import *
from torch import nn
import timm


class TestMixedPrecision:
    """
    Testing mixed-precision.
    """

    def setup_class(self):
        # Prepare dataset and model
        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model1, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS)
        self.model1.eval()

        self.model2 = copy.deepcopy(self.model1)
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        # Preprocessing to get layer_max
        self.layer_min, self.layer_max, self.actual_max = gather_min_max_per_layer(
            self.model1, self.dataiter, self.BATCH_SIZE, precision="fp32"
        )

    def test_uniform(self, params):

        # Prepare goldeneye models for inference
        self.num_sys_name = params["num_sys_name"]

        gmodel1 = goldeneye(
            self.model1,
            self.BATCH_SIZE,
            input_shape=[3, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            layer_max=self.layer_max,
            quant=True,
            inj_order=0,
            num_sys=getNumSysName(self.num_sys_name),
        )

        inf_model1 = gmodel1.declare_neuron_fi(
            function=gmodel1.apply_goldeneye_transformation
        )

        gmodel2 = goldeneye(
            self.model2,
            self.BATCH_SIZE,
            input_shape=[3, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            layer_max=self.layer_max,
            quant=True,
            inj_order=0,
            num_sys=getNumSysName(self.num_sys_name),
        )

        inf_model2 = gmodel1.declare_neuron_fi(
            function=gmodel1.apply_goldeneye_transformation
        )

        print("Testing uniform: ")
        print(inf_model2(self.images))

        print("Testing fake mixed: ")
        print(inf_model1(self.images))


#################################################################
################### HELPER METHODS FOR NUMSYS ###################
#################################################################
def getNumSysName(name, bits=16, radix_up=5, radix_down=10, bias=None):
    # common number systems in PyTorch
    if name == "fp32":
        return num_fp32(), name
    if name == "INT":
        # assert getQuantize_en()
        return num_fp32(), name
    elif name == "fp16":
        return num_fp16(), name
    elif name == "bfloat16":
        return num_bfloat16(), name

    # generic number systems in PyTorch
    elif name == "fp_n":
        return num_float_n(exp_len=radix_up, mant_len=radix_down), name
    elif name == "fxp_n":
        return num_fixed_pt(int_len=radix_up, frac_len=radix_down), name
    elif name == "block_fp":
        return block_fp(bit_width=bits, exp_len=radix_up, mant_len=radix_down), name
    elif name == "adaptive_fp":
        return (
            adaptive_float(
                bit_width=bits, exp_len=radix_up, mant_len=radix_down, exp_bias=bias
            ),
            name,
        )

    else:
        sys.exit("Number format not supported")


#################################################################
# From preprocess.py, had an error importing util so copying it
# here for the moment
# TODO: fix this
#################################################################
activations = []


def save_activations(module, input, output):
    activations.append(output)


def gather_min_max_per_layer(
    model,
    data_iter,
    batch_size,
    precision="FP16",
    cuda_en=True,
    debug=False,
    verbose=False,
):
    global activations
    layer_max = torch.Tensor([])
    layer_min = torch.Tensor([])

    if cuda_en:
        layer_max = layer_max.cuda()
        layer_min = layer_min.cuda()
    if precision == "FP16":
        layer_max = layer_max.half()
        layer_min = layer_min.half()

    # register forward hook to the model
    handles = []
    for param in model.modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            handles.append(param.register_forward_hook(save_activations))

    # main loops to gather ranges
    processed_elements = 0
    batch_num = 0

    for input_data in tqdm(data_iter):

        # prepare the next batch for inference
        images, labels = input_data
        if cuda_en:
            images = images.cuda()
            labels = labels.cuda()
        if precision == "FP16":
            images = images.half()

        activations = []  # reset before every inference
        model(images)  # run an inference

        # Range gathering: iterate through each layer

        min_vals = (
            torch.Tensor(list(map(lambda layer: layer.min().item(), activations)))
            .cuda()
            .half()
        )
        max_vals = (
            torch.Tensor(list(map(lambda layer: layer.max().item(), activations)))
            .cuda()
            .half()
        )
        if batch_num == 0:
            layer_max = max_vals
            layer_min = min_vals
        else:
            layer_max = torch.max(layer_max, max_vals)
            layer_min = torch.min(layer_min, min_vals)

        processed_elements += len(labels)
        batch_num += 1
        torch.cuda.empty_cache()

    # remove hooks
    for i in range(len(handles)):
        handles[i].remove()
    del activations

    actual_max = torch.max(torch.abs(layer_min), torch.abs(layer_max))

    return layer_min, layer_max, actual_max


if __name__ == "__main__":
    print("Testing mixed-precision...")
