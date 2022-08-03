from goldeneye.src.goldeneye import goldeneye
from ..pytorchfi.test.unit_tests.util_test import helper_setUp_CIFAR10
from ..src.num_sys_class import *
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

    def test_uniform(self, params):

        # Prepare goldeneye models for inference
        self.num_sys_name = params["num_sys_name"]

        gmodel1 = goldeneye(
            self.model1,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
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
            use_cuda=self.USE_GPU,
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


if __name__ == "__main__":
    print("Testing mixed-precision...")
