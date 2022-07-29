from goldeneye.src.goldeneye import goldeneye
from ..pytorchfi.test.unit_tests.util_test import helper_setUp_CIFAR10
from goldeneye.src.util import *
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

        self.model2 = self.model1
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

    def test_uniform(num_sys_name):

        # Prepare goldeneye models for inference

        gmodel1 = goldeneye(
            self.model1,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
            quant=True,
            layer_max=[],
            inj_order=0,
            num_sys=getNumSysName(num_sys_name),
        )

        inf_model1 = gmodel1.declare_neuron_fi(
            function=gmodel1.apply_goldeneye_transformation
        )

        gmodel2 = goldeneye(
            self.model2,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
            quant=True,
            layer_max=[],
            inj_order=0,
            num_sys=getNumSysName(num_sys_name),
        )

        inf_model2 = gmodel1.declare_neuron_fi(
            function=gmodel1.apply_goldeneye_transformation
        )

        print("Testing uniform: ")
        print(inf_model2(self.images))

        print("Testing fake mixed: ")
        print(inf_model1(self.images))


if __name__ == "__main__":
    print("Testing mixed-precision...")
