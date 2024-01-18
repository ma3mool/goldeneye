from pytorchfi.pytorchfi.error_models_num_sys import num_fixed_pt
import torch
import random
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.error_models import (
    single_bit_flip_func,
    random_inj_per_layer,
    random_inj_per_layer_batched,
    random_neuron_inj,
    random_neuron_inj_batched,
    random_neuron_single_bit_inj,
    random_neuron_single_bit_inj_batched,
)

from pytorchfi.error_models_num_sys import (
    single_bit_flip_func_with_num_sys,
    num_fp32,
    num_fp16,
    num_bfloat16,
)

from .util_test import helper_setUp_CIFAR10_same


class TestNeuronErrorModelsFuncFP32:
    """
    Testing neuron perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(1)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = single_bit_flip_func_with_num_sys(
            self.model,
            self.BATCH_SIZE,
            num_sys=num_fp32(),
            quant=False,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            bits=8,
        )
        self.ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453]

    def test_random_neuron_single_bit_inj_rand(self):
        random.seed(3)
        self.inj_model = random_neuron_single_bit_inj_batched(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_sameLoc(self):
        random.seed(2)
        self.inj_model = random_neuron_single_bit_inj_batched(
            self.p, self.ranges, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_single(self):
        random.seed(0)
        self.inj_model = random_neuron_single_bit_inj(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if not torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError


class TestNeuronErrorModelsFuncFP32:
    """
    Testing neuron perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(1)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = single_bit_flip_func_with_num_sys(
            self.model,
            self.BATCH_SIZE,
            num_sys=num_fp16(),
            quant=False,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            bits=8,
        )
        self.ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453]

    def test_random_neuron_single_bit_inj_rand(self):
        random.seed(3)
        self.inj_model = random_neuron_single_bit_inj_batched(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_sameLoc(self):
        random.seed(2)
        self.inj_model = random_neuron_single_bit_inj_batched(
            self.p, self.ranges, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_single(self):
        random.seed(0)
        self.inj_model = random_neuron_single_bit_inj(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if not torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError


class TestNeuronErrorModelsFuncFP32:
    """
    Testing neuron perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(1)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = single_bit_flip_func_with_num_sys(
            self.model,
            self.BATCH_SIZE,
            num_sys=num_bfloat16(),
            quant=False,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            bits=8,
        )
        self.ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453]

    def test_random_neuron_single_bit_inj_rand(self):
        random.seed(3)
        self.inj_model = random_neuron_single_bit_inj_batched(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_sameLoc(self):
        random.seed(2)
        self.inj_model = random_neuron_single_bit_inj_batched(
            self.p, self.ranges, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_single(self):
        random.seed(0)
        self.inj_model = random_neuron_single_bit_inj(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if not torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError


class TestNeuronErrorModelsFuncFP32:
    """
    Testing neuron perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(1)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = single_bit_flip_func_with_num_sys(
            self.model,
            self.BATCH_SIZE,
            num_sys=num_fixed_pt(),
            quant=False,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            bits=8,
        )
        self.ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453]

    def test_random_neuron_single_bit_inj_rand(self):
        random.seed(3)
        self.inj_model = random_neuron_single_bit_inj_batched(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_sameLoc(self):
        random.seed(2)
        self.inj_model = random_neuron_single_bit_inj_batched(
            self.p, self.ranges, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_single(self):
        random.seed(0)
        self.inj_model = random_neuron_single_bit_inj(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if not torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError
