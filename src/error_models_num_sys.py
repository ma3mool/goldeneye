"""
pytorchfi.error_models provides different error models out-of-the-box for use.
"""

import random
import logging
import numpy as np
import torch
import csv
from pytorchfi import core


"""
helper functions
"""


def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model, layer=-1):
    if layer == -1:
        layer = random.randint(0, pfi_model.get_total_layers() - 1)

    c = random.randint(0, pfi_model.get_fmaps_num(layer) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(layer) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(layer) - 1)

    return (layer, c, h, w)


def random_weight_location(pfi_model, layer=-1):
    loc = []

    if layer == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_layers() - 1)
    else:
        corrupt_layer = layer
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi_model.get_original_model().named_parameters():
        if "features" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1

    if curr_layer != pfi_model.get_total_layers():
        raise AssertionError
    if len(loc) != 5:
        raise AssertionError

    return tuple(loc)


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


"""
Number Systems
"""


class _number_sys:
    # General class for number systems, used to bit_flip using a specific format
    def bit_flip(self, bit_arr, bit_ind):
        # bit_arr to bit_arr
        bit_arr[bit_ind] = "0" if int(bit_arr[bit_ind]) else "1"
        return bit_arr

    def real_to_format(self, num):
        raise NotImplementedError

    def format_to_real(self, bit_arr):
        raise NotImplementedError

    def single_bit_flip_in_format(self, num, bit_ind):
        bit_arr = self.real_to_format(num)

        assert bit_ind >= 0 and bit_ind < len(bit_arr), "bit index out of range"

        bit_arr_corrupted = self.bit_flip(bit_arr, bit_ind)

        return self.format_to_real(bit_arr_corrupted)

    def convert_numsys_flip(self, num, bit_ind, flip=False):
        bit_arr = self.real_to_format(num)

        if flip:
            bit_arr = self.bit_flip(bit_arr, bit_ind)

        return self.format_to_real(bit_arr)

    def real_to_format_to_real_tensor(input_tensor):
        return output.apply_(lambda val: num_fp32().convert_numsys_flip(val))

    # HELPER FUNCTIONS

    def quantize_float(float_arr, n_bits=8, n_exp=3, use_denorm=True):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = np.sign(float_arr)
        float_arr = abs(float_arr)

        # 2. limits the range of output float point
        min_exp = -(2 ** (n_exp - 1)) + 2
        max_exp = 2 ** (n_exp - 1) - 1

        min_value = 2 ** min_exp
        max_value = (2 ** max_exp) * (2 - 2 ** (-n_mant))
        # print(min_value, max_value)
        ## 2.1. reduce too small values to zero

        ## Handle qunatization on denormalization values
        ### extract denorm terms
        denorm = float_arr * (float_arr < min_value)
        # print(denorm)
        ## round Denormalization values
        denorm_min = (2 ** min_exp) * (2 ** (-n_mant))
        denorm_out = (denorm / denorm_min).round() * denorm_min
        # print(denorm_out)

        # Non denormal part
        float_arr[float_arr < min_value] = 0

        ## 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = np.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        # exp should not be larger than max_exp
        assert exp.max() <= max_exp

        power_exp = np.exp2(exp)
        ## 4. quantize mantissa
        scale = 2 ** (-n_mant)  ## e.g. 2 bit, scale = 0.25
        mant = ((mant / scale).round()) * scale

        float_out = sign * power_exp * mant

        ## include the denormalization
        if use_denorm == True:
            float_out += sign * denorm_out
        float_out = float_out.astype("float32")
        return float_out

    def int_to_bin(num):
        # integer to its binary representation
        return str(bin(num))[2:]

    def frac_to_bin(frac):
        # a fraction (form: 0.sth) into its binary representation
        # exp: 0.5 -> "1", 0.25 -> "01", 0.125 -> "001"

        # Declaring an empty string
        # to store binary bits.
        binary = str()

        # Iterating through
        # fraction until it
        # becomes Zero.
        while frac:

            # Multiplying fraction by 2.
            frac *= 2

            # Storing Integer Part of
            # Fraction in int_part.
            if frac >= 1:
                int_part = 1
                frac -= 1
            else:
                int_part = 0

            # Adding int_part to binary
            # after every iteration.
            binary += str(int_part)

        # Returning the binary string.
        return binary

    def bin_to_frac(frac_str):
        # a binary form to a fraction: "01" -> 0.25

        power_count = -1
        frac = 0

        for i in frac_str:
            frac += int(i) * pow(2, power_count)
            power_count -= 1

        # returning mantissa in 0.M form.
        return frac


class _ieee754(_number_sys):
    def __init__(
        self, exp_len=8, mant_len=23, bias=None, denorm=True, max_val=None, min_val=None
    ):
        self.exp_len = exp_len
        self.mant_len = mant_len
        self.bias = bias
        self.denorm = denorm
        self.max_val = max_val
        self.min_val = min_val

    def real_to_format(self, num):
        # compute bias
        if self.bias is None:
            self.bias = 2 ** (self.exp_len - 1) - 1

        # Handle denorm
        if not self.denorm and self.max_val is not None and self.min_val is not None:
            if num < self.min_val:
                num = 0
            elif num > self.max_val:
                num = self.max_val

        # real to bit_arr
        sign = "1" if num < 0 else "0"

        num = abs(num)

        # # Quantize using Thierry's code
        # num = _number_sys.quantize_float(
        #     np.array([num]),
        #     n_bits=self.exp_len + 1 + self.mant_len,
        #     n_exp=self.exp_len,
        #     use_denorm=self.denorm,
        # ).item()

        int_str = _number_sys.int_to_bin(int(num))
        frac_str = _number_sys.frac_to_bin(num - int(num))

        ind = int_str.index("1") if int_str.find("1") != -1 else 0

        # init values and and
        exp_str = "0" * self.exp_len
        if int_str != "0":
            dec_shift = len(int_str) - ind - 1  # decimal shift
            exp_str = _number_sys.int_to_bin(dec_shift + self.bias)

        mant_str = int_str[ind + 1 :] + frac_str
        # Zero padding
        exp_str = ("0" * (self.exp_len - len(exp_str))) + exp_str
        mant_str = (mant_str + ("0" * (self.mant_len - len(mant_str))))[: self.mant_len]

        # asserts
        assert len(exp_str) == self.exp_len, "exp_len unknown error"
        assert len(mant_str) == self.mant_len, "mant_len unknown error"

        return list("".join([sign, exp_str, mant_str]))

    def format_to_real(self, bit_arr):
        def mant_to_int(mantissa_str):
            # mantissa in 1.M form
            return _number_sys.bin_to_frac(mantissa_str) + 1

        # compute bias
        if self.bias is None:
            self.bias = 2 ** (self.exp_len - 1) - 1

        sign = pow(-1, int(bit_arr[0]))

        exp_str = "".join(bit_arr[1 : self.exp_len + 1])
        exp = int(exp_str, 2) - self.bias

        mant_str = "".join(bit_arr[self.exp_len + 1 :])
        mant = mant_to_int(mant_str)

        # Exceptions
        if exp_str == "0" * self.exp_len and mant_str == "0" * self.mant_len:
            return 0
        if exp_str == "1" * self.exp_len and mant_str == "0" * self.mant_len:
            return sign * float("inf")
        if exp_str == "1" * self.exp_len and mant_str != "0" * self.mant_len:
            return float("nan")

        # Handling denormals
        if exp_str == "0" * self.exp_len and mant_str != "0" * self.mant_len:
            if self.denorm:
                # denormalized
                mant -= 1
            else:
                # not using denormals (like in AdaptivFloat)
                mant = 0

        return sign * mant * pow(2, exp)


class num_fp32(_ieee754):
    def __init__(self):
        super(num_fp32, self).__init__()


class num_fp16(_ieee754):
    def __init__(self):
        super(num_fp16, self).__init__(exp_len=5, mant_len=10)


class num_bfloat16(_ieee754):
    def __init__(self):
        super(num_fp16, self).__init__(exp_len=8, mant_len=7)


class num_fixed_pt(_number_sys):
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(self, int_len=3, frac_len=3):
        self.int_len = int_len
        self.frac_len = frac_len

    def real_to_format(self, num):
        # sign-magnitude is used for representing the sign
        sign = "1" if num < 0 else "0"
        num = abs(num)
        int_str = _number_sys.int_to_bin(int(num))
        if len(int_str) > self.int_len:
            int_str = "1" * self.int_len

        frac_str = _number_sys.frac_to_bin(num - int(num))[: self.frac_len]

        # Zero padding
        int_str = ("0" * (self.int_len - len(int_str))) + int_str
        frac_str = frac_str + ("0" * (self.frac_len - len(frac_str)))

        return list(sign) + list(int_str) + list(frac_str)

    def format_to_real(self, bit_arr):
        int_str, frac_str = map(
            lambda arr: "".join(arr),
            (bit_arr[1 : self.int_len + 1], bit_arr[self.int_len + 1 :]),
        )
        sign = 1 if bit_arr[0] == "0" else -1
        return sign * (int(int_str, 2) + _number_sys.bin_to_frac(frac_str))


"""
Neuron Perturbation Models
"""


# single random neuron error in single batch element
def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    (layer, C, H, W) = random_neuron_location(pfi_model)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_neuron_fi(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    if not randLoc:
        (layer, C, H, W) = random_neuron_location(pfi_model)
    if not randVal:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (layer, C, H, W) = random_neuron_location(pfi_model)
        if randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi_model, min_val=-1, max_val=1):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi_model)
    for i in range(pfi_model.get_total_layers()):
        (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
        batch.append(b)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi_model.get_total_layers()):
        if not randLoc:
            (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
        if not randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi_model.get_total_batches()):
            if randLoc:
                (layer, C, H, W) = random_neuron_location(pfi_model, layer=i)
            if randVal:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, batch_size, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.bits = kwargs.get("bits", 8)
        self.LayerRanges = []

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    def _twos_comp_shifted(self, val, nbits):
        if val < 0:
            val = (1 << nbits) + val
        else:
            val = self._twos_comp(val, nbits)
        return val

    def _twos_comp(self, val, bits):
        # compute the 2's complement of int value val
        if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)  # compute negative value
        return val  # return positive value as is

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("orig value:", orig_value)

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("quantum:", quantum)
        logging.info("twos_comple:", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("bits:", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logging.info("sign extend bits", bits)

        # flip a bit
        # use MSB -> LSB indexing
        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("bits", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("out", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("new_value", new_value)

        return torch.tensor(new_value, dtype=save_type)

    def single_bit_flip_signed_across_batch(self, module, input, output):
        corrupt_conv_set = self.get_corrupt_layer()
        range_max = self.get_conv_max(self.get_curr_layer())
        logging.info("curr_conv", self.get_curr_layer())
        logging.info("range_max", range_max)

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_curr_layer(),
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                prev_value = output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]][
                    self.CORRUPT_DIM2[i]
                ][self.CORRUPT_DIM3[i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]][
                    self.CORRUPT_DIM2[i]
                ][self.CORRUPT_DIM3[i]] = new_value

        else:
            self.assert_inj_bounds()
            if self.get_curr_layer() == corrupt_conv_set:
                prev_value = output[self.CORRUPT_BATCH][self.CORRUPT_DIM1][
                    self.CORRUPT_DIM2
                ][self.CORRUPT_DIM3]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH][self.CORRUPT_DIM1][self.CORRUPT_DIM2][
                    self.CORRUPT_DIM3
                ] = new_value

        self.updateLayer()
        if self.get_curr_layer() >= self.get_total_layers():
            self.reset_curr_layer()


class single_bit_flip_func_with_num_sys(single_bit_flip_func):
    def __init__(
        self,
        model,
        batch_size,
        input_shape=None,
        num_sys=num_fp32(),
        quant=False,
        **kwargs
    ):
        super(single_bit_flip_func_with_num_sys, self).__init__(
            model, batch_size, input_shape=input_shape
        )
        self.num_sys = num_sys
        self.quant = quant

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        in_num = orig_value.item()
        if self.quant:
            logging.info("orig value:", orig_value)
            quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
            in_num = self._twos_comp_shifted(quantum, total_bits)  # signed
            logging.info("quantum:", quantum)
            logging.info("twos_comple:", in_num)

        # Bit flip using twos_comple as input

        out_num = self.num_sys.single_bit_flip_in_format(in_num, bit_pos)

        if self.quant:
            # get FP equivalent from quantum
            out_num = out_num * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
            logging.info("new_value", out_num)

        return torch.tensor(out_num, dtype=save_type)


def random_neuron_single_bit_inj_batched(pfi_model, layer_ranges, randLoc=True):
    pfi_model.set_conv_max(layer_ranges)
    batch, layer_num, c_rand, h_rand, w_rand = ([] for i in range(5))

    if not randLoc:
        (layer, C, H, W) = random_neuron_location(pfi_model)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (layer, C, H, W) = random_neuron_location(pfi_model)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi_model, layer_ranges):
    # TODO Support multiple error models via list
    pfi_model.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi_model)
    (layer, C, H, W) = random_neuron_location(pfi_model)

    return pfi_model.declare_neuron_fi(
        batch=[batch],
        layer_num=[layer],
        dim1=[C],
        dim2=[H],
        dim3=[W],
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


"""
Weight Perturbation Models
"""


def random_weight_inj(pfi_model, corrupt_conv=-1, min_val=-1, max_val=1):
    (layer, k, c_in, kH, kW) = random_weight_location(pfi_model, corrupt_conv)
    faulty_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_weight_fi(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zeroFunc_rand_weight(pfi_model):
    (layer, k, c_in, kH, kW) = random_weight_location(pfi_model)
    return pfi_model.declare_weight_fi(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData
