import torch
# import numpy as np
import random
from qtorch.quant import float_quantize, fixed_point_quantize, block_quantize


class _number_sys:
    # General class for number systems, used to bit_flip using a specific format
    def bit_flip(self, bit_arr, bit_ind):
        # interpret index from least significant bit
        bit_ind_LSB = len(bit_arr) - 1 - bit_ind

        # bit_arr to bit_arr
        bit_arr[bit_ind_LSB] = "0" if int(bit_arr[bit_ind_LSB]) else "1"
        return bit_arr

    def real_to_format(self, num):
        raise NotImplementedError

    def real_to_format_tensor(self, tensor):
        raise NotImplementedError

    def real_to_format_tensor_meta(self, tensor):
        raise NotImplementedError

    def format_to_real(self, bit_arr):
        raise NotImplementedError

    def format_to_real_tensor(self, tensor):
        return tensor.to(torch.float32)

    def single_bit_flip_in_format(self, num, bit_ind):
        bit_arr = self.real_to_format(num)

        assert bit_ind >= 0 and bit_ind < len(bit_arr), "bit index out of range"
        bit_arr_corrupted = self.bit_flip(bit_arr, bit_ind)

        return self.format_to_real(bit_arr_corrupted)

    def convert_numsys_flip(self, num, bit_ind, flip=False):
        # print("Index: ", bit_ind)
        # print("Original: ", num)
        bit_arr = self.real_to_format(num)
        # print("B4: BitArray: ", bit_arr)

        if flip:
            bit_arr = self.bit_flip(bit_arr, bit_ind)

        # print("A4: BitArray: ", bit_arr)
        # print("Faulty: ", self.format_to_real(bit_arr))
        return self.format_to_real(bit_arr)


    def convert_numsys_tensor(self, tensor, meta_inj=False):
        if meta_inj:
            return self.format_to_real_tensor(self.real_to_format_tensor_meta(tensor))
        else:
            return self.format_to_real_tensor(self.real_to_format_tensor(tensor))

    # HELPER FUNCTIONS

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

        ind = len(int_str) - 1 - int_str.index("1") if int_str.find("1") != -1 else 0

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
        assert len(exp_str) == self.exp_len, "exp_len unknown error: %d != %d" %(len(exp_str), self.exp_len)
        assert len(mant_str) == self.mant_len, "mant_len unknown error: %d != %d" %(len(mant_str), self.mant_str)

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
            return sign * float('inf')
        if exp_str == "1" * self.exp_len and mant_str != "0" * self.mant_len:
            return float('nan')

        # Handling denormals
        if exp_str == "0" * self.exp_len and mant_str != "0" * self.mant_len:
            if self.denorm:
                # denormalized
                mant -= 1
            else:
                # not using denormals (like in AdaptivFloat)
                mant = 0

        return sign * mant * pow(2, exp)

    def int_to_bitstream(self, num):
        # sign-magnitude is used for representing the sign
        sign = "1" if num < 0 else "0"
        num = abs(num)
        int_str = _number_sys.int_to_bin(int(num))
        if len(int_str) > self.exp_len:
            int_str = "1" * self.exp_len

        # Zero padding
        int_str = ("0" * (self.exp_len - len(int_str))) + int_str
        return list(int_str)

    def bitstream_to_int(self, bit_arr):
        exp_str = "".join(bit_arr[1: self.exp_len + 1])
        exp = int(exp_str, 2)
        return exp


class num_fp32(_ieee754):
    def __init__(self):
        super(num_fp32, self).__init__()

    def real_to_format_tensor(self, tensor):
        return tensor.to(torch.float32)


class num_fp16(_ieee754):
    def __init__(self):
        super(num_fp16, self).__init__(exp_len=5, mant_len=10)

    def real_to_format_tensor(self, tensor):
        return tensor.to(torch.float16)

class num_float_n(_ieee754):
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(
        self,
        exp_len=5,
        mant_len=10,
    ):
        super(num_float_n, self).__init__(exp_len=exp_len, mant_len=mant_len)

    def real_to_format_tensor(self, tensor):
        return float_quantize(tensor, exp=self.exp_len, man=self.mant_len)


class num_bfloat16(_ieee754):
    def __init__(self):
        super(num_bfloat16, self).__init__(exp_len=8, mant_len=7)

    def real_to_format_tensor(self, tensor):
        return tensor.to(torch.bfloat16)


class num_fixed_pt(_number_sys):
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(
        self,
        int_len=3,
        frac_len=3,
    ):
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

    def real_to_format_tensor(self, tensor):
        return fixed_point_quantize(tensor, 1 + self.int_len + self.frac_len, self.frac_len)

    def format_to_real(self, bit_arr):
        int_str, frac_str = map(
            lambda arr: "".join(arr),
            (bit_arr[1 : self.int_len + 1], bit_arr[self.int_len + 1 :]),
        )
        sign = 1 if bit_arr[0] == "0" else -1
        return sign * (int(int_str, 2) + _number_sys.bin_to_frac(frac_str))


# class block_fp(_number_sys):
class block_fp(_ieee754):
    # 1 bit for sign + len(integer part) + len(frac part)

    def __init__(self, bit_width=32, exp_len=8, mant_len=23):
        super(block_fp, self).__init__(exp_len=exp_len, mant_len=mant_len)
        self.bit_width = bit_width

    def real_to_format_tensor(self, tensor):
        return self.quant_bfloat(float_arr = tensor, n_bits = self.bit_width, n_exp = self.exp_len)

    def real_to_format_tensor_meta(self, tensor):
        return self.quant_bfloat_meta(float_arr = tensor, n_bits = self.bit_width, n_exp = self.exp_len)

    def quant_bfloat(self, float_arr, n_bits=8, n_exp=3):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2
        max_exp = 2 ** (n_exp - 1) - 1

        min_value = 2 ** min_exp
        max_value = (2 ** max_exp) * (2 - 2 ** (-n_mant))

        # print ("min_value is {}, max_value is {}".format(min_value, max_value))
        # Non denormal part
        float_arr[float_arr < min_value] = 0

        ## 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        # print ("--before adjust-- exp is {}".format(exp))
        # print ("--before adjust-- mant is {}".format(mant))

        shared_exp = exp.max()
        exp_diff = shared_exp - exp
        power_exp_diff = torch.exp2(exp_diff)
        mant_adj = mant / power_exp_diff

        exp_adj = torch.full(exp.shape, shared_exp, device=float_arr.device)

        # print ("shared_exp is {} and exp_diff is {}".format(shared_exp, exp_diff))
        # print ("exp_adj is {}".format(exp_adj))
        # print ("mant_adj is {}".format(mant_adj))

        # exp should not be larger than max_exp
        assert (shared_exp <= max_exp)
        power_exp = torch.exp2(exp_adj)

        ## 4. quantize mantissa
        scale = 2 ** (-n_mant)  ## e.g. 2 bit, scale = 0.25
        mant_adj = ((mant_adj / scale).round()) * scale

        bfloat_out = sign * power_exp * mant_adj
        # print ("bfloat_out: ", bfloat_out)

        return bfloat_out

    def quant_bfloat_meta(self, float_arr, n_bits=8, n_exp=3):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2
        max_exp = 2 ** (n_exp - 1) - 1

        min_value = 2 ** min_exp
        max_value = (2 ** max_exp) * (2 - 2 ** (-n_mant))

        # print ("min_value is {}, max_value is {}".format(min_value, max_value))
        # Non denormal part
        float_arr[float_arr < min_value] = 0

        ## 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        # print ("--before adjust-- exp is {}".format(exp))
        # print ("--before adjust-- mant is {}".format(mant))

        shared_exp = exp.max()

        ##### ERROR INJECTION INTO META DATA
        # get bit array of shared exp
        exp_str = self.int_to_bitstream(shared_exp)

        # flip a random bit
        bit_ind = random.randint(0, self.exp_len - 1)
        bit_arr = self.bit_flip(exp_str, bit_ind)

        # get numerical value
        shared_exp = self.bitstream_to_int(bit_arr)
        # print("SHARED EXP: ", shared_exp)
        ##### ERROR INJECTION INTO META DATA


        exp_diff = shared_exp - exp
        power_exp_diff = torch.exp2(exp_diff)
        mant_adj = mant / power_exp_diff

        exp_adj = torch.full(exp.shape, shared_exp, device=float_arr.device)

        # exp should not be larger than max_exp
        assert (shared_exp <= max_exp)
        power_exp = torch.exp2(exp_adj)

        ## 4. quantize mantissa
        scale = 2 ** (-n_mant)  ## e.g. 2 bit, scale = 0.25
        mant_adj = ((mant_adj / scale).round()) * scale

        bfloat_out = sign * power_exp * mant_adj
        # print ("bfloat_out: ", bfloat_out)

        return bfloat_out

# ADAPTIVE FLOAT
class adaptive_float(_ieee754):
    # 1 bit for sign + len(integer part) + len(frac part)
    def __init__(self, bit_width=32, exp_len=8, mant_len=23, exp_bias=None):
        super(adaptive_float, self).__init__(exp_len=exp_len, mant_len=mant_len)
        self.bit_width = bit_width
        self.exp_bias = exp_bias

    def real_to_format_tensor(self, tensor):
        return self.quantize_adaptivfloat(
                float_arr=tensor, n_bits=self.bit_width, n_exp=self.exp_len, bias=self.exp_bias
        )

    def real_to_format_tensor_meta(self, tensor):
        return self.quantize_adaptivfloat_meta(
                float_arr=tensor, n_bits=self.bit_width, n_exp=self.exp_len, bias=self.exp_bias
        )

    def quantize_adaptivfloat(self, float_arr, n_bits=8, n_exp=4, bias=None):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        bias_temp = torch.frexp(float_arr.max())[1] - 1
        bias = (2 ** (n_exp - 1) - 1) - bias_temp

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2 - bias
        max_exp = 2 ** (n_exp - 1) - 1 - bias

        min_value = 2. ** min_exp
        max_value = (2. ** max_exp) * (2 - 2 ** (-n_mant))
        # Non denormal part
        float_arr[float_arr < min_value] = 0

        ## 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        power_exp = torch.exp2(exp)
        ## 4. quantize mantissa
        scale = 2 ** (-n_mant)  ## e.g. 2 bit, scale = 0.25
        mant = ((mant / scale).round()) * scale

        float_out = sign * power_exp * mant
        return float_out

    def quantize_adaptivfloat_meta(self, float_arr, n_bits=8, n_exp=4, bias=None):
        n_mant = n_bits - 1 - n_exp
        # 1. store sign value and do the following part as unsigned value
        sign = torch.sign(float_arr)
        float_arr = torch.abs(float_arr)

        bias_temp = torch.frexp(float_arr.max())[1] - 1
        bias_in = (2 ** (n_exp - 1) - 1) - bias_temp

        print("Before: ", bias_in)
        # ERROR INJECTION INTO META =============
        # get bit array of shared exp
        exp_str = self.int_to_bitstream(bias_in)

        # flip a random bit
        bit_ind = random.randint(0, 7)
        bit_arr = self.bit_flip(exp_str, bit_ind)

        # get numerical value
        bias = self.bitstream_to_int(bit_arr)
        print("After: ",bias)
        # ERROR INJECTION INTO META =============

        # 2. limits the range of output float point
        min_exp = -2 ** (n_exp - 1) + 2 - bias
        max_exp = 2 ** (n_exp - 1) - 1 - bias

        min_value = 2. ** min_exp
        max_value = (2. ** max_exp) * (2 - 2 ** (-n_mant))
        # Non denormal part
        float_arr[float_arr < min_value] = 0

        ## 2.2. reduce too large values to max value of output format
        float_arr[float_arr > max_value] = max_value

        # 3. get mant, exp (the format is different from IEEE float)
        mant, exp = torch.frexp(float_arr)

        # 3.1 change mant, and exp format to IEEE float format
        # no effect for exponent of 0 outputs
        mant = 2 * mant
        exp = exp - 1

        power_exp = torch.exp2(exp)
        ## 4. quantize mantissa
        scale = 2 ** (-n_mant)  ## e.g. 2 bit, scale = 0.25
        mant = ((mant / scale).round()) * scale

        float_out = sign * power_exp * mant
        return float_out
