import random, sys
import logging
import numpy as np
import torch
import csv

sys.path.append("./pytorchfi")
from pytorchfi import core
from num_sys_class import *


class goldeneye(core.fault_injection):
    def __init__(
        self,
        model,
        batch_size,
        input_shape=None,
        layer_types=None,
        num_sys=None,
        quant=None,
        layer_max=[],
        inj_order=0,
        **kwargs
    ):
        # higher abstraction than pytorchfi, decides when to inject and converts all numbers
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(
            model,
            batch_size,
            input_shape=input_shape,
            layer_types=layer_types,
            **kwargs
        )

        # Golden eye specific
        self.num_sys = num_sys
        self.quant = quant
        self.LayerRanges = layer_max
        self.inj_order = inj_order
        self.bits = kwargs.get("bits", 8)

        # the order of injecting within the goldeneye transformation
        # 0 -> no injection, 1 -> between quantization and de-quantization, 2 -> after converting to the number system, 3 -> after dequantization, 4 -> after converting num sys

    def set_layer_max(self, data):
        self.LayerRanges = data

    def reset_layer_max(self, data):
        self.LayerRanges = []

    def get_layer_max(self, layer):
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

    def hook1_num_sys_inj1(self, in_num, bit_pos, to_inj):
        # Transform to num sys and bit flip if inj_order == 1
        flip_val = to_inj and (self.inj_order == 1 or not self.quant)
        out_num = self.num_sys.convert_numsys_flip(in_num, bit_pos, flip=flip_val)
        return out_num

    def hook2_quant(self, in_num, max_value):
        if self.quant:
            logging.info("orig value:", in_num)
            quantum = int((in_num / max_value) * ((2.0 ** (self.bits - 1))))
            twos_comple = self._twos_comp_shifted(quantum, self.bits)  # signed
            logging.info("quantum:", quantum)
            logging.info("twos_comple:", twos_comple)
            return twos_comple
        return in_num

    def hook3_inj2(self, in_num, bit_pos):
        if self.inj_order == 1:
            assert (
                self.quant
            ), "Injection at location 2 not allowed if quantization is off"
            # binary representation
            bits = bin(in_num)[2:]
            logging.info("bits:", bits)

            # sign extend 0's
            temp = "0" * (self.bits - len(bits))
            bits = temp + bits
            if len(bits) != self.bits:
                raise AssertionError
            logging.info("sign extend bits", bits)

            # flip a bit
            # use MSB -> LSB indexing
            if bit_pos >= self.bits:
                raise AssertionError

            bits_new = list(bits)
            bit_loc = self.bits - bit_pos - 1
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
            out_num = self._twos_comp(new_quantum, self.bits)
            return out_num
        return in_num

    def hook4_dequant(self, in_num, max_value):
        if self.quant:
            # get FP equivalent from quantum
            out_num = in_num * ((2.0 ** (-1 * (self.bits - 1))) * max_value)
            logging.info("new_value", out_num)
            return out_num
        return in_num

    def hook5_num_sys_inj3(self, in_num, bit_pos, to_inj):
        return self.num_sys.convert_numsys_flip(
            in_num, bit_pos, flip=(to_inj and (self.quant and self.inj_order == 3))
        )

    def get_tensor_value(self, tensor, b, dim1, dim2, dim3):
        assert b is not None, "Batch is set to None!"

        assert dim1 is not None, "Dim1 is set to None!"

        if dim2 is None:
            return tensor[b][dim1]

        if dim3 is None:
            return tensor[b][dim1][dim2]

        return tensor[b][dim1][dim2][dim3]

    def set_tensor_value(self, tensor, new_value, b, dim1, dim2, dim3):
        assert b is not None, "Batch is set to None!"

        assert dim1 is not None, "Dim1 is set to None!"

        if dim2 is None:
            tensor[b][dim1] = new_value
            return

        if dim3 is None:
            tensor[b][dim1][dim2] = new_value
            return

        tensor[b][dim1][dim2][dim3] = new_value

    def _flip_bit_goldeneye(self, orig_value, max_value, bit_pos=-1, to_inj=False):
        if to_inj == False:
            assert bit_pos == -1

        # save_type = orig_value.dtype
        # in_num = orig_value.item()
        in_num = orig_value

        # If injection pos = 0, no injection

        # Num sys conversion + flip if inj_order = 1

        out_num = self.hook1_num_sys_inj1(in_num, bit_pos, to_inj)

        # Quantization (num_sys to int8)

        out_num = self.hook2_quant(out_num, max_value)

        # Bit flip if inj_order == 1
        if to_inj:
            out_num = self.hook3_inj2(out_num, bit_pos)

        # Dequant

        out_num = self.hook4_dequant(out_num, max_value)

        # Num sys conversion + flip if inj_order = 3

        out_num = self.hook5_num_sys_inj3(out_num, bit_pos, to_inj)

        # return torch.tensor(out_num, dtype=save_type)
        return torch.tensor(out_num)

    def apply_goldeneye_transformation(self, module, input, output):
        corrupt_layer_set = self.get_corrupt_layer()
        range_max = self.get_layer_max(self.get_curr_layer())
        logging.info("curr_conv", self.get_curr_layer())
        logging.info("range_max", range_max)

        inj_list = list(
            filter(
                lambda x: corrupt_layer_set[x] == self.get_curr_layer(),
                range(len(corrupt_layer_set)),
            )
        )
        for i in inj_list:
            prev_value = self.get_tensor_value(
                output,
                self.CORRUPT_BATCH[i],
                self.CORRUPT_DIM1[i],
                self.CORRUPT_DIM2[i],
                self.CORRUPT_DIM3[i],
            )
            rand_bit = random.randint(0, self.bits - 1)
            logging.info("rand_bit", rand_bit)
            new_value = self._flip_bit_goldeneye(
                prev_value, range_max, rand_bit, to_inj=True
            )
            self.set_tensor_value(
                output,
                new_value,
                self.CORRUPT_BATCH[i],
                self.CORRUPT_DIM1[i],
                self.CORRUPT_DIM2[i],
                self.CORRUPT_DIM3[i],
            )

        baseDevice = output.get_device()
        # TO OPTIMIZE (??). Must move to CPU, then back to_device
        print("Got here. Starting Layer Num: %d" % (self.get_curr_layer()))
        if self.use_cuda:
            output = output.cpu()

        # output.apply_(
        #     lambda val: self._flip_bit_goldeneye(
        #         val, range_max, to_inj=False
        #     )
        # )

        # apply is too slow, we replaced it by tensor-operations-based functions
        output = self.num_sys.convert_numsys_tensor(output)

        if self.use_cuda:
            output = output.cuda()

        # TODO: Double check that this does not change injected values

        print("Got here. Done with Layer Num: %d" % (self.get_curr_layer()))
        self.updateLayer()
        if self.get_curr_layer() >= self.get_total_layers():
            self.reset_curr_layer()
