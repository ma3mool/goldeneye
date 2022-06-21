from goldeneye.src.num_sys_class import *
from goldeneye.src.num_sys_class import _number_sys, _ieee754
import torch
import math


# _number_sys
def test_number_sys():
    test = _number_sys()

    # bit_flip tests
    assert(test.bit_flip(['1', '0', '1', '1', '1', '1'], 0)
           == ['1', '0', '1', '1', '1', '0'])
    assert(test.bit_flip(['1', '0', '1', '1', '1', '1'], 3)
           == ['1', '0', '0', '1', '1', '1'])
    assert(test.bit_flip(['1', '0', '1', '1', '1', '1'], 5)
           == ['0', '0', '1', '1', '1', '1'])

    # bitwidth of 6, 1 sign bit, 1 exponent bit, 4 mantissa bits
    fp6 = _ieee754(
        exp_len=1,
        mant_len=4,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # single_bit_flip_in_format
    assert(fp6.single_bit_flip_in_format(-0.9375, 0) == -0.875)
    assert(fp6.single_bit_flip_in_format(0.3125, 2) == 0.0625)

    # denormals
    assert(fp6.single_bit_flip_in_format(0, 4) == float('inf'))

    # NaN
    assert(math.isnan(fp6.single_bit_flip_in_format(0.5625, 4)))
    assert(math.isnan(fp6.single_bit_flip_in_format(-0.4375, 4)))

    # format_to_real_tensor
    test1 = torch.tensor([[-0.9375, 0.3125,  0.0],
                          [-0.4375, 0.5625, -0.0]])

    expected1 = torch.tensor([[-0.9375, 0.3125,  0.0],
                              [-0.4375, 0.5625, -0.0]])

    assert(torch.equal(fp6.format_to_real_tensor(test1), expected1))

    # convert_numsys_flip
    assert(fp6.convert_numsys_flip(-0.9375, 0) == -0.9375)
    assert(fp6.convert_numsys_flip(0.3125, 2, True) == 0.0625)

    # denormals
    assert(fp6.convert_numsys_flip(0, 4, True) == float('inf'))

    # NaN
    assert(fp6.convert_numsys_flip(0.5625, 4) == 0.5625)
    assert(math.isnan(fp6.convert_numsys_flip(-0.4375, 4, True)))


# IEEE754
def test_ieee754():
    # bitwidth of 8, 1 sign bit, 3 exponent bit, 4 mantissa bits
    fp8 = _ieee754(
        exp_len=3,
        mant_len=4,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # int and bitstream conversions
    assert(fp8.int_to_bitstream(6) == ['1', '1', '0'])
    assert(fp8.bitstream_to_int(['0', '1', '1', '0', '1', '0', '1']) == 6)

    assert(fp8.int_to_bitstream(3) == ['0', '1', '1'])
    assert(fp8.bitstream_to_int(['0', '0', '1', '1', '0', '0', '1']) == 3)


# num_fp32
def test_num_fp32():
    # bitwidth of 32, 1 sign bit, 8 exponent bits, 23 mantissa bits
    fp32 = num_fp32()

    # numbers
    assert(fp32.format_to_real(['1', '0', '1', '1', '1', '1', '1', '1', '0',
                                '0', '1', '0', '0', '0', '1', '0', '1', '0',
                                '1', '0', '1', '1', '1', '1', '1', '1', '0',
                                '1', '0', '0', '0', '0'])
           == -0.63549518585205078125)
    # assert(fp32.real_to_format())

    # denormals
    assert(fp32.format_to_real(['0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0']) == 0.0)
    assert(fp32.real_to_format(0.0) ==
           ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0'])

    assert(fp32.format_to_real(['1', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0']) == 0.0)

    # infinity
    assert(fp32.format_to_real(['0', '1', '1', '1', '1', '1', '1', '1', '1',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0'])
           == float('inf'))
    assert(fp32.format_to_real(['1', '1', '1', '1', '1', '1', '1', '1', '1',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0'])
           == float('-inf'))

    # NaN
    assert(math.isnan(fp32.format_to_real(
           ['0', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '1',
            '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '1', '0',
            '0', '0', '0', '1', '0', '0'])))
    assert(math.isnan(fp32.format_to_real(
           ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0',
            '0', '1', '1', '0', '0', '1', '1', '1', '0', '0', '0', '0', '1',
            '0', '0', '0', '0', '0', '0'])))

    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])

    expected1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                              [-1.14,  2.05,  1.01,  0.07],
                              [ 0.16, -0.03, -0.89, -0.87],
                              [-0.04, -0.39,  0.64, -2.89]])

    assert(torch.equal(fp32.real_to_format_tensor(test1), expected1))

    # int and bitstream conversions
    assert(fp32.int_to_bitstream(256) ==
            ['1', '1', '1', '1', '1', '1', '1', '1'])

    assert(fp32.int_to_bitstream(127) ==
           ['0', '1', '1', '1', '1', '1', '1', '1'])
    assert(fp32.bitstream_to_int(['0', '1', '1', '1', '1', '1', '1', '1'])
           == 127)

    # convert_numsys_tensor
    assert(torch.equal(fp32.convert_numsys_tensor(test1), expected1))


# num_fp16
def test_num_fp16():
    # bitwidth of 32, 1 sign bit, 8 exponent bits, 23 mantissa bits (FP32)
    fp16 = num_fp16()

    # numbers
    assert(fp16.format_to_real(['1', '0', '1', '1', '1', '1', '1', '0', '1',
                                '0', '1', '1', '1', '1', '0', '1'])
           == -1.6845703125)
    assert(fp16.real_to_format(-1.6845703125) ==
           ['1', '0', '1', '1', '1', '1', '1', '0', '1', '0', '1', '1', '1',
            '1', '0', '1'])

    # denormals
    assert(fp16.format_to_real(['0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0']) == 0.0)
    assert(fp16.format_to_real(['1', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0']) == 0.0)

    # infinity
    assert(fp16.format_to_real(['0', '1', '1', '1', '1', '1', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0'])
           == float('inf'))
    assert(fp16.format_to_real(['1', '1', '1', '1', '1', '1', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0', '0'])
           == float('-inf'))

    # NaN
    assert(math.isnan(fp16.format_to_real(['0', '1', '1', '1', '1', '1', '0',
                                           '0', '0', '1', '0', '1', '0', '0',
                                           '1', '0'])))

    assert(math.isnan(fp16.format_to_real(['1', '1', '1', '1', '1', '1', '0',
                                           '0', '0', '0', '0', '1', '1', '0',
                                           '0', '1'])))

    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])

    expected1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                              [-1.14,  2.05,  1.01,  0.07],
                              [ 0.16, -0.03, -0.89, -0.87],
                              [-0.04, -0.39,  0.64, -2.89]]).to(torch.float16)

    assert(torch.equal(fp16.real_to_format_tensor(test1), expected1))

    # int and bitstream conversions
    assert(fp16.int_to_bitstream(21) == ['1', '0', '1', '0', '1'])
    assert(fp16.bitstream_to_int(['0', '1', '0', '1', '0', '1', '0', '1', '0',
                                  '0', '0', '1', '1', '0', '1', '0']) == 21)


# num_float_n
def test_num_float_n():
    # bitwidth of 6, 1 sign bit, 1 exponent bit, 4 mantissa bits
    fp6 = num_float_n(
        exp_len=1,
        mant_len=4
    )

    # negative and positive numbers
    assert(fp6.format_to_real(['1', '0', '1', '1', '1', '1']) == -0.9375)
    assert(fp6.real_to_format(-0.9375) == ['1', '0', '1', '1', '1', '1'])
    assert(fp6.format_to_real(['0', '0', '0', '1', '0', '1']) == 0.3125)
    assert(fp6.real_to_format(0.3125) == ['0', '0', '0', '1', '0', '1'])

    # denormals
    assert(fp6.format_to_real(['0', '0', '0', '0', '0', '0']) == 0.0)
    assert(fp6.real_to_format(0.0) == ['0', '0', '0', '0', '0', '0'])
    assert(fp6.format_to_real(['1', '0', '0', '0', '0', '0']) == 0.0)

    # infinity
    assert(fp6.format_to_real(['0', '1', '0', '0', '0', '0']) == float('inf'))
    assert(fp6.format_to_real(['1', '1', '0', '0', '0', '0']) == float('-inf'))

    # NaN
    assert(math.isnan(fp6.format_to_real(['0', '1', '1', '0', '0', '1'])))
    assert(math.isnan(fp6.format_to_real(['1', '1', '0', '1', '1', '1'])))

    # bitwidth of 14, 1 sign bit, 4 exponent bits, 9 mantissa bits
    fp14 = num_float_n(
        exp_len=4,
        mant_len=9,
    )

    # negative and positive numbers
    assert(fp14.format_to_real(['1', '1', '1', '0', '0', '0', '1', '1', '0',
                                '1', '1', '0', '0', '0']) == -45.5)
    assert(fp14.format_to_real(['0', '0', '1', '1', '0', '1', '1', '1', '0',
                                '0', '1', '0', '0', '0']) == 0.9453125)

    # denormals
    assert(fp14.format_to_real(['0', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0']) == 0.0)
    assert(fp14.format_to_real(['1', '0', '0', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0']) == 0.0)

    # infinity
    assert(fp14.format_to_real(['0', '1', '1', '1', '1', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0']) == float('inf'))
    assert(fp14.format_to_real(['1', '1', '1', '1', '1', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0']) == float('-inf'))

    # NaN
    assert(math.isnan(fp14.format_to_real(['0', '1', '1', '1', '1', '0', '0',
                                           '0', '0', '0', '0', '0', '0', '1'])))
    assert(math.isnan(fp14.format_to_real(['1', '1', '1', '1', '1', '0', '0',
                                           '0', '0', '0', '0', '0', '1', '0'])))


# num_bfloat16
def test_num_bfloat16():
    bfloat = num_bfloat16()

    # numbers
    assert(bfloat.format_to_real(['1', '0', '1', '1', '1', '1', '1', '0', '1',
                                  '0', '1', '1', '1', '1', '0', '1'])
           == -0.369140625)

    # denormals
    assert(bfloat.format_to_real(['0', '0', '0', '0', '0', '0', '0', '0', '0',
                                  '0', '0', '0', '0', '0', '0', '0']) == 0.0)
    assert(bfloat.format_to_real(['1', '0', '0', '0', '0', '0', '0', '0', '0',
                                  '0', '0', '0', '0', '0', '0', '0']) == 0.0)

    # infinity
    assert(bfloat.format_to_real(['0', '1', '1', '1', '1', '1', '1', '1', '1',
                                  '0', '0', '0', '0', '0', '0', '0'])
           == float('inf'))
    assert(bfloat.format_to_real(['1', '1', '1', '1', '1', '1', '1', '1', '1',
                                  '0', '0', '0', '0', '0', '0', '0'])
           == float('-inf'))

    # NaN
    assert(math.isnan(bfloat.format_to_real(['0', '1', '1', '1', '1', '1', '1',
                                             '1', '1', '1', '0', '1', '0', '0',
                                             '1', '0'])))

    assert(math.isnan(bfloat.format_to_real(['1', '1', '1', '1', '1', '1', '1',
                                             '1', '1', '0', '0', '1', '1', '0',
                                             '0', '1'])))

    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])

    expected1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                              [-1.14,  2.05,  1.01,  0.07],
                              [ 0.16, -0.03, -0.89, -0.87],
                              [-0.04, -0.39,  0.64, -2.89]]).to(torch.bfloat16)

    assert(torch.equal(bfloat.real_to_format_tensor(test1), expected1))

    # int and bitstream conversions
    assert(bfloat.int_to_bitstream(255)
           == ['1', '1', '1', '1', '1', '1', '1', '1'])
    assert(bfloat.bitstream_to_int(['0', '1', '1', '1', '1', '1', '1', '1', '1',
                                    '0', '0', '1', '1', '0', '1', '0']) == 255)


# Fixed Point
def test_fixed():
    # bitwidth of 6, 1 sign bit, 2 integer bits, 3 fraction bits
    fixed6 = num_fixed_pt(
        int_len=2,
        frac_len=3
    )

    # negative and positive numbers
    assert(fixed6.format_to_real(['0', '1', '1', '0', '0', '1']) == 3.125)
    assert(fixed6.real_to_format(3.125) == ['0', '1', '1', '0', '0', '1'])

    assert(fixed6.format_to_real(['1', '1', '0', '0', '0', '0']) == -2.0)
    assert(fixed6.real_to_format(-2.0) == ['1', '1', '0', '0', '0', '0'])

    assert(fixed6.format_to_real(['1', '0', '0', '1', '0', '1']) == -0.625)
    assert(fixed6.real_to_format(-0.625) == ['1', '0', '0', '1', '0', '1'])

    assert(fixed6.format_to_real(['0', '1', '0', '1', '1', '1']) == 2.875)
    assert(fixed6.real_to_format(2.875) == ['0', '1', '0', '1', '1', '1'])

    assert(fixed6.format_to_real(['1', '0', '0', '0', '0', '0']) == -0.0)

    assert(fixed6.format_to_real(['1', '1', '1', '1', '1', '1']) == -3.875)
    assert(fixed6.real_to_format(-3.875) == ['1', '1', '1', '1', '1', '1'])

    # maximum
    assert(fixed6.real_to_format(8) == ['0', '1', '1', '0', '0', '0'])

    # bitwidth of 6, 1 sign bit, 3 integer bits, 2 fraction bits
    fixed6 = num_fixed_pt(
        int_len=3,
        frac_len=2
    )

    # negative and positive numbers
    assert(fixed6.format_to_real(['0', '1', '1', '0', '0', '1']) == 6.25)
    assert(fixed6.real_to_format(6.25) == ['0', '1', '1', '0', '0', '1'])

    assert(fixed6.format_to_real(['1', '1', '0', '0', '0', '0']) == -4.0)
    assert(fixed6.real_to_format(-4.0) == ['1', '1', '0', '0', '0', '0'])

    assert(fixed6.format_to_real(['1', '0', '0', '1', '0', '1']) == -1.25)
    assert(fixed6.real_to_format(-1.25) == ['1', '0', '0', '1', '0', '1'])

    assert(fixed6.format_to_real(['1', '0', '0', '0', '0', '0']) == -0.0)

    assert(fixed6.format_to_real(['0', '1', '0', '1', '1', '1']) == 5.75)
    assert(fixed6.real_to_format(5.75) == ['0', '1', '0', '1', '1', '1'])

    assert(fixed6.format_to_real(['1', '1', '1', '1', '1', '1']) == -7.75)
    assert(fixed6.real_to_format(-7.75) == ['1', '1', '1', '1', '1', '1'])

    # bitwidth of 10, 1 sign bit, 2 integer bits, 7 fraction bits
    fixed10 = num_fixed_pt(
        int_len=2,
        frac_len=7
    )

    # negative and positive numbers
    assert(fixed10.format_to_real(
           ['1', '1', '1', '1', '1', '1', '0', '0', '0', '0']) == -3.875)
    assert(fixed10.real_to_format(-3.875)
           == ['1', '1', '1', '1', '1', '1', '0', '0', '0', '0'])

    assert(fixed10.format_to_real(
           ['0', '1', '0', '0', '0', '1', '0', '0', '1', '1']) == 2.1484375)
    assert(fixed10.real_to_format(2.1484375)
           == ['0', '1', '0', '0', '0', '1', '0', '0', '1', '1'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '1', '1', '1', '0', '1', '1', '0']) == -1.921875)
    assert(fixed10.real_to_format(-1.921875)
           == ['1', '0', '1', '1', '1', '1', '0', '1', '1', '0'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '0', '0', '0', '0', '0', '0', '0']) == -1.0)
    assert(fixed10.real_to_format(-1.0)
           == ['1', '0', '1', '0', '0', '0', '0', '0', '0', '0'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '1', '1', '0', '0', '0', '0', '0']) == -1.75)
    assert(fixed10.real_to_format(-1.75)
           == ['1', '0', '1', '1', '1', '0', '0', '0', '0', '0'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '1', '0', '0', '1', '1', '1', '1']) == -1.6171875)
    assert(fixed10.real_to_format(-1.6171875)
           == ['1', '0', '1', '1', '0', '0', '1', '1', '1', '1'])

    # bitwidth of 10, 1 sign bit, 6 integer bits, 3 fraction bits
    fixed10 = num_fixed_pt(
        int_len=6,
        frac_len=3
    )

    # negative and positive numbers
    assert(fixed10.format_to_real(
           ['1', '1', '1', '1', '1', '1', '0', '0', '0', '0']) == -62.0)
    assert(fixed10.real_to_format(-62.0)
           == ['1', '1', '1', '1', '1', '1', '0', '0', '0', '0'])

    assert(fixed10.format_to_real(
           ['0', '1', '0', '0', '0', '1', '0', '0', '1', '1']) == 34.375)
    assert(fixed10.real_to_format(34.375)
           == ['0', '1', '0', '0', '0', '1', '0', '0', '1', '1'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '1', '1', '1', '0', '1', '1', '0']) == -30.75)
    assert(fixed10.real_to_format(-30.75)
           == ['1', '0', '1', '1', '1', '1', '0', '1', '1', '0'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '0', '0', '0', '0', '0', '0', '0']) == -16.0)
    assert(fixed10.real_to_format(-16.0)
           == ['1', '0', '1', '0', '0', '0', '0', '0', '0', '0'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '1', '1', '0', '0', '0', '0', '0']) == -28.0)
    assert(fixed10.real_to_format(-28.0)
           == ['1', '0', '1', '1', '1', '0', '0', '0', '0', '0'])

    assert(fixed10.format_to_real(
           ['1', '0', '1', '1', '0', '0', '1', '1', '1', '1']) == -25.875)
    assert(fixed10.real_to_format(-25.875)
           == ['1', '0', '1', '1', '0', '0', '1', '1', '1', '1'])

    # bitwidth of 23, 1 sign bit, 13 integer bits, 9 fraction bits
    fixed23 = num_fixed_pt(
        int_len=13,
        frac_len=9
    )

    # negative and positive numbers
    assert(fixed23.format_to_real(
           ['0', '0', '1', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1',
            '1', '1', '1', '0', '0', '0', '1', '0', '0', '0']) == 3803.765625)
    assert(fixed23.real_to_format(3803.765625)
           == ['0', '0', '1', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1',
               '1', '1', '1', '0', '0', '0', '1', '0', '0', '0'])

    assert(fixed23.format_to_real(
           ['1', '0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1',
            '0', '0', '0', '1', '0', '0', '0', '0', '0', '0']) == -446.125)
    assert(fixed23.real_to_format(-446.125)
           == ['1', '0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1',
               '0', '0', '0', '1', '0', '0', '0', '0', '0', '0'])

    assert(fixed23.format_to_real(
           ['0', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '0',
            '1', '1', '1', '0', '0', '1', '1', '1', '1', '0']) == 1797.80859375)
    assert(fixed23.real_to_format(1797.80859375)
           == ['0', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '0',
               '1', '1', '1', '0', '0', '1', '1', '1', '1', '0'])

    assert(fixed23.format_to_real(
           ['0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1',
            '0', '1', '1', '1', '0', '1', '0', '0', '0', '0']) == 3074.90625)
    assert(fixed23.real_to_format(3074.90625)
           == ['0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1',
               '0', '1', '1', '1', '0', '1', '0', '0', '0', '0'])

    assert(fixed23.format_to_real(
           ['0', '1', '1', '0', '0', '1', '1', '0', '0', '0', '0', '0', '1',
            '0', '1', '1', '1', '0', '0', '1', '1', '1', '0']) == 6530.90234375)
    assert(fixed23.real_to_format(6530.90234375)
           == ['0', '1', '1', '0', '0', '1', '1', '0', '0', '0', '0', '0', '1',
               '0', '1', '1', '1', '0', '0', '1', '1', '1', '0'])

    assert(fixed23.format_to_real(
           ['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '1', '0', '0',
            '0', '0', '1', '0', '1', '0', '0', '0', '1', '1'])
           == -8136.318359375)
    assert(fixed23.real_to_format(-8136.318359375)
           == ['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '1', '0', '0',
               '0', '0', '1', '0', '1', '0', '0', '0', '1', '1'])

    # bitwidth of 23, 1 sign bit, 15 integer bits, 7 fraction bits
    fixed23 = num_fixed_pt(
        int_len=15,
        frac_len=7
    )

    # negative and positive numbers
    assert(fixed23.format_to_real(
           ['0', '0', '1', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1',
            '1', '1', '1', '0', '0', '0', '1', '0', '0', '0']) == 15215.0625)
    assert(fixed23.real_to_format(15215.0625)
           == ['0', '0', '1', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1',
               '1', '1', '1', '0', '0', '0', '1', '0', '0', '0'])

    assert(fixed23.format_to_real(
           ['1', '0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1',
            '0', '0', '0', '1', '0', '0', '0', '0', '0', '0']) == -1784.5)
    assert(fixed23.real_to_format(-1784.5)
           == ['1', '0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1',
               '0', '0', '0', '1', '0', '0', '0', '0', '0', '0'])

    assert(fixed23.format_to_real(
           ['0', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '0',
            '1', '1', '1', '0', '0', '1', '1', '1', '1', '0']) == 7191.234375)
    assert(fixed23.real_to_format(7191.234375)
           == ['0', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '0',
               '1', '1', '1', '0', '0', '1', '1', '1', '1', '0'])

    assert(fixed23.format_to_real(
           ['0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1',
            '0', '1', '1', '1', '0', '1', '0', '0', '0', '0']) == 12299.625)
    assert(fixed23.real_to_format(12299.625)
           == ['0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1',
               '0', '1', '1', '1', '0', '1', '0', '0', '0', '0'])

    assert(fixed23.format_to_real(
           ['0', '1', '1', '0', '0', '1', '1', '0', '0', '0', '0', '0', '1',
            '0', '1', '1', '1', '0', '0', '1', '1', '1', '0']) == 26123.609375)
    assert(fixed23.real_to_format(26123.609375)
           == ['0', '1', '1', '0', '0', '1', '1', '0', '0', '0', '0', '0', '1',
               '0', '1', '1', '1', '0', '0', '1', '1', '1', '0'])

    assert(fixed23.format_to_real(
           ['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '1', '0', '0',
            '0', '0', '1', '0', '1', '0', '0', '0', '1', '1'])
           == -32545.2734375)
    assert(fixed23.real_to_format(-32545.2734375)
           == ['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '1', '0', '0',
               '0', '0', '1', '0', '1', '0', '0', '0', '1', '1'])


# Adaptive Float
def test_adaptive():
    # test tensors to use on different systems
    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])

    test2 = torch.tensor([[ 997.481,  188.034, -147.376, -277.766],
                          [-617.844, -755.696,   18.283,  670.539],
                          [-709.682, -841.260,  300.587,  837.047],
                          [ 347.082,   98.871, -775.379,  709.284]])

    # bitwidth of 4, 1 sign bit, 2 exponent bits, 1 mantissa bit
    adaptive4 = adaptive_float(
        bit_width=4,
        exp_len=2,
        mant_len=1,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.0,  3.0, -1.5,  0.0],
                              [-1.0,  2.0,  1.0,  0.0],
                              [ 0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0,  0.0, -3.0]])

    assert(torch.equal(adaptive4.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 768.0,    0.0,   -0.0, -256.0],
                              [-512.0, -768.0,    0.0,  768.0],
                              [-768.0, -768.0,  256.0,  768.0],
                              [ 384.0,    0.0, -768.0,  768.0]])

    assert(torch.equal(adaptive4.real_to_format_tensor(test2), expected2))

    # bitwidth of 6, 1 sign bit, 2 exponent bits, 3 mantissa bits
    adaptive6 = adaptive_float(
        bit_width=6,
        exp_len=2,
        mant_len=3,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.125, 2.750, -1.625,  0.0],
                              [-1.125,   2.0,    1.0,  0.0],
                              [   0.0,  -0.0,   -0.0, -0.0],
                              [  -0.0,  -0.0,    0.0, -3.0]])

    assert(torch.equal(adaptive6.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 960.0,    0.0,   -0.0, -288.0],
                              [-640.0, -768.0,    0.0,  640.0],
                              [-704.0, -832.0,  288.0,  832.0],
                              [ 352.0,    0.0, -768.0,  704.0]])

    assert(torch.equal(adaptive6.real_to_format_tensor(test2), expected2))

    # bitwidth of 11, 1 sign bit, 4 exponent bits, 6 mantissa bits
    adaptive11 = adaptive_float(
        bit_width=11,
        exp_len=4,
        mant_len=6,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = \
        torch.tensor([[    -1.171875,         2.71875,  -1.59375,  0.4296875],
                      [    -1.140625,          2.0625,  1.015625,  0.0703125],
                      [   0.16015625, -0.030029296875, -0.890625, -0.8671875],
                      [-0.0400390625,       -0.390625,  0.640625,     -2.875]])

    assert(torch.equal(adaptive11.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[1000.0,  188.0, -148.0, -276.0],
                              [-616.0, -752.0,  18.25,  672.0],
                              [-712.0, -840.0,  300.0,  840.0],
                              [ 348.0,   99.0, -776.0,  712.0]])

    assert(torch.equal(adaptive11.real_to_format_tensor(test2), expected2))

    # bitwidth of 11, 1 sign bit, 2 exponent bits, 8 mantissa bits
    adaptive11 = adaptive_float(
        bit_width=11,
        exp_len=2,
        mant_len=8,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.171875, 2.7109375, -1.6015625,       0.0],
                              [-1.140625,  2.046875, 1.01171875,       0.0],
                              [      0.0,      -0.0,       -0.0,      -0.0],
                              [     -0.0,      -0.0,        0.0, -2.890625]])

    assert(torch.equal(adaptive11.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 998.0,    0.0,   -0.0, -278.0],
                              [-618.0, -756.0,    0.0,  670.0],
                              [-710.0, -842.0,  301.0,  838.0],
                              [ 347.0,    0.0, -776.0,  710.0]])

    assert(torch.equal(adaptive11.real_to_format_tensor(test2), expected2))


# Block Float
def test_block():
    # test tensors to use on different systems
    test1 = torch.tensor([[-1.17,  2.71, -1.60,  0.43],
                          [-1.14,  2.05,  1.01,  0.07],
                          [ 0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39,  0.64, -2.89]])

    test2 = torch.tensor([[ 997.481,  188.034, -147.376, -277.766],
                          [-617.844, -755.696,   18.283,  670.539],
                          [-709.682, -841.260,  300.587,  837.047],
                          [ 347.082,   98.871, -775.379,  709.284]])

    # bitwidth of 4, 1 sign bit, 2 exponent bits, 1 mantissa bit
    block4 = block_fp(
        bit_width=4,
        exp_len=2,
        mant_len=1
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.0,  3.0, -2.0,  0.0],
                              [-1.0,  2.0,  1.0,  0.0],
                              [ 0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0,  0.0, -3.0]])

    assert(torch.equal(block4.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 3.0,  3.0, -3.0, -3.0],
                              [-3.0, -3.0,  3.0,  3.0],
                              [-3.0, -3.0,  3.0,  3.0],
                              [ 3.0,  3.0, -3.0,  3.0]])

    assert(torch.equal(block4.real_to_format_tensor(test2), expected2))

    # bitwidth of 7, 1 sign bit, 3 exponent bits, 3 mantissa bits
    block7 = block_fp(
        bit_width=7,
        exp_len=3,
        mant_len=3
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.25, 2.75, -1.5,   0.5],
                              [-1.25,  2.0,  1.0,   0.0],
                              [  0.0, -0.0, -1.0, -0.75],
                              [ -0.0, -0.5, 0.75,  -3.0]])

    assert(torch.equal(block7.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 15.0,  15.0, -15.0, -15.0],
                              [-15.0, -15.0,  15.0,  15.0],
                              [-15.0, -15.0,  15.0,  15.0],
                              [ 15.0,  15.0, -15.0,  15.0]])

    assert(torch.equal(block7.real_to_format_tensor(test2), expected2))

    # bitwidth of 10, 1 sign bit, 4 exponent bits, 5 mantissa bits
    block10 = block_fp(
        bit_width=10,
        exp_len=4,
        mant_len=5
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.1875, 2.6875, -1.625, 0.4375],
                              [ -1.125, 2.0625,    1.0, 0.0625],
                              [ 0.1875,   -0.0, -0.875, -0.875],
                              [-0.0625, -0.375,  0.625, -2.875]])

    assert(torch.equal(block10.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 252.0,  188.0, -148.0, -252.0],
                              [-252.0, -252.0,   20.0,  252.0],
                              [-252.0, -252.0,  252.0,  252.0],
                              [ 252.0,  100.0, -252.0,  252.0]])

    assert(torch.equal(block10.real_to_format_tensor(test2), expected2))

    # bitwidth of 10, 1 sign bit, 2 exponent bits, 7 mantissa bits
    block10 = block_fp(
        bit_width=10,
        exp_len=2,
        mant_len=7
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.171875, 2.703125, -1.59375,       0.0],
                              [-1.140625, 2.046875, 1.015625,       0.0],
                              [      0.0,     -0.0,     -0.0,      -0.0],
                              [     -0.0,     -0.0,      0.0, -2.890625]])

    assert(torch.equal(block10.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[ 3.984375,  3.984375, -3.984375, -3.984375],
                              [-3.984375, -3.984375,  3.984375,  3.984375],
                              [-3.984375, -3.984375,  3.984375,  3.984375],
                              [ 3.984375,  3.984375, -3.984375,  3.984375]])

    assert(torch.equal(block10.real_to_format_tensor(test2), expected2))