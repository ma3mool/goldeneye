from goldeneye.src.num_sys_class import _ieee754, num_fixed_pt, adaptive_float, block_fp
import torch
import math


# IEEE754
def test_ieee754():
    # bitwidth of 6, 1 sign bit, 1 exponent bit, 4 mantissa bits
    fp6 = _ieee754(
        exp_len=1,
        mant_len=4,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert(fp6.format_to_real("101111") == -0.9375)
    assert(fp6.format_to_real("000101") == 0.3125)

    # denormals
    assert(fp6.format_to_real("000000") == 0.0)
    assert(fp6.format_to_real("100000") == 0.0)

    # infinity
    assert(fp6.format_to_real("010000") == float('inf'))
    assert(fp6.format_to_real("110000") == float('-inf'))

    # NaN
    assert(math.isnan(fp6.format_to_real("011001")))
    assert(math.isnan(fp6.format_to_real("110111")))

    # bitwidth of 6, 1 sign bit, 3 exponent bits, 2 mantissa bits
    fp6 = _ieee754(
        exp_len=3,
        mant_len=2,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp6.format_to_real("101100") == -1.0)
    assert (fp6.format_to_real("011001") == 10.0)

    # denormals
    assert (fp6.format_to_real("000000") == 0.0)
    assert (fp6.format_to_real("100000") == 0.0)

    # infinity
    assert (fp6.format_to_real("011100") == float('inf'))
    assert (fp6.format_to_real("111100") == float('-inf'))

    # NaN
    assert (math.isnan(fp6.format_to_real("011101")))
    assert (math.isnan(fp6.format_to_real("111110")))

    # bitwidth of 14, 1 sign bit, 4 exponent bits, 9 mantissa bits
    fp14 = _ieee754(
        exp_len=4,
        mant_len=9,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp14.format_to_real("11100011011000") == -45.5)
    assert (fp14.format_to_real("00110111001000") == 0.9453125)

    # denormals
    assert (fp14.format_to_real("00000000000000") == 0.0)
    assert (fp14.format_to_real("10000000000000") == 0.0)

    # infinity
    assert (fp14.format_to_real("01111000000000") == float('inf'))
    assert (fp14.format_to_real("11111000000000") == float('-inf'))

    # NaN
    assert (math.isnan(fp14.format_to_real("01111000000001")))
    assert (math.isnan(fp14.format_to_real("11111000000010")))

    # bitwidth of 14, 1 sign bit, 6 exponent bits, 7 mantissa bits
    fp14 = _ieee754(
        exp_len=6,
        mant_len=7,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp14.format_to_real("10100111100100") == -0.00043487548828125)
    assert (fp14.format_to_real("00111000011010") == 0.150390625)

    # denormals
    assert (fp14.format_to_real("00000000000000") == 0.0)
    assert (fp14.format_to_real("10000000000000") == 0.0)

    # infinity
    assert (fp14.format_to_real("01111110000000") == float('inf'))
    assert (fp14.format_to_real("11111110000000") == float('-inf'))

    # NaN
    assert (math.isnan(fp14.format_to_real("01111110101001")))
    assert (math.isnan(fp14.format_to_real("11111110010110")))

    # bitwidth of 27, 1 sign bit, 6 exponent bits, 20 mantissa bits
    fp27 = _ieee754(
        exp_len=6,
        mant_len=20,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp27.format_to_real("001111110110000000010011100") == 1.687648773193359375)
    assert (fp27.format_to_real("110101001011011111100110011") == -2783.599609375)

    # denormals
    assert (fp27.format_to_real("000000000000000000000000000") == 0.0)
    assert (fp27.format_to_real("100000000000000000000000000") == 0.0)

    # infinity
    assert (fp27.format_to_real("011111100000000000000000000") == float('inf'))
    assert (fp27.format_to_real("111111100000000000000000000") == float('-inf'))

    # NaN
    assert (math.isnan(fp27.format_to_real("011111100010000100010010000")))
    assert (math.isnan(fp27.format_to_real("111111100000110000010111000")))

    # bitwidth of 27, 1 sign bit, 16 exponent bits, 10 mantissa bits
    fp27 = _ieee754(
        exp_len=16,
        mant_len=10,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp27.format_to_real("1100000000000001111110110011") == -31.3984375)
    assert (fp27.format_to_real("010000000000101000010011100") == 2416640)

    # denormals
    assert (fp27.format_to_real("000000000000000000000000000") == 0.0)
    assert (fp27.format_to_real("100000000000000000000000000") == 0.0)

    # infinity
    assert (fp27.format_to_real("011111111111111110000000000") == float('inf'))
    assert (fp27.format_to_real("111111111111111110000000000") == float('-inf'))

    # NaN
    assert (math.isnan(fp27.format_to_real("011111111111111110000111000")))
    assert (math.isnan(fp27.format_to_real("111111111111111110100100011")))

    # bitwidth of 32, 1 sign bit, 8 exponent bits, 23 mantissa bits (FP32)
    fp32 = _ieee754(
        exp_len=8,
        mant_len=23,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp32.format_to_real("10111111001000101010111111010000") == -0.63549518585205078125)
    assert (fp32.format_to_real("01000000000000000000000000000001") == 2.00000023841857910156)

    # denormals
    assert (fp32.format_to_real("00000000000000000000000000000000") == 0.0)
    assert (fp32.format_to_real("10000000000000000000000000000000") == 0.0)

    # infinity
    assert (fp32.format_to_real("01111111100000000000000000000000") == float('inf'))
    assert (fp32.format_to_real("11111111100000000000000000000000") == float('-inf'))

    # NaN
    assert (math.isnan(fp32.format_to_real("01111111100010100100000110000100")))
    assert (math.isnan(fp32.format_to_real("11111111100000110011100001000000")))

    # bitwidth of 6, 1 sign bit, 15 exponent bits, 16 mantissa bits
    fp32 = _ieee754(
        exp_len=15,
        mant_len=16,
        bias=None,
        denorm=True,
        max_val=None,
        min_val=None
    )

    # negative and positive numbers
    assert (fp32.format_to_real("10111111111011101010111111010000") == -0.00001286901533603668212890625)
    assert (fp32.format_to_real("01000000000011000001010010100010") == 8852.25)

    # denormals
    assert (fp32.format_to_real("00000000000000000000000000000000") == 0.0)
    assert (fp32.format_to_real("10000000000000000000000000000000") == 0.0)

    # infinity
    assert (fp32.format_to_real("01111111111111110000000000000000") == float('inf'))
    assert (fp32.format_to_real("11111111111111110000000000000000") == float('-inf'))

    # NaN
    assert (math.isnan(fp32.format_to_real("01111111111111110001101111001111")))
    assert (math.isnan(fp32.format_to_real("11111111111111111110010011111000")))

    return


# Fixed Point
def test_fixed():
    # bitwidth of 6, 1 sign bit, 2 integer bits, 3 fraction bits
    fixed6 = num_fixed_pt(
        int_len=2,
        frac_len=3
    )

    # negative and positive numbers
    assert (fixed6.format_to_real("011001") == 3.125)
    assert (fixed6.format_to_real("110000") == -2.0)
    assert (fixed6.format_to_real("100101") == -0.625)
    assert (fixed6.format_to_real("010111") == 2.875)
    assert (fixed6.format_to_real("100000") == -0.0)
    assert (fixed6.format_to_real("111111") == -3.875)

    # bitwidth of 6, 1 sign bit, 3 integer bits, 2 fraction bits
    fixed6 = num_fixed_pt(
        int_len=3,
        frac_len=2
    )

    # negative and positive numbers
    assert (fixed6.format_to_real("011001") == 6.25)
    assert (fixed6.format_to_real("110000") == -4.0)
    assert (fixed6.format_to_real("100101") == -1.25)
    assert (fixed6.format_to_real("100000") == -0.0)
    assert (fixed6.format_to_real("010111") == 5.75)
    assert (fixed6.format_to_real("111111") == -7.75)

    # bitwidth of 10, 1 sign bit, 2 integer bits, 7 fraction bits
    fixed10 = num_fixed_pt(
        int_len=2,
        frac_len=7
    )

    # negative and positive numbers
    assert (fixed10.format_to_real("1111110000") == -3.875)
    assert (fixed10.format_to_real("0100010011") == 2.1484375)
    assert (fixed10.format_to_real("1011110110") == -1.921875)
    assert (fixed10.format_to_real("1010000000") == -1.0)
    assert (fixed10.format_to_real("1011100000") == -1.75)
    assert (fixed10.format_to_real("1011001111") == -1.6171875)

    # bitwidth of 10, 1 sign bit, 6 integer bits, 3 fraction bits
    fixed10 = num_fixed_pt(
        int_len=6,
        frac_len=3
    )

    # negative and positive numbers
    assert (fixed10.format_to_real("1111110000") == -62.0)
    assert (fixed10.format_to_real("0100010011") == 34.375)
    assert (fixed10.format_to_real("1011110110") == -30.75)
    assert (fixed10.format_to_real("1010000000") == -16.0)
    assert (fixed10.format_to_real("1011100000") == -28.0)
    assert (fixed10.format_to_real("1011001111") == -25.875)

    # bitwidth of 23, 1 sign bit, 13 integer bits, 9 fraction bits
    fixed23 = num_fixed_pt(
        int_len=13,
        frac_len=9
    )

    # negative and positive numbers
    assert (fixed23.format_to_real("00111011011011110001000") == 3803.765625)
    assert (fixed23.format_to_real("10000110111110001000000") == -446.125)
    assert (fixed23.format_to_real("00011100000101110011110") == 1797.80859375)
    assert (fixed23.format_to_real("00110000000010111010000") == 3074.90625)
    assert (fixed23.format_to_real("01100110000010111001110") == 6530.90234375)
    assert (fixed23.format_to_real("11111111001000010100011") == -8136.318359375)

    # bitwidth of 23, 1 sign bit, 15 integer bits, 7 fraction bits
    fixed23 = num_fixed_pt(
        int_len=15,
        frac_len=7
    )

    # negative and positive numbers
    assert (fixed23.format_to_real("00111011011011110001000") == 15215.0625)
    assert (fixed23.format_to_real("10000110111110001000000") == -1784.5)
    assert (fixed23.format_to_real("00011100000101110011110") == 7191.234375)
    assert (fixed23.format_to_real("00110000000010111010000") == 12299.625)
    assert (fixed23.format_to_real("01100110000010111001110") == 26123.609375)
    assert (fixed23.format_to_real("11111111001000010100011") == -32545.2734375)

    # bitwidth of 32, 1 sign bit, 8 integer bits, 23 fraction bits
    fixed32 = num_fixed_pt(
        int_len=8,
        frac_len=23
    )

    # negative and positive numbers
    assert (fixed32.format_to_real("10100001001000101010111111010000") == -66.2709903717041015625)
    assert (fixed32.format_to_real("01000001101011000001010010100010") == 131.34437966346740722656)
    assert (fixed32.format_to_real("00110001111000000000010110101001") == 99.75017273426055908203)
    assert (fixed32.format_to_real("01101010010001010101011001101100") == 212.54169988632202148438)
    assert (fixed32.format_to_real("00010110110111000110111001100011") == 45.72211873531341552734)
    assert (fixed32.format_to_real("00001101010100001010100110000010") == 26.63017296791076660156)

    # bitwidth of 32, 1 sign bit, 2 integer bits, 29 fraction bits
    fixed32 = num_fixed_pt(
        int_len=2,
        frac_len=29
    )

    # negative and positive numbers
    assert (fixed32.format_to_real("10100001001000101010111111010000") == -1.03548422455787658691)
    assert (fixed32.format_to_real("01000001101011000001010010100010") == 2.05225593224167823792)
    assert (fixed32.format_to_real("00110001111000000000010110101001") == 1.55859644897282123566)
    assert (fixed32.format_to_real("01101010010001010101011001101100") == 3.32096406072378158569)
    assert (fixed32.format_to_real("00010110110111000110111001100011") == 0.71440810523927211761)
    assert (fixed32.format_to_real("00001101010100001010100110000010") == 0.41609645262360572815)

    return


# Adaptive Float
def test_adaptive():
    # test tensors to use on different systems
    test1 = torch.tensor([[-1.17, 2.71, -1.60, 0.43],
                          [-1.14, 2.05, 1.01, 0.07],
                          [0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39, 0.64, -2.89]])

    test2 = torch.tensor([[997.481, 188.034, -147.376, -277.766],
                         [-617.844, -755.696, 18.283, 670.539],
                         [-709.682, -841.260, 300.587, 837.047],
                         [347.082, 98.871, -775.379, 709.284]])

    # bitwidth of 4, 1 sign bit, 2 exponent bits, 1 mantissa bit
    adaptive4 = adaptive_float(
        bit_width=4,
        exp_len=2,
        mant_len=1,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.0, 3.0, -1.5, 0.0],
                              [-1.0, 2.0, 1.0, 0.0],
                              [0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0, 0.0, -3.0]])

    assert (torch.equal(adaptive4.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[768.0, 0.0, -0.0, -256.0],
                              [-512.0, -768.0, 0.0, 768.0],
                              [-768.0, -768.0, 256.0, 768.0],
                              [384.0, 0.0, -768.0, 768.0]])

    assert (torch.equal(adaptive4.real_to_format_tensor(test2), expected2))

    # bitwidth of 6, 1 sign bit, 2 exponent bits, 3 mantissa bits
    adaptive6 = adaptive_float(
        bit_width=6,
        exp_len=2,
        mant_len=3,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.125, 2.750, -1.625, 0.0],
                              [-1.125, 2.0, 1.0, 0.0],
                              [0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0, 0.0, -3.0]])

    assert (torch.equal(adaptive6.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[960.0, 0.0, -0.0, -288.0],
                              [-640.0, -768.0, 0.0, 640.0],
                              [-704.0, -832.0, 288.0, 832.0],
                              [352.0, 0.0, -768.0, 704.0]])

    assert (torch.equal(adaptive6.real_to_format_tensor(test2), expected2))

    # bitwidth of 11, 1 sign bit, 4 exponent bits, 6 mantissa bits
    adaptive11 = adaptive_float(
        bit_width=11,
        exp_len=4,
        mant_len=6,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.171875, 2.71875, -1.59375, 0.4296875],
                             [-1.140625, 2.0625, 1.015625, 0.0703125],
                             [0.16015625, -0.030029296875, -0.890625, -0.8671875],
                             [-0.0400390625, -0.390625, 0.640625, -2.875]])

    assert (torch.equal(adaptive11.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[1000.0, 188.0, -148.0, -276.0],
                              [-616.0, -752.0, 18.25, 672.0],
                              [-712.0, -840.0, 300.0, 840.0],
                              [348.0, 99.0, -776.0, 712.0]])

    assert (torch.equal(adaptive11.real_to_format_tensor(test2), expected2))

    # bitwidth of 11, 1 sign bit, 2 exponent bits, 8 mantissa bits
    adaptive11 = adaptive_float(
        bit_width=11,
        exp_len=2,
        mant_len=8,
        exp_bias=None
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.171875, 2.7109375, -1.6015625, 0.0],
                              [-1.140625, 2.046875, 1.01171875, 0.0],
                              [0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0, 0.0, -2.890625]])

    assert (torch.equal(adaptive11.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[998.0, 0.0, -0.0, -278.0],
                              [-618.0, -756.0, 0.0, 670.0],
                              [-710.0, -842.0, 301.0, 838.0],
                              [347.0, 0.0, -776.0, 710.0]])

    assert (torch.equal(adaptive11.real_to_format_tensor(test2), expected2))

    return


# Block Float
def test_block():
    # test tensors to use on different systems
    test1 = torch.tensor([[-1.17, 2.71, -1.60, 0.43],
                          [-1.14, 2.05, 1.01, 0.07],
                          [0.16, -0.03, -0.89, -0.87],
                          [-0.04, -0.39, 0.64, -2.89]])

    test2 = torch.tensor([[997.481, 188.034, -147.376, -277.766],
                          [-617.844, -755.696, 18.283, 670.539],
                          [-709.682, -841.260, 300.587, 837.047],
                          [347.082, 98.871, -775.379, 709.284]])

    # bitwidth of 4, 1 sign bit, 2 exponent bits, 1 mantissa bit
    block4 = block_fp(
        bit_width=4,
        exp_len=2,
        mant_len=1
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.0, 3.0, -2.0, 0.0],
                              [-1.0, 2.0, 1.0, 0.0],
                              [0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0, 0.0, -3.0]])

    assert (torch.equal(block4.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[3.0, 3.0, -3.0, -3.0],
                              [-3.0, -3.0, 3.0, 3.0],
                              [-3.0, -3.0, 3.0, 3.0],
                              [3.0, 3.0, -3.0, 3.0]])

    assert (torch.equal(block4.real_to_format_tensor(test2), expected2))

    # bitwidth of 7, 1 sign bit, 3 exponent bits, 3 mantissa bits
    block7 = block_fp(
        bit_width=7,
        exp_len=3,
        mant_len=3
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.25, 2.75, -1.5, 0.5],
                              [-1.25, 2.0, 1.0, 0.0],
                              [0.0, -0.0, -1.0, -0.75],
                              [-0.0, -0.5, 0.75, -3.0]])

    assert (torch.equal(block7.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[15.0, 15.0, -15.0, -15.0],
                              [-15.0, -15.0, 15.0, 15.0],
                              [-15.0, -15.0, 15.0, 15.0],
                              [15.0, 15.0, -15.0, 15.0]])

    assert (torch.equal(block7.real_to_format_tensor(test2), expected2))

    # bitwidth of 10, 1 sign bit, 4 exponent bits, 5 mantissa bits
    block10 = block_fp(
        bit_width=10,
        exp_len=4,
        mant_len=5
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.1875, 2.6875, -1.625, 0.4375],
                              [-1.125, 2.0625, 1.0, 0.0625],
                              [0.1875, -0.0, -0.875, -0.875],
                              [-0.0625, -0.375, 0.625, -2.875]])

    assert (torch.equal(block10.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[252.0, 188.0, -148.0, -252.0],
                              [-252.0, -252.0, 20.0, 252.0],
                              [-252.0, -252.0, 252.0, 252.0],
                              [252.0, 100.0, -252.0, 252.0]])

    assert (torch.equal(block10.real_to_format_tensor(test2), expected2))

    # bitwidth of 10, 1 sign bit, 2 exponent bits, 7 mantissa bits
    block10 = block_fp(
        bit_width=10,
        exp_len=2,
        mant_len=7
    )

    # expected tensors for tests
    expected1 = torch.tensor([[-1.171875, 2.703125, -1.59375, 0.0],
                              [-1.140625, 2.046875, 1.015625, 0.0],
                              [0.0, -0.0, -0.0, -0.0],
                              [-0.0, -0.0, 0.0, -2.890625]])

    assert (torch.equal(block10.real_to_format_tensor(test1), expected1))

    expected2 = torch.tensor([[3.984375, 3.984375, -3.984375, -3.984375],
                              [-3.984375, -3.984375, 3.984375, 3.984375],
                              [-3.984375, -3.984375, 3.984375, 3.984375],
                              [3.984375, 3.984375, -3.984375, 3.984375]])

    assert (torch.equal(block10.real_to_format_tensor(test2), expected2))

    return


if __name__ == '__main__':
    test_ieee754()
    test_fixed()
    test_adaptive()
    test_block()
