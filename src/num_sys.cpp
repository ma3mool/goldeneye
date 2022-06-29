#include <torch/extension.h>
#include <cstdlib>
#include "num_sys.h"
#include "num_sys_helper.h"

using namespace at;

Tensor quant_bfloat(Tensor tensor, int n_bits, int n_exp)
{
    int n_mant = n_bits - 1 - n_exp;
    // 1. store sign value and do the following part as unsigned value
    Tensor signs = sign(tensor);
    tensor = abs(tensor);

    // 2. limits the range of output float point
    int min_exp = -pow(2, n_exp-1) + 2;
    int max_exp = pow(2, n_exp-1) - 1;

    float min_value = pow(2., min_exp);
    float max_value = (pow(2., max_exp))*(2 - pow(2, -n_mant));

    // Non denormal part
    tensor = where(tensor < min_value, torch::tensor(0.0), tensor);

    // 2.2. reduce too large values to max value of output format
    tensor = where(tensor > max_value, torch::tensor(max_value), tensor);

    //# 3. get mant, exp (the format is different from IEEE float)
    Tensor mant, exp;
    std::tie(mant, exp) = frexp(tensor);

    // 3.1 change mant, and exp format to IEEE float format
    // no effect for exponent of 0 outputs
    mant = 2 * mant;
    exp = exp - 1;

    int shared_exp = (exp.max()).item<int>();
    Tensor exp_diff = shared_exp - exp;
    Tensor power_exp_diff = exp2(exp_diff);
    Tensor mant_adj = mant / power_exp_diff;

    Tensor exp_adj = full(exp.sizes(), shared_exp);

    assert(shared_exp <= max_exp);
    Tensor power_exp = exp2(exp_adj);

    // 4. quantize mantissa
    float scale = pow(2, -n_mant); // e.g. 2 bit, scale = 0.25
    mant_adj = ((mant_adj / scale).round()) * scale;

    Tensor bfloat_out = signs * power_exp * mant_adj;

    return bfloat_out;
}

Tensor quant_bfloat_meta(Tensor tensor, int n_bits, int n_exp)
{
    int n_mant = n_bits - 1 - n_exp;
    // 1. store sign value and do the following part as unsigned value
    Tensor signs = sign(tensor);
    tensor = abs(tensor);

    // 2. limits the range of output float point
    int min_exp = -pow(2, n_exp-1) + 2;
    int max_exp = pow(2, n_exp-1) - 1;

    float min_value = pow(2., min_exp);
    float max_value = (pow(2., max_exp))*(2 - pow(2, -n_mant));

    // Non denormal part
    tensor = where(tensor < min_value, torch::tensor(0.0), tensor);

    // 2.2. reduce too large values to max value of output format
    tensor = where(tensor > max_value, torch::tensor(max_value), tensor);

    //# 3. get mant, exp (the format is different from IEEE float)
    Tensor mant, exp;
    std::tie(mant, exp) = frexp(tensor);
    // auto [mant, exp] = torch::frexp(tensor);

    // 3.1 change mant, and exp format to IEEE float format
    // no effect for exponent of 0 outputs
    mant = 2 * mant;
    exp = exp - 1;

    int shared_exp = (exp.max()).item<int>();

    // ERROR INJECTION INTO META DATA
    // get bit array of shared exp
    std::string exp_str = int_to_bitstream(shared_exp, n_exp);

    // flip a random bit
    int bit_ind = rand() % n_exp;
    std::string bit_arr = bit_flip(exp_str, bit_ind);

    // get numerical value
    shared_exp = bitstream_to_int(bit_arr, n_exp);
    // ERROR INJECTION INTO META DATA

    Tensor exp_diff = shared_exp - exp;
    Tensor power_exp_diff = exp2(exp_diff);
    Tensor mant_adj = mant / power_exp_diff;

    Tensor exp_adj = full(exp.sizes(), shared_exp);

    assert(shared_exp <= max_exp);
    Tensor power_exp = exp2(exp_adj);

    // 4. quantize mantissa
    float scale = pow(2, -n_mant); // e.g. 2 bit, scale = 0.25
    mant_adj = ((mant_adj / scale).round()) * scale;

    Tensor bfloat_out = signs * power_exp * mant_adj;

    return bfloat_out;
}

Tensor quantize_adaptivfloat(Tensor tensor, int n_bits, int n_exp, int bias)
{
    int n_mant = n_bits - 1 - n_exp;
    // 1. store sign value and do the following part as unsigned value
    Tensor signs = sign(tensor);
    tensor = abs(tensor);

    if(bias == -1) {
        int bias_temp = (std::get<1>(frexp(tensor.max()))).item<int>() - 1;
        bias = std::pow(2, n_exp - 1) - 1 - bias_temp;
    }

    // 2. limits the range of output float point
    int min_exp = -pow(2, n_exp-1) + 2 - bias;
    int max_exp = pow(2, n_exp-1) - 1 - bias;

    float min_value = pow(2., min_exp);
    float max_value = (pow(2., max_exp))*(2 - pow(2, -n_mant));

    // Non denormal part
    tensor = where(tensor < min_value, torch::tensor(0.0), tensor);

    // 2.2. reduce too large values to max value of output format
    tensor = where(tensor > max_value, torch::tensor(max_value), tensor);

    //# 3. get mant, exp (the format is different from IEEE float)
    Tensor mant, exp;
    std::tie(mant, exp) = frexp(tensor);
    // auto [mant, exp] = torch::frexp(tensor);

    // 3.1 change mant, and exp format to IEEE float format
    // no effect for exponent of 0 outputs
    mant = 2 * mant;
    exp = exp - 1;

    Tensor power_exp = exp2(exp);

    // 4. quantize mantissa
    float scale = pow(2, -n_mant); // e.g. 2 bit, scale = 0.25
    mant = ((mant / scale).round()) * scale;

    Tensor float_out = signs * power_exp * mant;

    return float_out;
}

Tensor quantize_adaptivfloat_meta(Tensor tensor, int n_bits, int n_exp, int bias)
{
    int n_mant = n_bits - 1 - n_exp;
    // 1. store sign value and do the following part as unsigned value
    Tensor signs = sign(tensor);
    tensor = abs(tensor);

    int bias_temp = (std::get<1>(frexp(tensor.max()))).item<int>() - 1;
    int bias_in = std::pow(2, n_exp - 1) - 1 - bias_temp;

    // ERROR INJECTION INTO META =============
    // get bit array of shared exp
    std::string exp_str = int_to_bitstream(bias_in, n_exp);

    // flip a random bit
    int bit_ind = rand() % 8;
    std::string bit_arr = bit_flip(exp_str, bit_ind);

    // get numerical value
    bias = bitstream_to_int(bit_arr, n_exp);
    // ERROR INJECTION INTO META =============

    // 2. limits the range of output float point
    int min_exp = -pow(2, n_exp-1) + 2 - bias;
    int max_exp = pow(2, n_exp-1) - 1 - bias;

    float min_value = pow(2., min_exp);
    float max_value = (pow(2., max_exp))*(2 - pow(2, -n_mant));

    // Non denormal part
    tensor = where(tensor < min_value, torch::tensor(0.0), tensor);

    // 2.2. reduce too large values to max value of output format
    tensor = where(tensor > max_value, torch::tensor(max_value), tensor);

    //# 3. get mant, exp (the format is different from IEEE float)
    Tensor mant, exp;
    std::tie(mant, exp) = frexp(tensor);

    // 3.1 change mant, and exp format to IEEE float format
    // no effect for exponent of 0 outputs
    mant = 2 * mant;
    exp = exp - 1;

    Tensor power_exp = exp2(exp);

    // 4. quantize mantissa
    float scale = pow(2, -n_mant); // e.g. 2 bit, scale = 0.25
    mant = ((mant / scale).round()) * scale;

    Tensor float_out = signs * power_exp * mant;

    return float_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_bfloat", &quant_bfloat, "Quantize Block Float");
    m.def("quant_bfloat_meta", &quant_bfloat_meta, "Quantize Block Float Meta");
    m.def("quantize_adaptivfloat", &quantize_adaptivfloat, "Quantize AdaptivFloat");
    m.def("quantize_adaptivfloat_meta", &quantize_adaptivfloat_meta, "Quantize AdaptivFloat Meta");
}
