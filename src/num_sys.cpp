#include <torch/extension.h>
#include "num_sys.h"

using namespace at;

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

Tensor quantize_bfloat(Tensor tensor, int n_bits, int n_exp)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_adaptivfloat", &quantize_adaptivfloat, "Quantize AdaptivFloat");
    m.def("quantize_bfloat", &quantize_bfloat, "Quantize Block Float");
}

