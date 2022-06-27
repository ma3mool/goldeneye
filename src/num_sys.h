using namespace at;

Tensor quant_bfloat(Tensor tensor, int n_bits, int n_exp);
Tensor quant_bfloat_meat(Tensor tensor, int n_bits, int n_exp);
Tensor quantize_adaptivfloat(Tensor tensor, int n_bits, int n_exp, int bias);
Tensor quantize_adaptivfloat_meta(Tensor tensor, int n_bits, int n_exp, int bias);
