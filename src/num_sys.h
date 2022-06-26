using namespace at;

Tensor quantize_bfloat(Tensor tensor, int n_bits, int n_exp);
Tensor quantize_bfloat_meat(Tensor tensor, int n_bits, int n_exp);
Tensor quantize_adaptivfloat(Tensor tensor, int n_bits, int n_exp, int bias);
Tensor quantize_adaptivfloat_meta(Tensor tensor, int n_bits, int n_exp, int bias);
