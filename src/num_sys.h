#ifndef GOLDENEYE_NUM_SYS_H
#define GOLDENEYE_NUM_SYS_H 

using namespace at;

Tensor quantize_adaptivfloat(Tensor tensor, int n_bits, int n_exp, int bias);
Tensor quantize_bfloat(Tensor tensor, int n_bits, int n_exp);


#endif

