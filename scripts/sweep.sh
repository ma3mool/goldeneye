#!/bin/bash
set -x

script="python ../src/sweep_num_formats.py"

#echo "alexnet ..."
#time ${script} -n alexnet   -b 256 -d IMAGENET -w 16 -P FP32 -f fp32    #2&> ./log/alexnet_sweep.log

echo "vgg19_bn ..."
time ${script} -n vgg19_bn  -b 64   -d IMAGENET -w 16 -P FP32 -f fp32   #2&> ./log/vgg19_bn_sweep.log

echo "resnet50 ..."
time ${script} -n resnet50  -b 64   -d IMAGENET -w 16 -P FP32 -f fp32   #2&> ./log/resnet50_sweep.log

echo "deit_tiny ..."
time ${script} -n deit_tiny -b 128  -d IMAGENET -w 16 -P FP32 -f fp32   #2&> ./log/deit_ti_sweepny.log

echo "deit_base ..."
time ${script} -n deit_base -b 32   -d IMAGENET -w 16 -P FP32 -f fp32   #2&> ./log/deit_ba_sweepse.log

echo "vit_base ..."
time ${script} -n vit_base  -b 32   -d IMAGENET -w 16 -P FP32 -f fp32   #2&> ./log/vit_bas_sweepe.log
