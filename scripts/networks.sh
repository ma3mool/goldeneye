#!/bin/bash
#set -x

script="./end_to_end.sh"
network=${1}
batchsize=${2}


#time ${script} ${network} ${batchsize} block_fp 12 7  1               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 8 3 1               #2&> ./log/alexnet.log

#time ${script} ${network} ${batchsize} block_fp 12 7  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 8 3 2               #2&> ./log/alexnet.log
##
#time ${script} ${network} ${batchsize} fp32 32 23  1               #2&> ./log/alexnet.log
#
time ${script} ${network} ${batchsize} block_fp 16 8  1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} block_fp 16 10 1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} block_fp 16 11 1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} block_fp 12 6  1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} block_fp 10 5  1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} block_fp 11 5  1               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} block_fp 16 8  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} block_fp 16 10 2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} block_fp 16 11 2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} block_fp 12 6  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} block_fp 10 5  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} block_fp 11 5  2               #2&> ./log/alexnet.log
#
#
time ${script} ${network} ${batchsize} adaptive_fp 16 8  1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} adaptive_fp 16 10 1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} adaptive_fp 16 11 1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} adaptive_fp 8 2   1               #2&> ./log/alexnet.log  #START HERE FOR DEIT
time ${script} ${network} ${batchsize} adaptive_fp 12 6  1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} adaptive_fp 10 5  1               #2&> ./log/alexnet.log
time ${script} ${network} ${batchsize} adaptive_fp 9 4   1               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 16 8  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 16 10 2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 16 11 2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 8 2   2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 12 6  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 10 5  2               #2&> ./log/alexnet.log
#time ${script} ${network} ${batchsize} adaptive_fp 9 4   2               #2&> ./log/alexnet.log

#echo "alexnet ..."
##time ${script} alexnet   512    2&> ./log/alexnet.log

#echo "vgg19_bn ..."
##time ${script} vgg19_bn  64    2&> ./log/vgg19_bn.log

#echo "resnet50 ..."
##time ${script} resnet50  64    2&> ./log/resnet50.log

#echo "deit_tiny ..."
##time ${script} deit_tiny 128    2&> ./log/deit_tiny.log

#echo "deit_base ..."
#time ${script} deit_base 32     2&> ./log/deit_base.log

#echo "vit_base ..."
#time ${script} vit_base  32     2&> ./log/vit_base.log
