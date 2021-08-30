#!/bin/bash
#set -x

script="./end_to_end.sh"

echo "alexnet ..."
#time ${script} alexnet   512    2&> ./log/alexnet.log

echo "vgg19_bn ..."
#time ${script} vgg19_bn  64    2&> ./log/vgg19_bn.log

echo "resnet50 ..."
#time ${script} resnet50  64    2&> ./log/resnet50.log

echo "deit_tiny ..."
#time ${script} deit_tiny 128    2&> ./log/deit_tiny.log

echo "deit_base ..."
time ${script} deit_base 32     2&> ./log/deit_base.log

echo "vit_base ..."
time ${script} vit_base  32     2&> ./log/vit_base.log
