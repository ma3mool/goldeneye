#!/bin/bash
#set -x

script="./gather_accuracy.sh"

#${script} alexnet 256
${script} resnet18 128
${script} deit_tiny 128
#${script} deit_base 32
#${script} vit_base 32
