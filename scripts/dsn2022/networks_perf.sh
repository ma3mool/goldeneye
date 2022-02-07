#!/bin/bash
#set -x

script="./gather_perf.sh"

${script} alexnet 32
${script} resnet18 128
#${script} resnet50 128
${script} deit_tiny 32
#${script} deit_base 32
#${script} vit_base 32
