#!/bin/bash
#Command format: 
# bash resilience.sh Dataset-corruption outputpath
# e.g. bash resilience.sh C_IMAGENET-contrast /home/pfi/Documents/Data/CVPR_paper/goldeneye/output/task1a/C_IMAGENET/contrast

models=(
  # "resnet18_baseline"
  # "resnet50_baseline"
  # "resnet18_l2_eps0.01"
  "resnet18_l2_eps0.03"
  # "resnet18_l2_eps0.05"
  # "resnet18_l2_eps0.1"
  # "resnet18_l2_eps0.25"
  # "resnet18_l2_eps0.5"
  # "resnet18_l2_eps0"
  # "resnet18_l2_eps1"
  # "resnet18_l2_eps3"
  # "resnet18_l2_eps5"
  # "resnet18_linf_eps0.5"
  # "resnet18_linf_eps1.0"
  # "resnet18_linf_eps2.0"
  # "resnet18_linf_eps4.0"
  # "resnet18_linf_eps8.0"
  # "resnet50_l2_eps0.01"
  # "resnet50_l2_eps0.03"
  # "resnet50_l2_eps0.05"
  # "resnet50_l2_eps0.1"
  # "resnet50_l2_eps0.25"
  # "resnet50_l2_eps0.5"
  # "resnet50_l2_eps0"
  # "resnet50_l2_eps1"
  # "resnet50_l2_eps3"
  # "resnet50_l2_eps5"
  # "resnet50_linf_eps0.5"
  # "resnet50_linf_eps1.0"
  # "resnet50_linf_eps2.0"
  # "resnet50_linf_eps4.0"
  # "resnet50_linf_eps8.0"
)

DATASET=${1} #"IMAGENET"
OUTPUT_PATH=${2}

severity_levels=(1 2 3 4 5)

for severity in "${severity_levels[@]}"; do
  output="$OUTPUT_PATH/$severity"
  for model in "${models[@]}"; do
      script_args="$model $DATASET-$severity 128 fp_n 32 23 1 $output"
      echo "$script_args"
      bash end_to_end.sh $script_args
      echo "Running script with arguments: $script_args"
    done
done


