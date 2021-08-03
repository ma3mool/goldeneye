#!/bin/bash
set -x

NETWORK=${1}
DATASET="IMAGENET"

OUTPUT_PATH="../output/"
SCRIPT1="../src/preprocess.py"
SCRIPT2="../src/profile.py"
SCRIPT3="../src/injections.py"
SCRIPT4="../src/postprocess.py"
VERBOSE=""
DEBUG=""
#QUANT="-q" # -q leave empty if you do not want quantization
#BIT_FLIP="-e" # -e leave empty if you do not want bit flip model. NOTE: -q MUST BE ENABLED TOO WITH THIS
#RANKSET="-r"  # -r leave empty if using testset

WORKERS=8
BATCH=128

echo "==============================="
echo "STARTING ${NETWORK}-${DATASET}..."

# preprocess
echo "Preprocessing ..."

echo "Profiling ..."

echo "Error Injection Campaign ..."

echo "Postprocessing ..."

echo "Done!"
