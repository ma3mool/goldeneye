#!/bin/bash
#set -x

NETWORK=${1}
DATASET=${2} #"IMAGENET"
BATCH=${3} #128
FORMAT=${4}  #"fp_n"    # simulated format
BITWIDTH=${5}   #32
RADIX=${6}      #23
INJECTIONS_LOC=${7}  # {0, no injection}. {1: value} or {2, META}


OUTPUT_PATH="../output/"
SRC_PATH="../src/"
LOG_PATH="./log/"
SCRIPT1="../src/preprocess.py"
SCRIPT2="../src/profiling.py"
SCRIPT3="../src/split_data.py"
SCRIPT4="../src/injections.py"
SCRIPT5="../src/postprocess.py"
VERBOSE=""
DEBUG=""
PRECISION="FP32" # compute fabric
BIAS="" # leave empty, or include the flag with the number: "-a -8" | ""
QUANT="" # -q leave empty if you do not want quantization
#BIT_FLIP="-e" # -e leave empty if you do not want bit flip model. NOTE: -q MUST BE ENABLED TOO WITH THIS
TRAINSET="" # -r. leave empty if using testset
WORKERS=16

INJECTIONS=102400 #${BATCH}
#INJECTIONS_LOC=1  # {0, no injection}. {1: value} or {2, META}
                  #OLD {2, INT value}, or {3, INT scaling}, or {4, block meta}, or {5, adaptive meta}





# Script check
if [[ -z "$NETWORK" ]]
then
  echo "ERROR: No network provided."
  exit 1
fi
if [[ -z "$BATCH" ]]
then
  echo "ERROR: No batch size provided."
  exit 1
fi


if [[ ! -f "$SCRIPT1" ]] && [[ ! -f "$SCRIPT2" ]] && [[ ! -f "$SCRIPT3" ]] && [[ ! -f "$SCRIPT4" ]] && [[ ! -f "$SCRIPT5" ]]
then
  echo "ERROR: Can't find execution scripts."
  exit 1
fi

# File IO preprocessing
if [[ ! -d "$LOG_PATH" ]]
then
    mkdir -p $LOG_PATH
fi
RANGES="${NETWORK}_${DATASET}/ranges_trainset_layer.p.bz2"
RANGES_FILE="${OUTPUT_PATH}/networkRanges/${RANGES}"
GOLDEN="${NETWORK}_${DATASET}_real${PRECISION}_sim${FORMAT}_bw${BITWIDTH}_r${RADIX}_biasNone/golden_data.p.bz2"
GOLDEN_FILE="${OUTPUT_PATH}/networkProfiles/${GOLDEN}"


echo "!===================================!"
echo "! Launching ${NETWORK}-${DATASET}-${PRECISION}"
echo "!===================================!"

# preprocess
echo -n "Preprocessing ... "
if [[ ! -f "$RANGES_FILE" ]]
then
    python3 ${SCRIPT1} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -f ${FORMAT}
    echo "Complete!"
else
    echo "Skipped."
fi

# profiling
echo -n "Profiling ... "
if [[ ! -f "$GOLDEN_FILE" ]]
then
    python3 ${SCRIPT2} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -f ${FORMAT} -B ${BITWIDTH} -R ${RADIX} ${BIAS} ${QUANT}
    python3 ${SCRIPT3} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -f ${FORMAT} -B ${BITWIDTH} -R ${RADIX} ${BIAS} ${QUANT}
    echo "Complete!"
else
    echo "Skipped."
fi

# injections
echo "Error Injection Campaign ... "
#read -p "    About to launch an error injection campaign. Are you sure? " -n 1 -r
#echo
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
python3 ${SCRIPT4} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -i ${INJECTIONS} -I ${INJECTIONS_LOC} -f ${FORMAT}  -B ${BITWIDTH} -R ${RADIX} ${BIAS} ${QUANT}
#python3 ${SCRIPT4} -b 1 -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -i ${INJECTIONS} -I ${INJECTIONS_LOC} -f ${FORMAT}  -B ${BITWIDTH} -R ${RADIX} ${BIAS} ${QUANT}
#fi

# postprocessing
echo -n "Postprocessing ... "
python3 ${SCRIPT5} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -i ${INJECTIONS} -f ${FORMAT} -B ${BITWIDTH} -R ${RADIX} ${BIAS} ${QUANT}
echo -n "Done! "
