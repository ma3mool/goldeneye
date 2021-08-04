#!/bin/bash
#set -x

NETWORK=${1}
DATASET="IMAGENET"
OUTPUT_PATH="../output/"
SRC_PATH="../src/"
LOG_PATH="./log/"
SCRIPT1="../src/preprocess.py"
SCRIPT2="../src/profile.py"
SCRIPT3="../src/injections.py"
SCRIPT4="../src/postprocess.py"
VERBOSE=""
DEBUG=""
PRECISION="FP16"
#QUANT="-q" # -q leave empty if you do not want quantization
#BIT_FLIP="-e" # -e leave empty if you do not want bit flip model. NOTE: -q MUST BE ENABLED TOO WITH THIS
TRAINSET="" # -r. leave empty if using testset
WORKERS=8
BATCH=128

INJECTIONS=3200



# Script check
if [[ -z "$NETWORK" ]]
then
  echo "ERROR: No network provided."
  exit 1
fi
if [[ ! -f "$SCRIPT1" ]] && [[ ! -f "$SCRIPT2" ]] && [[ ! -f "$SCRIPT3" ]] && [[ ! -f "$SCRIPT4" ]]
then
  echo "ERROR: Can't find execution scripts."
  exit 1
fi

# File IO preprocessing
if [[ ! -d "$LOG_PATH" ]]
then
    mkdir -p $LOG_PATH
fi
RANGES="${NETWORK}_${DATASET}_${PRECISION}/ranges_trainset_layer.p.bz2"
RANGES_FILE="${OUTPUT_PATH}/networkRanges/${RANGES}"
GOLDEN="${NETWORK}_${DATASET}_${PRECISION}/golden_data.p.bz2"
GOLDEN_FILE="${OUTPUT_PATH}/networkProfiles/${GOLDEN}"


echo "!===================================!"
echo "! Launching ${NETWORK}-${DATASET}-${PRECISION}"
echo "!===================================!"

# preprocess
echo -n "Preprocessing ... "
if [[ ! -f "$RANGES_FILE" ]]
then
    python3 ${SCRIPT1} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION}
    echo "Complete!"
else
    echo "Skipped."
fi

# profiling
echo -n "Profiling ... "
if [[ ! -f "$GOLDEN_FILE" ]]
then
    python3 ${SCRIPT2} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION}
    echo "Complete!"
else
    echo "Skipped."
fi

# injections
echo "Error Injection Campaign ... "
read -p "    About to launch an error injection campaign. Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3 ${SCRIPT3} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -i ${INJECTIONS}
fi

# postprocessing
echo -n "Postprocessing ... "
python3 ${SCRIPT4} -b ${BATCH} -n ${NETWORK} -d ${DATASET} -o ${OUTPUT_PATH} ${TRAINSET} ${VERBOSE} ${DEBUG} -w ${WORKERS} -P ${PRECISION} -i ${INJECTIONS} -i ${INJECTIONS}
echo -n "Done! "