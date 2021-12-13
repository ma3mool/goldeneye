#!/bin/bash
#set -x

cd ../
script="./perf_measurements.sh"
NAME=${1}
BATCH=${2}

echo "${NAME} ..."
${script} ${NAME} ${BATCH} fp32 32 23 0      # baseline
${script} ${NAME} ${BATCH} fp_n 32 23 0      # simulated fp 32
${script} ${NAME} ${BATCH} fp_n 16 7 0       # bfloat
${script} ${NAME} ${BATCH} fp_n 16 7 1       # bfloat w/ errors
${script} ${NAME} ${BATCH} fxp_n 32 16 0     # fixed point
${script} ${NAME} ${BATCH} fxp_n 32 16 1     # fixed point w/ errors
${script} ${NAME} ${BATCH} INT 8 0 0         # INT8
${script} ${NAME} ${BATCH} INT 8 0 1         # INT8 w/ error 1
${script} ${NAME} ${BATCH} INT 8 0 0         # INT8 w/ error 2 TODO Fix
${script} ${NAME} ${BATCH} block_fp 12 7 0   # BFP e4m7
${script} ${NAME} ${BATCH} block_fp 12 7 1   # BFP e4m7 w/ error 1
${script} ${NAME} ${BATCH} block_fp 12 7 2   # BFP e4m7 w/ error 2
${script} ${NAME} ${BATCH} adaptive_fp 8 3 0 # AFP e4m3
${script} ${NAME} ${BATCH} adaptive_fp 8 3 1 # AFP e4m3 w/ error 1
${script} ${NAME} ${BATCH} adaptive_fp 8 3 2 # AFP e4m3 w/ error 2

echo "DONE"