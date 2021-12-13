#!/bin/bash
#set -x

cd ../
script="./accuracy_profile.sh"
NAME=${1}
BATCH=${2}

echo "${NAME} ..."
${script} ${NAME} ${BATCH} fp32 32 23 0     # baseline
#
${script} ${NAME} ${BATCH} fp_n 32 23 0     # baseline
${script} ${NAME} ${BATCH} fp_n 16 10 0     #
${script} ${NAME} ${BATCH} fp_n 12 7 0
${script} ${NAME} ${BATCH} fp_n 8 5 0
${script} ${NAME} ${BATCH} fp_n 4 2 0

${script} ${NAME} ${BATCH} fxp_n 32 16 0     # baseline
${script} ${NAME} ${BATCH} fxp_n 16 8 0     #
${script} ${NAME} ${BATCH} fxp_n 12 6 0
${script} ${NAME} ${BATCH} fxp_n 8 3 0
${script} ${NAME} ${BATCH} fxp_n 4 2 0

${script} ${NAME} ${BATCH} INT 32 0 0         # INT8
${script} ${NAME} ${BATCH} INT 16 0 0         # INT8
${script} ${NAME} ${BATCH} INT 12 0 0         # INT8
${script} ${NAME} ${BATCH} INT 8 0 0         # INT8
${script} ${NAME} ${BATCH} INT 4 0 0         # INT8
#
#${script} ${NAME} ${BATCH} block_fp 32 23 0     # baseline
#${script} ${NAME} ${BATCH} block_fp 16 10 0     #
#${script} ${NAME} ${BATCH} block_fp 12 7 0
#${script} ${NAME} ${BATCH} block_fp 8 3 0
#${script} ${NAME} ${BATCH} block_fp 4 2 0

${script} ${NAME} ${BATCH} adaptive_fp 32 23 0     # baseline
${script} ${NAME} ${BATCH} adaptive_fp 16 10 0     #
${script} ${NAME} ${BATCH} adaptive_fp 12 7 0
${script} ${NAME} ${BATCH} adaptive_fp 8 3 0
${script} ${NAME} ${BATCH} adaptive_fp 4 2 0

echo "DONE"