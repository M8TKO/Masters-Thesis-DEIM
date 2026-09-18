#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

N_BULK=56
N_CIRCLE=112
WAKE_REFINEMENT=0.25
T_END=8
DT=0.01
SAVE_EVERY=2
U_MAX=0.9
RAMP_TIME=4
PERTURBATION=1e-2
KICK_STRENGTH=0.04
KICK_UNTIL=6
KICK_PERIOD=0.6

U_MAX_LABEL="${U_MAX//./_}"
OUTPUT_DIR="run_u_max_${U_MAX_LABEL}"
OUTPUT_FILE="${OUTPUT_DIR}/wake_u_max_${U_MAX_LABEL}.xdmf"
PARAMS_FILE="${OUTPUT_DIR}/parameters.txt"

mkdir -p "${OUTPUT_DIR}"

cat > "${PARAMS_FILE}" <<EOF
python=${PYTHON}
n_bulk=${N_BULK}
n_circle=${N_CIRCLE}
wake_refinement=${WAKE_REFINEMENT}
t_end=${T_END}
dt=${DT}
save_every=${SAVE_EVERY}
u_max=${U_MAX}
ramp_time=${RAMP_TIME}
perturbation=${PERTURBATION}
kick_strength=${KICK_STRENGTH}
kick_until=${KICK_UNTIL}
kick_period=${KICK_PERIOD}
output=${OUTPUT_FILE}
EOF

"${PYTHON}" dfg.py \
  --n-bulk "${N_BULK}" \
  --n-circle "${N_CIRCLE}" \
  --wake-refinement "${WAKE_REFINEMENT}" \
  --t-end "${T_END}" \
  --dt "${DT}" \
  --save-every "${SAVE_EVERY}" \
  --u-max "${U_MAX}" \
  --ramp-time "${RAMP_TIME}" \
  --perturbation "${PERTURBATION}" \
  --kick-strength "${KICK_STRENGTH}" \
  --kick-until "${KICK_UNTIL}" \
  --kick-period "${KICK_PERIOD}" \
  --output "${OUTPUT_FILE}"
