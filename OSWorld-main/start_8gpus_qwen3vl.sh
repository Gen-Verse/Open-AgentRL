#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"   
BASE_PORT=8000
DTYPE="bfloat16"
MAX_BATCH=4
QUEUE_MS=20

NUM_GPU_PER_MODEL=${NUM_GPU_PER_MODEL:-2}
DEVICE_MAP=${DEVICE_MAP:-auto}   
MAX_GPU_MEM=${MAX_GPU_MEM:-}

IDLE_UNLOAD_S=0
OFFLOAD_MODE="none"
PRELOAD="--preload"


MAX_MODEL_LEN=${MAX_MODEL_LEN:-0}        
ENFORCE_EAGER=${ENFORCE_EAGER:-0}        

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME:-}}"

unset CUDA_VISIBLE_DEVICES

python - <<'PY'
import torch, sys
print(f"[check] cuda_available={torch.cuda.is_available()}, torch_cuda={getattr(torch.version,'cuda',None)}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY

mkdir -p /tmp/opencua_pids   

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d '[:space:]')
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" == "0" ]]; then
  echo "[fatal] no GPU visible"; exit 1
fi

if (( NUM_GPU_PER_MODEL < 1 )); then NUM_GPU_PER_MODEL=1; fi
if (( NUM_GPU_PER_MODEL > GPU_COUNT )); then NUM_GPU_PER_MODEL=${GPU_COUNT}; fi

echo "[groups] GPU_COUNT=${GPU_COUNT}, NUM_GPU_PER_MODEL=${NUM_GPU_PER_MODEL}"

GROUP_STRS=()
start=0
while (( start < GPU_COUNT )); do
  end=$(( start + NUM_GPU_PER_MODEL - 1 ))
  grp=""
  i=${start}
  while (( i < GPU_COUNT && i <= end )); do
    if [[ -z "$grp" ]]; then grp="${i}"; else grp="${grp},${i}"; fi
    i=$(( i + 1 ))
  done
  GROUP_STRS+=( "$grp" )
  start=$(( start + NUM_GPU_PER_MODEL ))
done

echo -n "[groups] GROUPS:"
for g in "${GROUP_STRS[@]}"; do echo -n " {$g}"; done
echo

PORTS=()
idx=0
for grp in "${GROUP_STRS[@]}"; do
  PORT=$(( BASE_PORT + idx ))
  PORTS+=( "$PORT" )
  LOG="/tmp/qwen3vl_${PORT}.log"     
  echo "Starting GPUs {${grp}} on port ${PORT} ..."
  CUDA_VISIBLE_DEVICES="${grp}" \
  nohup python -u serve_qwen3vl.py \
    --model "${MODEL}" \
    --port  "${PORT}" \
    --dtype "${DTYPE}" \
    --max-batch ${MAX_BATCH} \
    --queue-ms ${QUEUE_MS} \
    --idle-unload-s ${IDLE_UNLOAD_S} \
    --offload-mode ${OFFLOAD_MODE} \
    --num-gpu-per-model ${NUM_GPU_PER_MODEL} \
    --device-map ${DEVICE_MAP} \
    ${MAX_GPU_MEM:+--max-gpu-mem "${MAX_GPU_MEM}"} \
    $([[ "${MAX_MODEL_LEN}" != "0" ]] && echo --max-model-len "${MAX_MODEL_LEN}") \
    $([[ "${ENFORCE_EAGER}" == "1" ]] && echo --enforce-eager) \
    ${PRELOAD} \
    > "${LOG}" 2>&1 &

  echo $! > "/tmp/opencua_pids/${PORT}.pid"
  echo "  -> PID $(cat /tmp/opencua_pids/${PORT}.pid), log ${LOG}"
  idx=$(( idx + 1 ))
done

EP=""
for p in "${PORTS[@]}"; do
  [[ -z "${EP}" ]] && EP="http://127.0.0.1:${p}" || EP="${EP},http://127.0.0.1:${p}"
done

cat > .env <<EOF
OPENCUA_LOCAL_ENDPOINTS=${EP}
OPENCUA_API_KEY=dummy
EOF

echo "[start] wrote .env with endpoints:"
cat .env

echo "[start] waiting all servers to be ready..."
for p in "${PORTS[@]}"; do
  echo -n "  - port ${p} "
  for t in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${p}/ready" >/tmp/ready_${p}.json 2>/dev/null; then
      python - <<PY
import json,sys
d=json.load(open("/tmp/ready_${p}.json"))
sys.exit(0 if d.get("loaded") and d.get("on_gpu") else 1)
PY
      if [[ $? -eq 0 ]]; then
        echo "READY"
        break
      fi
    fi
    sleep 1
    if [[ $t -eq 600 ]]; then
      echo "TIMEOUT (see /tmp/qwen3vl_${p}.log)"
      exit 1
    fi
  done
done
echo "[start] all servers ready."
