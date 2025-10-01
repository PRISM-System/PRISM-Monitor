#!/usr/bin/env bash
set -euo pipefail

# ▶ 바꿔야 하는 부분들
MODEL="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}/v1"
CSV="/home/minjoo/Github/PRISM-Monitor/InstructionRF/data/Semiconductor_intent_dataset__preview_.csv"

# 4bit + 4B면 1장으로도 넉넉. 병렬 오버헤드 줄이기 위해 1 추천
export CUDA_VISIBLE_DEVICES="0"
TP_SIZE=1

MAX_MODEL_LEN=65536      
GPU_UTIL=0.95
TORCH_FLAGS="--enforce-eager"   # torch.compile 비활성화 원하면 유지, 아니면 빈값 ""

LOG_DIR="/tmp/vllm_oneclick"
SRV_LOG="${LOG_DIR}/vllm_server.log"
PID_FILE="${LOG_DIR}/vllm_server.pid"
EVAL_LOG="${LOG_DIR}/evaluate.log"
mkdir -p "$LOG_DIR"

stdbuf_oL() { stdbuf -oL -eL "$@"; }

cleanup() {
  echo "[CLEANUP] 종료 처리…"
  [[ -n "${TAIL_PID:-}" ]] && kill "${TAIL_PID}" 2>/dev/null || true
  if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}" || true)
    if [[ -n "${PID}" ]] && ps -p "${PID}" >/dev/null 2>&1; then
      kill "${PID}" || true
      sleep 2
      ps -p "${PID}" >/dev/null 2>&1 && kill -9 "${PID}" || true
    fi
    rm -f "${PID_FILE}"
  fi
}
trap cleanup EXIT

echo "[1/4] vLLM 서버 시작(백그라운드)…"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --trust-remote-code \
  ${TORCH_FLAGS} \
  > "${SRV_LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "  - PID: $(cat ${PID_FILE})"
echo "  - 로그: ${SRV_LOG}"

echo "[LOG] vLLM 서버 로그 팔로우 시작… (Ctrl+C로 전체 작업 종료됨)"
tail -n 0 -f "${SRV_LOG}" &
TAIL_PID=$!

echo "[2/4] 서버 준비 대기(/v1/models 헬스체크)…"
for i in {1..120}; do
  if curl -sS "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "  - 서버 준비 완료"
    break
  fi
  sleep 1
  if [[ $i -eq 120 ]]; then
    echo "  - 서버 준비 실패. 최근 로그:"
    tail -n 100 "${SRV_LOG}" || true
    exit 2
  fi
done

echo "[3/4] 성능 평가 실행(evaluate_intent_accuracy.py)… (로그: ${EVAL_LOG})"
stdbuf_oL python evaluate_intent_accuracy.py \
  --base_url "${BASE_URL}" \
  --csv "${CSV}" \
  --model "${MODEL}" \
  --timeout 60 2>&1 | tee "${EVAL_LOG}"

echo "[4/4] 완료. 리소스 정리 중…"
echo "✅ 완료! 서버 로그: ${SRV_LOG}, 평가 로그: ${EVAL_LOG}"