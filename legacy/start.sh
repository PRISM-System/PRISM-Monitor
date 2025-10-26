#!/bin/bash

# 첫 번째 명령어: API 서버 실행 (백그라운드에서 실행)
echo "Starting vLLM OpenAI API server..."
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B\
  --port 8001 \
  --host 0.0.0.0 &

# PID 저장
API_SERVER_PID=$!

# 두 번째 명령어: main_legacy.py 실행 (포그라운드에서 실행)
echo "Starting main.py..."
python main.py

# main_legacy.py가 끝나면 백그라운드 프로세스 종료 (선택사항)
echo "Stopping vLLM API server (PID $API_SERVER_PID)..."
kill $API_SERVER_PID