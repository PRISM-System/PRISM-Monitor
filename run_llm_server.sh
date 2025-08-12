#!/bin/bash

# vLLM 서버 실행 스크립트
# 모델: NousResearch/Meta-Llama-3-8B-Instruct
# dtype: auto
# API 키: token-abc123

vllm serve \
  Qwen/Qwen3-0.6B \
  --port 8001 
