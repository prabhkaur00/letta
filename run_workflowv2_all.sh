#!/usr/bin/env bash
set -euo pipefail

# Edit these lists as needed (comment out entries to skip)
DATASETS=(
  gsm8k
  # agentgym
  # prm800k
  # ultrachat
  # ultrafeedback
  # xlam_function_calling
)

MODES=(
  search_then_step_insert
  step_search_then_insert
  one_search_one_insert
  search_only
)

FAISS_INDEX="/data/IVF.direct.index"
LLM_PARALLEL=100

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "[run] dataset=${dataset} mode=${mode}"
    python letta_llm_agent_workflowv3.py \
      --dataset "${dataset}" \
      --mode "${mode}" \
      --faiss-index "${FAISS_INDEX}" \
      --llm-parallel "${LLM_PARALLEL}" \
      --llm-url "http://localhost:8000/v1/chat/completions" \
      --faiss-limit 8000
  done
done
