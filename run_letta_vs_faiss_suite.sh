#!/usr/bin/env bash
#
# Convenience driver that sweeps multiple dataset/mode combinations through
# letta_vs_faiss.py. Mirrors the M3 suite runner: configure env vars for Faiss
# indexes + parameters and the script will emit JSON summaries per dataset/mode.

set -euo pipefail

topk="${TOPK:-5}"
search_batch="${SEARCH_BATCH:-256}"
insert_batch="${INSERT_BATCH:-256}"
hydrate_batch="${HYDRATE_BATCH:-2048}"
ops_per_run="${OPS_PER_RUN:-200}"
query_limit="${QUERY_LIMIT:-10000}"
limit="${DATASET_LIMIT:-0}"
modes="${MODES:-all}"
log_root="${LOG_ROOT:-benchmarks/letta_vs_faiss}"
faiss_root="${FAISS_INDEX_ROOT:-}"
m3_repo="${M3_REPO:-}"
agent_model="${AGENT_MODEL:-letta/letta-free}"
embedding_model="${EMBEDDING_MODEL:-letta/letta-free}"
datasets_env="${DATASETS:-gsm8k agentgym prm800k ultrachat ultrafeedback xlam_function_calling}"
extra_args="${LETTABENCH_EXTRA_ARGS:-}"

if [[ -z "${faiss_root}" ]]; then
  echo "FAISS_INDEX_ROOT must be set (either a single .index file or a directory containing <dataset>.index files)." >&2
  exit 1
fi

mkdir -p "${log_root}"

IFS=' ' read -r -a datasets <<< "${datasets_env}"

for ds in "${datasets[@]}"; do
  faiss_index="${faiss_root}"
  if [[ ! -f "${faiss_index}" ]]; then
    candidate="${faiss_root}/${ds}.index"
    if [[ -f "${candidate}" ]]; then
      faiss_index="${candidate}"
    else
      echo "[skip] dataset=${ds} (Faiss index not found at ${faiss_root} or ${candidate})" >&2
      continue
    fi
  fi

  log_base="${log_root}/${ds}.json"
  printf '[suite] dataset=%s modes=%s -> %s\n' "${ds}" "${modes}" "${log_base}"

  cmd=(
    python3 letta_vs_faiss.py
    --dataset "${ds}"
    --mode "${modes}"
    --faiss-index "${faiss_index}"
    --search-batch "${search_batch}"
    --insert-batch "${insert_batch}"
    --hydrate-batch "${hydrate_batch}"
    --ops-per-run "${ops_per_run}"
    --top-k "${topk}"
    --query-limit "${query_limit}"
    --limit "${limit}"
    --agent-model "${agent_model}"
    --embedding-model "${embedding_model}"
    --log-file "${log_base}"
  )
  if [[ -n "${m3_repo}" ]]; then
    cmd+=(--m3-repo "${m3_repo}")
  fi
  if [[ -n "${extra_args}" ]]; then
    # shellcheck disable=SC2206
    extra_split=(${extra_args})
    cmd+=("${extra_split[@]}")
  fi

  "${cmd[@]}"
done
