#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRACTOR_DIR="$(cd "$SCRIPT_DIR/../../extractors" && pwd)"

export MPLCONFIGDIR=/tmp/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"

python "$EXTRACTOR_DIR/extract_effective_across_methods_csv.py" \
  --base ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --base n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --base n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --base annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --post n2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --post n3=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --post annealed=/DATA/disk1/zhurui/ablation_study_step_2_3/annealed/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex_csv/effective_across_methods_2_3 \
  --subset all \
  --abs_T 3072 \
  --rel_bins 128 \
  --boot 1000 \
  --max_len 3072 \
  --pad_id 151643 \
  --prototype n2 \
  --dist_metric cosine
