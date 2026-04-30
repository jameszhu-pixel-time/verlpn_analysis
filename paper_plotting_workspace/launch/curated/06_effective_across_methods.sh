#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"
infer_name=grpo_step200-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl
python "$CORE_DIR/effective_rollout_across_methods.py" \
  --base n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/verl_rollouts/2.jsonl \
  --base n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/verl_rollouts/2.jsonl \
  --base n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/verl_rollouts/2.jsonl \
  --post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/vllm_rollouts/$infer_name \
  --post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/vllm_rollouts/$infer_name \
  --post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/vllm_rollouts/$infer_name \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex/effective_across_methods_2_13 \
  --subset all \
  --abs_T 3072 \
  --rel_bins 128 \
  --boot 1000 \
  --max_len 3072 \
  --pad_id 151643 \
  --prototype n2 \
  --dist_metric cosine
