#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRACTOR_DIR="$(cd "$SCRIPT_DIR/../../extractors" && pwd)"

export MPLCONFIGDIR=/tmp/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"

python "$EXTRACTOR_DIR/extract_compare_entropy_csv.py" \
  --train_base /DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
  --A_name normal_2 \
  --A_post /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --B_name normal_3 \
  --B_post /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex_csv/compare_normal_2_vs_normal_3 \
  --M 256 \
  --early_frac 0.2 \
  --boot 2000 \
  --max_len 3072 \
  --pad_id 151643 \
  --band_mode ci \
  --make_matched \
  --matched_seed 0 \
  --plot_abs \
  --abs_T 1024
