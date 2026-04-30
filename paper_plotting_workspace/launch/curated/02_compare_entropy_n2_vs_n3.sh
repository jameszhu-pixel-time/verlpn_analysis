#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"

export MPLCONFIGDIR=/tmp/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"

python "$CORE_DIR/compare_entropy_paper.py" \
  --A_base /DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --A_post /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --B_base /DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --B_post /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --A_name normal_2 \
  --B_name normal_3 \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex/compare_entropy_n2_vs_n3 \
  --M 256 \
  --early_frac 0.2 \
  --max_len 3072 \
  --pad_id 151643 \
  --abs_T 3072 \
  --zoom_skip_frac 0.02 \
  --zoom_skip_abs 32

# Paper-oriented comparison:
# - own base for each strategy
# - writes four separate panels: relative curve, relative shift, absolute curve, absolute shift
# - stats JSON includes input-shift indicators from shift_indicator.tex: msk, bsk, rt
