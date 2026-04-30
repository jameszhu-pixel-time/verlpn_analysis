#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"

export MPLCONFIGDIR=/DATA/disk1/zhurui/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"
strata=normal_1
stratb=normal_2
python "$CORE_DIR/compare_entropy_paper.py" \
  --A_base /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/$strata/verl_rollouts/2.jsonl \
  --A_post /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/$strata/vllm_rollouts/grpo_step200-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --B_base /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/$stratb/verl_rollouts/2.jsonl\
  --B_post /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/$stratb/vllm_rollouts/grpo_step200-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --A_name $strata \
  --B_name $stratb \
  --outdir "/DATA/disk2/zhurui/A_entry/paper_codex/compare_entropy_${strata}vs${stratb}"  \
  --M 256 \
  --early_frac 0.2 \
  --max_len 3072 \
  --pad_id 151643 \
  --abs_T 3072 \
  --zoom_skip_frac 0.02 \
  --zoom_skip_abs 32

# Paper-oriented comparison:
# - own base for each strategy
# - full curve + early-distribution shift
