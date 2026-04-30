#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"
DRIVERS_PY=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py

python "$CORE_DIR/compare_g.py" \
  --train normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --np_base normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --np_base normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --np_post normal_2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post normal_3=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --pair normal_2,normal_3 \
  --drivers_py "$DRIVERS_PY" \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew

# This is the active replacement for the old n1-vs-n3 naming drift.
