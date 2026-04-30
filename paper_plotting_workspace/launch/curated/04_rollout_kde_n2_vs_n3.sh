#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"
DRIVERS_PY=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py
infer_name=grpo_step200-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl

python "$CORE_DIR/compare_g.py" \
  --train n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/verl_rollouts/2.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/verl_rollouts/2.jsonl \
  --np_base n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/verl_rollouts/2.jsonl \
  --np_base n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/verl_rollouts/2.jsonl \
  --np_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/vllm_rollouts/$infer_name \
  --np_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/vllm_rollouts/$infer_name \
  --pair n1,n2 \
  --drivers_py "$DRIVERS_PY" \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew

python "$CORE_DIR/compare_g.py" \
  --train n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/verl_rollouts/2.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/verl_rollouts/2.jsonl \
  --np_base n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/verl_rollouts/2.jsonl \
  --np_base n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/verl_rollouts/2.jsonl \
  --np_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/vllm_rollouts/$infer_name \
  --np_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/vllm_rollouts/$infer_name \
  --pair n1,n3 \
  --drivers_py "$DRIVERS_PY" \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew

python "$CORE_DIR/compare_g.py" \
  --train n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/verl_rollouts/2.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/verl_rollouts/2.jsonl \
  --np_base n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/verl_rollouts/2.jsonl \
  --np_base n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/verl_rollouts/2.jsonl \
  --np_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/vllm_rollouts/$infer_name \
  --np_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/vllm_rollouts/$infer_name \
  --pair n2,n3 \
  --drivers_py "$DRIVERS_PY" \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew
# This is the active replacement for the old n1-vs-n3 naming drift.
