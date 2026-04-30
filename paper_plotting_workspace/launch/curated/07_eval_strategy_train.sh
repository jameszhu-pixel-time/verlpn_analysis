#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"
DRIVERS_PY=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py
infer_name=grpo_step200-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl
python "$CORE_DIR/eval_strategy_np_driver.py" \
  --train n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/verl_rollouts/2.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/verl_rollouts/2.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/verl_rollouts/2.jsonl \
  --train_ref n2 \
  --val_base /DATA/disk2/zhurui/A_entry/new_ablation/GRPO_ablation/2.jsonl\
  --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_1/vllm_rollouts/$infer_name \
  --val_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_2/vllm_rollouts/$infer_name \
  --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/normal_3/vllm_rollouts/$infer_name \
  --drivers_py "$DRIVERS_PY" \
  --outdir /DATA/disk2/zhurui/A_entry/results/train_np_driver_out_n2n3_n1 \
  --x_metric tail_mass \
  --subset ALL4 \
  --max_len 3072 \
  --pad_id 151643

# This is not a direct figure exporter, but it is a useful scaffold for the
# next round of analysis after the plotting redesign.
