#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRACTOR_DIR="$(cd "$SCRIPT_DIR/../../extractors" && pwd)"

export MPLCONFIGDIR=/tmp/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"

DRIVERS_PY=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py

python "$EXTRACTOR_DIR/extract_rollout_pair_csv.py" \
  --train normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --np_base normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --np_base annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --np_post normal_2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post annealed=/DATA/disk1/zhurui/ablation_study_step_2_3/annealed/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --pair normal_2,annealed \
  --drivers_py "$DRIVERS_PY" \
  --outdir /DATA/disk2/zhurui/A_entry/paper_codex_csv/rollout_pairs \
  --pad_id 151643 \
  --max_len 3072
