#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/DATA/disk2/zhurui/A_entry/Inference_pipeline/ent_pos/compare_g.py
ROOT_OUT=/DATA/disk2/zhurui/A_entry/paper_codex
OUTDIR=${ROOT_OUT}/compare_normal2_vs_normal3
DRIVERS=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py

mkdir -p "$OUTDIR"

python "$SCRIPT" \
  --train origin=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --train normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  \
  --np_base origin=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --np_base normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  \
  --np_post origin=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post normal_3=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  \
  --pair origin,normal_3 \
  --drivers_py "$DRIVERS" \
  --outdir "$OUTDIR" \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew
#normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl
#normal_2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \

#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/DATA/disk2/zhurui/A_entry/Inference_pipeline/ent_pos/compare_g.py
ROOT_OUT=/DATA/disk2/zhurui/A_entry/paper_codex
OUTDIR=${ROOT_OUT}/compare_normal2_vs_normal3
DRIVERS=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py

mkdir -p "$OUTDIR"

python "$SCRIPT" \
  --train normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  \
  --np_base normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --np_base normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  \
  --np_post normal_2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post normal_3=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  \
  --pair normal_2,normal_3 \
  --drivers_py "$DRIVERS" \
  --outdir "$OUTDIR" \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew
#normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl
#normal_2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \

SCRIPT=/DATA/disk2/zhurui/A_entry/Inference_pipeline/ent_pos/compare_g.py
ROOT_OUT=/DATA/disk2/zhurui/A_entry/paper_codex
OUTDIR=${ROOT_OUT}/compare_annealed_vs_normal3
DRIVERS=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py

mkdir -p "$OUTDIR"

python "$SCRIPT" \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --train normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  \
  --np_base annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --np_base normal_3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  \
  --np_post annealed=/DATA/disk1/zhurui/ablation_study_step_2_3/annealed/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post normal_3=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  \
  --pair annealed,normal_3 \
  --drivers_py "$DRIVERS" \
  --outdir "$OUTDIR" \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew \
  --legend_show_skew
SCRIPT=/DATA/disk2/zhurui/A_entry/Inference_pipeline/ent_pos/compare_g.py
ROOT_OUT=/DATA/disk2/zhurui/A_entry/paper_codex
OUTDIR=${ROOT_OUT}/compare_annealed_vs_normal2
DRIVERS=/DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py


python "$SCRIPT" \
  --train normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  \
  --np_base normal_2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --np_base annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  \
  --np_post normal_2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post annealed=/DATA/disk1/zhurui/ablation_study_step_2_3/annealed/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  \
  --pair normal_2,annealed \
  --drivers_py "$DRIVERS" \
  --outdir "$OUTDIR" \
  --pad_id 151643 \
  --max_len 3072 \
  --save_right_skew 
