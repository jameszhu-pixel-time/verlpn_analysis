  --train original=/DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --val_base /DATA/disk2/zhurui/A_entry/results/aime_2_3/grpo_step2-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts2/grpo_step3-马刺3-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post original=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
/DATA/disk2/zhurui/A_entry/ablation_train/dapo
# for strat in normal_1 normal_2 normal_3;do
#     python ent_pos.py \
#         --train_base /DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
#         --train_post /DATA/disk2/zhurui/A_entry/results/verl_2_3/${strat}__3.jsonl \
#         --strategy $strat \
#         --outdir /DATA/disk2/zhurui/A_entry/plots/ent_pos$strat \
#         --M 256 \
#         --early_frac 0.2 \
#         --boot 2000 \
#         --max_len 3072 \
#         --pad_id 151643 \
#         --plot_abs \
#         --abs_T 1024
# done
# /DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/verl_rollouts
#!/usr/bin/env bash
/DATA/disk1/zhurui/ablation_study_step_2_3/normal_1/verl_rollouts
set -euo pipefail
SCRIPT=ent_pos.py
OUTDIR=/DATA/disk2/zhurui/A_entry/paper_codex/intraplot
BASE=/DATA/disk2/zhurui/A_entry/results/verl_2_3
POST_ROOT=/DATA/disk1/zhurui/ablation_study_step_2_3

for strat in normal_2 normal_3; do
  python "$SCRIPT" \
    --train_base "/DATA/disk2/zhurui/A_entry/results/verl_2_3/${strat}__3.jsonl" \
    --train_post "${POST_ROOT}/${strat}/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl" \
    --strategy "$strat" \
    --outdir "$OUTDIR" \
    --M 256 \
    --early_frac 0.2 \
    --boot 2000 \
    --max_len 3072 \
    --pad_id 151643 \
    --make_matched \
    --matched_seed 0 \
    --plot_abs \
    --abs_T 1024
done

SCRIPT=ent_pos.py
OUTDIR=/DATA/disk2/zhurui/A_entry/plots
python "$SCRIPT" \
    --train_base "/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl" \
    --train_post "/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl" \
    --strategy "original" \
    --outdir "$OUTDIR" \
    --M 256 \
    --early_frac 0.2 \
    --boot 2000 \
    --max_len 3072 \
    --pad_id 151643 \
    --make_matched \
    --matched_seed 0 \
    --plot_abs \
    --abs_T 1024
