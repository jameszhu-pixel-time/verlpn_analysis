#!/usr/bin/env bash
# c

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../../core" && pwd)"

OUTDIR=/DATA/disk2/zhurui/A_entry/new_paper_1_2/intraplot
POST_ROOT=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2

export MPLCONFIGDIR=/DATA/disk1/zhurui/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"

for strat in normal_1 normal_2 normal_3; do
  label="${strat/normal_/n}"
  python "$CORE_DIR/ent_pos_paper.py" \
    --train_base "/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_1_2/$strat/verl_rollouts/2.jsonl" \
    --train_post "${POST_ROOT}/${strat}/vllm_rollouts/grpo_step200-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl" \
    --strategy "$strat" \
    --display_name "$label" \
    --outdir "$OUTDIR" \
    --token_metric entropy \
    --M 256 \
    --early_frac 0.2 \
    --max_len 3072 \
    --pad_id 151643 \
    --abs_T 3072 \
    --zoom_skip_frac 0.02 \
    --zoom_skip_abs 32
done

# Historical note:
# `n1` was produced by the same script family, but that checkpoint is now lost.
