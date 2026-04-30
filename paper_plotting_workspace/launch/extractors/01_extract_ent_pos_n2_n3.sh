#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRACTOR_DIR="$(cd "$SCRIPT_DIR/../../extractors" && pwd)"

export MPLCONFIGDIR=/tmp/matplotlib-codex
mkdir -p "$MPLCONFIGDIR"

CSV_ROOT=/DATA/disk2/zhurui/A_entry/paper_codex_csv/ent_pos
POST_ROOT=/DATA/disk1/zhurui/ablation_study_step_2_3

for strat in normal_2 normal_3; do
  tag="${strat/normal_/n}"
  python "$EXTRACTOR_DIR/extract_ent_pos_csv.py" \
    --train_base "/DATA/disk2/zhurui/A_entry/results/verl_2_3/${strat}__3.jsonl" \
    --train_post "${POST_ROOT}/${strat}/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl" \
    --strategy "$tag" \
    --outdir "$CSV_ROOT" \
    --token_metric entropy \
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
