# Paper Plotting Workspace

This folder is a focused extraction of the useful plotting and analysis code from `Inference_pipeline/` for the eight paper figures listed in `paper/piece_of_main.tex`.

Current working assumption:
- `n1` is lost and cannot be reproduced now.
- Active analysis should be based on `n2`, `n3`, `annealed`, and `original/ref`.
- Historical `n1` references are preserved only for provenance in `launch/legacy/`.

## Directory Layout

- `core/`
  Core Python scripts copied from `Inference_pipeline/` and kept together so later refactors can happen in one place.

  - `ent_pos.py`
    Single-strategy entropy-position plots. Splits prompts into `N2P` / `N2N` using base -> post labels and plots curves from the base rollouts.
  - `compare.py`
    Two-strategy entropy-position comparison on a shared base file. This is the most likely source for `compared_entropy.png`.
  - `compare_g.py`
    Rollout-level KDE comparison across two strategies. Useful for the batch-level driver-shift figures.
  - `effective_rollout_across_methods.py`
    Cross-method effective vs ineffective rollout comparison.
  - `simple_essay.py`
    Patched paper-oriented intra-strategy driver plots. Only keeps the `all` and `neg` subsets. This is the cleanest local entry for the within-batch separation figures.
  - `simple.py`
    Original intra-strategy driver script kept for provenance.
  - `eval_strategy_np_driver.py`
    Strategy-level driver comparison plus validation N2P-rate analysis.
  - `train_rollout_driver_with_nplabels.py`
    Training-side bridge script. Compares driver deltas with N2P/N2N labels.
  - `drivers.py`
    Local driver definitions copied from `Inference_pipeline/driver-NP/drivers.py`.

- `launch/legacy/`
  Verbatim copies of the old shell launchers from `Inference_pipeline/`. These are useful as provenance, but some of them are stale.

- `launch/curated/`
  Cleaned launchers that point to scripts in `./core/`, keep the currently useful remote paths, and avoid the lost `n1` dependency where possible.

- `extractors/`
  CSV exporters for the current paper-facing plot families. See
  [extractors/README.md](/Users/zhurui/Desktop/verlpn_analysis/paper_plotting_workspace/extractors/README.md).

## Important Findings

### 1. The code is not one unified framework

The plotting pipeline is really a small family of scripts that all reuse the same ideas:

- load rollout JSONL, optionally recover token entropy from NPZ
- build `any_correct(qid)` labels on base/post files
- split prompts or rollouts into `N2P` vs `N2N`
- aggregate either:
  - entropy-position curves, or
  - rollout/qid-level driver distributions

### 2. There are two driver sources

- Local fallback: `core/drivers.py`
- Remote paper driver file used by several legacy launchers:
  - `/DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py`

This matters because the local `drivers.py` does **not** currently expose the exact paper driver names such as:
- `early_mean_ppl`
- `early_128_token`
- likely also the late-stage counterpart used for the temperature-decay figure

So for paper-faithful reproduction, the remote external `drivers.py` is still part of the effective runtime.

### 3. One launcher is stale by filename

`launch/legacy/compare.sh` calls `ent_pos_compare2.py`, but the local file that matches that docstring is `core/compare.py`.

The curated replacement already fixes this.

### 4. The paper asset names are partly stale

`paper/piece_of_main.tex` still includes assets such as:
- `early_ppl_n1vsn3.png`

But the surrounding caption text says the comparison is between the stronger batch `n2` and the weaker batch `n3`. Given your note that `n1` is lost, this asset name should be treated as historical naming drift, not as the new analysis target.

## Figure Map

The mapping below is based on direct inspection of:
- `paper/piece_of_main.tex`
- the copied shell launchers
- the CLI behavior of the local scripts

When a row says `inferred`, that means the paper caption and the launcher behavior match strongly, but there is no local image file to verify the exact final asset export step.

| Paper asset | Paper caption | Likely script | Launcher in this folder | Remote data pairing | Status |
| --- | --- | --- | --- | --- | --- |
| `ent_position.png` | `Entropy-position plot for n1 checkpoint.` | `core/ent_pos.py` | historical only: see `launch/legacy/ent_pos.sh` | `normal_1__3.jsonl` + `normal_1/vllm_rollouts_training/...` | historical only, blocked by lost `n1` |
| `compared_entropy.png` | `Entropy-position plot for n2(good) vs n3(bad) checkpoint.` | `core/compare.py` | `launch/curated/02_compare_entropy_n2_vs_n3.sh` | shared base `results/verl_2_3/3.jsonl` plus post files for `normal_2` and `normal_3` | active |
| `early_ppl_n2.png` | `Rollout groups from n2: driver early_mean_ppl` | `core/simple_essay.py` or `core/simple.py` | `launch/curated/03_intra_driver_groups_n2_n3_annealed.sh` | train `normal_2__3.jsonl`, labels from base/post training rollout files | active, but depends on remote external `drivers.py` |
| `early128_token.png` | `Rollout groups from n2: driver early_128_token` | `core/simple_essay.py` or `core/simple.py` | `launch/curated/03_intra_driver_groups_n2_n3_annealed.sh` | same as above | active, but depends on remote external `drivers.py` |
| `early_ppl_n1vsn3.png` | `Rollout compared groups: driver early_mean_ppl` | `core/compare_g.py` | `launch/curated/04_rollout_kde_n2_vs_n3.sh` | pairwise rollout comparison for `normal_2` vs `normal_3` | active target should now be `n2` vs `n3`; paper asset name is stale |
| `norm_vs_anneal.png` | `Rollout compared groups: normal vs annealed(intervene)` | `core/compare_g.py` | `launch/curated/05_rollout_kde_n2_vs_annealed.sh` | pairwise rollout comparison for `normal_2` vs `annealed` | active |
| `temp_decayed_early.png` | `Early-stage uncertainty` | inferred: `core/simple_essay.py` or `core/compare_g.py` with an early driver | `launch/curated/03_intra_driver_groups_n2_n3_annealed.sh` and `launch/curated/05_rollout_kde_n2_vs_annealed.sh` | `normal_2` vs `annealed` | active, exact final driver name still lives in remote `drivers.py` |
| `temp_decayed_late.png` | `Late-stage uncertainty` | inferred: same family as above, but late-stage driver | `launch/curated/03_intra_driver_groups_n2_n3_annealed.sh` and `launch/curated/05_rollout_kde_n2_vs_annealed.sh` | `normal_2` vs `annealed` | active, exact final driver name still lives in remote `drivers.py` |

## Main Remote Paths

These are the recurring remote JSONL sources that appear across the useful launchers:

- `original/ref` train base
  - `/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl`
- `original/ref` post labels
  - `/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl`

- `n2` train base
  - `/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl`
- `n2` post labels
  - `/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl`

- `n3` train base
  - `/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl`
- `n3` post labels
  - `/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl`

- `annealed` train base
  - `/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl`
- `annealed` post labels
  - `/DATA/disk1/zhurui/ablation_study_step_2_3/annealed/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl`

There are a few legacy variants under `disk2` and with `vllm_rollouts_train` instead of `vllm_rollouts_training`; those are preserved in `launch/legacy/` exactly as found.

## Recommended Starting Points

If the next step is to redesign the figures around the actual insights, start here:

1. `core/ent_pos.py`
   For rethinking the entropy-position curves themselves.

2. `core/simple_essay.py`
   For the within-batch `N2P` vs `N2N` driver separation figures.

3. `core/compare_g.py`
   For the batch-level shift and normal-vs-annealed comparisons.

4. `core/effective_rollout_across_methods.py`
   If you want method-level overlay plots rather than scalar driver histograms.

5. `core/drivers.py` plus the remote external `drivers.py`
   Because the real figure semantics are encoded in the driver definitions.

## What To Ignore For Now

- Any workflow that requires `n1` for the main branch of analysis
- The raw `launch/legacy/compare.sh` script name as written; use the curated replacement instead
- The local `core/drivers.py` as the only source of truth for paper figures

## Next Refactor Direction

The cleanest next step is to split this workspace into:

- `io.py`
  shared JSONL/NPZ loading and masking
- `labels.py`
  `any_correct`, `N2P`, `N2N`, effective/ineffective grouping
- `drivers.py`
  all early/late metrics in one place
- `plots/`
  consistent plotting API for:
  - entropy-position curves
  - within-strategy driver separation
  - cross-strategy rollout KDE
  - temperature-intervention comparisons

That will make it much easier to redesign the plots around the insights rather than around the current script boundaries.
