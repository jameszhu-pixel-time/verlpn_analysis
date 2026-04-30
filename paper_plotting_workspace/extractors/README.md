# CSV Extractors

These scripts export plotting data as CSV so you can redesign each figure without re-reading the raw JSONL/NPZ every time.

Matching launchers live in:
- `paper_plotting_workspace/launch/extractors/`

## Coverage

- `extract_ent_pos_csv.py`
  Single-strategy entropy-position curves from `core/ent_pos.py`

- `extract_compare_entropy_csv.py`
  Two-strategy entropy-position comparison from `core/compare.py`

- `extract_intra_driver_csv.py`
  Intra-strategy qid-level driver separation from `core/simple_essay.py`

- `extract_rollout_pair_csv.py`
  Pairwise rollout-level driver KDE inputs from `core/compare_g.py`

- `extract_effective_across_methods_csv.py`
  Cross-method effective vs ineffective rollout curves from `core/effective_rollout_across_methods.py`

These five cover the current paper-facing figure families:
- entropy-position
- within-batch driver separation
- stronger-vs-weaker batch driver shift
- normal-vs-annealed comparison
- cross-method effective/ineffective overlays

## Output Style

Each extractor writes two kinds of CSV:

- raw or semi-raw long tables
  Examples:
  - per-qid curves
  - per-rollout driver values
  - per-rollout entropy curves

- aggregated plotting tables
  Examples:
  - mean curve with confidence band
  - histogram bin densities
  - KDE values on a common x-grid
  - ECDF tables

This means later plotting can happen directly from CSV, without recomputing labels and grouping logic.

## Notes

- `extract_compare_entropy_csv.py` intentionally matches the current behavior of `core/compare.py`.
  The current plot code treats `--band_mode ci` as a quantile band in practice.

- Driver-based figures may still depend on the remote external driver file:
  `/DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py`

- These scripts do not render figures.
  They only export the data needed to render them.
