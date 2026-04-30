#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export CSV tables for the single-strategy entropy-position plots used by `core/ent_pos.py`.

Outputs:
- qid_labels.csv
- group_counts.csv
- norm_prompt_curves.csv
- norm_group_summary.csv
- norm_matched_prompt_curves.csv
- norm_matched_group_summary.csv
- abs_prompt_curves.csv
- abs_group_summary.csv
- abs_matched_prompt_curves.csv
- abs_matched_group_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import ent_pos as core  # type: ignore

from _common import ensure_dir, group_summary_rows, transition_group, write_csv


def build_curve_rows(curve_map, qids, group_name: str, axis_values: np.ndarray):
    for qid in qids:
        curve = curve_map.get(qid)
        if curve is None:
            continue
        for axis_idx, (axis_val, value) in enumerate(zip(axis_values, curve)):
            if not np.isfinite(value):
                continue
            yield {
                "qid": qid,
                "group": group_name,
                "axis_index": int(axis_idx),
                "axis_value": float(axis_val),
                "value": float(value),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_base", type=str, required=True)
    ap.add_argument("--train_post", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--strategy", type=str, required=True)
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--token_metric", type=str, default="entropy", choices=["entropy", "gini"])
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--early_frac", type=float, default=0.2)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--make_matched", action="store_true")
    ap.add_argument("--matched_seed", type=int, default=0)
    ap.add_argument("--plot_abs", action="store_true")
    ap.add_argument("--abs_T", type=int, default=1024)
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.strategy / "csv"
    ensure_dir(outdir)

    base_samples = core.load_samples(
        Path(args.train_base),
        pad_id=args.pad_id,
        max_len=args.max_len,
        metric=args.token_metric,
    )
    post_samples = core.load_samples(
        Path(args.train_post),
        pad_id=args.pad_id,
        max_len=args.max_len,
        metric=args.token_metric,
    )

    base_anyc = core.per_qid_any_correct(base_samples)
    post_anyc = core.per_qid_any_correct(post_samples)
    all_qids = sorted(set(base_anyc.keys()) | set(post_anyc.keys()))
    np_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and post_anyc.get(q, False)])
    nn_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and (not post_anyc.get(q, False))])
    keep = set(np_qids) | set(nn_qids)

    write_csv(
        outdir / "qid_labels.csv",
        ["qid", "base_anyc", "post_anyc", "transition_group"],
        (
            {
                "qid": qid,
                "base_anyc": int(bool(base_anyc.get(qid, False))),
                "post_anyc": int(bool(post_anyc.get(qid, False))),
                "transition_group": transition_group(bool(base_anyc.get(qid, False)), bool(post_anyc.get(qid, False))),
            }
            for qid in all_qids
        ),
    )

    q_curve_norm = core.prompt_level_mean_curves_normalized(base_samples, qids_keep=keep, M=args.M)
    np_mat = np.stack([q_curve_norm[q] for q in np_qids if q in q_curve_norm], axis=0).astype(np.float64)
    nn_mat = np.stack([q_curve_norm[q] for q in nn_qids if q in q_curve_norm], axis=0).astype(np.float64)
    x_norm = np.linspace(0.0, 1.0, args.M)

    write_csv(
        outdir / "group_counts.csv",
        ["curve_set", "group", "n_qids", "n_curves"],
        [
            {"curve_set": "norm_full", "group": "N2P", "n_qids": len(np_qids), "n_curves": int(np_mat.shape[0])},
            {"curve_set": "norm_full", "group": "N2N", "n_qids": len(nn_qids), "n_curves": int(nn_mat.shape[0])},
        ],
    )

    write_csv(
        outdir / "norm_prompt_curves.csv",
        ["qid", "group", "axis_index", "axis_value", "value"],
        list(build_curve_rows(q_curve_norm, np_qids, "N2P", x_norm))
        + list(build_curve_rows(q_curve_norm, nn_qids, "N2N", x_norm)),
    )
    write_csv(
        outdir / "norm_group_summary.csv",
        ["group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode"],
        group_summary_rows({"N2P": np_mat, "N2N": nn_mat}, x_norm, band_mode="quantile"),
    )

    if args.make_matched:
        np_qids_m, nn_qids_m, _ = core.downsample_matched(np_qids, nn_qids, seed=args.matched_seed)
        np_mat_m = np.stack([q_curve_norm[q] for q in np_qids_m if q in q_curve_norm], axis=0).astype(np.float64)
        nn_mat_m = np.stack([q_curve_norm[q] for q in nn_qids_m if q in q_curve_norm], axis=0).astype(np.float64)
        write_csv(
            outdir / "norm_matched_prompt_curves.csv",
            ["qid", "group", "axis_index", "axis_value", "value"],
            list(build_curve_rows(q_curve_norm, np_qids_m, "N2P", x_norm))
            + list(build_curve_rows(q_curve_norm, nn_qids_m, "N2N", x_norm)),
        )
        write_csv(
            outdir / "norm_matched_group_summary.csv",
            ["group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode"],
            group_summary_rows({"N2P": np_mat_m, "N2N": nn_mat_m}, x_norm, band_mode="quantile"),
        )

    if args.plot_abs:
        q_curve_abs = core.prompt_level_mean_curves_absolute(base_samples, qids_keep=keep, abs_T=args.abs_T)
        np_abs = np.stack([q_curve_abs[q] for q in np_qids if q in q_curve_abs], axis=0).astype(np.float64)
        nn_abs = np.stack([q_curve_abs[q] for q in nn_qids if q in q_curve_abs], axis=0).astype(np.float64)
        x_abs = np.arange(args.abs_T, dtype=np.float64)
        write_csv(
            outdir / "abs_prompt_curves.csv",
            ["qid", "group", "axis_index", "axis_value", "value"],
            list(build_curve_rows(q_curve_abs, np_qids, "N2P", x_abs))
            + list(build_curve_rows(q_curve_abs, nn_qids, "N2N", x_abs)),
        )
        write_csv(
            outdir / "abs_group_summary.csv",
            ["group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode"],
            group_summary_rows({"N2P": np_abs, "N2N": nn_abs}, x_abs, band_mode="quantile"),
        )

        if args.make_matched:
            np_qids_m, nn_qids_m, _ = core.downsample_matched(np_qids, nn_qids, seed=args.matched_seed)
            np_abs_m = np.stack([q_curve_abs[q] for q in np_qids_m if q in q_curve_abs], axis=0).astype(np.float64)
            nn_abs_m = np.stack([q_curve_abs[q] for q in nn_qids_m if q in q_curve_abs], axis=0).astype(np.float64)
            write_csv(
                outdir / "abs_matched_prompt_curves.csv",
                ["qid", "group", "axis_index", "axis_value", "value"],
                list(build_curve_rows(q_curve_abs, np_qids_m, "N2P", x_abs))
                + list(build_curve_rows(q_curve_abs, nn_qids_m, "N2N", x_abs)),
            )
            write_csv(
                outdir / "abs_matched_group_summary.csv",
                ["group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode"],
                group_summary_rows({"N2P": np_abs_m, "N2N": nn_abs_m}, x_abs, band_mode="quantile"),
            )

    print(f"[OK] csv outdir = {outdir}")


if __name__ == "__main__":
    main()
