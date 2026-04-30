#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export CSV tables for the two-strategy entropy-position comparison used by `core/compare.py`.

Important:
- `core/compare.py` currently treats `--band_mode ci` the same as quantile band in practice.
- This extractor matches the script's current behavior, not the intended label.
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

import compare as core  # type: ignore

from _common import ensure_dir, group_summary_rows, transition_group, write_csv


def build_curve_rows(curve_map, qids, strategy: str, group_name: str, axis_values: np.ndarray):
    for qid in qids:
        curve = curve_map.get(qid)
        if curve is None:
            continue
        for axis_idx, (axis_val, value) in enumerate(zip(axis_values, curve)):
            if not np.isfinite(value):
                continue
            yield {
                "strategy": strategy,
                "qid": qid,
                "group": group_name,
                "axis_index": int(axis_idx),
                "axis_value": float(axis_val),
                "value": float(value),
            }


def applied_band_mode(requested: str) -> str:
    if requested == "std":
        return "std"
    if requested == "bootstrap":
        return "bootstrap"
    return "quantile"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_base", type=str, required=True)
    ap.add_argument("--A_name", type=str, required=True)
    ap.add_argument("--A_post", type=str, required=True)
    ap.add_argument("--B_name", type=str, required=True)
    ap.add_argument("--B_post", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--early_frac", type=float, default=0.2)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--band_mode", choices=["ci", "std", "quantile", "bootstrap"], default="ci")
    ap.add_argument("--make_matched", action="store_true")
    ap.add_argument("--matched_seed", type=int, default=0)
    ap.add_argument("--plot_abs", action="store_true")
    ap.add_argument("--abs_T", type=int, default=1024)
    args = ap.parse_args()

    outdir = Path(args.outdir) / "csv"
    ensure_dir(outdir)

    base_samples = core.load_samples(Path(args.train_base), pad_id=args.pad_id, max_len=args.max_len)
    base_anyc = core.per_qid_any_correct(base_samples)
    A_post_samples = core.load_samples(Path(args.A_post), pad_id=args.pad_id, max_len=args.max_len)
    B_post_samples = core.load_samples(Path(args.B_post), pad_id=args.pad_id, max_len=args.max_len)
    A_anyc = core.per_qid_any_correct(A_post_samples)
    B_anyc = core.per_qid_any_correct(B_post_samples)
    A_np, A_nn = core.collect_strategy_np_nn_qids(base_anyc, A_anyc)
    B_np, B_nn = core.collect_strategy_np_nn_qids(base_anyc, B_anyc)

    write_csv(
        outdir / "qid_labels.csv",
        ["strategy", "qid", "base_anyc", "post_anyc", "transition_group"],
        [
            {
                "strategy": args.A_name,
                "qid": qid,
                "base_anyc": int(bool(base_anyc.get(qid, False))),
                "post_anyc": int(bool(A_anyc.get(qid, False))),
                "transition_group": transition_group(bool(base_anyc.get(qid, False)), bool(A_anyc.get(qid, False))),
            }
            for qid in sorted(set(base_anyc.keys()) | set(A_anyc.keys()))
        ]
        + [
            {
                "strategy": args.B_name,
                "qid": qid,
                "base_anyc": int(bool(base_anyc.get(qid, False))),
                "post_anyc": int(bool(B_anyc.get(qid, False))),
                "transition_group": transition_group(bool(base_anyc.get(qid, False)), bool(B_anyc.get(qid, False))),
            }
            for qid in sorted(set(base_anyc.keys()) | set(B_anyc.keys()))
        ],
    )

    keep_all = set(A_np) | set(A_nn) | set(B_np) | set(B_nn)
    q_curve_norm = core.prompt_level_mean_curves_normalized(base_samples, qids_keep=keep_all, M=args.M)
    A_np_mat = np.stack([q_curve_norm[q] for q in A_np if q in q_curve_norm], axis=0).astype(np.float64)
    A_nn_mat = np.stack([q_curve_norm[q] for q in A_nn if q in q_curve_norm], axis=0).astype(np.float64)
    B_np_mat = np.stack([q_curve_norm[q] for q in B_np if q in q_curve_norm], axis=0).astype(np.float64)
    B_nn_mat = np.stack([q_curve_norm[q] for q in B_nn if q in q_curve_norm], axis=0).astype(np.float64)
    x_norm = np.linspace(0.0, 1.0, args.M)

    write_csv(
        outdir / "group_counts.csv",
        ["curve_set", "strategy", "group", "n_qids", "n_curves"],
        [
            {"curve_set": "norm_full", "strategy": args.A_name, "group": "N2P", "n_qids": len(A_np), "n_curves": int(A_np_mat.shape[0])},
            {"curve_set": "norm_full", "strategy": args.A_name, "group": "N2N", "n_qids": len(A_nn), "n_curves": int(A_nn_mat.shape[0])},
            {"curve_set": "norm_full", "strategy": args.B_name, "group": "N2P", "n_qids": len(B_np), "n_curves": int(B_np_mat.shape[0])},
            {"curve_set": "norm_full", "strategy": args.B_name, "group": "N2N", "n_qids": len(B_nn), "n_curves": int(B_nn_mat.shape[0])},
        ],
    )

    write_csv(
        outdir / "norm_prompt_curves.csv",
        ["strategy", "qid", "group", "axis_index", "axis_value", "value"],
        list(build_curve_rows(q_curve_norm, A_np, args.A_name, "N2P", x_norm))
        + list(build_curve_rows(q_curve_norm, A_nn, args.A_name, "N2N", x_norm))
        + list(build_curve_rows(q_curve_norm, B_np, args.B_name, "N2P", x_norm))
        + list(build_curve_rows(q_curve_norm, B_nn, args.B_name, "N2N", x_norm)),
    )

    band_mode = applied_band_mode(args.band_mode)
    write_csv(
        outdir / "norm_group_summary.csv",
        ["strategy", "group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode", "band_mode_requested"],
        list(
            group_summary_rows(
                {"N2P": A_np_mat, "N2N": A_nn_mat},
                x_norm,
                band_mode=band_mode,
                boot=args.boot,
                extra_columns={"strategy": args.A_name, "band_mode_requested": args.band_mode},
            )
        )
        + list(
            group_summary_rows(
                {"N2P": B_np_mat, "N2N": B_nn_mat},
                x_norm,
                band_mode=band_mode,
                boot=args.boot,
                extra_columns={"strategy": args.B_name, "band_mode_requested": args.band_mode},
            )
        ),
    )

    if args.make_matched:
        A_np_m, A_nn_m, _ = core.downsample_qids(A_np, A_nn, seed=args.matched_seed)
        B_np_m, B_nn_m, _ = core.downsample_qids(B_np, B_nn, seed=args.matched_seed)
        A_np_m_mat = np.stack([q_curve_norm[q] for q in A_np_m if q in q_curve_norm], axis=0).astype(np.float64)
        A_nn_m_mat = np.stack([q_curve_norm[q] for q in A_nn_m if q in q_curve_norm], axis=0).astype(np.float64)
        B_np_m_mat = np.stack([q_curve_norm[q] for q in B_np_m if q in q_curve_norm], axis=0).astype(np.float64)
        B_nn_m_mat = np.stack([q_curve_norm[q] for q in B_nn_m if q in q_curve_norm], axis=0).astype(np.float64)
        write_csv(
            outdir / "norm_matched_prompt_curves.csv",
            ["strategy", "qid", "group", "axis_index", "axis_value", "value"],
            list(build_curve_rows(q_curve_norm, A_np_m, args.A_name, "N2P", x_norm))
            + list(build_curve_rows(q_curve_norm, A_nn_m, args.A_name, "N2N", x_norm))
            + list(build_curve_rows(q_curve_norm, B_np_m, args.B_name, "N2P", x_norm))
            + list(build_curve_rows(q_curve_norm, B_nn_m, args.B_name, "N2N", x_norm)),
        )
        write_csv(
            outdir / "norm_matched_group_summary.csv",
            ["strategy", "group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode", "band_mode_requested"],
            list(
                group_summary_rows(
                    {"N2P": A_np_m_mat, "N2N": A_nn_m_mat},
                    x_norm,
                    band_mode=band_mode,
                    boot=args.boot,
                    extra_columns={"strategy": args.A_name, "band_mode_requested": args.band_mode},
                )
            )
            + list(
                group_summary_rows(
                    {"N2P": B_np_m_mat, "N2N": B_nn_m_mat},
                    x_norm,
                    band_mode=band_mode,
                    boot=args.boot,
                    extra_columns={"strategy": args.B_name, "band_mode_requested": args.band_mode},
                )
            ),
        )

    if args.plot_abs:
        q_curve_abs = core.prompt_level_mean_curves_absolute(base_samples, qids_keep=keep_all, abs_T=args.abs_T)
        A_np_abs = np.stack([q_curve_abs[q] for q in A_np if q in q_curve_abs], axis=0).astype(np.float64)
        A_nn_abs = np.stack([q_curve_abs[q] for q in A_nn if q in q_curve_abs], axis=0).astype(np.float64)
        B_np_abs = np.stack([q_curve_abs[q] for q in B_np if q in q_curve_abs], axis=0).astype(np.float64)
        B_nn_abs = np.stack([q_curve_abs[q] for q in B_nn if q in q_curve_abs], axis=0).astype(np.float64)
        x_abs = np.arange(args.abs_T, dtype=np.float64)
        write_csv(
            outdir / "abs_prompt_curves.csv",
            ["strategy", "qid", "group", "axis_index", "axis_value", "value"],
            list(build_curve_rows(q_curve_abs, A_np, args.A_name, "N2P", x_abs))
            + list(build_curve_rows(q_curve_abs, A_nn, args.A_name, "N2N", x_abs))
            + list(build_curve_rows(q_curve_abs, B_np, args.B_name, "N2P", x_abs))
            + list(build_curve_rows(q_curve_abs, B_nn, args.B_name, "N2N", x_abs)),
        )
        write_csv(
            outdir / "abs_group_summary.csv",
            ["strategy", "group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode", "band_mode_requested"],
            list(
                group_summary_rows(
                    {"N2P": A_np_abs, "N2N": A_nn_abs},
                    x_abs,
                    band_mode=band_mode,
                    boot=args.boot,
                    extra_columns={"strategy": args.A_name, "band_mode_requested": args.band_mode},
                )
            )
            + list(
                group_summary_rows(
                    {"N2P": B_np_abs, "N2N": B_nn_abs},
                    x_abs,
                    band_mode=band_mode,
                    boot=args.boot,
                    extra_columns={"strategy": args.B_name, "band_mode_requested": args.band_mode},
                )
            ),
        )

        if args.make_matched:
            A_np_m, A_nn_m, _ = core.downsample_qids(A_np, A_nn, seed=args.matched_seed)
            B_np_m, B_nn_m, _ = core.downsample_qids(B_np, B_nn, seed=args.matched_seed)
            A_np_abs_m = np.stack([q_curve_abs[q] for q in A_np_m if q in q_curve_abs], axis=0).astype(np.float64)
            A_nn_abs_m = np.stack([q_curve_abs[q] for q in A_nn_m if q in q_curve_abs], axis=0).astype(np.float64)
            B_np_abs_m = np.stack([q_curve_abs[q] for q in B_np_m if q in q_curve_abs], axis=0).astype(np.float64)
            B_nn_abs_m = np.stack([q_curve_abs[q] for q in B_nn_m if q in q_curve_abs], axis=0).astype(np.float64)
            write_csv(
                outdir / "abs_matched_prompt_curves.csv",
                ["strategy", "qid", "group", "axis_index", "axis_value", "value"],
                list(build_curve_rows(q_curve_abs, A_np_m, args.A_name, "N2P", x_abs))
                + list(build_curve_rows(q_curve_abs, A_nn_m, args.A_name, "N2N", x_abs))
                + list(build_curve_rows(q_curve_abs, B_np_m, args.B_name, "N2P", x_abs))
                + list(build_curve_rows(q_curve_abs, B_nn_m, args.B_name, "N2N", x_abs)),
            )
            write_csv(
                outdir / "abs_matched_group_summary.csv",
                ["strategy", "group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode", "band_mode_requested"],
                list(
                    group_summary_rows(
                        {"N2P": A_np_abs_m, "N2N": A_nn_abs_m},
                        x_abs,
                        band_mode=band_mode,
                        boot=args.boot,
                        extra_columns={"strategy": args.A_name, "band_mode_requested": args.band_mode},
                    )
                )
                + list(
                    group_summary_rows(
                        {"N2P": B_np_abs_m, "N2N": B_nn_abs_m},
                        x_abs,
                        band_mode=band_mode,
                        boot=args.boot,
                        extra_columns={"strategy": args.B_name, "band_mode_requested": args.band_mode},
                    )
                ),
            )

    print(f"[OK] csv outdir = {outdir}")


if __name__ == "__main__":
    main()
