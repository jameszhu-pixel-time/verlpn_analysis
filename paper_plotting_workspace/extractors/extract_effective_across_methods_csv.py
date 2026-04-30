#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export CSV tables for the cross-method effective-vs-ineffective plots used by
`core/effective_rollout_across_methods.py`.
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

import effective_rollout_across_methods as core  # type: ignore

from _common import ensure_dir, group_summary_rows, summarize_1d, write_csv


def abs_curve_rows(strategy_to_samples, abs_T: int):
    axis_values = np.arange(abs_T, dtype=np.float64)
    for strategy, groups in strategy_to_samples.items():
        for group_name, samples in groups.items():
            for sample in samples:
                curve = core._finite_prefix(sample.ent, abs_T)
                for axis_idx, (axis_val, value) in enumerate(zip(axis_values, curve)):
                    if not np.isfinite(value):
                        continue
                    yield {
                        "strategy": strategy,
                        "group": group_name,
                        "qid": sample.qid,
                        "rid": sample.rid,
                        "axis_index": int(axis_idx),
                        "axis_value": float(axis_val),
                        "value": float(value),
                    }


def rel_curve_rows(strategy_to_samples, rel_bins: int):
    axis_values = np.linspace(0.0, 1.0, rel_bins)
    for strategy, groups in strategy_to_samples.items():
        for group_name, samples in groups.items():
            for sample in samples:
                curve = core.sample_to_relative_curve(sample.ent, rel_bins)
                for axis_idx, (axis_val, value) in enumerate(zip(axis_values, curve)):
                    if not np.isfinite(value):
                        continue
                    yield {
                        "strategy": strategy,
                        "group": group_name,
                        "qid": sample.qid,
                        "rid": sample.rid,
                        "axis_index": int(axis_idx),
                        "axis_value": float(axis_val),
                        "value": float(value),
                    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, action="append", required=True)
    ap.add_argument("--post", type=str, action="append", required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--subset", type=str, default="neg", choices=["neg", "all"])
    ap.add_argument("--abs_T", type=int, default=256)
    ap.add_argument("--rel_bins", type=int, default=128)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--prototype", type=str, default="")
    ap.add_argument("--dist_metric", type=str, default="cosine", choices=["cosine", "l2"])
    args = ap.parse_args()

    outdir = Path(args.outdir) / "csv"
    ensure_dir(outdir)

    base_map = core.parse_name_path_list(args.base, "--base")
    post_map = core.parse_name_path_list(args.post, "--post")
    strategies = [s for s in base_map.keys() if s in post_map]
    if not strategies:
        raise ValueError("No overlapping strategies between --base and --post.")

    abs_x = np.arange(args.abs_T, dtype=np.float64)
    rel_x = np.linspace(0.0, 1.0, args.rel_bins)

    qid_label_rows = []
    strategy_summary_rows = []
    sample_groups = {}
    eff_abs_by_strategy = {}
    eff_rel_by_strategy = {}
    ineff_abs_by_strategy = {}
    ineff_rel_by_strategy = {}

    for strategy in strategies:
        base_samples = core.load_samples(base_map[strategy], pad_id=args.pad_id, max_len=args.max_len)
        post_samples = core.load_samples(post_map[strategy], pad_id=args.pad_id, max_len=args.max_len)
        base_anyc = core.per_qid_any_correct(base_samples)
        post_anyc = core.per_qid_any_correct(post_samples)
        eff_qids = core.build_effective_qids(base_anyc, post_anyc)
        overlap_qids = set(eff_qids.keys())

        for qid in sorted(overlap_qids):
            is_eff = bool(eff_qids.get(qid, False))
            qid_label_rows.append(
                {
                    "strategy": strategy,
                    "qid": qid,
                    "base_anyc": int(bool(base_anyc.get(qid, False))),
                    "post_anyc": int(bool(post_anyc.get(qid, False))),
                    "effective_group": "effective" if is_eff else "ineffective",
                }
            )

        if args.subset == "neg":
            base_use = [s for s in base_samples if (not s.correct) and (s.qid in overlap_qids)]
        else:
            base_use = [s for s in base_samples if s.qid in overlap_qids]

        eff_samples = [s for s in base_use if bool(eff_qids.get(s.qid, False))]
        ineff_samples = [s for s in base_use if not bool(eff_qids.get(s.qid, False))]
        sample_groups[strategy] = {"effective": eff_samples, "ineffective": ineff_samples}

        eff_abs = core.make_abs_mat(eff_samples, T=args.abs_T)
        ineff_abs = core.make_abs_mat(ineff_samples, T=args.abs_T)
        eff_rel = core.make_rel_mat(eff_samples, bins=args.rel_bins)
        ineff_rel = core.make_rel_mat(ineff_samples, bins=args.rel_bins)

        eff_abs_by_strategy[strategy] = eff_abs
        ineff_abs_by_strategy[strategy] = ineff_abs
        eff_rel_by_strategy[strategy] = eff_rel
        ineff_rel_by_strategy[strategy] = ineff_rel

        strategy_summary_rows.append(
            {
                "strategy": strategy,
                "subset": args.subset,
                "base_file": str(base_map[strategy]),
                "post_file": str(post_map[strategy]),
                "base_qids": int(len(base_anyc)),
                "post_qids": int(len(post_anyc)),
                "qid_overlap": int(len(overlap_qids)),
                "effective_qids": int(sum(1 for _, v in eff_qids.items() if v)),
                "ineffective_qids": int(sum(1 for _, v in eff_qids.items() if not v)),
                "effective_rollouts": int(eff_abs.shape[0]),
                "ineffective_rollouts": int(ineff_abs.shape[0]),
            }
        )

    write_csv(
        outdir / "qid_labels.csv",
        ["strategy", "qid", "base_anyc", "post_anyc", "effective_group"],
        qid_label_rows,
    )
    write_csv(
        outdir / "strategy_summary.csv",
        [
            "strategy",
            "subset",
            "base_file",
            "post_file",
            "base_qids",
            "post_qids",
            "qid_overlap",
            "effective_qids",
            "ineffective_qids",
            "effective_rollouts",
            "ineffective_rollouts",
        ],
        strategy_summary_rows,
    )

    write_csv(
        outdir / "rollout_curves_abs.csv",
        ["strategy", "group", "qid", "rid", "axis_index", "axis_value", "value"],
        abs_curve_rows(sample_groups, args.abs_T),
    )
    write_csv(
        outdir / "rollout_curves_rel.csv",
        ["strategy", "group", "qid", "rid", "axis_index", "axis_value", "value"],
        rel_curve_rows(sample_groups, args.rel_bins),
    )

    write_csv(
        outdir / "group_summary_abs.csv",
        ["strategy", "group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode"],
        list(
            row
            for strategy in strategies
            for row in group_summary_rows(
                {"effective": eff_abs_by_strategy[strategy], "ineffective": ineff_abs_by_strategy[strategy]},
                abs_x,
                band_mode="bootstrap",
                boot=args.boot,
                extra_columns={"strategy": strategy},
            )
        ),
    )
    write_csv(
        outdir / "group_summary_rel.csv",
        ["strategy", "group", "axis_index", "axis_value", "mean", "band_low", "band_high", "n", "band_mode"],
        list(
            row
            for strategy in strategies
            for row in group_summary_rows(
                {"effective": eff_rel_by_strategy[strategy], "ineffective": ineff_rel_by_strategy[strategy]},
                rel_x,
                band_mode="bootstrap",
                boot=args.boot,
                extra_columns={"strategy": strategy},
            )
        ),
    )

    proto_name = args.prototype if args.prototype else strategies[0]
    prototype_rows = []
    if proto_name in eff_rel_by_strategy and core._finite_rows(eff_rel_by_strategy[proto_name]).shape[0] > 0:
        proto = np.nanmean(core._finite_rows(eff_rel_by_strategy[proto_name]), axis=0)
        for strategy in strategies:
            eff_rel = core._finite_rows(eff_rel_by_strategy[strategy])
            ineff_rel = core._finite_rows(ineff_rel_by_strategy[strategy])
            if args.dist_metric == "cosine":
                eff_dist = core.rowwise_cosine_distance(eff_rel, proto)
                ineff_dist = core.rowwise_cosine_distance(ineff_rel, proto)
            else:
                eff_dist = core.rowwise_l2_distance(eff_rel, proto)
                ineff_dist = core.rowwise_l2_distance(ineff_rel, proto)

            eff_stats = summarize_1d(eff_dist)
            ineff_stats = summarize_1d(ineff_dist)
            prototype_rows.append(
                {
                    "strategy": strategy,
                    "prototype_strategy": proto_name,
                    "subset": args.subset,
                    "dist_metric": args.dist_metric,
                    "effective_n": eff_stats["n"],
                    "effective_mean": eff_stats["mean"],
                    "effective_median": eff_stats["median"],
                    "ineffective_n": ineff_stats["n"],
                    "ineffective_mean": ineff_stats["mean"],
                    "ineffective_median": ineff_stats["median"],
                }
            )

    write_csv(
        outdir / "prototype_transfer.csv",
        [
            "strategy",
            "prototype_strategy",
            "subset",
            "dist_metric",
            "effective_n",
            "effective_mean",
            "effective_median",
            "ineffective_n",
            "ineffective_mean",
            "ineffective_median",
        ],
        prototype_rows,
    )
    print(f"[OK] csv outdir = {outdir}")


if __name__ == "__main__":
    main()
