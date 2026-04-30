#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export CSV tables for the rollout-level pairwise driver comparison used by `core/compare_g.py`.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import compare_g as core  # type: ignore

from _common import ensure_dir, transition_group, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, action="append", default=[], required=True)
    ap.add_argument("--np_base", type=str, action="append", default=[], required=True)
    ap.add_argument("--np_post", type=str, action="append", default=[], required=True)
    ap.add_argument("--pair", type=str, required=True)
    ap.add_argument("--drivers_py", type=str, default="")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    args = ap.parse_args()

    train_map = core.parse_name_path_list(args.train, "--train")
    base_map = core.parse_name_path_list(args.np_base, "--np_base")
    post_map = core.parse_name_path_list(args.np_post, "--np_post")
    A_name, B_name = [x.strip() for x in args.pair.split(",", 1)]

    outdir = Path(args.outdir) / f"compare_{A_name}_vs_{B_name}" / "csv"
    ensure_dir(outdir)

    drivers = {}
    if args.drivers_py:
        drivers.update(core.load_drivers_from_py(Path(args.drivers_py)))
    drivers.update(core.load_drivers_from_py(CORE_DIR / "drivers.py"))
    if not drivers:
        raise RuntimeError("No drivers loaded.")

    base_anyc_A = core.per_qid_any_correct_from_file(base_map[A_name])
    post_anyc_A = core.per_qid_any_correct_from_file(post_map[A_name])
    base_anyc_B = core.per_qid_any_correct_from_file(base_map[B_name])
    post_anyc_B = core.per_qid_any_correct_from_file(post_map[B_name])

    def build_sets(base_anyc, post_anyc):
        base_n = {q for q, v in base_anyc.items() if not bool(v)}
        base_n &= set(post_anyc.keys())
        n2p = {q for q in base_n if bool(post_anyc.get(q, False))}
        n2n = base_n - n2p
        return n2p, n2n

    A_n2p_qids, A_n2n_qids = build_sets(base_anyc_A, post_anyc_A)
    B_n2p_qids, B_n2n_qids = build_sets(base_anyc_B, post_anyc_B)

    write_csv(
        outdir / "qid_labels.csv",
        ["strategy", "qid", "base_anyc", "post_anyc", "transition_group"],
        [
            {
                "strategy": A_name,
                "qid": qid,
                "base_anyc": int(bool(base_anyc_A.get(qid, False))),
                "post_anyc": int(bool(post_anyc_A.get(qid, False))),
                "transition_group": transition_group(bool(base_anyc_A.get(qid, False)), bool(post_anyc_A.get(qid, False))),
            }
            for qid in sorted(set(base_anyc_A.keys()) | set(post_anyc_A.keys()))
        ]
        + [
            {
                "strategy": B_name,
                "qid": qid,
                "base_anyc": int(bool(base_anyc_B.get(qid, False))),
                "post_anyc": int(bool(post_anyc_B.get(qid, False))),
                "transition_group": transition_group(bool(base_anyc_B.get(qid, False)), bool(post_anyc_B.get(qid, False))),
            }
            for qid in sorted(set(base_anyc_B.keys()) | set(post_anyc_B.keys()))
        ],
    )

    A_rollouts = core.load_rollouts_for_driver(train_map[A_name], pad_id=args.pad_id, max_len=args.max_len)
    B_rollouts = core.load_rollouts_for_driver(train_map[B_name], pad_id=args.pad_id, max_len=args.max_len)

    summary_rows = []
    kde_rows = []

    raw_path = outdir / "rollout_driver_values.csv"
    ensure_dir(raw_path.parent)
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["driver", "strategy", "qid", "rid", "transition_group", "driver_value"],
        )
        writer.writeheader()

        for dname, fn in drivers.items():
            try:
                n_params = len(inspect.signature(fn).parameters)
            except Exception:
                n_params = 1

            group_values = {
                (A_name, "N2P"): [],
                (A_name, "N2N"): [],
                (B_name, "N2P"): [],
                (B_name, "N2N"): [],
            }

            for strategy_name, rollouts, q_n2p, q_n2n in [
                (A_name, A_rollouts, A_n2p_qids, A_n2n_qids),
                (B_name, B_rollouts, B_n2p_qids, B_n2n_qids),
            ]:
                for rollout in rollouts:
                    if rollout.qid in q_n2p:
                        group = "N2P"
                    elif rollout.qid in q_n2n:
                        group = "N2N"
                    else:
                        continue

                    try:
                        if n_params >= 2:
                            value = float(fn(rollout.ent, rollout.topk_logprobs_per_token))
                        else:
                            value = float(fn(rollout.ent))
                    except Exception:
                        continue
                    if not np.isfinite(value):
                        continue

                    group_values[(strategy_name, group)].append(value)
                    writer.writerow(
                        {
                            "driver": dname,
                            "strategy": strategy_name,
                            "qid": rollout.qid,
                            "rid": rollout.rid,
                            "transition_group": group,
                            "driver_value": float(value),
                        }
                    )

            A_N2P = np.asarray(group_values[(A_name, "N2P")], dtype=np.float64)
            A_N2N = np.asarray(group_values[(A_name, "N2N")], dtype=np.float64)
            B_N2P = np.asarray(group_values[(B_name, "N2P")], dtype=np.float64)
            B_N2N = np.asarray(group_values[(B_name, "N2N")], dtype=np.float64)

            for strategy_name, group_name, arr in [
                (A_name, "N2P", A_N2P),
                (A_name, "N2N", A_N2N),
                (B_name, "N2P", B_N2P),
                (B_name, "N2N", B_N2N),
            ]:
                skew = core._summary_right_skew(arr)
                summary_rows.append(
                    {
                        "driver": dname,
                        "strategy": strategy_name,
                        "transition_group": group_name,
                        **skew,
                    }
                )

            total = core._finite(A_N2P).size + core._finite(A_N2N).size + core._finite(B_N2P).size + core._finite(B_N2N).size
            if total < 4:
                continue

            lo, hi = core._robust_xlim([A_N2P, A_N2N, B_N2P, B_N2N])
            grid = np.linspace(lo, hi, 600)
            for strategy_name, group_name, arr in [
                (A_name, "N2P", A_N2P),
                (A_name, "N2N", A_N2N),
                (B_name, "N2P", B_N2P),
                (B_name, "N2N", B_N2N),
            ]:
                dens = core._kde_gaussian(core._finite(arr), grid)
                for idx, x in enumerate(grid):
                    kde_rows.append(
                        {
                            "driver": dname,
                            "strategy": strategy_name,
                            "transition_group": group_name,
                            "axis_index": int(idx),
                            "axis_value": float(x),
                            "density": float(dens[idx]),
                        }
                    )

    write_csv(
        outdir / "kde_curve.csv",
        ["driver", "strategy", "transition_group", "axis_index", "axis_value", "density"],
        kde_rows,
    )
    write_csv(
        outdir / "right_skew_summary.csv",
        [
            "driver",
            "strategy",
            "transition_group",
            "n",
            "mean",
            "std",
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
            "moment_skewness",
            "bowley_skewness",
            "right_tail_cutoff_q3_1p5iqr",
            "right_tail_mass_q3_1p5iqr",
            "right_tail_mass_q90",
            "mean_minus_median",
        ],
        summary_rows,
    )
    print(f"[OK] csv outdir = {outdir}")


if __name__ == "__main__":
    main()
