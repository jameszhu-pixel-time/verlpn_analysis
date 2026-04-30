#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export CSV tables for the intra-strategy driver plots used by `core/simple_essay.py`.

This matches the paper-oriented behavior:
- subsets are restricted to `all` and `neg`
- qid-level driver mean is compared between N2P and N2N
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

import simple_essay as core  # type: ignore

from _common import ensure_dir, summarize_1d, transition_group, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, action="append", default=[], required=True)
    ap.add_argument("--np_base", type=str, action="append", required=True)
    ap.add_argument("--np_post", type=str, action="append", default=[], required=True)
    ap.add_argument("--drivers_py", type=str, default="")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    outdir = Path(args.outdir) / "csv"
    ensure_dir(outdir)

    drivers = {}
    if args.drivers_py:
        drivers.update(core.load_drivers_from_py(Path(args.drivers_py)))
    drivers.update(core.load_drivers_from_py(CORE_DIR / "drivers.py"))
    if not drivers:
        raise RuntimeError("No drivers loaded.")

    train_map = core.parse_name_path_list(args.train, "--train")
    np_post_map = core.parse_name_path_list(args.np_post, "--np_post")
    shared_np_base, np_base_map = core.parse_np_base_arg(args.np_base)
    subset_list = ["all", "neg"]

    post_anyc = {}
    for name, path in np_post_map.items():
        ss = core.load_samples(path, pad_id=args.pad_id, max_len=args.max_len)
        post_anyc[name] = core.per_qid_any_correct(ss)

    shared_base_anyc = None
    if shared_np_base is not None:
        shared_base_anyc = core.per_qid_any_correct(
            core.load_samples(shared_np_base, pad_id=args.pad_id, max_len=args.max_len)
        )

    qid_label_rows = []
    qid_driver_rows = []
    summary_rows = []
    density_rows = []
    kde_rows = []
    ecdf_rows = []

    for sname, train_path in train_map.items():
        if sname not in post_anyc:
            continue

        if sname in np_base_map:
            base_anyc = core.per_qid_any_correct(
                core.load_samples(np_base_map[sname], pad_id=args.pad_id, max_len=args.max_len)
            )
            np_base_used = str(np_base_map[sname])
        else:
            if shared_base_anyc is None:
                continue
            base_anyc = shared_base_anyc
            np_base_used = str(shared_np_base)

        post_anyc_s = post_anyc[sname]
        for qid in sorted(set(base_anyc.keys()) | set(post_anyc_s.keys())):
            qid_label_rows.append(
                {
                    "strategy": sname,
                    "qid": qid,
                    "base_anyc": int(bool(base_anyc.get(qid, False))),
                    "post_anyc": int(bool(post_anyc_s.get(qid, False))),
                    "transition_group": transition_group(
                        bool(base_anyc.get(qid, False)),
                        bool(post_anyc_s.get(qid, False)),
                    ),
                }
            )

        train_samples = core.load_samples(train_path, pad_id=args.pad_id, max_len=args.max_len)
        train_subsets = core.split_subsets(train_samples)

        for subset in subset_list:
            sb_samples = train_subsets.get(subset, [])
            if not sb_samples:
                continue

            for dname, fn in drivers.items():
                qmean = core.qid_mean_driver(sb_samples, fn)
                if not qmean:
                    continue

                overlap = set(base_anyc.keys()) & set(post_anyc_s.keys()) & set(qmean.keys())
                n2p_vals = []
                n2n_vals = []
                for qid in sorted(overlap):
                    if bool(base_anyc[qid]):
                        continue
                    value = qmean.get(qid)
                    if value is None or (not np.isfinite(value)):
                        continue
                    group = "N2P" if bool(post_anyc_s[qid]) else "N2N"
                    qid_driver_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "qid": qid,
                            "transition_group": group,
                            "qid_mean_driver": float(value),
                        }
                    )
                    if group == "N2P":
                        n2p_vals.append(value)
                    else:
                        n2n_vals.append(value)

                a = np.asarray(n2p_vals, dtype=np.float64)
                b = np.asarray(n2n_vals, dtype=np.float64)
                a_stats = summarize_1d(a)
                b_stats = summarize_1d(b)
                w1 = core.wasserstein_1d(a, b)
                ks = core.ks_statistic(a, b)
                summary_rows.append(
                    {
                        "strategy": sname,
                        "subset": subset,
                        "driver": dname,
                        "qid_overlap_used": int(len(overlap)),
                        "np_base": np_base_used,
                        "np_post": str(np_post_map[sname]),
                        "train_file": str(train_path),
                        "N2P_n": a_stats["n"],
                        "N2P_mean": a_stats["mean"],
                        "N2P_median": a_stats["median"],
                        "N2N_n": b_stats["n"],
                        "N2N_mean": b_stats["mean"],
                        "N2N_median": b_stats["median"],
                        "w1": None if not np.isfinite(w1) else float(w1),
                        "ks": None if not np.isfinite(ks) else float(ks),
                    }
                )

                aa = core._finite(a)
                bb = core._finite(b)
                if aa.size == 0 or bb.size == 0:
                    continue

                xs = np.concatenate([aa, bb])
                bins_use = max(int(args.bins), core._freedman_diaconis_bins(xs))
                bins_use = int(np.clip(bins_use, 30, 180))
                lo, hi = core._robust_xlim(aa, bb)
                edges = np.linspace(lo, hi, bins_use + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ha, _ = np.histogram(aa, bins=edges, density=True)
                hb, _ = np.histogram(bb, bins=edges, density=True)

                for idx, center in enumerate(centers):
                    density_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "group": "N2P",
                            "bin_index": int(idx),
                            "bin_left": float(edges[idx]),
                            "bin_right": float(edges[idx + 1]),
                            "bin_center": float(center),
                            "density": float(ha[idx]),
                        }
                    )
                    density_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "group": "N2N",
                            "bin_index": int(idx),
                            "bin_left": float(edges[idx]),
                            "bin_right": float(edges[idx + 1]),
                            "bin_center": float(center),
                            "density": float(hb[idx]),
                        }
                    )

                grid = np.linspace(lo, hi, 400)
                da = core._kde_gaussian(aa, grid)
                db = core._kde_gaussian(bb, grid)
                for idx, x in enumerate(grid):
                    kde_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "group": "N2P",
                            "axis_index": int(idx),
                            "axis_value": float(x),
                            "density": float(da[idx]),
                        }
                    )
                    kde_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "group": "N2N",
                            "axis_index": int(idx),
                            "axis_value": float(x),
                            "density": float(db[idx]),
                        }
                    )

                xa, ya = core._ecdf(aa)
                xb, yb = core._ecdf(bb)
                for idx, (x, y) in enumerate(zip(xa, ya)):
                    ecdf_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "group": "N2P",
                            "axis_index": int(idx),
                            "axis_value": float(x),
                            "ecdf": float(y),
                        }
                    )
                for idx, (x, y) in enumerate(zip(xb, yb)):
                    ecdf_rows.append(
                        {
                            "strategy": sname,
                            "subset": subset,
                            "driver": dname,
                            "group": "N2N",
                            "axis_index": int(idx),
                            "axis_value": float(x),
                            "ecdf": float(y),
                        }
                    )

    write_csv(
        outdir / "qid_labels.csv",
        ["strategy", "qid", "base_anyc", "post_anyc", "transition_group"],
        qid_label_rows,
    )
    write_csv(
        outdir / "qid_driver_values.csv",
        ["strategy", "subset", "driver", "qid", "transition_group", "qid_mean_driver"],
        qid_driver_rows,
    )
    write_csv(
        outdir / "summary.csv",
        [
            "strategy",
            "subset",
            "driver",
            "qid_overlap_used",
            "np_base",
            "np_post",
            "train_file",
            "N2P_n",
            "N2P_mean",
            "N2P_median",
            "N2N_n",
            "N2N_mean",
            "N2N_median",
            "w1",
            "ks",
        ],
        summary_rows,
    )
    write_csv(
        outdir / "density_hist.csv",
        ["strategy", "subset", "driver", "group", "bin_index", "bin_left", "bin_right", "bin_center", "density"],
        density_rows,
    )
    write_csv(
        outdir / "density_kde.csv",
        ["strategy", "subset", "driver", "group", "axis_index", "axis_value", "density"],
        kde_rows,
    )
    write_csv(
        outdir / "ecdf.csv",
        ["strategy", "subset", "driver", "group", "axis_index", "axis_value", "ecdf"],
        ecdf_rows,
    )
    print(f"[OK] csv outdir = {outdir}")


if __name__ == "__main__":
    main()
