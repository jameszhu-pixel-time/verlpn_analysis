#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def transition_group(base_anyc: bool, post_anyc: bool) -> str:
    if (not base_anyc) and post_anyc:
        return "N2P"
    if (not base_anyc) and (not post_anyc):
        return "N2N"
    if base_anyc and post_anyc:
        return "P2P"
    return "P2N"


def summarize_1d(values: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def pointwise_quantile_band(mat: np.ndarray, q_low: float = 2.5, q_high: float = 97.5):
    lo = np.nanpercentile(mat, q_low, axis=0)
    hi = np.nanpercentile(mat, q_high, axis=0)
    return lo, hi


def mean_std_band(mat: np.ndarray):
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1)
    return mean - std, mean + std


def bootstrap_ci(mat: np.ndarray, boot: int = 1000, seed: int = 0):
    if mat.shape[0] == 0:
        return np.full((mat.shape[1],), np.nan), np.full((mat.shape[1],), np.nan)
    rng = np.random.default_rng(seed)
    means = []
    n = mat.shape[0]
    for _ in range(boot):
        idx = rng.integers(0, n, size=n)
        means.append(np.nanmean(mat[idx], axis=0))
    means = np.stack(means, axis=0)
    lo = np.nanpercentile(means, 2.5, axis=0)
    hi = np.nanpercentile(means, 97.5, axis=0)
    return lo, hi


def group_summary_rows(
    group_to_mat: Mapping[str, np.ndarray],
    axis_values: np.ndarray,
    *,
    band_mode: str,
    boot: int = 1000,
    seed_base: int = 0,
    extra_columns: Optional[Mapping[str, object]] = None,
) -> Iterator[dict[str, object]]:
    extra = dict(extra_columns or {})
    for idx_group, (group, mat) in enumerate(group_to_mat.items()):
        if mat is None or mat.ndim != 2 or mat.shape[0] == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        if band_mode == "std":
            lo, hi = mean_std_band(mat)
        elif band_mode == "bootstrap":
            lo, hi = bootstrap_ci(mat, boot=boot, seed=seed_base + idx_group)
        else:
            lo, hi = pointwise_quantile_band(mat)

        for axis_idx, axis_val in enumerate(axis_values):
            row = dict(extra)
            row.update(
                {
                    "group": group,
                    "axis_index": int(axis_idx),
                    "axis_value": safe_float(axis_val),
                    "mean": safe_float(mean[axis_idx]),
                    "band_low": safe_float(lo[axis_idx]),
                    "band_high": safe_float(hi[axis_idx]),
                    "n": int(mat.shape[0]),
                    "band_mode": band_mode,
                }
            )
            yield row
