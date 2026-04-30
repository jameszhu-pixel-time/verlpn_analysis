#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-oriented n2 vs n3 entropy-position comparison.

Compared with the legacy compare.py:
- each strategy uses its own base file
- the figure emphasizes distribution shift in the early 20% region
- both Relative and Absolute (T=3072) views are shown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

import ent_pos as core


PALETTE = {
    "np_line": "#C75B12",
    "np_fill": "#F1B584",
    "nn_line": "#2E6EA6",
    "nn_fill": "#A6C8E6",
    "n2_text": "#22303C",
    "n3_text": "#5D6873",
    "early_fill": "#F5EBDD",
    "early_edge": "#C49A5A",
    "grid": "#D9E1EA",
    "spine": "#A9B4C2",
    "text": "#22303C",
    "subtle": "#5E6B77",
    "bg": "#FBFCFD",
}


plt.rcParams.update(
    {
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["spine"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "xtick.color": PALETTE["subtle"],
        "ytick.color": PALETTE["subtle"],
        "text.color": PALETTE["text"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
    }
)


def _short_name(name: str) -> str:
    if name.startswith("normal_"):
        return "n" + name.split("_", 1)[1]
    return name


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _robust_xlim(arrs: Iterable[np.ndarray]) -> Tuple[float, float]:
    finite_chunks = []
    for arr in arrs:
        vals = _finite(arr)
        if vals.size > 0:
            finite_chunks.append(vals)
    if not finite_chunks:
        return -1.0, 1.0
    vals = np.concatenate(finite_chunks, axis=0)
    lo = float(np.quantile(vals, 0.01))
    hi = float(np.quantile(vals, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
    return lo, hi


def _kde_gaussian(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = _finite(x)
    if vals.size < 2:
        return np.zeros_like(grid)
    std = float(np.std(vals))
    if (not np.isfinite(std)) or std <= 1e-12:
        return np.zeros_like(grid)
    h = max(1.06 * std * (vals.size ** (-1.0 / 5.0)), 1e-6)
    z = (grid[:, None] - vals[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (vals.size * h * np.sqrt(2.0 * np.pi))
    return dens


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sort(_finite(a))
    bb = np.sort(_finite(b))
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    xs = np.sort(np.unique(np.concatenate([aa, bb])))
    if xs.size < 2:
        return 0.0
    Fa = np.searchsorted(aa, xs, side="right") / aa.size
    Fb = np.searchsorted(bb, xs, side="right") / bb.size
    dx = xs[1:] - xs[:-1]
    return float(np.sum(np.abs(Fa - Fb)[:-1] * dx))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sort(_finite(a))
    bb = np.sort(_finite(b))
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    xs = np.sort(np.unique(np.concatenate([aa, bb])))
    Fa = np.searchsorted(aa, xs, side="right") / aa.size
    Fb = np.searchsorted(bb, xs, side="right") / bb.size
    return float(np.max(np.abs(Fa - Fb)))


def _probability_superiority(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sort(_finite(a))
    bb = np.sort(_finite(b))
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    wins = np.searchsorted(bb, aa, side="left").astype(np.float64)
    ties = (
        np.searchsorted(bb, aa, side="right").astype(np.float64)
        - np.searchsorted(bb, aa, side="left").astype(np.float64)
    )
    total = float(aa.size * bb.size)
    return float((np.sum(wins) + 0.5 * np.sum(ties)) / max(total, 1.0))


def _summary_shift(a: np.ndarray, b: np.ndarray) -> Dict[str, Optional[float]]:
    aa = _finite(a)
    bb = _finite(b)
    if aa.size == 0 or bb.size == 0:
        return {"delta_mean": None, "delta_median": None, "w1": None, "ks": None, "ps": None}
    delta_mean = float(np.mean(aa) - np.mean(bb))
    delta_median = float(np.median(aa) - np.median(bb))
    w1 = _wasserstein_1d(aa, bb)
    ks = _ks_statistic(aa, bb)
    ps = _probability_superiority(aa, bb)
    return {
        "delta_mean": delta_mean if np.isfinite(delta_mean) else None,
        "delta_median": delta_median if np.isfinite(delta_median) else None,
        "w1": w1 if np.isfinite(w1) else None,
        "ks": ks if np.isfinite(ks) else None,
        "ps": ps if np.isfinite(ps) else None,
    }


def _driver_stats(values: np.ndarray, *, d_ref: Optional[float], tau: Optional[float]) -> Dict[str, Optional[float]]:
    vals = _finite(values)
    if vals.size == 0:
        return {"n": 0, "msk": None, "bsk": None, "rt": None, "d_ref": d_ref, "tau": tau}

    mean = float(np.mean(vals))
    bsk = None
    if d_ref is not None and np.isfinite(d_ref):
        bsk = mean - float(d_ref)
    rt = None
    if tau is not None and np.isfinite(tau):
        rt = float(np.mean(vals > float(tau)))
    return {
        "n": int(vals.size),
        "msk": mean if np.isfinite(mean) else None,
        "bsk": bsk if bsk is not None and np.isfinite(bsk) else None,
        "rt": rt if rt is not None and np.isfinite(rt) else None,
        "d_ref": float(d_ref) if d_ref is not None and np.isfinite(d_ref) else None,
        "tau": float(tau) if tau is not None and np.isfinite(tau) else None,
    }


def _input_shift_indicators(
    distributions: Dict[str, np.ndarray], strategy_short: str, *, tail_quantile: float = 0.90
) -> Dict[str, Dict[str, Optional[float]]]:
    """Batch-level input shift indicators from shift_indicator.tex.

    For each strategy, NN is the reference input distribution: d_ref is mean(NN),
    and tau is the selected high-tail quantile of NN.
    """
    nn_key = f"{strategy_short}_nn"
    np_key = f"{strategy_short}_np"
    nn_vals = _finite(distributions[nn_key])
    d_ref = float(np.mean(nn_vals)) if nn_vals.size else None
    tau = float(np.quantile(nn_vals, tail_quantile)) if nn_vals.size else None
    return {
        "NP": _driver_stats(distributions[np_key], d_ref=d_ref, tau=tau),
        "NN": _driver_stats(distributions[nn_key], d_ref=d_ref, tau=tau),
    }


def _quantile_band(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(mat, axis=0)
    lo = np.nanpercentile(mat, 2.5, axis=0)
    hi = np.nanpercentile(mat, 97.5, axis=0)
    return mean, lo, hi


def _focus_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    chunks = [a[np.isfinite(a)] for a in arrays if np.any(np.isfinite(a))]
    if not chunks:
        return 0.0, 1.0
    vals = np.concatenate(chunks, axis=0)
    lo = float(np.nanpercentile(vals, 5.0))
    hi = float(np.nanpercentile(vals, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    pad = 0.12 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _compute_strategy_curves(base_path: Path, post_path: Path, *, pad_id: int, max_len: int, M: int, abs_T: int):
    base_samples = core.load_samples(base_path, pad_id=pad_id, max_len=max_len, metric="entropy")
    post_samples = core.load_samples(post_path, pad_id=pad_id, max_len=max_len, metric="entropy")
    base_anyc = core.per_qid_any_correct(base_samples)
    post_anyc = core.per_qid_any_correct(post_samples)
    all_qids = set(base_anyc.keys()) | set(post_anyc.keys())
    np_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and post_anyc.get(q, False)])
    nn_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and (not post_anyc.get(q, False))])
    keep = set(np_qids) | set(nn_qids)

    norm_map = core.prompt_level_mean_curves_normalized(base_samples, qids_keep=keep, M=M)
    abs_map = core.prompt_level_mean_curves_absolute(base_samples, qids_keep=keep, abs_T=abs_T)

    def _stack(curve_map: Dict[str, np.ndarray], qids: List[str], width: int) -> np.ndarray:
        curves = [curve_map[q] for q in qids if q in curve_map]
        if not curves:
            return np.zeros((0, width), dtype=np.float64)
        return np.stack(curves, axis=0).astype(np.float64)

    return {
        "np_qids": np_qids,
        "nn_qids": nn_qids,
        "norm_np": _stack(norm_map, np_qids, M),
        "norm_nn": _stack(norm_map, nn_qids, M),
        "abs_np": _stack(abs_map, np_qids, abs_T),
        "abs_nn": _stack(abs_map, nn_qids, abs_T),
    }


def _early_ratio(mat: np.ndarray, early_points: int) -> np.ndarray:
    out = []
    for row in mat:
        vals = np.asarray(row, dtype=np.float64)
        valid = vals[np.isfinite(vals)]
        if valid.size < 3:
            continue
        k = max(1, min(int(early_points), valid.size))
        num = float(np.mean(valid[:k]))
        den = float(np.mean(valid))
        if np.isfinite(num) and np.isfinite(den) and den > 1e-12:
            out.append(num / den)
    return np.asarray(out, dtype=np.float64)


def _configure_axis(ax, ylabel: Optional[str] = None):
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _style_for(strategy_short: str, group: str):
    if group == "NP":
        color = PALETTE["np_line"]
        fill = PALETTE["np_fill"]
    else:
        color = PALETTE["nn_line"]
        fill = PALETTE["nn_fill"]
    linestyle = "-" if strategy_short == "n2" else (0, (5, 3))
    linewidth = 2.4 if strategy_short == "n2" else 2.1
    alpha_fill = 0.18 if strategy_short == "n2" else 0.12
    return color, fill, linestyle, linewidth, alpha_fill


def _draw_curve_panel(
    ax,
    *,
    x: np.ndarray,
    data: Dict[str, np.ndarray],
    panel_title: str,
    xlabel: str,
    ylabel: str,
    early_end: float,
    zoom_start: float,
    zoom_end: float,
    relative: bool,
):
    ax.axvspan(0.0, early_end, facecolor=PALETTE["early_fill"], alpha=0.92, zorder=0)
    ax.axvline(early_end, color=PALETTE["early_edge"], linewidth=1.1, linestyle=(0, (3, 3)))

    plotted = []
    for key, label in [
        ("n2_np", "n2 · NP"),
        ("n2_nn", "n2 · NN"),
        ("n3_np", "n3 · NP"),
        ("n3_nn", "n3 · NN"),
    ]:
        mat = data[key]
        if mat.ndim != 2 or mat.shape[0] == 0:
            continue
        strategy_short = key.split("_")[0]
        group = key.split("_")[1].upper()
        mean, lo, hi = _quantile_band(mat)
        color, fill, linestyle, linewidth, alpha_fill = _style_for(strategy_short, group)
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth, label=f"{label} (n={mat.shape[0]})")
        ax.fill_between(x, lo, hi, color=fill, alpha=alpha_fill)
        plotted.append((mean, lo, hi, color, linestyle))

    _configure_axis(ax, ylabel=ylabel)
    ax.set_title(panel_title, loc="left", pad=8, fontweight="semibold")
    ax.set_xlabel(xlabel)

    if relative:
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))}%"))
    else:
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_xticks([0, 512, 1024, 1536, 2048, 2560, 3072])

    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=2.4, columnspacing=1.2)

    mask = (x >= zoom_start) & (x <= zoom_end)
    if np.sum(mask) >= 3 and plotted:
        axins = inset_axes(ax, width="47%", height="48%", loc="upper right", borderpad=1.0)
        axins.axvspan(zoom_start, zoom_end, facecolor=PALETTE["early_fill"], alpha=0.6, zorder=0)
        local_chunks = []
        for mean, lo, hi, color, linestyle in plotted:
            axins.plot(x[mask], mean[mask], color=color, linestyle=linestyle, linewidth=1.8)
            axins.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=0.10)
            local_chunks.extend([mean[mask], lo[mask], hi[mask]])
        y0, y1 = _focus_ylim(*local_chunks)
        axins.set_ylim(y0, y1)
        axins.set_xlim(zoom_start, zoom_end)
        axins.grid(axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.8)
        axins.grid(axis="x", color=PALETTE["grid"], linewidth=0.35, alpha=0.25)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)
        axins.spines["left"].set_color(PALETTE["spine"])
        axins.spines["bottom"].set_color(PALETTE["spine"])
        axins.tick_params(axis="both", labelsize=8, length=2.5)
        if relative:
            axins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))}%"))
        axins.set_title("early 20%", fontsize=8.5, loc="left", pad=2)


def _draw_distribution_panel(
    ax,
    *,
    distributions: List[Tuple[str, str, np.ndarray]],
    panel_title: str,
    xlabel: str,
    pair_annotations: List[Tuple[str, Dict[str, Optional[float]]]],
    input_shift: Optional[Dict[str, Dict[str, Dict[str, Optional[float]]]]] = None,
    reference_x: Optional[float] = None,
):
    xs = [vals for _, _, vals in distributions if _finite(vals).size > 0]
    lo, hi = _robust_xlim(xs)
    span = max(hi - lo, 1e-6)
    lo_pad = lo - 0.02 * span
    hi_pad = hi + 0.34 * span
    grid = np.linspace(lo, hi, 500)
    width = 0.34
    positions = list(range(len(distributions), 0, -1))
    label_x = lo_pad + 0.012 * (hi_pad - lo_pad)
    median_map = {}

    if reference_x is not None and lo <= reference_x <= hi:
        ax.axvline(reference_x, color=PALETTE["spine"], linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.95, zorder=0.5)

    for idx in range(0, len(positions), 2):
        if idx + 1 >= len(positions):
            break
        ax.axhspan(
            positions[idx + 1] - 0.48,
            positions[idx] + 0.48,
            color="#F7F9FB" if (idx // 2) % 2 == 0 else "#FCFDFE",
            zorder=0,
        )

    for pos, (group_label, style_key, vals) in zip(positions, distributions):
        values = _finite(vals)
        if values.size == 0:
            continue
        strategy_short, group = style_key.split("_")
        color, fill, linestyle, linewidth, _ = _style_for(strategy_short, group.upper())
        dens = _kde_gaussian(values, grid)
        scale = width / max(np.max(dens), 1e-9)
        y_top = pos + dens * scale
        y_bot = pos - dens * scale

        ax.fill_between(grid, y_bot, y_top, color=fill, alpha=0.35, linewidth=0.0)
        ax.plot(grid, y_top, color=color, linestyle=linestyle, linewidth=1.4)
        ax.plot(grid, y_bot, color=color, linestyle=linestyle, linewidth=1.4)

        median = float(np.median(values))
        mean = float(np.mean(values))
        q1, q3 = [float(np.quantile(values, q)) for q in (0.25, 0.75)]
        ax.plot([q1, q3], [pos, pos], color=color, linewidth=3.0, solid_capstyle="round")
        ax.plot([median, median], [pos - 0.13, pos + 0.13], color=color, linewidth=1.5)
        ax.scatter([mean], [pos], s=18, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        median_map[group_label] = {"median": median, "pos": pos}
        ax.text(label_x, pos + 0.36, f"{group_label}  (n={values.size})", ha="left", va="bottom", fontsize=9.2, color=PALETTE["subtle"])

    for strategy_label, summary in pair_annotations:
        np_key = f"{strategy_label} · NP"
        nn_key = f"{strategy_label} · NN"
        if np_key not in median_map or nn_key not in median_map:
            continue
        x0 = median_map[nn_key]["median"]
        x1 = median_map[np_key]["median"]
        y_mid = 0.5 * (median_map[np_key]["pos"] + median_map[nn_key]["pos"])
        color = PALETTE["np_line"] if x1 >= x0 else PALETTE["nn_line"]
        ax.annotate(
            "",
            xy=(x1, y_mid),
            xytext=(x0, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6, shrinkA=2.0, shrinkB=2.0),
        )
        if summary.get("delta_median") is not None or summary.get("ps") is not None:
            delta_txt = f"Δmed={summary['delta_median']:+.3f}" if summary.get("delta_median") is not None else "Δmed=n/a"
            ps_txt = f"P(NP>NN)={summary['ps']:.2f}" if summary.get("ps") is not None else "P(NP>NN)=n/a"
            input_txt = ""
            if input_shift is not None:
                stats = input_shift.get(strategy_label, {}).get("NP", {})
                if stats.get("msk") is not None and stats.get("bsk") is not None and stats.get("rt") is not None:
                    input_txt = f"\nmsk={stats['msk']:.3f}  bsk={stats['bsk']:+.3f}  rt={stats['rt']:.2f}"
            ax.text(
                max(x0, x1) + 0.02 * max(hi - lo, 1e-6),
                y_mid + 0.06,
                f"{strategy_label}: {delta_txt}  {ps_txt}{input_txt}",
                ha="left",
                va="center",
                fontsize=8.7,
                color=PALETTE["subtle"],
            )

    _configure_axis(ax)
    ax.set_title(panel_title, loc="left", pad=8, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_xlim(lo_pad, hi_pad)
    ax.set_ylim(0.5, len(distributions) + 0.9)


def _save_single_panel(
    out_png: Path,
    out_pdf: Path,
    *,
    title: str,
    subtitle: str,
    draw,
    figsize: Tuple[float, float],
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=220)
    fig.subplots_adjust(top=0.82)
    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.92, subtitle, ha="left", va="top", fontsize=10.5, color=PALETTE["subtle"])
    draw(ax)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_base", type=str, required=True)
    ap.add_argument("--A_post", type=str, required=True)
    ap.add_argument("--B_base", type=str, required=True)
    ap.add_argument("--B_post", type=str, required=True)
    ap.add_argument("--A_name", type=str, default="normal_2")
    ap.add_argument("--B_name", type=str, default="normal_3")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--early_frac", type=float, default=0.2)
    ap.add_argument("--abs_T", type=int, default=3072)
    ap.add_argument("--zoom_skip_frac", type=float, default=0.02)
    ap.add_argument("--zoom_skip_abs", type=int, default=32)
    args = ap.parse_args()

    A_short = _short_name(args.A_name)
    B_short = _short_name(args.B_name)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    A = _compute_strategy_curves(
        Path(args.A_base), Path(args.A_post), pad_id=args.pad_id, max_len=args.max_len, M=args.M, abs_T=args.abs_T
    )
    B = _compute_strategy_curves(
        Path(args.B_base), Path(args.B_post), pad_id=args.pad_id, max_len=args.max_len, M=args.M, abs_T=args.abs_T
    )

    if min(
        A["norm_np"].shape[0],
        A["norm_nn"].shape[0],
        B["norm_np"].shape[0],
        B["norm_nn"].shape[0],
        A["abs_np"].shape[0],
        A["abs_nn"].shape[0],
        B["abs_np"].shape[0],
        B["abs_nn"].shape[0],
    ) == 0:
        raise RuntimeError("Some compared groups are empty. Check base/post overlap or entropy extraction.")

    x_norm = np.linspace(0.0, 1.0, args.M)
    x_abs = np.arange(args.abs_T, dtype=np.float64)
    early_points_norm = max(1, int(round(args.early_frac * args.M)))
    early_points_abs = max(1, int(round(args.early_frac * args.abs_T)))

    rel_dist = {
        f"{A_short}_np": _early_ratio(A["norm_np"], early_points_norm),
        f"{A_short}_nn": _early_ratio(A["norm_nn"], early_points_norm),
        f"{B_short}_np": _early_ratio(B["norm_np"], early_points_norm),
        f"{B_short}_nn": _early_ratio(B["norm_nn"], early_points_norm),
    }
    abs_dist = {
        f"{A_short}_np": _early_ratio(A["abs_np"], early_points_abs),
        f"{A_short}_nn": _early_ratio(A["abs_nn"], early_points_abs),
        f"{B_short}_np": _early_ratio(B["abs_np"], early_points_abs),
        f"{B_short}_nn": _early_ratio(B["abs_nn"], early_points_abs),
    }

    rel_shift_A = _summary_shift(rel_dist[f"{A_short}_np"], rel_dist[f"{A_short}_nn"])
    rel_shift_B = _summary_shift(rel_dist[f"{B_short}_np"], rel_dist[f"{B_short}_nn"])
    abs_shift_A = _summary_shift(abs_dist[f"{A_short}_np"], abs_dist[f"{A_short}_nn"])
    abs_shift_B = _summary_shift(abs_dist[f"{B_short}_np"], abs_dist[f"{B_short}_nn"])

    rel_input_shift = {
        A_short: _input_shift_indicators(rel_dist, A_short),
        B_short: _input_shift_indicators(rel_dist, B_short),
    }
    abs_input_shift = {
        A_short: _input_shift_indicators(abs_dist, A_short),
        B_short: _input_shift_indicators(abs_dist, B_short),
    }

    output_files = {
        "relative_curve_png": outdir / f"compare_entropy_{A_short}_vs_{B_short}_relative_curve.png",
        "relative_curve_pdf": outdir / f"compare_entropy_{A_short}_vs_{B_short}_relative_curve.pdf",
        "relative_shift_png": outdir / f"compare_entropy_{A_short}_vs_{B_short}_relative_shift.png",
        "relative_shift_pdf": outdir / f"compare_entropy_{A_short}_vs_{B_short}_relative_shift.pdf",
        "absolute_curve_png": outdir / f"compare_entropy_{A_short}_vs_{B_short}_absolute_curve_T{args.abs_T}.png",
        "absolute_curve_pdf": outdir / f"compare_entropy_{A_short}_vs_{B_short}_absolute_curve_T{args.abs_T}.pdf",
        "absolute_shift_png": outdir / f"compare_entropy_{A_short}_vs_{B_short}_absolute_shift_T{args.abs_T}.png",
        "absolute_shift_pdf": outdir / f"compare_entropy_{A_short}_vs_{B_short}_absolute_shift_T{args.abs_T}.pdf",
    }

    title = f"{A_short} vs {B_short}"
    _save_single_panel(
        output_files["relative_curve_png"],
        output_files["relative_curve_pdf"],
        title=title,
        subtitle="Relative entropy-position curve",
        figsize=(7.6, 5.2),
        draw=lambda ax: _draw_curve_panel(
            ax,
            x=x_norm,
            data={"n2_np": A["norm_np"], "n2_nn": A["norm_nn"], "n3_np": B["norm_np"], "n3_nn": B["norm_nn"]},
            panel_title="Relative",
            xlabel="Position in response",
            ylabel="Token entropy",
            early_end=float(args.early_frac),
            zoom_start=float(args.zoom_skip_frac),
            zoom_end=float(args.early_frac),
            relative=True,
        ),
    )
    _save_single_panel(
        output_files["relative_shift_png"],
        output_files["relative_shift_pdf"],
        title=title,
        subtitle="Relative input shift",
        figsize=(8.3, 5.2),
        draw=lambda ax: _draw_distribution_panel(
            ax,
            distributions=[
                (f"{A_short} · NP", "n2_np", rel_dist[f"{A_short}_np"]),
                (f"{A_short} · NN", "n2_nn", rel_dist[f"{A_short}_nn"]),
                (f"{B_short} · NP", "n3_np", rel_dist[f"{B_short}_np"]),
                (f"{B_short} · NN", "n3_nn", rel_dist[f"{B_short}_nn"]),
            ],
            panel_title="Relative shift",
            xlabel="mean(entropy, early 20%) / mean(entropy, full)",
            pair_annotations=[(A_short, rel_shift_A), (B_short, rel_shift_B)],
            input_shift=rel_input_shift,
            reference_x=1.0,
        ),
    )
    _save_single_panel(
        output_files["absolute_curve_png"],
        output_files["absolute_curve_pdf"],
        title=title,
        subtitle=f"Absolute entropy-position curve · T={args.abs_T}",
        figsize=(7.6, 5.2),
        draw=lambda ax: _draw_curve_panel(
            ax,
            x=x_abs,
            data={"n2_np": A["abs_np"], "n2_nn": A["abs_nn"], "n3_np": B["abs_np"], "n3_nn": B["abs_nn"]},
            panel_title=f"Absolute · T={args.abs_T}",
            xlabel="Token index",
            ylabel="Token entropy",
            early_end=float(args.early_frac) * float(args.abs_T),
            zoom_start=float(args.zoom_skip_abs),
            zoom_end=float(args.early_frac) * float(args.abs_T),
            relative=False,
        ),
    )
    _save_single_panel(
        output_files["absolute_shift_png"],
        output_files["absolute_shift_pdf"],
        title=title,
        subtitle=f"Absolute input shift · T={args.abs_T}",
        figsize=(8.3, 5.2),
        draw=lambda ax: _draw_distribution_panel(
            ax,
            distributions=[
                (f"{A_short} · NP", "n2_np", abs_dist[f"{A_short}_np"]),
                (f"{A_short} · NN", "n2_nn", abs_dist[f"{A_short}_nn"]),
                (f"{B_short} · NP", "n3_np", abs_dist[f"{B_short}_np"]),
                (f"{B_short} · NN", "n3_nn", abs_dist[f"{B_short}_nn"]),
            ],
            panel_title=f"Absolute shift · T={args.abs_T}",
            xlabel=f"mean(entropy, first {early_points_abs} tokens) / mean(entropy, full)",
            pair_annotations=[(A_short, abs_shift_A), (B_short, abs_shift_B)],
            input_shift=abs_input_shift,
            reference_x=1.0,
        ),
    )

    stat = {
        "A_name": args.A_name,
        "B_name": args.B_name,
        "A_base": args.A_base,
        "A_post": args.A_post,
        "B_base": args.B_base,
        "B_post": args.B_post,
        "A_short": A_short,
        "B_short": B_short,
        "early_frac": float(args.early_frac),
        "abs_T": int(args.abs_T),
        "relative_shift": {A_short: rel_shift_A, B_short: rel_shift_B},
        "absolute_shift": {A_short: abs_shift_A, B_short: abs_shift_B},
        "input_shift_indicators": {
            "definition": "msk=mean(d_i); bsk=mean(d_i)-d_ref; rt=mean(1[d_i>tau]); d_ref and tau use the same-strategy NN distribution.",
            "tail_quantile": 0.90,
            "relative": rel_input_shift,
            "absolute": abs_input_shift,
        },
        "counts": {
            A_short: {
                "norm_np": int(A["norm_np"].shape[0]),
                "norm_nn": int(A["norm_nn"].shape[0]),
                "abs_np": int(A["abs_np"].shape[0]),
                "abs_nn": int(A["abs_nn"].shape[0]),
            },
            B_short: {
                "norm_np": int(B["norm_np"].shape[0]),
                "norm_nn": int(B["norm_nn"].shape[0]),
                "abs_np": int(B["abs_np"].shape[0]),
                "abs_nn": int(B["abs_nn"].shape[0]),
            },
        },
        "outputs": {k: str(v) for k, v in output_files.items()},
    }
    with (outdir / f"compare_entropy_{A_short}_vs_{B_short}_paper_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    for path in output_files.values():
        print(f"[OK] saved: {path}")
    print(f"[OK] stats: {outdir / f'compare_entropy_{A_short}_vs_{B_short}_paper_stats.json'}")


if __name__ == "__main__":
    main()
