#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-oriented entropy-position figure.

Design goals:
- one figure per strategy
- split the old mixed figure into separate token-level curve and
  distribution-shift figures
- each strategy uses its own base file
- add a token-level exp(entropy) curve when the source metric is entropy
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
    "np_line": "#D55E00",
    "np_fill": "#F1B27A",
    "nn_line": "#1F77B4",
    "nn_fill": "#9FC3E6",
    "early_fill": "#F3EBDD",
    "early_edge": "#C49A5A",
    "grid": "#D8DEE7",
    "spine": "#A9B4C2",
    "text": "#22303C",
    "subtle": "#5E6B77",
}


plt.rcParams.update(
    {
        "figure.facecolor": "white",
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
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
    }
)


def _label_strategy(name: str) -> str:
    if name.startswith("normal_"):
        return "n" + name.split("_", 1)[1]
    return name


def _stack_curves(curve_map: Dict[str, np.ndarray], qids: List[str], width: int) -> np.ndarray:
    curves = [curve_map[q] for q in qids if q in curve_map]
    if not curves:
        return np.zeros((0, width), dtype=np.float64)
    return np.stack(curves, axis=0).astype(np.float64)


def _quantile_band(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(mat, axis=0)
    lo = np.nanpercentile(mat, 2.5, axis=0)
    hi = np.nanpercentile(mat, 97.5, axis=0)
    return mean, lo, hi


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _robust_xlim(arrs: Iterable[np.ndarray]) -> Tuple[float, float]:
    chunks = []
    for arr in arrs:
        vals = _finite(arr)
        if vals.size > 0:
            chunks.append(vals)
    if not chunks:
        return -1.0, 1.0
    vals = np.concatenate(chunks, axis=0)
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


def _early_ratio(mat: np.ndarray, early_points: int) -> np.ndarray:
    early_points = max(1, min(int(early_points), int(mat.shape[1])))
    early = np.nanmean(mat[:, :early_points], axis=1)
    full = np.nanmean(mat, axis=1)
    out = early / np.maximum(full, 1e-12)
    out[~np.isfinite(out)] = np.nan
    return out


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
    return float((np.sum(wins) + 0.5 * np.sum(ties)) / max(float(aa.size * bb.size), 1.0))


def _summary_shift(a: np.ndarray, b: np.ndarray) -> Dict[str, Optional[float]]:
    aa = _finite(a)
    bb = _finite(b)
    if aa.size == 0 or bb.size == 0:
        return {"delta_mean": None, "delta_median": None, "ps": None}
    delta_mean = float(np.mean(aa) - np.mean(bb))
    delta_median = float(np.median(aa) - np.median(bb))
    ps = _probability_superiority(aa, bb)
    return {
        "delta_mean": delta_mean if np.isfinite(delta_mean) else None,
        "delta_median": delta_median if np.isfinite(delta_median) else None,
        "ps": ps if np.isfinite(ps) else None,
    }


def _configure_axis(ax, ylabel: Optional[str] = None):
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _focus_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    finite_chunks = [a[np.isfinite(a)] for a in arrays if np.any(np.isfinite(a))]
    if not finite_chunks:
        return 0.0, 1.0
    vals = np.concatenate(finite_chunks, axis=0)
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(vals, 5.0))
    hi = float(np.nanpercentile(vals, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    pad = 0.12 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _draw_panel(
    ax,
    *,
    x: np.ndarray,
    np_mat: np.ndarray,
    nn_mat: np.ndarray,
    panel_title: str,
    xlabel: str,
    ylabel: Optional[str],
    early_end: float,
    zoom_start: float,
    zoom_end: float,
    relative: bool,
) -> float:
    np_mean, np_lo, np_hi = _quantile_band(np_mat)
    nn_mean, nn_lo, nn_hi = _quantile_band(nn_mat)

    ax.axvspan(0.0, early_end, facecolor=PALETTE["early_fill"], alpha=0.9, zorder=0)
    ax.axvline(early_end, color=PALETTE["early_edge"], linewidth=1.1, linestyle=(0, (3, 3)))

    ax.plot(x, np_mean, color=PALETTE["np_line"], linewidth=2.4, label=f"N2P (n={np_mat.shape[0]})")
    ax.fill_between(x, np_lo, np_hi, color=PALETTE["np_fill"], alpha=0.22)

    ax.plot(x, nn_mean, color=PALETTE["nn_line"], linewidth=2.4, label=f"N2N (n={nn_mat.shape[0]})")
    ax.fill_between(x, nn_lo, nn_hi, color=PALETTE["nn_fill"], alpha=0.22)

    _configure_axis(ax, ylabel=ylabel)
    ax.set_title(panel_title, loc="left", pad=8, fontweight="semibold")
    ax.set_xlabel(xlabel)

    if relative:
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))}%"))
        focus_mask = (x >= zoom_start) & (x <= zoom_end)
    else:
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_xticks(np.linspace(float(x[0]), float(x[-1]), 7))
        focus_mask = (x >= zoom_start) & (x <= zoom_end)

    gap = float(np.nanmean(np_mean[focus_mask] - nn_mean[focus_mask])) if np.any(focus_mask) else float("nan")
    if np.isfinite(gap):
        ax.text(
            0.02,
            0.96,
            f"early gap {gap:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            color=PALETTE["subtle"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE["grid"], alpha=0.95),
        )

    if np.sum(focus_mask) >= 3:
        axins = inset_axes(ax, width="48%", height="49%", loc="upper right", borderpad=1.0)
        axins.axvspan(zoom_start, zoom_end, facecolor=PALETTE["early_fill"], alpha=0.55, zorder=0)
        axins.plot(x[focus_mask], np_mean[focus_mask], color=PALETTE["np_line"], linewidth=1.9)
        axins.fill_between(x[focus_mask], np_lo[focus_mask], np_hi[focus_mask], color=PALETTE["np_fill"], alpha=0.22)
        axins.plot(x[focus_mask], nn_mean[focus_mask], color=PALETTE["nn_line"], linewidth=1.9)
        axins.fill_between(x[focus_mask], nn_lo[focus_mask], nn_hi[focus_mask], color=PALETTE["nn_fill"], alpha=0.22)
        axins.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
        axins.grid(axis="x", color=PALETTE["grid"], linewidth=0.35, alpha=0.25)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)
        axins.spines["left"].set_color(PALETTE["spine"])
        axins.spines["bottom"].set_color(PALETTE["spine"])
        axins.tick_params(axis="both", labelsize=8, length=2.5)
        axins.set_xlim(zoom_start, zoom_end)
        y0, y1 = _focus_ylim(np_mean[focus_mask], np_lo[focus_mask], np_hi[focus_mask], nn_mean[focus_mask], nn_lo[focus_mask], nn_hi[focus_mask])
        axins.set_ylim(y0, y1)
        if relative:
            axins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))}%"))
        axins.set_title("early 20%", fontsize=8.5, loc="left", pad=2)

    return gap


def _draw_token_curve_figure(
    *,
    out_png: Path,
    out_pdf: Path,
    display_name: str,
    subtitle: str,
    x_abs: np.ndarray,
    np_abs: np.ndarray,
    nn_abs: np.ndarray,
    ylabel: str,
    early_end: float,
    zoom_start: float,
    zoom_end: float,
    abs_T: int,
) -> float:
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.9), dpi=220)
    fig.subplots_adjust(top=0.80)
    fig.text(0.08, 0.965, display_name, ha="left", va="top", fontsize=16, fontweight="bold", color=PALETTE["text"])
    fig.text(0.08, 0.925, subtitle, ha="left", va="top", fontsize=11, color=PALETTE["subtle"])

    gap = _draw_panel(
        ax,
        x=x_abs,
        np_mat=np_abs,
        nn_mat=nn_abs,
        panel_title=f"Token level · T={abs_T}",
        xlabel="Token index",
        ylabel=ylabel,
        early_end=early_end,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        relative=False,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.94, 0.965),
        ncol=1,
        frameon=False,
        handlelength=2.2,
    )

    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)
    return gap


def _draw_distribution_figure(
    *,
    out_png: Path,
    out_pdf: Path,
    display_name: str,
    subtitle: str,
    np_dist: np.ndarray,
    nn_dist: np.ndarray,
    xlabel: str,
    pair_summary: Dict[str, Optional[float]],
):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4), dpi=220)
    fig.subplots_adjust(top=0.78)
    fig.text(0.08, 0.965, display_name, ha="left", va="top", fontsize=16, fontweight="bold", color=PALETTE["text"])
    fig.text(0.08, 0.925, subtitle, ha="left", va="top", fontsize=11, color=PALETTE["subtle"])

    distributions = [
        ("N2P", "np", np_dist),
        ("N2N", "nn", nn_dist),
    ]
    xs = [vals for _, _, vals in distributions if _finite(vals).size > 0]
    lo, hi = _robust_xlim(xs)
    span = max(hi - lo, 1e-6)
    lo_pad = lo - 0.03 * span
    hi_pad = hi + 0.36 * span
    grid = np.linspace(lo, hi, 500)
    positions = [2, 1]
    width = 0.32
    median_map = {}

    if lo <= 1.0 <= hi:
        ax.axvline(1.0, color=PALETTE["spine"], linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.95, zorder=0.5)

    ax.axhspan(0.52, 2.48, color="#F7F9FB", zorder=0)
    label_x = lo_pad + 0.012 * (hi_pad - lo_pad)

    for pos, (group_label, group_key, vals) in zip(positions, distributions):
        values = _finite(vals)
        if values.size == 0:
            continue
        color = PALETTE["np_line"] if group_key == "np" else PALETTE["nn_line"]
        fill = PALETTE["np_fill"] if group_key == "np" else PALETTE["nn_fill"]
        dens = _kde_gaussian(values, grid)
        scale = width / max(float(np.max(dens)), 1e-9)
        y_top = pos + dens * scale
        y_bot = pos - dens * scale

        ax.fill_between(grid, y_bot, y_top, color=fill, alpha=0.35, linewidth=0.0)
        ax.plot(grid, y_top, color=color, linewidth=1.5)
        ax.plot(grid, y_bot, color=color, linewidth=1.5)

        median = float(np.median(values))
        mean = float(np.mean(values))
        q1, q3 = [float(np.quantile(values, q)) for q in (0.25, 0.75)]
        ax.plot([q1, q3], [pos, pos], color=color, linewidth=3.0, solid_capstyle="round")
        ax.plot([median, median], [pos - 0.13, pos + 0.13], color=color, linewidth=1.5)
        ax.scatter([mean], [pos], s=18, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(label_x, pos + 0.36, f"{group_label}  (n={values.size})", ha="left", va="bottom", fontsize=9.2, color=PALETTE["subtle"])
        median_map[group_key] = {"median": median, "pos": pos}

    if "np" in median_map and "nn" in median_map:
        x0 = median_map["nn"]["median"]
        x1 = median_map["np"]["median"]
        y_mid = 0.5 * (median_map["np"]["pos"] + median_map["nn"]["pos"])
        color = PALETTE["np_line"] if x1 >= x0 else PALETTE["nn_line"]
        ax.annotate(
            "",
            xy=(x1, y_mid),
            xytext=(x0, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6, shrinkA=2.0, shrinkB=2.0),
        )
        delta_txt = "Δmed=n/a" if pair_summary.get("delta_median") is None else f"Δmed={pair_summary['delta_median']:+.3f}"
        ps_txt = "P(N2P>N2N)=n/a" if pair_summary.get("ps") is None else f"P(N2P>N2N)={pair_summary['ps']:.2f}"
        ax.text(
            max(x0, x1) + 0.02 * span,
            y_mid + 0.06,
            f"{delta_txt}  {ps_txt}",
            ha="left",
            va="center",
            fontsize=8.8,
            color=PALETTE["subtle"],
        )

    _configure_axis(ax)
    ax.set_title("Early/full distribution shift", loc="left", pad=8, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_xlim(lo_pad, hi_pad)
    ax.set_ylim(0.5, 2.9)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_base", type=str, required=True)
    ap.add_argument("--train_post", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--strategy", type=str, required=True)
    ap.add_argument("--display_name", type=str, default="")
    ap.add_argument("--token_metric", type=str, default="entropy", choices=["entropy", "gini"])
    ap.add_argument("--pad_id", type=int, default=core.PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--early_frac", type=float, default=0.2)
    ap.add_argument("--abs_T", type=int, default=3072)
    ap.add_argument("--zoom_skip_frac", type=float, default=0.02)
    ap.add_argument("--zoom_skip_abs", type=int, default=32)
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.strategy
    outdir.mkdir(parents=True, exist_ok=True)

    metric_label = "Token entropy" if args.token_metric == "entropy" else "Token gini"
    display_name = args.display_name.strip() or _label_strategy(args.strategy)

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

    all_qids = set(base_anyc.keys()) | set(post_anyc.keys())
    np_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and post_anyc.get(q, False)])
    nn_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and (not post_anyc.get(q, False))])
    keep = set(np_qids) | set(nn_qids)

    q_curve_abs = core.prompt_level_mean_curves_absolute(base_samples, qids_keep=keep, abs_T=args.abs_T)

    np_abs = _stack_curves(q_curve_abs, np_qids, args.abs_T)
    nn_abs = _stack_curves(q_curve_abs, nn_qids, args.abs_T)

    if min(np_abs.shape[0], nn_abs.shape[0]) == 0:
        raise RuntimeError("Some curve groups are empty. Check qid overlap or entropy extraction.")

    x_abs = np.arange(args.abs_T, dtype=np.float64)
    early_points_abs = max(1, int(round(args.early_frac * args.abs_T)))
    early_end_abs = float(args.early_frac) * float(args.abs_T)

    curve_png = outdir / f"{args.token_metric}_token_level_paper.png"
    curve_pdf = outdir / f"{args.token_metric}_token_level_paper.pdf"
    gap_abs = _draw_token_curve_figure(
        out_png=curve_png,
        out_pdf=curve_pdf,
        display_name=display_name,
        subtitle=f"Token-level {args.token_metric}",
        x_abs=x_abs,
        np_abs=np_abs,
        nn_abs=nn_abs,
        ylabel=metric_label,
        early_end=early_end_abs,
        zoom_start=float(args.zoom_skip_abs),
        zoom_end=early_end_abs,
        abs_T=args.abs_T,
    )

    abs_dist_np = _early_ratio(np_abs, early_points_abs)
    abs_dist_nn = _early_ratio(nn_abs, early_points_abs)
    abs_shift = _summary_shift(abs_dist_np, abs_dist_nn)
    dist_png = outdir / f"{args.token_metric}_distribution_shift_paper.png"
    dist_pdf = outdir / f"{args.token_metric}_distribution_shift_paper.pdf"
    _draw_distribution_figure(
        out_png=dist_png,
        out_pdf=dist_pdf,
        display_name=display_name,
        subtitle="Distribution shift",
        np_dist=abs_dist_np,
        nn_dist=abs_dist_nn,
        xlabel=f"mean({args.token_metric}, first {early_points_abs} tokens) / mean({args.token_metric}, full)",
        pair_summary=abs_shift,
    )

    exp_curve_png = None
    exp_curve_pdf = None
    exp_gap_abs = None
    if args.token_metric == "entropy":
        exp_np_abs = np.exp(np_abs)
        exp_nn_abs = np.exp(nn_abs)
        exp_curve_png = outdir / "exp_entropy_token_level_paper.png"
        exp_curve_pdf = outdir / "exp_entropy_token_level_paper.pdf"
        exp_gap_abs = _draw_token_curve_figure(
            out_png=exp_curve_png,
            out_pdf=exp_curve_pdf,
            display_name=display_name,
            subtitle="Token-level exponential entropy",
            x_abs=x_abs,
            np_abs=exp_np_abs,
            nn_abs=exp_nn_abs,
            ylabel="exp(Token entropy)",
            early_end=early_end_abs,
            zoom_start=float(args.zoom_skip_abs),
            zoom_end=early_end_abs,
            abs_T=args.abs_T,
        )

    stat = {
        "display_name": display_name,
        "strategy": args.strategy,
        "train_base": args.train_base,
        "train_post": args.train_post,
        "token_metric": args.token_metric,
        "np_qids": len(np_qids),
        "nn_qids": len(nn_qids),
        "np_abs_curves_used": int(np_abs.shape[0]),
        "nn_abs_curves_used": int(nn_abs.shape[0]),
        "M": int(args.M),
        "abs_T": int(args.abs_T),
        "early_frac": float(args.early_frac),
        "zoom_skip_frac": float(args.zoom_skip_frac),
        "zoom_skip_abs": int(args.zoom_skip_abs),
        "absolute_early_gap": gap_abs if np.isfinite(gap_abs) else None,
        "absolute_shift": abs_shift,
        "exp_entropy_absolute_early_gap": exp_gap_abs if exp_gap_abs is not None and np.isfinite(exp_gap_abs) else None,
        "curve_png": str(curve_png),
        "curve_pdf": str(curve_pdf),
        "distribution_png": str(dist_png),
        "distribution_pdf": str(dist_pdf),
        "exp_entropy_curve_png": str(exp_curve_png) if exp_curve_png is not None else None,
        "exp_entropy_curve_pdf": str(exp_curve_pdf) if exp_curve_pdf is not None else None,
    }
    with (outdir / "entropy_position_paper_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {curve_png}")
    print(f"[OK] saved: {curve_pdf}")
    print(f"[OK] saved: {dist_png}")
    print(f"[OK] saved: {dist_pdf}")
    if exp_curve_png is not None and exp_curve_pdf is not None:
        print(f"[OK] saved: {exp_curve_png}")
        print(f"[OK] saved: {exp_curve_pdf}")
    print(f"[OK] stats: {outdir / 'entropy_position_paper_stats.json'}")


if __name__ == "__main__":
    main()
