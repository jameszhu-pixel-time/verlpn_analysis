#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ent_pos_compare2.py

Compare TWO strategies on the SAME base rollouts:
- For each strategy i:
  - Use base vs post_i to label qids NP_i / NN_i (base=N only).
  - Use ONLY train_base rollouts to compute prompt-level mean entropy curves.
- Plot 4 curves on one figure: (sA: NP, NN) + (sB: NP, NN)
  with variance shading (default: 95% bootstrap CI over qids; optionally: +/-1 std).

Also supports:
- matched-size version per strategy (downsample its NP/NN to equal counts)
- absolute-index curves (entropy vs token index) if --plot_abs is set

Protocol is aligned with your reference intra_strategy script.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PAD_ID_DEFAULT = 151643


# ---------------- IO ----------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _get_first(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
    v = _get_first(rec, ["q_id", "question_id", "qid", "id"])
    return None if v is None else str(v)


def extract_rid(rec: Dict[str, Any]) -> Optional[str]:
    v = _get_first(rec, ["generation_id", "rid", "rollout_id", "sample_id"])
    if v is not None:
        return str(v)
    p = _get_first(rec, ["full_logprobs_path", "npz_path", "rollout_npz", "npz", "path", "file", "save_path"])
    if p is None:
        return None
    s = str(p)
    m = re.search(r"(?:^|[^a-zA-Z0-9])rid(?:=|:|[_-])?(\d+)(?:[^a-zA-Z0-9]|$)", s)
    return m.group(1) if m else None


def extract_correct(rec: Dict[str, Any]) -> Optional[bool]:
    v = _get_first(rec, ["label", "is_correct", "correct", "success"])
    if v is not None:
        if isinstance(v, bool):
            return v
        try:
            return bool(int(v))
        except Exception:
            return None
    s = _get_first(rec, ["score", "reward", "final_reward"])
    if s is None:
        return None
    try:
        return float(s) > 0.0
    except Exception:
        return None


def extract_token_ids(rec: Dict[str, Any]) -> Optional[List[int]]:
    v = _get_first(rec, ["token_ids", "response_token_ids", "tokens"])
    return v if isinstance(v, list) else None


def build_response_mask_from_tokens(token_ids: Optional[List[int]], pad_id: int) -> Optional[np.ndarray]:
    if token_ids is None:
        return None
    toks = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if toks.size == 0:
        return None
    is_pad = (toks == pad_id)
    if not np.any(is_pad):
        return np.ones_like(toks, dtype=np.bool_)
    first_pad = int(np.argmax(is_pad))
    m = np.zeros_like(toks, dtype=np.bool_)
    m[:first_pad] = True
    return m


def extract_valid_len_hint(rec: Dict[str, Any]) -> Optional[int]:
    v = _get_first(rec, ["valid_len", "response_len", "gen_len", "length", "output_len"])
    if v is None:
        return None
    try:
        x = int(v)
        return x if x > 0 else None
    except Exception:
        return None


def extract_response_mask(rec: Dict[str, Any], T: int, pad_id: int) -> Optional[np.ndarray]:
    m = rec.get("response_mask", None)
    if m is not None:
        try:
            m = np.asarray(m).astype(bool).reshape(-1)
            if m.size == T:
                return m
        except Exception:
            pass

    L = extract_valid_len_hint(rec)
    if L is not None:
        L = max(0, min(T, int(L)))
        m = np.zeros((T,), dtype=bool)
        m[:L] = True
        return m

    token_ids = extract_token_ids(rec)
    if token_ids is not None:
        m = build_response_mask_from_tokens(token_ids, pad_id)
        if m is not None and m.size == T:
            return m

    return None


# ---------------- Entropy compute (same as your reference) ----------------
def entropy_from_topk_logprobs_vec(logp_topk: np.ndarray) -> np.ndarray:
    x = np.asarray(logp_topk, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    m = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - m)
    Z = np.sum(ex, axis=1, keepdims=True) + 1e-20
    p = ex / Z
    ent = -np.sum(p * np.log(p + 1e-20), axis=1)
    return ent.astype(np.float64)


def entropy_from_topk_logprobs_json(topk_logprobs_per_token) -> Optional[np.ndarray]:
    if topk_logprobs_per_token is None:
        return None
    try:
        T = len(topk_logprobs_per_token)
        if T == 0:
            return None
        out = np.full((T,), np.nan, dtype=np.float64)
        for t in range(T):
            pairs = topk_logprobs_per_token[t]
            if not pairs:
                continue
            lps = np.array([float(p[1]) for p in pairs], dtype=np.float64)
            mm = np.max(lps)
            p = np.exp(lps - mm)
            Z = np.sum(p) + 1e-20
            p = p / Z
            out[t] = -np.sum(p * np.log(p + 1e-20))
        return out
    except Exception:
        return None


_NPZ_NAME_PATTERNS = [
    re.compile(r".*?(?:step|global_step)?[-_=]?\d*?_?qid(\d+)_rid(\d+)\.npz$", re.IGNORECASE),
    re.compile(r".*?qid(\d+)_rid(\d+)\.npz$", re.IGNORECASE),
]


def _try_parse_qid_rid_from_npz_name(name: str) -> Optional[Tuple[str, str]]:
    for pat in _NPZ_NAME_PATTERNS:
        m = pat.match(name)
        if m:
            return str(m.group(1)), str(m.group(2))
    return None


def build_npz_index(jsonl_path: Path) -> Dict[Tuple[str, str], Path]:
    idx: Dict[Tuple[str, str], Path] = {}
    cand_dirs = [jsonl_path.parent, jsonl_path.parent / "npz"]
    for d in cand_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in d.glob("*.npz"):
            pr = _try_parse_qid_rid_from_npz_name(p.name)
            if pr is None:
                continue
            idx[(pr[0], pr[1])] = p
    return idx


def load_entropy_from_npz(npz_path: Path) -> Optional[np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False, mmap_mode="r") as z:
            if "full_logprobs" in z.files:
                lp = np.asarray(z["full_logprobs"], dtype=np.float64)
                if lp.ndim == 2 and lp.shape[0] > 0:
                    return entropy_from_topk_logprobs_vec(lp)
            if "token_entropy" in z.files:
                ent = np.asarray(z["token_entropy"], dtype=np.float64).reshape(-1)
                if ent.size > 0:
                    return ent
    except Exception:
        return None
    return None


def _safe_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return None if arr.size == 0 else arr
    except Exception:
        return None


def extract_entropy(rec: Dict[str, Any],
                    qid: Optional[str],
                    rid: Optional[str],
                    npz_index: Optional[Dict[Tuple[str, str], Path]]) -> Optional[np.ndarray]:
    flp = rec.get("full_logprobs", None)
    if flp is not None:
        try:
            lp = np.asarray(flp, dtype=np.float64)
            if lp.ndim == 2 and lp.shape[0] > 0:
                return entropy_from_topk_logprobs_vec(lp)
        except Exception:
            pass

    ent = _safe_float_array(_get_first(rec, ["token_entropies_topk", "entropy", "entropies"]))
    if ent is not None:
        return ent

    ent2 = entropy_from_topk_logprobs_json(rec.get("topk_logprobs_per_token"))
    if ent2 is not None:
        return ent2

    npz_path = _get_first(rec, ["full_logprobs_path", "npz_path", "npz_file", "npz"])
    if isinstance(npz_path, str) and npz_path:
        p = Path(npz_path)
        if p.exists():
            e = load_entropy_from_npz(p)
            if e is not None:
                return e

    if npz_index is not None and qid is not None and rid is not None:
        p = npz_index.get((qid, rid))
        if p is not None and p.exists():
            e = load_entropy_from_npz(p)
            if e is not None:
                return e

    return None


# ---------------- Core data ----------------
@dataclass
class Sample:
    qid: str
    rid: str
    correct: bool
    ent: np.ndarray     # (T,) with NaNs for invalid
    valid_len: int


def load_samples(path: Path, pad_id: int, max_len: int) -> List[Sample]:
    samples: List[Sample] = []
    npz_index = build_npz_index(path)

    for rec in tqdm(iter_jsonl(path), desc=f"load {path.name}"):
        qid = extract_qid(rec)
        rid = extract_rid(rec) or "unknown"
        correct = extract_correct(rec)
        if qid is None or correct is None:
            continue

        ent = extract_entropy(rec, qid=qid, rid=rid, npz_index=npz_index)
        if ent is None or ent.size == 0:
            continue

        ent = np.asarray(ent, dtype=np.float64).reshape(-1)
        if ent.size > max_len:
            ent = ent[:max_len]
        T = int(ent.size)

        mask = extract_response_mask(rec, T=T, pad_id=pad_id)
        if mask is not None:
            if mask.size > max_len:
                mask = mask[:max_len]
            valid = mask & np.isfinite(ent)
        else:
            valid = np.isfinite(ent)

        if not np.any(valid):
            continue

        ent2 = ent.astype(np.float64, copy=True)
        ent2[~valid] = np.nan
        samples.append(Sample(qid=str(qid), rid=str(rid), correct=bool(correct),
                              ent=ent2, valid_len=int(np.sum(valid))))

    return samples


def per_qid_any_correct(samples: List[Sample]) -> Dict[str, bool]:
    anyc: Dict[str, bool] = {}
    for s in samples:
        anyc[s.qid] = anyc.get(s.qid, False) or bool(s.correct)
    return anyc


# ---------------- Curve builders ----------------
def interp_entropy_to_M(ent: np.ndarray, M: int) -> Optional[np.ndarray]:
    x = np.asarray(ent, dtype=np.float64).reshape(-1)
    T = x.size
    if T < 2:
        return None
    idx = np.where(np.isfinite(x))[0]
    if idx.size < 2:
        return None
    v = x[idx]
    p = idx.astype(np.float64) / max(1.0, (T - 1.0))
    pt = np.linspace(0.0, 1.0, M, dtype=np.float64)
    return np.interp(pt, p, v).astype(np.float64)


def prompt_level_mean_curves_normalized(samples: List[Sample], qids_keep: set, M: int) -> Dict[str, np.ndarray]:
    bucket: Dict[str, List[np.ndarray]] = {}
    for s in samples:
        if s.qid not in qids_keep:
            continue
        y = interp_entropy_to_M(s.ent, M=M)
        if y is None:
            continue
        bucket.setdefault(s.qid, []).append(y)
    out: Dict[str, np.ndarray] = {}
    for q, lst in bucket.items():
        if lst:
            out[q] = np.mean(np.stack(lst, axis=0), axis=0)
    return out


def prompt_level_mean_curves_absolute(samples: List[Sample], qids_keep: set, abs_T: int) -> Dict[str, np.ndarray]:
    bucket: Dict[str, List[np.ndarray]] = {}
    for s in samples:
        if s.qid not in qids_keep:
            continue
        x = np.asarray(s.ent, dtype=np.float64).reshape(-1)
        if x.size == 0:
            continue
        y = np.full((abs_T,), np.nan, dtype=np.float64)
        L = min(abs_T, x.size)
        y[:L] = x[:L]
        if not np.any(np.isfinite(y)):
            continue
        bucket.setdefault(s.qid, []).append(y)
    out: Dict[str, np.ndarray] = {}
    for q, lst in bucket.items():
        if not lst:
            continue
        A = np.stack(lst, axis=0)
        m = np.nanmean(A, axis=0)
        if np.any(np.isfinite(m)):
            out[q] = m
    return out


# ---------------- Variance / CI ----------------
def ci_bootstrap(mat: np.ndarray, boot: int, alpha: float = 0.05, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N, L = mat.shape
    if N <= 1:
        return mat[0], mat[0]
    idx = rng.integers(0, N, size=(boot, N))
    means = np.nanmean(mat[idx], axis=1)  # (boot, L)
    lo = np.nanquantile(means, alpha / 2, axis=0)
    hi = np.nanquantile(means, 1 - alpha / 2, axis=0)
    return lo, hi


def band_std(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.nanmean(mat, axis=0)
    s = np.nanstd(mat, axis=0, ddof=1)
    return m - s, m + s


def downsample_qids(qids_a: List[str], qids_b: List[str], seed: int) -> Tuple[List[str], List[str], int]:
    n = int(min(len(qids_a), len(qids_b)))
    if n <= 0:
        return [], [], 0
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(qids_a), size=n, replace=False)
    ib = rng.choice(len(qids_b), size=n, replace=False)
    return [qids_a[i] for i in ia], [qids_b[i] for i in ib], n

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def pointwise_quantile_band(mat: np.ndarray, q_low=2.5, q_high=97.5):
    lo = np.nanpercentile(mat, q_low, axis=0)
    hi = np.nanpercentile(mat, q_high, axis=0)
    return lo, hi


def plot_4lines_with_band(
    series: List[Tuple[str, np.ndarray]],  # (label, mat[N,L])
    x: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    early_region: Optional[Tuple[float, float]],
    band_mode: str,
    boot: int,
):
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=170)

    if early_region is not None:
        ax.axvspan(early_region[0], early_region[1], alpha=0.12)

    plotted = []  # store (label, mean, lo, hi, n)

    for i, (label, mat) in enumerate(series):
        if mat is None or mat.ndim != 2 or mat.shape[0] == 0:
            continue

        mean = np.nanmean(mat, axis=0)

        if band_mode == "std":
            lo, hi = band_std(mat)
        elif band_mode == "bootstrap":
            lo, hi = ci_bootstrap(mat, boot=boot, seed=100 + i)
        else:
            # default: quantile band
            lo, hi = pointwise_quantile_band(mat, q_low=2.5, q_high=97.5)

        line, = ax.plot(x, mean, label=f"{label} (count={mat.shape[0]})", linewidth=2.0)
        ax.fill_between(x, lo, hi, alpha=0.18, color=line.get_color())

        plotted.append((label, mean, lo, hi, mat.shape[0], line.get_color()))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # inset zoom
    if early_region is not None and len(plotted) > 0:
        x0, x1 = early_region
        x0 = 50 # avoid early confusion
        # normalized vs absolute x handling
        if np.nanmax(x) <= 1.0 + 1e-8:
            x0 = max(x0, 0.03)   # skip first ~3% normalized positions
        else:
            x0 = max(x0, 50.0)   # skip first 50 absolute tokens

        mask = (x >= x0) & (x <= x1)

        if np.sum(mask) >= 2:
            axins = inset_axes(ax, width="42%", height="42%", loc="upper right")

            local_vals = []
            for label, mean, lo, hi, n, color in plotted:
                axins.plot(x[mask], mean[mask], linewidth=1.8, color=color)
                axins.fill_between(x[mask], lo[mask], hi[mask], alpha=0.18, color=color)

                local_vals.append(lo[mask])
                local_vals.append(hi[mask])

            local_vals = np.concatenate(local_vals, axis=0)
            local_vals = local_vals[np.isfinite(local_vals)]
            if local_vals.size > 0:
                y_min = np.min(local_vals)
                y_max = np.max(local_vals)
                pad = 0.08 * (y_max - y_min + 1e-8)
                axins.set_ylim(y_min - pad, y_max + pad)

            axins.set_xlim(x0, x1)
            axins.grid(True, linewidth=0.25, alpha=0.35)
            axins.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
# def plot_4lines_with_band(
#     series: List[Tuple[str, np.ndarray]],  # (label, mat[N,L])
#     x: np.ndarray,
#     out_png: Path,
#     title: str,
#     xlabel: str,
#     ylabel: str,
#     early_region: Optional[Tuple[float, float]],
#     band_mode: str,
#     boot: int,
# ):
#     plt.figure(figsize=(8.4, 4.6), dpi=170)
#     if early_region is not None:
#         plt.axvspan(early_region[0], early_region[1], alpha=0.12)

#     for i, (label, mat) in enumerate(series):
#         mean = np.nanmean(mat, axis=0)
#         if band_mode == "std":
#             lo, hi = band_std(mat)
#         else:
#             lo, hi = ci_bootstrap(mat, boot=boot, seed=100 + i)
#         plt.plot(x, mean, label=f"{label} (n={mat.shape[0]})")
#         plt.fill_between(x, lo, hi, alpha=0.18)

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend(frameon=False, ncol=2)
#     plt.grid(True, linewidth=0.3, alpha=0.4)
#     plt.tight_layout()
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_png, dpi=200)
#     plt.close()


def collect_strategy_np_nn_qids(base_anyc: Dict[str, bool], post_anyc: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    all_qids = set(base_anyc.keys()) | set(post_anyc.keys())
    np_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and post_anyc.get(q, False)])
    nn_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and (not post_anyc.get(q, False))])
    return np_qids, nn_qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_base", type=str, required=True)

    ap.add_argument("--A_name", type=str, required=True)
    ap.add_argument("--A_post", type=str, required=True)

    ap.add_argument("--B_name", type=str, required=True)
    ap.add_argument("--B_post", type=str, required=True)

    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)

    # normalized
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--early_frac", type=float, default=0.2)
    ap.add_argument("--boot", type=int, default=2000)

    # band type
    ap.add_argument("--band_mode", choices=["ci", "std", "quantile"], default="quantile",
                    help="Variance band: ci=95% bootstrap CI over qids; std=mean±1std over qids.")

    # matched per strategy
    ap.add_argument("--make_matched", action="store_true")
    ap.add_argument("--matched_seed", type=int, default=0)

    # absolute
    ap.add_argument("--plot_abs", action="store_true")
    ap.add_argument("--abs_T", type=int, default=1024)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load base once (entropy source)
    base_samples = load_samples(Path(args.train_base), pad_id=args.pad_id, max_len=args.max_len)
    base_anyc = per_qid_any_correct(base_samples)

    # load each post file only for labeling
    A_post_samples = load_samples(Path(args.A_post), pad_id=args.pad_id, max_len=args.max_len)
    B_post_samples = load_samples(Path(args.B_post), pad_id=args.pad_id, max_len=args.max_len)
    A_anyc = per_qid_any_correct(A_post_samples)
    B_anyc = per_qid_any_correct(B_post_samples)

    A_np, A_nn = collect_strategy_np_nn_qids(base_anyc, A_anyc)
    B_np, B_nn = collect_strategy_np_nn_qids(base_anyc, B_anyc)

    # build prompt-level curves from base once, then slice
    keep_all = set(A_np) | set(A_nn) | set(B_np) | set(B_nn)
    q_curve_norm = prompt_level_mean_curves_normalized(base_samples, qids_keep=keep_all, M=args.M)

    def mat(qids: List[str]) -> np.ndarray:
        xs = [q_curve_norm[q] for q in qids if q in q_curve_norm]
        if not xs:
            return np.zeros((0, args.M), dtype=np.float64)
        return np.stack(xs, axis=0).astype(np.float64)

    A_np_mat, A_nn_mat = mat(A_np), mat(A_nn)
    B_np_mat, B_nn_mat = mat(B_np), mat(B_nn)

    if min(A_np_mat.shape[0], A_nn_mat.shape[0], B_np_mat.shape[0], B_nn_mat.shape[0]) == 0:
        raise RuntimeError("Some group has 0 curves after extraction. Check qid overlap / entropy extraction.")

    x_norm = np.linspace(0.0, 1.0, args.M)
    out_png = outdir / f"compare2_{args.A_name}_vs_{args.B_name}_norm.png"
    title = f"NP/NN entropy-position (plot=train_base) | {args.A_name} vs {args.B_name}"
    series = [
        (f"{args.A_name}-NP", A_np_mat),
        (f"{args.A_name}-NN", A_nn_mat),
        (f"{args.B_name}-NP", B_np_mat),
        (f"{args.B_name}-NN", B_nn_mat),
    ]
    plot_4lines_with_band(
        series=series,
        x=x_norm,
        out_png=out_png,
        title=title,
        xlabel="Normalized response position",
        ylabel="Token entropy",
        early_region=(0.0, float(args.early_frac)),
        band_mode=args.band_mode,
        boot=args.boot,
    )
    print(f"[OK] saved: {out_png}")

    # matched normalized (per-strategy NP/NN equalized)
    if args.make_matched:
        A_np_m, A_nn_m, nA = downsample_qids(A_np, A_nn, seed=args.matched_seed)
        B_np_m, B_nn_m, nB = downsample_qids(B_np, B_nn, seed=args.matched_seed)
        A_np_m_mat, A_nn_m_mat = mat(A_np_m), mat(A_nn_m)
        B_np_m_mat, B_nn_m_mat = mat(B_np_m), mat(B_nn_m)

        out_png_m = outdir / f"compare2_{args.A_name}_vs_{args.B_name}_norm_MATCHED.png"
        title_m = f"NP/NN entropy-position MATCHED (plot=train_base) | {args.A_name}(n={nA}) vs {args.B_name}(n={nB}) | seed={args.matched_seed}"
        series_m = [
            (f"{args.A_name}-NPm", A_np_m_mat),
            (f"{args.A_name}-NNm", A_nn_m_mat),
            (f"{args.B_name}-NPm", B_np_m_mat),
            (f"{args.B_name}-NNm", B_nn_m_mat),
        ]
        plot_4lines_with_band(
            series=series_m,
            x=x_norm,
            out_png=out_png_m,
            title=title_m,
            xlabel="Normalized response position",
            ylabel="Token entropy",
            early_region=(0.0, float(args.early_frac)),
            band_mode=args.band_mode,
            boot=args.boot,
        )
        print(f"[OK] saved: {out_png_m}")

    # absolute plot
    if args.plot_abs:
        q_curve_abs = prompt_level_mean_curves_absolute(base_samples, qids_keep=keep_all, abs_T=args.abs_T)

        def mat_abs(qids: List[str]) -> np.ndarray:
            xs = [q_curve_abs[q] for q in qids if q in q_curve_abs]
            if not xs:
                return np.zeros((0, args.abs_T), dtype=np.float64)
            return np.stack(xs, axis=0).astype(np.float64)

        A_np_abs, A_nn_abs = mat_abs(A_np), mat_abs(A_nn)
        B_np_abs, B_nn_abs = mat_abs(B_np), mat_abs(B_nn)

        x_abs = np.arange(args.abs_T, dtype=np.float64)
        out_png_abs = outdir / f"compare2_{args.A_name}_vs_{args.B_name}_absT{args.abs_T}.png"
        title_abs = f"NP/NN entropy-abs-index (plot=train_base) | {args.A_name} vs {args.B_name} | T={args.abs_T}"
        early_abs = (0.0, float(args.early_frac) * float(args.abs_T))
        series_abs = [
            (f"{args.A_name}-NP", A_np_abs),
            (f"{args.A_name}-NN", A_nn_abs),
            (f"{args.B_name}-NP", B_np_abs),
            (f"{args.B_name}-NN", B_nn_abs),
        ]
        plot_4lines_with_band(
            series=series_abs,
            x=x_abs,
            out_png=out_png_abs,
            title=title_abs,
            xlabel="Absolute token index",
            ylabel="Token entropy",
            early_region=early_abs,
            band_mode=args.band_mode,
            boot=args.boot,
        )
        print(f"[OK] saved: {out_png_abs}")

        if args.make_matched:
            A_np_m, A_nn_m, nA = downsample_qids(A_np, A_nn, seed=args.matched_seed)
            B_np_m, B_nn_m, nB = downsample_qids(B_np, B_nn, seed=args.matched_seed)

            A_np_abs_m, A_nn_abs_m = mat_abs(A_np_m), mat_abs(A_nn_m)
            B_np_abs_m, B_nn_abs_m = mat_abs(B_np_m), mat_abs(B_nn_m)

            out_png_abs_m = outdir / f"compare2_{args.A_name}_vs_{args.B_name}_absT{args.abs_T}_MATCHED.png"
            title_abs_m = f"NP/NN abs-index MATCHED | {args.A_name}(n={nA}) vs {args.B_name}(n={nB}) | seed={args.matched_seed} | T={args.abs_T}"
            series_abs_m = [
                (f"{args.A_name}-NPm", A_np_abs_m),
                (f"{args.A_name}-NNm", A_nn_abs_m),
                (f"{args.B_name}-NPm", B_np_abs_m),
                (f"{args.B_name}-NNm", B_nn_abs_m),
            ]
            plot_4lines_with_band(
                series=series_abs_m,
                x=x_abs,
                out_png=out_png_abs_m,
                title=title_abs_m,
                xlabel="Absolute token index",
                ylabel="Token entropy",
                early_region=early_abs,
                band_mode=args.band_mode,
                boot=args.boot,
            )
            print(f"[OK] saved: {out_png_abs_m}")


if __name__ == "__main__":
    main()