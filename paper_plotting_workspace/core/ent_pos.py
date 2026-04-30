#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_intra_strategy_np_entropy_curve.py

ONE strategy:
1) Determine NP/NN qids by any_correct transition between train_base -> train_post:
   - NP: base_anyc=False and post_anyc=True
   - NN: base_anyc=False and post_anyc=False

2) Plot curves using ONLY train_base rollouts:
   A) Normalized position curve (interpolate each rollout to M points in [0,1])
   B) Absolute length curve (no interpolation): token metric vs absolute token index 0..abs_T-1

   Aggregation:
   rollout -> prompt mean curve -> group mean curve
   95% CI / band: pointwise quantile band over prompts/qids

3) Optionally also plot matched-size versions (downsample NP/NN qids to equal size).

Protocol handling follows your reference exactly:
- extract_qid/extract_rid/extract_entropy/build_npz_index/load_entropy_from_npz/extract_response_mask/load_samples

New:
- support --token_metric {entropy,gini}
  * entropy: per-token entropy
  * gini:    per-token Gini impurity = 1 - sum_i p_i^2
             where p is reconstructed from provided top-k logprobs by local softmax
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


# ---------------- Token metric compute ----------------
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


def gini_from_topk_logprobs_vec(logp_topk: np.ndarray) -> np.ndarray:
    x = np.asarray(logp_topk, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    m = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - m)
    Z = np.sum(ex, axis=1, keepdims=True) + 1e-20
    p = ex / Z
    g = 1.0 - np.sum(p * p, axis=1)
    return g.astype(np.float64)


def gini_from_topk_logprobs_json(topk_logprobs_per_token) -> Optional[np.ndarray]:
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
            out[t] = 1.0 - np.sum(p * p)
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


def load_gini_from_npz(npz_path: Path) -> Optional[np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False, mmap_mode="r") as z:
            if "full_logprobs" in z.files:
                lp = np.asarray(z["full_logprobs"], dtype=np.float64)
                if lp.ndim == 2 and lp.shape[0] > 0:
                    return gini_from_topk_logprobs_vec(lp)
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


def extract_token_metric(
    rec: Dict[str, Any],
    qid: Optional[str],
    rid: Optional[str],
    npz_index: Optional[Dict[Tuple[str, str], Path]],
    metric: str = "entropy",
) -> Optional[np.ndarray]:
    if metric == "entropy":
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

    if metric == "gini":
        flp = rec.get("full_logprobs", None)
        if flp is not None:
            try:
                lp = np.asarray(flp, dtype=np.float64)
                if lp.ndim == 2 and lp.shape[0] > 0:
                    return gini_from_topk_logprobs_vec(lp)
            except Exception:
                pass

        g2 = gini_from_topk_logprobs_json(rec.get("topk_logprobs_per_token"))
        if g2 is not None:
            return g2

        npz_path = _get_first(rec, ["full_logprobs_path", "npz_path", "npz_file", "npz"])
        if isinstance(npz_path, str) and npz_path:
            p = Path(npz_path)
            if p.exists():
                g = load_gini_from_npz(p)
                if g is not None:
                    return g

        if npz_index is not None and qid is not None and rid is not None:
            p = npz_index.get((qid, rid))
            if p is not None and p.exists():
                g = load_gini_from_npz(p)
                if g is not None:
                    return g

        return None

    raise ValueError(f"Unsupported metric: {metric}")


# ---------------- Core data ----------------
@dataclass
class Sample:
    qid: str
    rid: str
    correct: bool
    tok_metric: np.ndarray   # (T,) with NaNs for invalid
    valid_len: int


def load_samples(path: Path, pad_id: int, max_len: int, metric: str = "entropy") -> List[Sample]:
    samples: List[Sample] = []
    npz_index = build_npz_index(path)

    for rec in tqdm(iter_jsonl(path), desc=f"load {path.name}"):
        qid = extract_qid(rec)
        rid = extract_rid(rec) or "unknown"
        correct = extract_correct(rec)
        if qid is None or correct is None:
            continue

        tok_metric = extract_token_metric(rec, qid=qid, rid=rid, npz_index=npz_index, metric=metric)
        if tok_metric is None or tok_metric.size == 0:
            continue

        tok_metric = np.asarray(tok_metric, dtype=np.float64).reshape(-1)
        if tok_metric.size > max_len:
            tok_metric = tok_metric[:max_len]
        T = int(tok_metric.size)

        mask = extract_response_mask(rec, T=T, pad_id=pad_id)
        if mask is not None:
            if mask.size > max_len:
                mask = mask[:max_len]
            valid = mask & np.isfinite(tok_metric)
        else:
            valid = np.isfinite(tok_metric)

        if not np.any(valid):
            continue

        tok_metric2 = tok_metric.astype(np.float64, copy=True)
        tok_metric2[~valid] = np.nan
        samples.append(
            Sample(
                qid=str(qid),
                rid=str(rid),
                correct=bool(correct),
                tok_metric=tok_metric2,
                valid_len=int(np.sum(valid)),
            )
        )

    return samples


def per_qid_any_correct(samples: List[Sample]) -> Dict[str, bool]:
    anyc: Dict[str, bool] = {}
    for s in samples:
        anyc[s.qid] = anyc.get(s.qid, False) or bool(s.correct)
    return anyc


# ---------------- Curve utilities ----------------
def interp_metric_to_M(x_in: np.ndarray, M: int) -> Optional[np.ndarray]:
    """
    x_in: (T,) with NaNs marking invalid tokens.
    Linear interpolate over normalized position to length M.
    """
    x = np.asarray(x_in, dtype=np.float64).reshape(-1)
    T = x.size
    if T < 2:
        return None
    idx = np.where(np.isfinite(x))[0]
    if idx.size < 2:
        return None
    v = x[idx]
    p = idx.astype(np.float64) / max(1.0, (T - 1.0))
    pt = np.linspace(0.0, 1.0, M, dtype=np.float64)
    y = np.interp(pt, p, v).astype(np.float64)
    return y


def prompt_level_mean_curves_normalized(samples: List[Sample], qids_keep: set, M: int) -> Dict[str, np.ndarray]:
    """
    For each qid, mean of interpolated curves over rollouts -> prompt curve (M,)
    """
    bucket: Dict[str, List[np.ndarray]] = {}
    for s in samples:
        if s.qid not in qids_keep:
            continue
        y = interp_metric_to_M(s.tok_metric, M=M)
        if y is None:
            continue
        bucket.setdefault(s.qid, []).append(y)

    out: Dict[str, np.ndarray] = {}
    for q, lst in bucket.items():
        if lst:
            out[q] = np.mean(np.stack(lst, axis=0), axis=0)
    return out


def prompt_level_mean_curves_absolute(samples: List[Sample], qids_keep: set, abs_T: int) -> Dict[str, np.ndarray]:
    """
    Absolute token index curve: token metric vs index 0..abs_T-1 (no interpolation).
    For each rollout, we take first abs_T values (NaN for missing/invalid),
    then prompt-level mean across rollouts via nanmean.
    """
    bucket: Dict[str, List[np.ndarray]] = {}
    for s in samples:
        if s.qid not in qids_keep:
            continue
        x = np.asarray(s.tok_metric, dtype=np.float64).reshape(-1)
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
        A = np.stack(lst, axis=0)  # (R, abs_T)
        m = np.nanmean(A, axis=0)
        if np.any(np.isfinite(m)):
            out[q] = m
    return out


def pointwise_quantile_band(mat: np.ndarray, q_low=2.5, q_high=97.5):
    lo = np.nanpercentile(mat, q_low, axis=0)
    hi = np.nanpercentile(mat, q_high, axis=0)
    return lo, hi


def _plot_two_groups(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    x: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    early_region: Optional[Tuple[float, float]],
):
    a_mean = np.nanmean(a_mat, axis=0)
    b_mean = np.nanmean(b_mat, axis=0)

    a_lo, a_hi = pointwise_quantile_band(a_mat, q_low=2.5, q_high=97.5)
    b_lo, b_hi = pointwise_quantile_band(b_mat, q_low=2.5, q_high=97.5)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)

    if early_region is not None:
        ax.axvspan(early_region[0], early_region[1], alpha=0.12)

    ax.plot(x, a_mean, label=f"NP (N→P), count={a_mat.shape[0]}", linewidth=2.0)
    ax.fill_between(x, a_lo, a_hi, alpha=0.22)

    ax.plot(x, b_mean, label=f"NN (N→N), count={b_mat.shape[0]}", linewidth=2.0)
    ax.fill_between(x, b_lo, b_hi, alpha=0.22)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    if early_region is not None:
        if np.nanmax(x) <= 1.0 + 1e-8:
            x0, x1 = 0.01, 0.2
        else:
            x0, x1 = early_region
            x0 = max(x0, 50.0)

        mask = (x >= x0) & (x <= x1)
        if np.sum(mask) >= 2:
            axins = inset_axes(ax, width="42%", height="42%", loc="upper right")

            axins.plot(x[mask], a_mean[mask], linewidth=1.8)
            axins.fill_between(x[mask], a_lo[mask], a_hi[mask], alpha=0.22)

            axins.plot(x[mask], b_mean[mask], linewidth=1.8)
            axins.fill_between(x[mask], b_lo[mask], b_hi[mask], alpha=0.22)

            local_vals = np.concatenate([a_lo[mask], a_hi[mask], b_lo[mask], b_hi[mask]])
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
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def downsample_matched(
    np_qids: List[str],
    nn_qids: List[str],
    seed: int,
) -> Tuple[List[str], List[str], int]:
    n = int(min(len(np_qids), len(nn_qids)))
    if n <= 0:
        return [], [], 0
    rng = np.random.default_rng(seed)
    np_sel = rng.choice(len(np_qids), size=n, replace=False)
    nn_sel = rng.choice(len(nn_qids), size=n, replace=False)
    return [np_qids[i] for i in np_sel], [nn_qids[i] for i in nn_sel], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_base", type=str, required=True, help="train rollouts (base) for the strategy")
    ap.add_argument("--train_post", type=str, required=True, help="train rollouts (post) for the same strategy")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--strategy", type=str, default="strategy")

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)

    ap.add_argument("--token_metric", type=str, default="entropy", choices=["entropy", "gini"],
                    help="Per-token metric used for plotting.")

    # normalized plot
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--early_frac", type=float, default=0.2)
    ap.add_argument("--boot", type=int, default=2000)

    # matched plot
    ap.add_argument("--make_matched", action="store_true",
                    help="Also plot a matched-size NP vs NN curve by downsampling to min(n_np, n_nn).")
    ap.add_argument("--matched_seed", type=int, default=0)

    # absolute length plot
    ap.add_argument("--plot_abs", action="store_true",
                    help="Also plot token metric vs absolute token index (0..abs_T-1) using train_base.")
    ap.add_argument("--abs_T", type=int, default=1024,
                    help="Absolute length horizon for the absolute-index plot.")

    args = ap.parse_args()

    outdir = Path(args.outdir) / args.strategy
    outdir.mkdir(parents=True, exist_ok=True)

    metric_label = "Token entropy" if args.token_metric == "entropy" else "Token Gini (1 - sum p^2)"

    base_samples = load_samples(
        Path(args.train_base),
        pad_id=args.pad_id,
        max_len=args.max_len,
        metric=args.token_metric,
    )
    post_samples = load_samples(
        Path(args.train_post),
        pad_id=args.pad_id,
        max_len=args.max_len,
        metric=args.token_metric,
    )

    base_anyc = per_qid_any_correct(base_samples)
    post_anyc = per_qid_any_correct(post_samples)

    all_qids = set(base_anyc.keys()) | set(post_anyc.keys())
    np_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and post_anyc.get(q, False)])
    nn_qids = sorted([q for q in all_qids if (not base_anyc.get(q, False)) and (not post_anyc.get(q, False))])

    keep = set(np_qids) | set(nn_qids)

    # ---------- normalized plot ----------
    q_curve_norm = prompt_level_mean_curves_normalized(base_samples, qids_keep=keep, M=args.M)

    np_list = [q_curve_norm[q] for q in np_qids if q in q_curve_norm]
    nn_list = [q_curve_norm[q] for q in nn_qids if q in q_curve_norm]
    if len(np_list) == 0 or len(nn_list) == 0:
        raise RuntimeError("No curves available for NP or NN (normalized). Check token metric extraction and qid overlap.")

    np_mat = np.stack(np_list, axis=0).astype(np.float64)
    nn_mat = np.stack(nn_list, axis=0).astype(np.float64)

    x_norm = np.linspace(0.0, 1.0, args.M)
    out_png = outdir / f"{args.token_metric}_np_vs_nn_norm_from_train_base.png"
    title = f"{args.strategy} | {args.token_metric}"
    _plot_two_groups(
        a_mat=np_mat,
        b_mat=nn_mat,
        x=x_norm,
        out_png=out_png,
        title=title,
        xlabel="Normalized response position",
        ylabel=metric_label,
        early_region=(0.0, float(args.early_frac)),
    )

    # matched normalized
    out_png_m = None
    matched_n = None
    if args.make_matched:
        np_qids_m, nn_qids_m, n = downsample_matched(np_qids, nn_qids, seed=args.matched_seed)
        matched_n = n
        if n >= 3:
            np_m = np.stack([q_curve_norm[q] for q in np_qids_m if q in q_curve_norm], axis=0)
            nn_m = np.stack([q_curve_norm[q] for q in nn_qids_m if q in q_curve_norm], axis=0)
            out_png_m = outdir / f"{args.token_metric}_np_vs_nn_norm_from_train_base_MATCHED.png"
            title2 = f"{args.strategy} | {args.token_metric} | NP vs NN MATCHED (n={n}) | seed={args.matched_seed} | normalized"
            _plot_two_groups(
                a_mat=np_m.astype(np.float64),
                b_mat=nn_m.astype(np.float64),
                x=x_norm,
                out_png=out_png_m,
                title=title2,
                xlabel="Normalized response position",
                ylabel=metric_label,
                early_region=(0.0, float(args.early_frac)),
            )

    # ---------- absolute-length plot ----------
    out_png_abs = None
    out_png_abs_m = None
    if args.plot_abs:
        q_curve_abs = prompt_level_mean_curves_absolute(base_samples, qids_keep=keep, abs_T=args.abs_T)

        np_list_abs = [q_curve_abs[q] for q in np_qids if q in q_curve_abs]
        nn_list_abs = [q_curve_abs[q] for q in nn_qids if q in q_curve_abs]
        if len(np_list_abs) == 0 or len(nn_list_abs) == 0:
            raise RuntimeError("No curves available for NP or NN (absolute). Check abs_T and data coverage.")

        np_abs = np.stack(np_list_abs, axis=0).astype(np.float64)
        nn_abs = np.stack(nn_list_abs, axis=0).astype(np.float64)

        x_abs = np.arange(args.abs_T, dtype=np.float64)
        out_png_abs = outdir / f"{args.token_metric}_np_vs_nn_absT{args.abs_T}_from_train_base.png"
        title_abs = f"{args.strategy} | {args.token_metric} | NP vs NN (split base→post) | plot=train_base | absolute (T={args.abs_T})"
        early_abs = (0.0, float(args.early_frac) * float(args.abs_T))
        _plot_two_groups(
            a_mat=np_abs,
            b_mat=nn_abs,
            x=x_abs,
            out_png=out_png_abs,
            title=title_abs,
            xlabel="Absolute token index",
            ylabel=metric_label,
            early_region=early_abs,
        )

        if args.make_matched and matched_n is not None and matched_n >= 3:
            np_qids_m, nn_qids_m, n = downsample_matched(np_qids, nn_qids, seed=args.matched_seed)
            np_abs_m = np.stack([q_curve_abs[q] for q in np_qids_m if q in q_curve_abs], axis=0).astype(np.float64)
            nn_abs_m = np.stack([q_curve_abs[q] for q in nn_qids_m if q in q_curve_abs], axis=0).astype(np.float64)

            out_png_abs_m = outdir / f"{args.token_metric}_np_vs_nn_absT{args.abs_T}_from_train_base_MATCHED.png"
            title_abs_m = f"{args.strategy} | {args.token_metric} | NP vs NN MATCHED (n={n}) | seed={args.matched_seed} | absolute (T={args.abs_T})"
            _plot_two_groups(
                a_mat=np_abs_m,
                b_mat=nn_abs_m,
                x=x_abs,
                out_png=out_png_abs_m,
                title=title_abs_m,
                xlabel="Absolute token index",
                ylabel=metric_label,
                early_region=early_abs,
            )

    stat = {
        "strategy": args.strategy,
        "train_base": args.train_base,
        "train_post": args.train_post,
        "token_metric": args.token_metric,
        "NP_qids": len(np_qids),
        "NN_qids": len(nn_qids),
        "NP_norm_curves_used": int(np_mat.shape[0]),
        "NN_norm_curves_used": int(nn_mat.shape[0]),
        "M": args.M,
        "early_frac": args.early_frac,
        "boot": args.boot,
        "pad_id": args.pad_id,
        "max_len": args.max_len,
        "make_matched": bool(args.make_matched),
        "matched_seed": int(args.matched_seed),
        "matched_n_target": int(min(len(np_qids), len(nn_qids))) if args.make_matched else None,
        "plot_abs": bool(args.plot_abs),
        "abs_T": int(args.abs_T),
        "out_norm": str(out_png),
        "out_norm_matched": str(out_png_m) if out_png_m is not None else None,
        "out_abs": str(out_png_abs) if out_png_abs is not None else None,
        "out_abs_matched": str(out_png_abs_m) if out_png_abs_m is not None else None,
    }
    with (outdir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_png}")
    if out_png_m is not None:
        print(f"[OK] saved: {out_png_m}")
    if out_png_abs is not None:
        print(f"[OK] saved: {out_png_abs}")
    if out_png_abs_m is not None:
        print(f"[OK] saved: {out_png_abs_m}")
    print(f"[OK] stats: {outdir / 'split_stats.json'}")


if __name__ == "__main__":
    main()