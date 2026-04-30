#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
effective_rollout_across_methods.py

Goal
----
For each strategy:
  1) Use its own base rollout file to define current solved/unsolved status
  2) Use its own post rollout file to define future transition
  3) Mark effective qids:
         effective_qid := (base unsolved) and (post solved)
  4) On the BASE rollouts of this strategy, split samples into:
         effective vs ineffective
     and plot entropy-position curves
  5) Horizontally compare effective curves across strategies

This version includes NPZ fallback logic for base verl_rollouts jsonl files.

Typical use
-----------
python effective_rollout_across_methods.py \
  --base ref=/path/to/ref_base.jsonl \
  --base n2=/path/to/n2_base.jsonl \
  --base n3=/path/to/n3_base.jsonl \
  --base annealed=/path/to/annealed_base.jsonl \
  --post ref=/path/to/ref_post.jsonl \
  --post n2=/path/to/n2_post.jsonl \
  --post n3=/path/to/n3_post.jsonl \
  --post annealed=/path/to/annealed_post.jsonl \
  --outdir /path/to/out \
  --subset neg \
  --abs_T 256 \
  --rel_bins 128 \
  --boot 1000 \
  --max_len 3072 \
  --pad_id 151643 \
  --prototype n2 \
  --dist_metric cosine
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


PAD_ID_DEFAULT = 151643

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# =========================================================
# IO
# =========================================================
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


# =========================================================
# Entropy extraction
# =========================================================
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


def _safe_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return None if arr.size == 0 else arr
    except Exception:
        return None


# ---------------- NPZ fallback ----------------
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


def extract_entropy(
    rec: Dict[str, Any],
    qid: Optional[str],
    rid: Optional[str],
    npz_index: Optional[Dict[Tuple[str, str], Path]],
) -> Optional[np.ndarray]:
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


# =========================================================
# Data structure
# =========================================================
@dataclass
class Sample:
    qid: str
    rid: str
    correct: bool
    ent: np.ndarray
    valid_len: int


def load_samples(path: Path, pad_id: int, max_len: int) -> List[Sample]:
    samples: List[Sample] = []
    npz_index = build_npz_index(path)

    print(f"[DEBUG] load_samples: {path} | npz_index={len(npz_index)}")

    for rec in iter_jsonl(path):
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
        samples.append(Sample(
            qid=str(qid),
            rid=str(rid),
            correct=bool(correct),
            ent=ent2,
            valid_len=int(np.sum(valid)),
        ))

    print(f"[DEBUG] loaded samples from {path}: {len(samples)}")
    return samples


def per_qid_any_correct(samples: List[Sample]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for s in samples:
        out[s.qid] = out.get(s.qid, False) or bool(s.correct)
    return out


# =========================================================
# Effective qids for each strategy
# =========================================================
def build_effective_qids(base_anyc: Dict[str, bool], post_anyc: Dict[str, bool]) -> Dict[str, bool]:
    overlap = set(base_anyc.keys()) & set(post_anyc.keys())
    eff = {}
    for q in overlap:
        eff[q] = (not bool(base_anyc[q])) and bool(post_anyc[q])
    return eff


# =========================================================
# Curve matrix
# =========================================================
def _finite_prefix(x: np.ndarray, T: int) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    y = y[:min(T, y.size)]
    out = np.full((T,), np.nan, dtype=np.float64)
    out[:y.size] = y
    return out


def sample_to_relative_curve(ent: np.ndarray, bins: int) -> np.ndarray:
    v = np.asarray(ent, dtype=np.float64)
    mask = np.isfinite(v)
    vv = v[mask]
    if vv.size < 2:
        return np.full((bins,), np.nan, dtype=np.float64)
    src_x = np.linspace(0.0, 1.0, vv.size)
    dst_x = np.linspace(0.0, 1.0, bins)
    out = np.interp(dst_x, src_x, vv)
    return out.astype(np.float64)


def make_abs_mat(samples: List[Sample], T: int) -> np.ndarray:
    mats = []
    for s in samples:
        mats.append(_finite_prefix(s.ent, T))
    if not mats:
        return np.zeros((0, T), dtype=np.float64)
    return np.stack(mats, axis=0)


def make_rel_mat(samples: List[Sample], bins: int) -> np.ndarray:
    mats = []
    for s in samples:
        mats.append(sample_to_relative_curve(s.ent, bins))
    if not mats:
        return np.zeros((0, bins), dtype=np.float64)
    return np.stack(mats, axis=0)


def _finite_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    ok = np.any(np.isfinite(mat), axis=1)
    return mat[ok]


# =========================================================
# Stats / CI
# =========================================================
def bootstrap_ci(mat: np.ndarray, boot: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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


def _stat_1d(x: np.ndarray) -> Dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
    }


# =========================================================
# Plotting
# =========================================================
def plot_two_groups(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    x: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    boot: int = 1000,
    a_label: str = "effective",
    b_label: str = "ineffective",
):
    a_mat = _finite_rows(a_mat)
    b_mat = _finite_rows(b_mat)
    if a_mat.shape[0] == 0 or b_mat.shape[0] == 0:
        return

    a_mean = np.nanmean(a_mat, axis=0)
    b_mean = np.nanmean(b_mat, axis=0)
    a_lo, a_hi = bootstrap_ci(a_mat, boot=boot, seed=1)
    b_lo, b_hi = bootstrap_ci(b_mat, boot=boot, seed=2)

    plt.figure(figsize=(7.0, 4.3), dpi=180)
    plt.plot(x, a_mean, label=f"{a_label} (n={a_mat.shape[0]})", color="#ff7f0e")
    plt.fill_between(x, a_lo, a_hi, alpha=0.20, color="#ff7f0e")

    plt.plot(x, b_mean, label=f"{b_label} (n={b_mat.shape[0]})", color="#1f77b4")
    plt.fill_between(x, b_lo, b_hi, alpha=0.20, color="#1f77b4")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_cross_strategy(
    mats: Dict[str, np.ndarray],
    x: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
):
    if not mats:
        return

    plt.figure(figsize=(7.2, 4.4), dpi=180)
    plotted = 0
    for name, mat in mats.items():
        mat = _finite_rows(mat)
        if mat.shape[0] == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        plt.plot(x, mean, label=f"{name} (n={mat.shape[0]})")
        plotted += 1

    if plotted == 0:
        plt.close()
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_combined_cross_strategy(
    eff_mats: Dict[str, np.ndarray],
    ineff_mats: Dict[str, np.ndarray],
    x: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt.figure(figsize=(8.2, 5.0), dpi=180)
    plotted = 0

    for name, mat in eff_mats.items():
        mat = _finite_rows(mat)
        if mat.shape[0] == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        plt.plot(
            x, mean,
            linewidth=2.0,
            linestyle="-",
            label=f"{name} effective (n={mat.shape[0]})"
        )
        plotted += 1

    for name, mat in ineff_mats.items():
        mat = _finite_rows(mat)
        if mat.shape[0] == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        plt.plot(
            x, mean,
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
            label=f"{name} ineffective (n={mat.shape[0]})"
        )
        plotted += 1

    if plotted == 0:
        plt.close()
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# =========================================================
# Prototype distance
# =========================================================
def rowwise_cosine_distance(mat: np.ndarray, proto: np.ndarray) -> np.ndarray:
    out = np.full((mat.shape[0],), np.nan, dtype=np.float64)
    for i in range(mat.shape[0]):
        a = mat[i]
        mask = np.isfinite(a) & np.isfinite(proto)
        if np.sum(mask) < 3:
            continue
        aa = a[mask]
        bb = proto[mask]
        na = np.linalg.norm(aa)
        nb = np.linalg.norm(bb)
        if na <= 1e-12 or nb <= 1e-12:
            continue
        cos = np.dot(aa, bb) / (na * nb)
        out[i] = 1.0 - cos
    return out


def rowwise_l2_distance(mat: np.ndarray, proto: np.ndarray) -> np.ndarray:
    out = np.full((mat.shape[0],), np.nan, dtype=np.float64)
    for i in range(mat.shape[0]):
        a = mat[i]
        mask = np.isfinite(a) & np.isfinite(proto)
        if np.sum(mask) < 3:
            continue
        aa = a[mask]
        bb = proto[mask]
        out[i] = float(np.sqrt(np.mean((aa - bb) ** 2)))
    return out


# =========================================================
# CLI helpers
# =========================================================
def parse_name_path_list(items: List[str], flag_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"{flag_name} expects name=path, got: {spec}")
        name, p = spec.split("=", 1)
        name = name.strip()
        p = p.strip()
        if not name:
            raise ValueError(f"{flag_name}: empty name in spec {spec}")
        out[name] = Path(p)
    return out


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", type=str, action="append", required=True,
                    help="repeatable: strategy=base_rollout_path")
    ap.add_argument("--post", type=str, action="append", required=True,
                    help="repeatable: strategy=post_rollout_path")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--subset", type=str, default="neg", choices=["neg", "all"])
    ap.add_argument("--abs_T", type=int, default=256)
    ap.add_argument("--rel_bins", type=int, default=128)
    ap.add_argument("--boot", type=int, default=1000)

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)

    ap.add_argument("--prototype", type=str, default="",
                    help="strategy name whose effective relative curve is used as prototype; default=first strategy")
    ap.add_argument("--dist_metric", type=str, default="cosine", choices=["cosine", "l2"])

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_map = parse_name_path_list(args.base, "--base")
    post_map = parse_name_path_list(args.post, "--post")

    strategies = [s for s in base_map.keys() if s in post_map]
    if not strategies:
        raise ValueError("No overlapping strategy names between --base and --post.")

    missing_post = sorted(set(base_map.keys()) - set(post_map.keys()))
    missing_base = sorted(set(post_map.keys()) - set(base_map.keys()))
    if missing_post:
        print(f"[WARN] strategies with base but no post: {missing_post}")
    if missing_base:
        print(f"[WARN] strategies with post but no base: {missing_base}")

    abs_x = np.arange(args.abs_T)
    rel_x = np.linspace(0.0, 1.0, args.rel_bins)

    eff_abs_by_strategy: Dict[str, np.ndarray] = {}
    eff_rel_by_strategy: Dict[str, np.ndarray] = {}
    ineff_abs_by_strategy: Dict[str, np.ndarray] = {}
    ineff_rel_by_strategy: Dict[str, np.ndarray] = {}

    summary_path = outdir / "effective_across_methods_summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    print("[INFO] loading and processing strategies...")
    for sname in strategies:
        print(f"[INFO] strategy={sname}")

        base_samples = load_samples(base_map[sname], pad_id=args.pad_id, max_len=args.max_len)
        post_samples = load_samples(post_map[sname], pad_id=args.pad_id, max_len=args.max_len)

        base_anyc = per_qid_any_correct(base_samples)
        post_anyc = per_qid_any_correct(post_samples)
        eff_qids = build_effective_qids(base_anyc, post_anyc)

        overlap_qids = set(eff_qids.keys())

        if args.subset == "neg":
            base_use = [s for s in base_samples if (not s.correct) and (s.qid in overlap_qids)]
        else:
            base_use = [s for s in base_samples if s.qid in overlap_qids]

        eff_samples = [s for s in base_use if bool(eff_qids.get(s.qid, False))]
        ineff_samples = [s for s in base_use if not bool(eff_qids.get(s.qid, False))]

        eff_abs = make_abs_mat(eff_samples, T=args.abs_T)
        ineff_abs = make_abs_mat(ineff_samples, T=args.abs_T)
        eff_rel = make_rel_mat(eff_samples, bins=args.rel_bins)
        ineff_rel = make_rel_mat(ineff_samples, bins=args.rel_bins)

        eff_abs_by_strategy[sname] = eff_abs
        eff_rel_by_strategy[sname] = eff_rel
        ineff_abs_by_strategy[sname] = ineff_abs
        ineff_rel_by_strategy[sname] = ineff_rel

        print(
            f"[DEBUG] {sname}: "
            f"base_qids={len(base_anyc)}, post_qids={len(post_anyc)}, "
            f"overlap={len(overlap_qids)}, "
            f"effective_qids={sum(1 for _, v in eff_qids.items() if v)}, "
            f"effective_rollouts={eff_abs.shape[0]}, "
            f"ineffective_rollouts={ineff_abs.shape[0]}"
        )

        strat_dir = outdir / sname
        strat_dir.mkdir(parents=True, exist_ok=True)

        title_prefix = f"{sname} | subset={args.subset}"

        plot_two_groups(
            a_mat=eff_abs,
            b_mat=ineff_abs,
            x=abs_x,
            out_png=strat_dir / f"abs_{sname}_{args.subset}_effective_vs_ineffective.png",
            title=f"{title_prefix} | absolute",
            xlabel="token position",
            ylabel="token entropy",
            boot=args.boot,
            a_label="effective",
            b_label="ineffective",
        )

        plot_two_groups(
            a_mat=eff_rel,
            b_mat=ineff_rel,
            x=rel_x,
            out_png=strat_dir / f"rel_{sname}_{args.subset}_effective_vs_ineffective.png",
            title=f"{title_prefix} | relative",
            xlabel="relative position",
            ylabel="token entropy",
            boot=args.boot,
            a_label="effective",
            b_label="ineffective",
        )

        rec = {
            "strategy": sname,
            "subset": args.subset,
            "base_file": str(base_map[sname]),
            "post_file": str(post_map[sname]),
            "base_qids": int(len(base_anyc)),
            "post_qids": int(len(post_anyc)),
            "qid_overlap": int(len(overlap_qids)),
            "effective_qids": int(sum(1 for _, v in eff_qids.items() if v)),
            "ineffective_qids": int(sum(1 for _, v in eff_qids.items() if not v)),
            "effective_rollouts": int(eff_abs.shape[0]),
            "ineffective_rollouts": int(ineff_abs.shape[0]),
        }
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[INFO] plotting cross-strategy effective curves...")
    plot_cross_strategy(
        mats=eff_abs_by_strategy,
        x=abs_x,
        out_png=outdir / f"cross_strategy_effective_abs_{args.subset}.png",
        title=f"effective rollout comparison across strategies | absolute | subset={args.subset}",
        xlabel="token position",
        ylabel="token entropy",
    )

    plot_cross_strategy(
        mats=eff_rel_by_strategy,
        x=rel_x,
        out_png=outdir / f"cross_strategy_effective_rel_{args.subset}.png",
        title=f"effective rollout comparison across strategies | relative | subset={args.subset}",
        xlabel="relative position",
        ylabel="token entropy",
    )

    plot_cross_strategy(
        mats=ineff_abs_by_strategy,
        x=abs_x,
        out_png=outdir / f"cross_strategy_ineffective_abs_{args.subset}.png",
        title=f"ineffective rollout comparison across strategies | absolute | subset={args.subset}",
        xlabel="token position",
        ylabel="token entropy",
    )

    plot_cross_strategy(
        mats=ineff_rel_by_strategy,
        x=rel_x,
        out_png=outdir / f"cross_strategy_ineffective_rel_{args.subset}.png",
        title=f"ineffective rollout comparison across strategies | relative | subset={args.subset}",
        xlabel="relative position",
        ylabel="token entropy",
    )

    plot_combined_cross_strategy(
        eff_mats=eff_abs_by_strategy,
        ineff_mats=ineff_abs_by_strategy,
        x=abs_x,
        out_png=outdir / f"cross_strategy_combined_abs_{args.subset}.png",
        title=f"all curves across strategies | absolute | subset={args.subset}",
        xlabel="token position",
        ylabel="token entropy",
    )

    plot_combined_cross_strategy(
        eff_mats=eff_rel_by_strategy,
        ineff_mats=ineff_rel_by_strategy,
        x=rel_x,
        out_png=outdir / f"cross_strategy_combined_rel_{args.subset}.png",
        title=f"all curves across strategies | relative | subset={args.subset}",
        xlabel="relative position",
        ylabel="token entropy",
    )

    proto_name = args.prototype if args.prototype else strategies[0]
    transfer_path = outdir / "prototype_transfer_across_methods.jsonl"
    if transfer_path.exists():
        transfer_path.unlink()

    if proto_name not in eff_rel_by_strategy or _finite_rows(eff_rel_by_strategy[proto_name]).shape[0] == 0:
        print(f"[WARN] prototype strategy '{proto_name}' has no effective relative curves; skip transfer.")
    else:
        print(f"[INFO] prototype transfer using strategy={proto_name}")
        proto = np.nanmean(_finite_rows(eff_rel_by_strategy[proto_name]), axis=0)

        for sname in strategies:
            eff_rel = _finite_rows(eff_rel_by_strategy[sname])
            ineff_rel = _finite_rows(ineff_rel_by_strategy[sname])

            if args.dist_metric == "cosine":
                eff_dist = rowwise_cosine_distance(eff_rel, proto)
                ineff_dist = rowwise_cosine_distance(ineff_rel, proto)
            else:
                eff_dist = rowwise_l2_distance(eff_rel, proto)
                ineff_dist = rowwise_l2_distance(ineff_rel, proto)

            rec = {
                "strategy": sname,
                "prototype_strategy": proto_name,
                "subset": args.subset,
                "dist_metric": args.dist_metric,
                "effective_dist": _stat_1d(eff_dist),
                "ineffective_dist": _stat_1d(ineff_dist),
            }
            with transfer_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] outdir = {outdir}")
    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {transfer_path}")
    print(f"[OK] strategies = {strategies}")


if __name__ == "__main__":
    main()