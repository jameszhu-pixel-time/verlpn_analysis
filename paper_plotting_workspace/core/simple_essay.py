#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
intra_strategy_np_driver.py (patched)

Patch goals (per your request)
1) Support your launch script:
   - --np_base name=PATH  (repeatable, per-strategy base labels)
   - still accepts legacy single-path --np_base PATH (shared base), for backward compatibility.

2) Plotting:
   - density and ECDF are saved as TWO separate figures (no subplots):
       density_qid_{driver}_{subset}_N2P_vs_N2N.png
       ecdf_qid_{driver}_{subset}_N2P_vs_N2N.png
   - blue/orange theme
   - legend smaller
   - concise title (strategy | driver | subset)

3) Subsets:
   - only draw "all" and "neg"
   - ignore effective/effective_neg even if user passes ALL4

No change to your existing --train / --np_post / --drivers_py / --outdir format.
"""

import argparse
import json
import re
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PAD_ID_DEFAULT = 151643

# Global matplotlib style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,     # smaller legend
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


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


# ---------------- Entropy compute ----------------
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


# ---------------- Core data ----------------
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
        samples.append(Sample(qid=str(qid), rid=str(rid), correct=bool(correct), ent=ent2, valid_len=int(np.sum(valid))))

    return samples


def per_qid_any_correct(samples: List[Sample]) -> Dict[str, bool]:
    anyc: Dict[str, bool] = {}
    for s in samples:
        anyc[s.qid] = anyc.get(s.qid, False) or bool(s.correct)
    return anyc


def split_subsets(samples: List[Sample]) -> Dict[str, List[Sample]]:
    # we still compute all,neg (effective ones often empty under your setup)
    anyc = per_qid_any_correct(samples)
    return {
        "all": list(samples),
        "neg": [s for s in samples if not s.correct],
        # kept for completeness (not used by default)
        "effective": [s for s in samples if anyc.get(s.qid, False)],
        "effective_neg": [s for s in samples if anyc.get(s.qid, False) and (not s.correct)],
    }


# ---------------- Drivers ----------------
DriverFn = Callable[[np.ndarray], float]


def load_drivers_from_py(py_path: Path) -> Dict[str, DriverFn]:
    if not py_path.exists():
        return {}
    try:
        spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
        if spec is None or spec.loader is None:
            return {}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        out: Dict[str, DriverFn] = {}

        if hasattr(mod, "get_drivers"):
            d = mod.get_drivers()
            if isinstance(d, dict):
                for k, fn in d.items():
                    if callable(fn):
                        out[str(k)] = fn

        for name in dir(mod):
            if name.startswith("driver_"):
                fn = getattr(mod, name)
                if callable(fn):
                    out[name[len("driver_"):]] = fn

        return out
    except Exception as e:
        print(f"[WARN] failed to import drivers from {py_path}: {e}")
        return {}


# ---------------- Stats + distances ----------------
def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(_finite(a))
    b = np.sort(_finite(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    na, nb = a.size, b.size
    xs = np.sort(np.unique(np.concatenate([a, b])))
    ia = np.searchsorted(a, xs, side="right")
    ib = np.searchsorted(b, xs, side="right")
    Fa = ia / na
    Fb = ib / nb
    if xs.size < 2:
        return 0.0
    dx = xs[1:] - xs[:-1]
    y = np.abs(Fa - Fb)
    return float(np.sum(y[:-1] * dx))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(_finite(a))
    b = np.sort(_finite(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    na, nb = a.size, b.size
    xs = np.sort(np.unique(np.concatenate([a, b])))
    Fa = np.searchsorted(a, xs, side="right") / na
    Fb = np.searchsorted(b, xs, side="right") / nb
    return float(np.max(np.abs(Fa - Fb)))


def qid_mean_driver(samples: List[Sample], driver_fn: DriverFn) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for s in samples:
        try:
            v = float(driver_fn(s.ent))
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        sums[s.qid] = sums.get(s.qid, 0.0) + v
        cnts[s.qid] = cnts.get(s.qid, 0) + 1
    return {q: sums[q] / cnts[q] for q in cnts if cnts[q] > 0}


# ---------------- Plotting (density + ECDF separated) ----------------
def _freedman_diaconis_bins(x: np.ndarray, max_bins: int = 140, min_bins: int = 25) -> int:
    x = _finite(x)
    if x.size < 2:
        return min_bins
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 1e-12:
        return min_bins
    bw = 2.0 * iqr * (x.size ** (-1.0 / 3.0))
    if bw <= 1e-12:
        return min_bins
    bins = int(np.ceil((np.max(x) - np.min(x)) / bw))
    return int(np.clip(bins, min_bins, max_bins))


def _kde_gaussian(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = _finite(x)
    n = x.size
    if n < 2:
        return np.zeros_like(grid)
    std = np.std(x)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(grid)
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    h = max(h, 1e-6)
    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))
    return dens


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(_finite(x))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _robust_xlim(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    aa = _finite(a)
    bb = _finite(b)
    xs = np.concatenate([aa, bb])
    lo = float(np.quantile(xs, 0.01))
    hi = float(np.quantile(xs, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(xs))
        hi = float(np.max(xs))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def plot_density_only(a: np.ndarray, b: np.ndarray, out_png: Path, title: str, bins: int):
    aa = _finite(a)
    bb = _finite(b)
    if aa.size == 0 or bb.size == 0:
        return

    lo, hi = _robust_xlim(aa, bb)

    xs = np.concatenate([aa, bb])
    try:
        bins_use = max(int(bins), _freedman_diaconis_bins(xs))
        bins_use = int(np.clip(bins_use, 30, 180))
    except Exception:
        bins_use = int(max(30, bins))

    edges = np.linspace(lo, hi, bins_use + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ha, _ = np.histogram(aa, bins=edges, density=True)
    hb, _ = np.histogram(bb, bins=edges, density=True)

    c_blue = "#1f77b4"
    c_orng = "#ff7f0e"

    w1 = wasserstein_1d(aa, bb)
    ks = ks_statistic(aa, bb)

    fig = plt.figure(figsize=(6.6, 4.2), dpi=180)
    ax = fig.add_subplot(111)

    ax.fill_between(centers, ha, step="mid", alpha=0.22, color=c_orng, label="N2P")
    ax.plot(centers, ha, linewidth=1.5, color=c_orng)

    ax.fill_between(centers, hb, step="mid", alpha=0.18, color=c_blue, label="N2N")
    ax.plot(centers, hb, linewidth=1.5, color=c_blue)

    # KDE (optional)
    grid = np.linspace(lo, hi, 400)
    da = _kde_gaussian(aa, grid)
    db = _kde_gaussian(bb, grid)
    if np.max(da) > 0:
        ax.plot(grid, da, linestyle="--", linewidth=1.1, color=c_orng, alpha=0.9)
    if np.max(db) > 0:
        ax.plot(grid, db, linestyle="--", linewidth=1.1, color=c_blue, alpha=0.9)

    # medians
    ax.axvline(np.median(aa), color=c_orng, linewidth=1.1, linestyle=":")
    ax.axvline(np.median(bb), color=c_blue, linewidth=1.1, linestyle=":")

    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_title(title, pad=8)
    ax.set_xlabel("qid_mean_driver")
    ax.set_ylabel("density")
    ax.set_xlim(lo, hi)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # small legend
    ax.legend(frameon=False, loc="upper right", handlelength=1.2, handletextpad=0.4, borderpad=0.2)

    # compact metrics
    ann = f"n={aa.size}/{bb.size}  W1={w1:.3g}  KS={ks:.3g}"
    ax.text(
        0.01, 0.98, ann,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.85")
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf_only(a: np.ndarray, b: np.ndarray, out_png: Path, title: str):
    aa = _finite(a)
    bb = _finite(b)
    if aa.size == 0 or bb.size == 0:
        return

    lo, hi = _robust_xlim(aa, bb)

    c_blue = "#1f77b4"
    c_orng = "#ff7f0e"

    w1 = wasserstein_1d(aa, bb)
    ks = ks_statistic(aa, bb)

    xa, ya = _ecdf(aa)
    xb, yb = _ecdf(bb)

    fig = plt.figure(figsize=(6.6, 4.2), dpi=180)
    ax = fig.add_subplot(111)

    ax.plot(xa, ya, color=c_orng, linewidth=1.6, label="N2P")
    ax.plot(xb, yb, color=c_blue, linewidth=1.6, label="N2N")
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_title(title, pad=8)
    ax.set_xlabel("qid_mean_driver")
    ax.set_ylabel("ECDF")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="lower right", handlelength=1.2, handletextpad=0.4, borderpad=0.2)

    ann = f"n={aa.size}/{bb.size}  W1={w1:.3g}  KS={ks:.3g}"
    ax.text(
        0.01, 0.05, ann,
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.85")
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------- CLI helpers ----------------
def parse_name_path_list(items: List[str], flag_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"{flag_name} expects name=path, got: {spec}")
        name, p = spec.split("=", 1)
        out[name.strip()] = Path(p.strip())
    return out


def parse_np_base_arg(np_base_items: List[str]) -> Tuple[Optional[Path], Dict[str, Path]]:
    """
    Support BOTH:
      legacy: --np_base /path/to/base.jsonl         (single shared base)
      new:    --np_base name=/path/to/base.jsonl    (repeatable per strategy)

    Returns:
      (shared_base_path_or_None, per_strategy_map)
    """
    if not np_base_items:
        raise ValueError("No --np_base provided.")

    per_map: Dict[str, Path] = {}
    shared: Optional[Path] = None

    for item in np_base_items:
        if "=" in item:
            name, p = item.split("=", 1)
            per_map[name.strip()] = Path(p.strip())
        else:
            # legacy single path
            if shared is not None:
                raise ValueError("Legacy --np_base PATH provided multiple times; use name=path style.")
            shared = Path(item.strip())

    return shared, per_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, action="append", default=[],
                    help="repeatable: name=path for TRAIN rollouts (driver source)")

    # changed: make np_base repeatable, supporting name=path (per-strategy) or single path (legacy)
    ap.add_argument("--np_base", type=str, action="append", required=True,
                    help="NP-label base rollouts. Either single PATH, or repeatable name=PATH per strategy.")

    ap.add_argument("--np_post", type=str, action="append", default=[],
                    help="repeatable: name=path for NP-label post rollouts (step3 inference results per strategy)")
    ap.add_argument("--drivers_py", type=str, default="", help="external drivers.py")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--subset", type=str, default="ALL4",
                    help="ignored now; we only plot all & neg to match your request")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # drivers
    drivers: Dict[str, DriverFn] = {}
    if args.drivers_py:
        drivers.update(load_drivers_from_py(Path(args.drivers_py)))
    script_dir = Path(__file__).resolve().parent
    drivers.update(load_drivers_from_py(script_dir / "drivers.py"))
    if not drivers:
        raise RuntimeError("No drivers loaded. Provide --drivers_py or put drivers.py next to this script.")

    train_map = parse_name_path_list(args.train, "--train")
    if not train_map:
        raise ValueError("No --train provided.")

    np_post_map = parse_name_path_list(args.np_post, "--np_post")
    if not np_post_map:
        raise ValueError("No --np_post provided.")

    # subset: forced to only ["all","neg"]
    subset_list = ["all", "neg"]

    # parse np_base (shared or per-strategy)
    shared_np_base, np_base_map = parse_np_base_arg(args.np_base)

    # load post anyc once
    post_anyc: Dict[str, Dict[str, bool]] = {}
    for name, p in np_post_map.items():
        ss = load_samples(p, pad_id=args.pad_id, max_len=args.max_len)
        post_anyc[name] = per_qid_any_correct(ss)

    # if shared base, load once
    shared_base_anyc: Optional[Dict[str, bool]] = None
    if shared_np_base is not None:
        base_samples = load_samples(shared_np_base, pad_id=args.pad_id, max_len=args.max_len)
        shared_base_anyc = per_qid_any_correct(base_samples)

    summary_path = outdir / "intra_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as sf:
        for sname, train_path in train_map.items():
            if sname not in post_anyc:
                print(f"[WARN] strategy '{sname}' has train but no np_post; skip.")
                continue

            # choose base_anyc per strategy if provided; else fallback to shared base
            if sname in np_base_map:
                base_samples = load_samples(np_base_map[sname], pad_id=args.pad_id, max_len=args.max_len)
                base_anyc = per_qid_any_correct(base_samples)
                np_base_used = str(np_base_map[sname])
            else:
                if shared_base_anyc is None:
                    print(f"[WARN] strategy '{sname}' has no per-strategy np_base and no shared np_base; skip.")
                    continue
                base_anyc = shared_base_anyc
                np_base_used = str(shared_np_base)

            strat_dir = outdir / sname
            strat_dir.mkdir(parents=True, exist_ok=True)

            # load train rollouts for driver source
            train_samples = load_samples(Path(train_path), pad_id=args.pad_id, max_len=args.max_len)
            train_subsets = split_subsets(train_samples)

            for sb in subset_list:
                sb_samples = train_subsets.get(sb, [])
                if not sb_samples:
                    continue

                for dname, fn in drivers.items():
                    qmean = qid_mean_driver(sb_samples, fn)
                    if not qmean:
                        continue

                    overlap = set(base_anyc.keys()) & set(post_anyc[sname].keys()) & set(qmean.keys())

                    n2p_vals = []
                    n2n_vals = []
                    for q in overlap:
                        b0 = bool(base_anyc[q])
                        p0 = bool(post_anyc[sname][q])
                        if b0:
                            continue  # only base=N
                        v = qmean.get(q, None)
                        if v is None or (not np.isfinite(v)):
                            continue
                        if p0:
                            n2p_vals.append(v)
                        else:
                            n2n_vals.append(v)

                    a = np.asarray(n2p_vals, dtype=np.float64)
                    b = np.asarray(n2n_vals, dtype=np.float64)

                    def _stat(xarr):
                        xarr = _finite(xarr)
                        if xarr.size == 0:
                            return {"n": 0, "mean": None, "median": None}
                        return {"n": int(xarr.size), "mean": float(np.mean(xarr)), "median": float(np.median(xarr))}

                    w1v = wasserstein_1d(a, b)
                    ksv = ks_statistic(a, b)

                    rec = {
                        "strategy": sname,
                        "subset": sb,
                        "driver": dname,
                        "qid_overlap_used": int(len(overlap)),
                        "N2P_driver": _stat(a),
                        "N2N_driver": _stat(b),
                        "dist_N2P_vs_N2N": {
                            "w1": None if not np.isfinite(w1v) else float(w1v),
                            "ks": None if not np.isfinite(ksv) else float(ksv),
                        },
                        "np_base": np_base_used,
                        "np_post": str(np_post_map[sname]),
                        "train_file": str(train_path),
                    }
                    sf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    # concise title
                    title = f"{sname} | {dname} | {sb}"

                    # density + ecdf separated
                    out_den = strat_dir / f"density_qid_{dname}_{sb}_N2P_vs_N2N.png"
                    out_ecdf = strat_dir / f"ecdf_qid_{dname}_{sb}_N2P_vs_N2N.png"

                    plot_density_only(a=a, b=b, out_png=out_den, title=title, bins=int(args.bins))
                    plot_ecdf_only(a=a, b=b, out_png=out_ecdf, title=title)

    print(f"[OK] outdir = {outdir}")
    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] drivers = {sorted(list(drivers.keys()))}")
    if shared_np_base is not None:
        print(f"[OK] shared np_base = {shared_np_base}")
    if np_base_map:
        print(f"[OK] per-strategy np_base keys = {sorted(list(np_base_map.keys()))}")


if __name__ == "__main__":
    main()