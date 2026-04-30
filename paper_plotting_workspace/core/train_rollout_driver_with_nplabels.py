#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_rollout_driver_with_nplabels.py

Goal (your intended logic):
- Use TRAIN rollouts (usually generated from step2 policy, used for training updates) to compute driver distributions.
- Use NP-label rollouts (step2 inference -> step3 inference per strategy) to define N2P/N2N/P2N/P2P at qid level.
- Bridge: for each strategy+driver+subset, compare Δdriver_q distribution between N2P vs N2N (where N/P comes from NP-label files).

Inputs
A) TRAIN rollouts (for driver):
  --train name=PATH   (repeatable)
  --train_ref NAME    (one of the --train names)

B) NP labels (for N2P definition):
  --np_base PATH                  (step2 inference results on same dataset)
  --np_post name=PATH (repeatable) (step3 inference results per strategy)

Outputs
1) outdir/entropy_position/{strategy}/abs_mean_4subsets.png + rel_mean_4subsets.png + stats.json
2) outdir/drivers/strategy_summary.jsonl
   - per strategy × driver × subset: deviation vs train_ref + NP N2P_rate (from np_base->np_post)
   - driver_overlay_{driver}_{subset}_{strategy}_vs_{train_ref}.png
   - scatter_{driver}_{subset}.png (x=selected x_metric, y=N2P_rate)
3) (if --bridge) outdir/bridge_qid/
   - delta_qid_hist_{driver}_{subset}_{strategy}.png   (Δdriver_q overlay: N2P vs N2N)
   - bridge_summary.jsonl (stats + KS/W1 comparing N2P vs N2N)

Notes / policies
- Entropy prefer full_logprobs->entropy whenever available (NPZ or JSONL).
- Must mask: response_mask -> valid_len/response_len -> PAD token -> finite(entropy)
- Truncate to --max_len (default 3072)
- Driver functions loaded from --drivers_py and/or colocated drivers.py:
    A) get_drivers()->dict[name]=fn(ent)->float
    B) driver_xxx(ent)->float

x_metric options (for scatter x-axis):
- d_mean      : mean(strategy_driver) - mean(ref_driver)
- d_median    : median(strategy_driver) - median(ref_driver)
- d_skew      : skew(strategy_driver) - skew(ref_driver)
- d_tail_mass : tail_mass(strategy > ref_q90) - ref_tail_mass (ref_q90 is 90% quantile of ref)
- js          : Jensen-Shannon divergence between histograms
- w1          : Wasserstein-1 (earth mover) distance
- ks          : Kolmogorov–Smirnov statistic

"""

import argparse
import json
import re
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

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
    # 1) response_mask
    m = rec.get("response_mask", None)
    if m is not None:
        try:
            m = np.asarray(m).astype(bool).reshape(-1)
            if m.size == T:
                return m
        except Exception:
            pass

    # 2) valid_len
    L = extract_valid_len_hint(rec)
    if L is not None:
        L = max(0, min(T, int(L)))
        m = np.zeros((T,), dtype=bool)
        m[:L] = True
        return m

    # 3) PAD
    token_ids = extract_token_ids(rec)
    if token_ids is not None:
        m = build_response_mask_from_tokens(token_ids, pad_id)
        if m is not None and m.size == T:
            return m

    return None


# ---------------- Entropy compute (prefer full_logprobs) ----------------
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
    # 1) JSONL full_logprobs (T,K) -> entropy
    flp = rec.get("full_logprobs", None)
    if flp is not None:
        try:
            lp = np.asarray(flp, dtype=np.float64)
            if lp.ndim == 2 and lp.shape[0] > 0:
                return entropy_from_topk_logprobs_vec(lp)
        except Exception:
            pass

    # 2) direct arrays
    ent = _safe_float_array(_get_first(rec, ["token_entropies_topk", "entropy", "entropies"]))
    if ent is not None:
        return ent

    # 3) vLLM topk list (if happens)
    ent2 = entropy_from_topk_logprobs_json(rec.get("topk_logprobs_per_token"))
    if ent2 is not None:
        return ent2

    # 4) explicit npz path
    npz_path = _get_first(rec, ["full_logprobs_path", "npz_path", "npz_file", "npz"])
    if isinstance(npz_path, str) and npz_path:
        p = Path(npz_path)
        if p.exists():
            e = load_entropy_from_npz(p)
            if e is not None:
                return e

    # 5) index by (qid,rid)
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
    ent: np.ndarray     # (T,) NaN-masked
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
    anyc = per_qid_any_correct(samples)
    return {
        "all": list(samples),
        "effective": [s for s in samples if anyc.get(s.qid, False)],
        "neg": [s for s in samples if not s.correct],
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


# ---------------- Entropy-position plots ----------------
def abs_mean_curve(samples: List[Sample], max_len: int) -> np.ndarray:
    ssum = np.zeros((max_len,), dtype=np.float64)
    cnt = np.zeros((max_len,), dtype=np.float64)
    for s in samples:
        e = s.ent
        T = min(e.size, max_len)
        e = e[:T]
        m = np.isfinite(e)
        if np.any(m):
            ssum[:T][m] += e[m]
            cnt[:T][m] += 1.0
    mean = ssum / np.maximum(cnt, 1.0)
    mean[cnt == 0] = np.nan
    return mean


def rel_mean_curve(samples: List[Sample], rel_bins: int) -> np.ndarray:
    ssum = np.zeros((rel_bins,), dtype=np.float64)
    cnt = np.zeros((rel_bins,), dtype=np.float64)
    for s in samples:
        ent = s.ent
        valid = np.isfinite(ent)
        if not np.any(valid):
            continue
        idx = np.where(valid)[0]
        denom = max(1, int(idx.max()) + 1)
        rel = idx / denom
        b = np.minimum((rel * rel_bins).astype(int), rel_bins - 1)
        for bi, ti in zip(b, idx):
            ssum[bi] += float(ent[ti])
            cnt[bi] += 1.0
    mean = ssum / np.maximum(cnt, 1.0)
    mean[cnt == 0] = np.nan
    return mean


def write_entropy_position_merged(strategy: str,
                                  subsets: Dict[str, List[Sample]],
                                  out_root: Path,
                                  max_len: int,
                                  rel_bins: int):
    od = out_root / "entropy_position" / strategy
    od.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {"strategy": strategy, "subsets": {}}
    for k, ss in subsets.items():
        vlen = [s.valid_len for s in ss]
        stats["subsets"][k] = {
            "n_samples": int(len(ss)),
            "avg_valid_len": float(np.mean(vlen)) if vlen else None,
            "median_valid_len": float(np.median(vlen)) if vlen else None,
        }
    (od / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # abs mean
    plt.figure()
    xs = np.arange(max_len)
    for k in ["all", "effective", "neg", "effective_neg"]:
        y = abs_mean_curve(subsets.get(k, []), max_len=max_len)
        plt.plot(xs, y, label=k)
    plt.xlabel("absolute token position")
    plt.ylabel("mean entropy (masked)")
    plt.title(f"{strategy} | abs mean entropy | 4 subsets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(od / "abs_mean_4subsets.png", dpi=180)
    plt.close()

    # rel mean
    plt.figure()
    xr = np.linspace(0, 1, rel_bins, endpoint=False)
    for k in ["all", "effective", "neg", "effective_neg"]:
        y = rel_mean_curve(subsets.get(k, []), rel_bins=rel_bins)
        plt.plot(xr, y, label=k)
    plt.xlabel("relative position (0..1)")
    plt.ylabel("mean entropy (masked)")
    plt.title(f"{strategy} | rel mean entropy | 4 subsets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(od / "rel_mean_4subsets.png", dpi=180)
    plt.close()


# ---------------- Driver metrics + deviations ----------------
def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def skewness(x: np.ndarray) -> float:
    x = _finite(x)
    if x.size < 3:
        return float("nan")
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-12:
        return float("nan")
    z3 = float(np.mean(((x - m) / s) ** 3))
    return z3


def q90(x: np.ndarray) -> float:
    x = _finite(x)
    return float(np.quantile(x, 0.90)) if x.size else float("nan")


def tail_mass_over_tau(x: np.ndarray, tau: float) -> float:
    x = _finite(x)
    if x.size == 0 or (not np.isfinite(tau)):
        return float("nan")
    return float(np.mean(x > tau))


def js_divergence_from_hist(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


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


def dist_metrics(a: np.ndarray, b: np.ndarray, bins: int = 80) -> Dict[str, float]:
    af = _finite(a)
    bf = _finite(b)
    if af.size == 0 or bf.size == 0:
        return {"js": float("nan"), "w1": float("nan"), "ks": float("nan")}
    xs = np.concatenate([af, bf])
    lo = float(np.quantile(xs, 0.01))
    hi = float(np.quantile(xs, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(xs))
        hi = float(np.max(xs))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(af, bins=edges, density=False)
    pb, _ = np.histogram(bf, bins=edges, density=False)
    return {"js": js_divergence_from_hist(pa, pb), "w1": wasserstein_1d(af, bf), "ks": ks_statistic(af, bf)}


def compute_driver_array(samples: List[Sample], fn: DriverFn) -> np.ndarray:
    vals: List[float] = []
    for s in samples:
        try:
            vals.append(float(fn(s.ent)))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def deviation_against_ref(strategy_vals: np.ndarray, ref_vals: np.ndarray) -> Dict[str, float]:
    s = _finite(strategy_vals)
    r = _finite(ref_vals)
    out: Dict[str, float] = {}

    out["mean"] = float(np.mean(s)) if s.size else float("nan")
    out["median"] = float(np.median(s)) if s.size else float("nan")
    out["skew"] = skewness(s)

    out["ref_mean"] = float(np.mean(r)) if r.size else float("nan")
    out["ref_median"] = float(np.median(r)) if r.size else float("nan")
    out["ref_skew"] = skewness(r)

    out["d_mean"] = out["mean"] - out["ref_mean"] if (np.isfinite(out["mean"]) and np.isfinite(out["ref_mean"])) else float("nan")
    out["d_median"] = out["median"] - out["ref_median"] if (np.isfinite(out["median"]) and np.isfinite(out["ref_median"])) else float("nan")
    out["d_skew"] = out["skew"] - out["ref_skew"] if (np.isfinite(out["skew"]) and np.isfinite(out["ref_skew"])) else float("nan")

    tau = q90(r)
    out["ref_q90"] = tau
    out["tail_mass"] = tail_mass_over_tau(s, tau)
    out["ref_tail_mass"] = tail_mass_over_tau(r, tau)
    out["d_tail_mass"] = out["tail_mass"] - out["ref_tail_mass"] if (np.isfinite(out["tail_mass"]) and np.isfinite(out["ref_tail_mass"])) else float("nan")

    out.update(dist_metrics(s, r, bins=80))
    return out


def pick_x_metric(dev: Dict[str, float], metric: str) -> float:
    return float(dev.get(metric, float("nan")))


# ---------------- NP events (qid label) ----------------
def np_event_counts(base_anyc: Dict[str, bool], post_anyc: Dict[str, bool]) -> Tuple[Optional[float], Dict[str, int]]:
    qids = set(base_anyc.keys()) & set(post_anyc.keys())
    n2p = n2n = p2n = p2p = 0
    for q in qids:
        b = bool(base_anyc[q])
        p = bool(post_anyc[q])
        if (not b) and p:
            n2p += 1
        elif (not b) and (not p):
            n2n += 1
        elif b and (not p):
            p2n += 1
        else:
            p2p += 1
    denom = n2p + n2n
    rate = (n2p / denom) if denom > 0 else None
    return rate, {"N2P": n2p, "N2N": n2n, "P2N": p2n, "P2P": p2p, "overlap_qids": len(qids)}


# ---------------- QID-level bridge: Δdriver_q distribution by event ----------------
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
    out: Dict[str, float] = {}
    for q, c in cnts.items():
        if c > 0:
            out[q] = sums[q] / c
    return out


def plot_delta_hist(delta_n2p: np.ndarray, delta_n2n: np.ndarray, out_png: Path, title: str, bins: int = 60):
    a = _finite(delta_n2p)
    b = _finite(delta_n2n)
    if a.size == 0 or b.size == 0:
        return
    xs = np.concatenate([a, b])
    lo = float(np.quantile(xs, 0.01))
    hi = float(np.quantile(xs, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(xs))
        hi = float(np.max(xs))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)

    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure()
    plt.plot(centers, ha, label="N2P")
    plt.plot(centers, hb, label="N2N")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Δdriver_q = mean_q(train_strategy) - mean_q(train_ref)")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def scatter_strategy_level(points: List[Tuple[str, float, float]],
                           out_png: Path,
                           title: str,
                           xlabel: str):
    pts = [(n, x, y) for (n, x, y) in points if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 2:
        return
    xs = np.asarray([p[1] for p in pts], dtype=np.float64)
    ys = np.asarray([p[2] for p in pts], dtype=np.float64)

    plt.figure()
    plt.scatter(xs, ys)
    for n, x, y in pts:
        plt.text(x, y, n, fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel("NP N2P_rate = N2P/(N2P+N2N)  [from np_base->np_post]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def parse_name_path_list(items: List[str], flag_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"{flag_name} expects name=path, got: {spec}")
        name, p = spec.split("=", 1)
        out[name.strip()] = Path(p.strip())
    return out


def _robust_range(a: np.ndarray, b: np.ndarray, lo_q=0.01, hi_q=0.99) -> Tuple[float, float]:
    x = np.concatenate([_finite(a), _finite(b)], axis=0)
    if x.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def plot_driver_hist_overlay(ref_vals: np.ndarray,
                             strat_vals: np.ndarray,
                             title: str,
                             out_png: Path,
                             out_tsv: Optional[Path] = None,
                             bins: int = 80,
                             density: bool = True):
    a = _finite(ref_vals)
    b = _finite(strat_vals)
    if a.size == 0 or b.size == 0:
        return

    lo, hi = _robust_range(a, b)
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=density)
    hb, _ = np.histogram(b, bins=edges, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0])

    plt.figure()
    plt.bar(centers, ha, width=width, alpha=0.45, label="train_ref")
    plt.bar(centers, hb, width=width, alpha=0.45, label="train_strategy")
    plt.xlabel("driver value")
    plt.ylabel("density" if density else "count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    if out_tsv is not None:
        import pandas as pd
        df = pd.DataFrame({
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": centers,
            "train_ref": ha,
            "train_strategy": hb,
        })
        df.to_csv(out_tsv, sep="\t", index=False)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, action="append", default=[],
                    help="repeatable: name=path for TRAIN rollouts (driver source)")
    ap.add_argument("--train_ref", type=str, required=True,
                    help="reference strategy name among --train (used for deviation + Δdriver_q ref)")

    ap.add_argument("--np_base", type=str, required=True,
                    help="NP-label base rollouts (step2 inference results)")
    ap.add_argument("--np_post", type=str, action="append", default=[],
                    help="repeatable: name=path for NP-label post rollouts (step3 inference results per strategy)")

    ap.add_argument("--drivers_py", type=str, default="", help="external drivers.py")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--rel_bins", type=int, default=100)

    ap.add_argument("--subset", type=str, default="all",
                    help="subset to compute strategy scatter/summary: all|effective|neg|effective_neg|ALL4")
    ap.add_argument("--x_metric", type=str, default="d_tail_mass",
                    help="x-axis metric: d_skew|d_tail_mass|d_mean|d_median|js|w1|ks")

    ap.add_argument("--bridge", action="store_true", help="enable qid-level bridge plots/summaries")
    ap.add_argument("--bridge_bins", type=int, default=60)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "entropy_position").mkdir(parents=True, exist_ok=True)
    (outdir / "drivers").mkdir(parents=True, exist_ok=True)
    (outdir / "bridge_qid").mkdir(parents=True, exist_ok=True)

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
    if args.train_ref not in train_map:
        raise ValueError(f"--train_ref '{args.train_ref}' not found in --train names: {sorted(train_map.keys())}")

    np_post_map = parse_name_path_list(args.np_post, "--np_post")
    if not np_post_map:
        raise ValueError("No --np_post provided.")

    # recommended: train strategies and np_post strategies should overlap
    overlap_names = sorted(set(train_map.keys()) & set(np_post_map.keys()))
    if not overlap_names:
        print("[WARN] No overlap between --train names and --np_post names. "
              "We will still compute deviations for all train strategies, but NP rates may be missing.")
    else:
        # we'll iterate on overlap for joint records; but still allow missing
        pass

    subset_list = ["all", "effective", "neg", "effective_neg"] if args.subset == "ALL4" else [args.subset]
    for sb in subset_list:
        if sb not in ["all", "effective", "neg", "effective_neg"]:
            raise ValueError(f"--subset invalid: {sb}")

    # load TRAIN samples for each strategy (driver source)
    train_samples: Dict[str, List[Sample]] = {}
    train_subsets: Dict[str, Dict[str, List[Sample]]] = {}

    for name, p in train_map.items():
        ss = load_samples(p, pad_id=args.pad_id, max_len=args.max_len)
        train_samples[name] = ss
        train_subsets[name] = split_subsets(ss)

        # entropy-position per strategy (train)
        write_entropy_position_merged(
            strategy=name,
            subsets=train_subsets[name],
            out_root=outdir,
            max_len=args.max_len,
            rel_bins=args.rel_bins,
        )

    # NP labels: base_anyc from np_base; post_anyc per strategy from np_post[name]
    np_base_samples = load_samples(Path(args.np_base), pad_id=args.pad_id, max_len=args.max_len)
    np_base_anyc = per_qid_any_correct(np_base_samples)

    np_post_anyc: Dict[str, Dict[str, bool]] = {}
    for name, p in np_post_map.items():
        ss = load_samples(p, pad_id=args.pad_id, max_len=args.max_len)
        np_post_anyc[name] = per_qid_any_correct(ss)

    # compute N2P_rate per strategy (np_base -> np_post[strategy])
    n2p_rate_by_strategy: Dict[str, Optional[float]] = {}
    np_counts_by_strategy: Dict[str, Dict[str, int]] = {}
    for name in train_map.keys():
        if name in np_post_anyc:
            r, c = np_event_counts(np_base_anyc, np_post_anyc[name])
            n2p_rate_by_strategy[name] = r
            np_counts_by_strategy[name] = c
        else:
            n2p_rate_by_strategy[name] = None
            np_counts_by_strategy[name] = {"N2P": 0, "N2N": 0, "P2N": 0, "P2P": 0, "overlap_qids": 0}

    # precompute ref arrays per subset per driver (train ref)
    ref_name = args.train_ref
    ref_arrays: Dict[str, Dict[str, np.ndarray]] = {sb: {} for sb in subset_list}
    for sb in subset_list:
        ref_ss = train_subsets[ref_name][sb]
        for dname, fn in drivers.items():
            ref_arrays[sb][dname] = compute_driver_array(ref_ss, fn)

    # qid-level ref means per subset per driver (train ref)
    ref_qmean: Dict[str, Dict[str, Dict[str, float]]] = {sb: {} for sb in subset_list}
    if args.bridge:
        for sb in subset_list:
            for dname, fn in drivers.items():
                ref_qmean[sb][dname] = qid_mean_driver(train_subsets[ref_name][sb], fn)

    drivers_root = outdir / "drivers"
    summary_path = drivers_root / "strategy_summary.jsonl"
    bridge_path = outdir / "bridge_qid" / "bridge_summary.jsonl"

    with summary_path.open("w", encoding="utf-8") as sf, bridge_path.open("w", encoding="utf-8") as bf:
        for sb in subset_list:
            for dname, fn in drivers.items():
                points: List[Tuple[str, float, float]] = []

                for sname in train_map.keys():
                    ss = train_subsets[sname][sb]
                    arr = compute_driver_array(ss, fn)
                    dev = deviation_against_ref(arr, ref_arrays[sb][dname])
                    x = pick_x_metric(dev, args.x_metric)
                    y = n2p_rate_by_strategy.get(sname, None)
                    yv = float("nan") if (y is None) else float(y)

                    rec = {
                        "strategy": sname,
                        "train_ref": ref_name,
                        "subset": sb,
                        "driver": dname,
                        "x_metric": args.x_metric,
                        "x_value": None if not np.isfinite(x) else float(x),
                        "np_N2P_rate": None if not np.isfinite(yv) else float(yv),
                        "np_counts": np_counts_by_strategy.get(sname, None),
                        "dev": {k: (None if (not np.isfinite(v)) else float(v)) for k, v in dev.items()},
                        "n_train_samples": int(len(ss)),
                        "n_train_samples_ref": int(len(train_subsets[ref_name][sb])),
                        "np_base": str(args.np_base),
                        "np_post": str(np_post_map.get(sname, "")) if sname in np_post_map else None,
                        "train_file": str(train_map[sname]),
                    }
                    sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    points.append((sname, x, yv))

                    # histogram overlay (train strategy vs train ref)
                    out_png = drivers_root / f"driver_overlay_{dname}_{sb}_{sname}_vs_{ref_name}.png"
                    out_tsv = drivers_root / f"driver_overlay_{dname}_{sb}_{sname}_vs_{ref_name}.tsv"
                    plot_driver_hist_overlay(
                        ref_vals=ref_arrays[sb][dname],
                        strat_vals=arr,
                        title=f"{dname} | subset={sb} | train:{sname} vs train_ref:{ref_name}",
                        out_png=out_png,
                        out_tsv=out_tsv,
                        bins=80,
                        density=True,
                    )

                    # bridge: label from NP files, driver_q from TRAIN files
                    if args.bridge:
                        if sname not in np_post_anyc:
                            continue

                        # N/P labels
                        base_anyc = np_base_anyc
                        post_anyc = np_post_anyc[sname]

                        # driver qmeans
                        qmean_s = qid_mean_driver(ss, fn)
                        qmean_r = ref_qmean[sb][dname]

                        overlap = set(base_anyc.keys()) & set(post_anyc.keys()) & set(qmean_s.keys()) & set(qmean_r.keys())

                        deltas_n2p = []
                        deltas_n2n = []
                        for q in overlap:
                            b = bool(base_anyc[q])
                            p = bool(post_anyc[q])
                            if b:
                                continue  # only base=N
                            delta = float(qmean_s[q] - qmean_r[q])
                            if not np.isfinite(delta):
                                continue
                            if p:
                                deltas_n2p.append(delta)
                            else:
                                deltas_n2n.append(delta)

                        dn2p = np.asarray(deltas_n2p, dtype=np.float64)
                        dn2n = np.asarray(deltas_n2n, dtype=np.float64)

                        def _stat(xarr):
                            xarr = _finite(xarr)
                            if xarr.size == 0:
                                return {"n": 0, "mean": None, "median": None}
                            return {"n": int(xarr.size), "mean": float(np.mean(xarr)), "median": float(np.median(xarr))}

                        dist = {
                            "w1": wasserstein_1d(dn2p, dn2n),
                            "ks": ks_statistic(dn2p, dn2n),
                        }

                        brec = {
                            "strategy": sname,
                            "train_ref": ref_name,
                            "subset": sb,
                            "driver": dname,
                            "qid_overlap_used": int(len(overlap)),
                            "N2P": _stat(dn2p),
                            "N2N": _stat(dn2n),
                            "dist_N2P_vs_N2N": {k: (None if (not np.isfinite(v)) else float(v)) for k, v in dist.items()},
                            "np_base": str(args.np_base),
                            "np_post": str(np_post_map[sname]),
                            "train_strategy_file": str(train_map[sname]),
                            "train_ref_file": str(train_map[ref_name]),
                        }
                        bf.write(json.dumps(brec, ensure_ascii=False) + "\n")

                        out_png2 = outdir / "bridge_qid" / f"delta_qid_hist_{dname}_{sb}_{sname}.png"
                        plot_delta_hist(
                            delta_n2p=dn2p,
                            delta_n2n=dn2n,
                            out_png=out_png2,
                            title=f"{sname} | {dname} | subset={sb} | Δqid(train driver) | label from np(step2->step3)",
                            bins=int(args.bridge_bins),
                        )

                # scatter
                out_png = outdir / "drivers" / f"scatter_{dname}_{sb}.png"
                scatter_strategy_level(
                    points=points,
                    out_png=out_png,
                    title=f"{dname} | subset={sb} | x={args.x_metric} vs NP N2P_rate (from np_base->np_post)",
                    xlabel=f"{args.x_metric} (train strategy vs train_ref={ref_name})",
                )

    (outdir / "README_train_driver_np_labels.txt").write_text(
        "\n".join([
            "This script separates:",
            "1) TRAIN rollouts for driver distributions (and Δdriver_q).",
            "2) NP-label rollouts (step2->step3 inference) for N2P/N2N labels.",
            "",
            f"train_ref={ref_name}",
            f"x_metric={args.x_metric}",
            f"subset={args.subset}",
            f"drivers={sorted(list(drivers.keys()))}",
            "",
            "x_metric options: d_mean, d_median, d_skew, d_tail_mass, js, w1, ks",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] outdir = {outdir}")
    print(f"[OK] train strategies = {sorted(list(train_map.keys()))}")
    print(f"[OK] np_post strategies = {sorted(list(np_post_map.keys()))}")
    print(f"[OK] train_ref = {ref_name}")
    print(f"[OK] drivers = {sorted(list(drivers.keys()))}")
    print(f"[OK] wrote: {summary_path}")
    if args.bridge:
        print(f"[OK] wrote: {bridge_path}")


if __name__ == "__main__":
    main()