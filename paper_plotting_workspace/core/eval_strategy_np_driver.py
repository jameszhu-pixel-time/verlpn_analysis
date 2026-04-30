#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_strategy_np_driver.py

Goal (per your requirement)
- TRAIN side: compare *same-step* rollouts across strategies (verl protocol) via drivers/entropy-position.
  We do NOT compare prev vs post steps here. We compare each strategy against a chosen TRAIN reference strategy.
- EVAL side: define NP events on validation using any_correct from step2->step3:
    val_base (step2) vs val_post[strategy] (step3, same qids)
  and compute:
    N2P_rate = #N2P / (#N2P + #N2N)

Outputs (organized)
1) outdir/entropy_position/{strategy}/
   - abs_mean_4subsets.png
   - rel_mean_4subsets.png
   - stats.json

2) outdir/drivers/
   summary/
     - strategy_summary.jsonl
   scatter/{subset}/
     - scatter_{driver}_{subset}.png
   overlay_by_strategy/{strategy}/{subset}/
     - {driver}_vs_{train_ref}.png
     - {driver}_vs_{train_ref}.tsv

Key policies
- Training entropy: prefer full_logprobs->entropy whenever available (NPZ or JSONL), MUST mask.
- Mask priority: response_mask -> valid_len/response_len -> PAD in token_ids -> finite(entropy)
- Truncate absolute length to --max_len (default 3072)

Drivers
- Load from --drivers_py and/or colocated drivers.py
  APIs:
    A) get_drivers()-> dict[name]=fn(ent)->float
    B) functions driver_xxx(ent)->float  (registered as "xxx")

Recommended usage
- train_ref can be any strategy you want as baseline of deviation.
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


# =========================
# JSONL IO + field extract
# =========================
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
    if isinstance(v, list):
        return v
    return None


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

    # 2) valid_len hint
    L = extract_valid_len_hint(rec)
    if L is not None:
        L = max(0, min(T, int(L)))
        m = np.zeros((T,), dtype=bool)
        m[:L] = True
        return m

    # 3) PAD in token_ids
    token_ids = extract_token_ids(rec)
    if token_ids is not None:
        m = build_response_mask_from_tokens(token_ids, pad_id)
        if m is not None and m.size == T:
            return m

    return None


# =========================
# Entropy from full_logprobs (preferred)
# =========================
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


def extract_entropy_train(rec: Dict[str, Any],
                          qid: Optional[str],
                          rid: Optional[str],
                          npz_index: Optional[Dict[Tuple[str, str], Path]]) -> Optional[np.ndarray]:
    """
    Training (verl) entropy priority:
      1) JSONL full_logprobs (T,K) -> entropy
      2) JSONL token_entropies_topk/entropy/entropies
      3) JSONL topk_logprobs_per_token
      4) NPZ explicit path (full_logprobs preferred)
      5) NPZ index by (qid,rid)
    """
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


# =========================
# Samples + subsets (training)
# =========================
@dataclass
class Sample:
    qid: str
    rid: str
    correct: bool
    ent: np.ndarray     # NaN-masked
    valid_len: int


def load_train_samples_verl(path: Path, pad_id: int, max_len: int) -> List[Sample]:
    samples: List[Sample] = []
    npz_index = build_npz_index(path)

    for rec in tqdm(iter_jsonl(path), desc=f"train load {path.name}"):
        qid = extract_qid(rec)
        rid = extract_rid(rec) or "unknown"
        correct = extract_correct(rec)
        if qid is None or correct is None:
            continue

        ent = extract_entropy_train(rec, qid=qid, rid=rid, npz_index=npz_index)
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


# =========================
# Validation: N2P_rate from step2->step3 any_correct
# =========================
def load_val_any_correct(path: Path) -> Dict[str, bool]:
    anyc: Dict[str, bool] = {}
    for rec in iter_jsonl(path):
        qid = extract_qid(rec)
        cor = extract_correct(rec)
        if qid is None or cor is None:
            continue
        anyc[qid] = anyc.get(qid, False) or bool(cor)
    return anyc


def n2p_rate(val_base_anyc: Dict[str, bool], val_post_anyc: Dict[str, bool]) -> Tuple[Optional[float], Dict[str, int]]:
    qids = set(val_base_anyc.keys()) & set(val_post_anyc.keys())
    n2p = n2n = p2n = p2p = 0
    for q in qids:
        b = bool(val_base_anyc[q])
        p = bool(val_post_anyc[q])
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


# =========================
# Drivers loading
# =========================
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
            if not name.startswith("driver_"):
                continue
            fn = getattr(mod, name)
            if callable(fn):
                out[name[len("driver_"):]] = fn

        return out
    except Exception as e:
        print(f"[WARN] failed to import drivers from {py_path}: {e}")
        return {}


# =========================
# Metrics: skewness + tail etc.
# =========================
def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def skewness(x: np.ndarray) -> float:
    x = _finite(x)
    if x.size < 3:
        return float("nan")
    m = np.mean(x)
    s = np.std(x)
    if s <= 1e-12:
        return float("nan")
    z3 = np.mean(((x - m) / s) ** 3)
    return float(z3)


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


# =========================
# Entropy-position plots (4 subsets in one fig)
# =========================
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
    with np.errstate(divide="ignore", invalid="ignore"):
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
    with np.errstate(divide="ignore", invalid="ignore"):
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

    # stats.json
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


# =========================
# Driver aggregation + overlays
# =========================
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
    """
    metric choices: d_skew, d_tail_mass, d_mean, d_median, js, w1, ks, skew, tail_mass
    """
    return float(dev.get(metric, float("nan")))


def scatter_strategy_level(points: List[Tuple[str, float, float]],
                           out_png: Path,
                           title: str,
                           xlabel: str,
                           ylabel: str):
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
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


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
    """
    Bar overlay: ref vs strategy (same driver, same subset).
    Saved into overlay_by_strategy/{strategy}/{subset}/
    """
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
    plt.bar(centers, ha, width=width, alpha=0.45, label="ref")
    plt.bar(centers, hb, width=width, alpha=0.45, label="strategy")
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
            "ref": ha,
            "strategy": hb,
        })
        df.to_csv(out_tsv, sep="\t", index=False)


# =========================
# CLI helpers
# =========================
def parse_name_path_list(items: List[str], flag_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"{flag_name} expects name=path, got: {spec}")
        name, p = spec.split("=", 1)
        out[name.strip()] = Path(p.strip())
    return out


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    # TRAIN: same-step strategies (verl protocol)
    ap.add_argument("--train", type=str, action="append", default=[], help="repeatable: name=path (verl JSONL)")
    ap.add_argument("--train_ref", type=str, required=True, help="which training strategy name is reference baseline")
    # VAL: NP events from step2->step3 any_correct (vLLM protocol typically)
    ap.add_argument("--val_base", type=str, required=True, help="validation step2 base JSONL")
    ap.add_argument("--val_post", type=str, action="append", default=[], help="repeatable: name=path (validation step3 per strategy)")
    # output + drivers
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--drivers_py", type=str, default="", help="path to external drivers.py")
    # knobs
    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--rel_bins", type=int, default=100)
    ap.add_argument("--x_metric", type=str, default="d_skew",
                    help="x-axis metric for scatter: d_skew|d_tail_mass|d_mean|d_median|js|w1|ks|skew|tail_mass")
    ap.add_argument("--subset", type=str, default="all",
                    help="which training subset to aggregate driver for scatter: all|effective|neg|effective_neg|ALL4 (do all 4)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # folders (organized)
    (outdir / "entropy_position").mkdir(parents=True, exist_ok=True)

    drivers_root = outdir / "drivers"
    drivers_root.mkdir(parents=True, exist_ok=True)

    summary_root = drivers_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    scatter_root = drivers_root / "scatter"
    scatter_root.mkdir(parents=True, exist_ok=True)

    overlay_root = drivers_root / "overlay_by_strategy"
    overlay_root.mkdir(parents=True, exist_ok=True)

    summary_path = summary_root / "strategy_summary.jsonl"

    # drivers
    drivers: Dict[str, DriverFn] = {}
    if args.drivers_py:
        drivers.update(load_drivers_from_py(Path(args.drivers_py)))
    script_dir = Path(__file__).resolve().parent
    drivers.update(load_drivers_from_py(script_dir / "drivers.py"))
    if not drivers:
        raise RuntimeError("No drivers loaded. Provide --drivers_py or put drivers.py next to this script.")

    # parse train + val maps
    train_map = parse_name_path_list(args.train, "--train")
    val_post_map = parse_name_path_list(args.val_post, "--val_post")

    if args.train_ref not in train_map:
        raise ValueError(f"--train_ref '{args.train_ref}' not found in --train names: {sorted(train_map.keys())}")

    # load validation any_correct (NP definition)
    val_base_anyc = load_val_any_correct(Path(args.val_base))
    val_np: Dict[str, Dict[str, Any]] = {}
    for sname, p in val_post_map.items():
        post_anyc = load_val_any_correct(p)
        rate, counts = n2p_rate(val_base_anyc, post_anyc)
        val_np[sname] = {"N2P_rate": rate, "counts": counts}

    # load training samples (verl) for each strategy
    train_subsets: Dict[str, Dict[str, List[Sample]]] = {}
    for sname, p in train_map.items():
        ss = load_train_samples_verl(p, pad_id=args.pad_id, max_len=args.max_len)
        train_subsets[sname] = split_subsets(ss)
        write_entropy_position_merged(
            strategy=sname,
            subsets=train_subsets[sname],
            out_root=outdir,
            max_len=args.max_len,
            rel_bins=args.rel_bins,
        )

    # subset list for analysis
    subset_list = ["all", "effective", "neg", "effective_neg"] if args.subset == "ALL4" else [args.subset]
    for sb in subset_list:
        if sb not in ["all", "effective", "neg", "effective_neg"]:
            raise ValueError(f"--subset invalid: {sb}")

    # precompute ref arrays for each driver per subset
    ref_name = args.train_ref
    ref_arrays: Dict[str, Dict[str, np.ndarray]] = {sb: {} for sb in subset_list}
    for sb in subset_list:
        ref_ss = train_subsets[ref_name].get(sb, [])
        for dname, fn in drivers.items():
            ref_arrays[sb][dname] = compute_driver_array(ref_ss, fn)

    # write summary + generate overlays organized by strategy
    with summary_path.open("w", encoding="utf-8") as sf:
        for sb in subset_list:
            for dname, fn in drivers.items():
                points: List[Tuple[str, float, float]] = []

                for sname in train_map.keys():
                    ss = train_subsets[sname].get(sb, [])
                    arr = compute_driver_array(ss, fn)
                    dev = deviation_against_ref(arr, ref_arrays[sb][dname])
                    x = pick_x_metric(dev, args.x_metric)

                    # y: N2P_rate from validation (step2->step3)
                    y = val_np.get(sname, {}).get("N2P_rate", None)
                    yv = float("nan") if (y is None) else float(y)

                    rec = {
                        "strategy": sname,
                        "train_ref": ref_name,
                        "subset": sb,
                        "driver": dname,
                        "x_metric": args.x_metric,
                        "x_value": None if not np.isfinite(x) else float(x),
                        "val_N2P_rate": None if not np.isfinite(yv) else float(yv),
                        "val_counts": val_np.get(sname, {}).get("counts", None),
                        "dev": {k: (None if (not np.isfinite(v)) else float(v)) for k, v in dev.items()},
                        "n_train_samples": int(len(ss)),
                        "n_train_samples_ref": int(len(train_subsets[ref_name].get(sb, []))),
                    }
                    sf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    points.append((sname, x, yv))

                    # ---- overlay_by_strategy/{strategy}/{subset}/{driver}_vs_ref.* ----
                    od = overlay_root / sname / sb
                    od.mkdir(parents=True, exist_ok=True)

                    out_png = od / f"{dname}_vs_{ref_name}.png"
                    out_tsv = od / f"{dname}_vs_{ref_name}.tsv"
                    plot_driver_hist_overlay(
                        ref_vals=ref_arrays[sb][dname],
                        strat_vals=arr,
                        title=f"{dname} | subset={sb} | {sname} vs ref={ref_name}",
                        out_png=out_png,
                        out_tsv=out_tsv,
                        bins=80,
                        density=True,
                    )

                # scatter per driver+subset (organized)
                sd = scatter_root / sb
                sd.mkdir(parents=True, exist_ok=True)
                out_png = sd / f"scatter_{dname}_{sb}.png"
                scatter_strategy_level(
                    points=points,
                    out_png=out_png,
                    title=f"{dname} | subset={sb} | x={args.x_metric} vs N2P_rate",
                    xlabel=f"{args.x_metric} (strategy vs ref={ref_name})",
                    ylabel="N2P_rate = N2P/(N2P+N2N)",
                )

    # quick README
    (outdir / "README_eval_strategy_np_driver.txt").write_text(
        "\n".join([
            "This script compares SAME-STEP training rollouts across strategies (verl protocol) against a chosen train_ref.",
            "Validation NP events are defined by any_correct from val_base(step2) vs val_post(step3,strategy).",
            "",
            "Outputs (organized):",
            "1) entropy_position/{strategy}/abs_mean_4subsets.png + rel_mean_4subsets.png + stats.json",
            "2) drivers/summary/strategy_summary.jsonl (deviation vs train_ref + val N2P_rate)",
            "3) drivers/scatter/{subset}/scatter_{driver}_{subset}.png",
            "4) drivers/overlay_by_strategy/{strategy}/{subset}/{driver}_vs_{train_ref}.png(.tsv)",
            "",
            f"train_ref={ref_name}",
            f"x_metric={args.x_metric}",
            f"subset={args.subset}",
            f"drivers={sorted(list(drivers.keys()))}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] outdir = {outdir}")
    print(f"[OK] loaded drivers = {sorted(list(drivers.keys()))}")
    print(f"[OK] wrote: {summary_path}")


if __name__ == "__main__":
    main()