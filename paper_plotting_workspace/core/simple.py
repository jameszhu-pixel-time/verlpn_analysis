#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
intra_strategy_np_driver.py

Intra-strategy NP vs driver pipeline (NO delta driver, NO ref comparison).

You want:
- For each strategy:
  - Use NP label files: np_base (step2 inference) -> np_post[strategy] (step3 inference)
    to label qids as N2P / N2N / P2N / P2P by any_correct(qid).
  - Use TRAIN rollouts of the SAME strategy to compute driver at qid-level (mean over rollouts).
  - Compare within the SAME strategy:
      driver_q distribution for N2P vs N2N (base=N only)
    with:
      - histogram overlay
      - Wasserstein-1 distance
      - KS statistic
      - basic stats (n/mean/median)

Inputs
- --train name=PATH         (repeatable)  train rollouts per strategy (verl rollouts)
- --np_base PATH            step2 inference results (same dataset)
- --np_post name=PATH       (repeatable)  step3 inference results per strategy (vLLM rollouts)
- --drivers_py PATH         your drivers.py

Outputs (per strategy × driver × subset)
outdir/{strategy}/
  - hist_qid_{driver}_{subset}_N2P_vs_N2N.png
  - intra_summary.jsonl  (one line per strategy×driver×subset)

Subsets are defined on TRAIN rollouts:
- all / effective / neg / effective_neg
Then we compute qid_mean_driver inside that subset and intersect with N2P/N2N qids.

Important:
- No "ref", no "delta", no strategy-vs-ref deviation.
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


def plot_hist_overlay(a: np.ndarray, b: np.ndarray, out_png: Path, title: str, bins: int = 60):
    aa = _finite(a)
    bb = _finite(b)
    if aa.size == 0 or bb.size == 0:
        return
    xs = np.concatenate([aa, bb])
    lo = float(np.quantile(xs, 0.01))
    hi = float(np.quantile(xs, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(xs))
        hi = float(np.max(xs))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(aa, bins=edges, density=True)
    hb, _ = np.histogram(bb, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure()
    plt.plot(centers, ha, label="N2P")
    plt.plot(centers, hb, label="N2N")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("qid_mean_driver (within strategy)")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, action="append", default=[],
                    help="repeatable: name=path for TRAIN rollouts (driver source)")
    ap.add_argument("--np_base", type=str, required=True,
                    help="NP-label base rollouts (step2 inference results)")
    ap.add_argument("--np_post", type=str, action="append", default=[],
                    help="repeatable: name=path for NP-label post rollouts (step3 inference results per strategy)")
    ap.add_argument("--drivers_py", type=str, default="", help="external drivers.py")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--subset", type=str, default="ALL4",
                    help="all|effective|neg|effective_neg|ALL4")
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

    subset_list = ["all", "effective", "neg", "effective_neg"] if args.subset == "ALL4" else [args.subset]
    for sb in subset_list:
        if sb not in ["all", "effective", "neg", "effective_neg"]:
            raise ValueError(f"--subset invalid: {sb}")

    # NP labels
    base_samples = load_samples(Path(args.np_base), pad_id=args.pad_id, max_len=args.max_len)
    base_anyc = per_qid_any_correct(base_samples)

    post_anyc: Dict[str, Dict[str, bool]] = {}
    for name, p in np_post_map.items():
        ss = load_samples(p, pad_id=args.pad_id, max_len=args.max_len)
        post_anyc[name] = per_qid_any_correct(ss)

    summary_path = outdir / "intra_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as sf:
        for sname, train_path in train_map.items():
            if sname not in post_anyc:
                print(f"[WARN] strategy '{sname}' has train but no np_post; skip NP grouping.")
                continue

            strat_dir = outdir / sname
            strat_dir.mkdir(parents=True, exist_ok=True)

            # load train rollouts for driver source
            train_samples = load_samples(Path(train_path), pad_id=args.pad_id, max_len=args.max_len)
            train_subsets = split_subsets(train_samples)

            for sb in subset_list:
                sb_samples = train_subsets[sb]
                if not sb_samples:
                    continue

                # qid -> mean(driver over rollouts), within this strategy+subset
                for dname, fn in drivers.items():
                    qmean = qid_mean_driver(sb_samples, fn)
                    if not qmean:
                        continue

                    # label qids by base_anyc -> post_anyc[strategy]
                    overlap = set(base_anyc.keys()) & set(post_anyc[sname].keys()) & set(qmean.keys())

                    n2p_vals = []
                    n2n_vals = []
                    for q in overlap:
                        b = bool(base_anyc[q])
                        p = bool(post_anyc[sname][q])
                        if b:
                            continue  # only base=N
                        v = qmean.get(q, None)
                        if v is None or (not np.isfinite(v)):
                            continue
                        if p:
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

                    rec = {
                        "strategy": sname,
                        "subset": sb,
                        "driver": dname,
                        "qid_overlap_used": int(len(overlap)),
                        "N2P_driver": _stat(a),
                        "N2N_driver": _stat(b),
                        "dist_N2P_vs_N2N": {
                            "w1": None if not np.isfinite(wasserstein_1d(a, b)) else float(wasserstein_1d(a, b)),
                            "ks": None if not np.isfinite(ks_statistic(a, b)) else float(ks_statistic(a, b)),
                        },
                        "np_base": str(args.np_base),
                        "np_post": str(np_post_map[sname]),
                        "train_file": str(train_path),
                    }
                    sf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    out_png = strat_dir / f"hist_qid_{dname}_{sb}_N2P_vs_N2N.png"
                    plot_hist_overlay(
                        a=a,
                        b=b,
                        out_png=out_png,
                        title=f"{sname} | {dname} | subset={sb} | qid_mean_driver: N2P vs N2N (within strategy)",
                        bins=int(args.bins),
                    )

    print(f"[OK] outdir = {outdir}")
    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] drivers = {sorted(list(drivers.keys()))}")


if __name__ == "__main__":
    main()