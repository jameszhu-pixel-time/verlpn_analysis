#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_npnn_kde_multi.py

For each strategy (name), use its own:
  - driver source:   --train name=PATH
  - base NP labels:  --np_base name=PATH
  - post NP labels:  --np_post name=PATH

Then compare TWO strategies A and B, plotting 4 KDE curves on ONE figure (rollout-level driver):
  A: N2P (orange solid)
  A: N2N (orange dashed)
  B: N2P (blue solid)
  B: N2N (blue dashed)

NP/NN definition per strategy:
  base_anyc[name] = any_correct(qid) from np_base[name]
  post_anyc[name] = any_correct(qid) from np_post[name]
  base=N: base_anyc[name][qid] == False
  N2P: base=N and post_anyc[name][qid] == True
  N2N: base=N and post_anyc[name][qid] == False

Driver values:
  rollout-level: for each rollout record in train[name], compute driver -> scalar;
  keep the rollout if its qid is in the corresponding N2P/N2N qid set.

No histogram background. No distribution metrics except optional right-skew metrics.

CLI:
  --train name=PATH   (repeatable)
  --np_base name=PATH (repeatable)
  --np_post name=PATH (repeatable)
  --pair A,B
  --drivers_py PATH
  --outdir OUT

New features:
  --save_right_skew
      save right-skew metrics JSON for each driver

  --legend_show_skew
      append skew summary into legend labels

Driver interface compatibility:
  1) old style:
        def driver_xxx(ent: np.ndarray) -> float
  2) new style:
        def driver_xxx(ent: np.ndarray, topk_logprobs_per_token) -> float

where topk_logprobs_per_token is:
    Optional[List[np.ndarray]]
and each element is the token-level top-k logprob vector (already aligned to valid response tokens).

This means you do NOT need to change the launch command.
"""

import argparse
import inspect
import json
import re
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable, Set

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PAD_ID_DEFAULT = 151643

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 7.5,
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


# ---------------- Entropy / topk logprobs compute ----------------
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


def extract_topk_logprobs_per_token(rec: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    """
    Convert rec['topk_logprobs_per_token'] into:
        List[np.ndarray], each element is shape [K_t] logprob vector for one token.
    Return None if unavailable / malformed.
    """
    raw = rec.get("topk_logprobs_per_token", None)
    if raw is None:
        return None
    try:
        out: List[np.ndarray] = []
        for pairs in raw:
            if not pairs:
                out.append(np.asarray([], dtype=np.float64))
                continue
            lps = []
            for p in pairs:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        lps.append(float(p[1]))
                    except Exception:
                        continue
            out.append(np.asarray(lps, dtype=np.float64))
        return out
    except Exception:
        return None


def mask_topk_logprobs_per_token(
    topk_lps: Optional[List[np.ndarray]],
    valid_mask: np.ndarray,
    max_len: int,
) -> Optional[List[np.ndarray]]:
    if topk_lps is None:
        return None
    try:
        T = min(len(topk_lps), int(valid_mask.size), int(max_len))
        out: List[np.ndarray] = []
        for t in range(T):
            if bool(valid_mask[t]):
                out.append(np.asarray(topk_lps[t], dtype=np.float64).reshape(-1))
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


# ---------------- Driver-source rollouts ----------------
@dataclass
class Rollout:
    qid: str
    rid: str
    ent: np.ndarray
    topk_logprobs_per_token: Optional[List[np.ndarray]] = None


def load_rollouts_for_driver(path: Path, pad_id: int, max_len: int) -> List[Rollout]:
    outs: List[Rollout] = []
    npz_index = build_npz_index(path)

    for rec in tqdm(iter_jsonl(path), desc=f"load {path.name}"):
        qid = extract_qid(rec)
        rid = extract_rid(rec) or "unknown"
        if qid is None:
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

        topk_lps = extract_topk_logprobs_per_token(rec)
        topk_lps = mask_topk_logprobs_per_token(topk_lps, valid_mask=valid, max_len=max_len)

        outs.append(
            Rollout(
                qid=str(qid),
                rid=str(rid),
                ent=ent2,
                topk_logprobs_per_token=topk_lps,
            )
        )

    return outs


# ---------------- NP label helpers ----------------
def per_qid_any_correct_from_file(path: Path) -> Dict[str, bool]:
    anyc: Dict[str, bool] = {}
    for rec in tqdm(iter_jsonl(path), desc=f"label {path.name}"):
        qid = extract_qid(rec)
        correct = extract_correct(rec)
        if qid is None or correct is None:
            continue
        q = str(qid)
        anyc[q] = anyc.get(q, False) or bool(correct)
    return anyc


# ---------------- Drivers ----------------
DriverFn = Callable[..., float]


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


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


# ---------------- Right-skew metrics ----------------
def _summary_right_skew(x: np.ndarray) -> Dict[str, float]:
    x = _finite(x)
    out = {
        "n": int(x.size),
        "mean": float("nan"),
        "std": float("nan"),
        "q10": float("nan"),
        "q25": float("nan"),
        "q50": float("nan"),
        "q75": float("nan"),
        "q90": float("nan"),
        "moment_skewness": float("nan"),
        "bowley_skewness": float("nan"),
        "right_tail_cutoff_q3_1p5iqr": float("nan"),
        "right_tail_mass_q3_1p5iqr": float("nan"),
        "right_tail_mass_q90": float("nan"),
        "mean_minus_median": float("nan"),
    }
    if x.size == 0:
        return out

    mu = float(np.mean(x))
    sd = float(np.std(x))
    q10, q25, q50, q75, q90 = [float(np.quantile(x, q)) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]
    iqr = q75 - q25

    out["mean"] = mu
    out["std"] = sd
    out["q10"] = q10
    out["q25"] = q25
    out["q50"] = q50
    out["q75"] = q75
    out["q90"] = q90
    out["mean_minus_median"] = mu - q50

    if np.isfinite(sd) and sd > 1e-12:
        m3 = float(np.mean((x - mu) ** 3))
        out["moment_skewness"] = m3 / (sd ** 3)

    if np.isfinite(iqr) and iqr > 1e-12:
        out["bowley_skewness"] = (q75 + q25 - 2.0 * q50) / iqr

    if np.isfinite(iqr):
        cutoff = q75 + 1.5 * iqr
        out["right_tail_cutoff_q3_1p5iqr"] = cutoff
        out["right_tail_mass_q3_1p5iqr"] = float(np.mean(x > cutoff))

    out["right_tail_mass_q90"] = float(np.mean(x > q90)) if x.size > 0 else float("nan")
    return out


def save_right_skew_metrics(
    out_json: Path,
    A_name: str,
    B_name: str,
    A_N2P: np.ndarray,
    A_N2N: np.ndarray,
    B_N2P: np.ndarray,
    B_N2N: np.ndarray,
):
    obj = {
        f"{A_name}:N2P": _summary_right_skew(A_N2P),
        f"{A_name}:N2N": _summary_right_skew(A_N2N),
        f"{B_name}:N2P": _summary_right_skew(B_N2P),
        f"{B_name}:N2N": _summary_right_skew(B_N2N),
    }

    bowley_rank = sorted(
        [(k, v.get("bowley_skewness", float("nan"))) for k, v in obj.items()],
        key=lambda t: (-1e18 if not np.isfinite(t[1]) else -t[1], t[0])
    )
    moment_rank = sorted(
        [(k, v.get("moment_skewness", float("nan"))) for k, v in obj.items()],
        key=lambda t: (-1e18 if not np.isfinite(t[1]) else -t[1], t[0])
    )
    tail_rank = sorted(
        [(k, v.get("right_tail_mass_q3_1p5iqr", float("nan"))) for k, v in obj.items()],
        key=lambda t: (-1e18 if not np.isfinite(t[1]) else -t[1], t[0])
    )

    obj["_ranking"] = {
        "by_bowley_skewness_desc": [[k, float(v)] for k, v in bowley_rank],
        "by_moment_skewness_desc": [[k, float(v)] for k, v in moment_rank],
        "by_right_tail_mass_q3_1p5iqr_desc": [[k, float(v)] for k, v in tail_rank],
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _legend_label_with_skew(base_label: str, x: np.ndarray, show_skew: bool) -> str:
    x = _finite(x)
    if not show_skew:
        return base_label
    s = _summary_right_skew(x)
    n = s["n"]
    ms = s["moment_skewness"]
    bs = s["bowley_skewness"]
    rt = s["right_tail_mass_q3_1p5iqr"]

    def _fmt(v):
        return "nan" if not np.isfinite(v) else f"{v:.3f}"

    return (
        f"{base_label} "
        f"(n={n}, msk={_fmt(ms)}, bsk={_fmt(bs)}, rt={_fmt(rt)})"
    )


# ---------------- KDE plot ----------------
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


def _robust_xlim(arrs: List[np.ndarray]) -> Tuple[float, float]:
    xs_list = []
    for a in arrs:
        fa = _finite(a)
        if fa.size > 0:
            xs_list.append(fa)
    if not xs_list:
        return -1.0, 1.0
    xs = np.concatenate(xs_list)
    lo = float(np.quantile(xs, 0.01))
    hi = float(np.quantile(xs, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(xs))
        hi = float(np.max(xs))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def plot_4kde_npnn(
    A_N2P: np.ndarray, A_N2N: np.ndarray,
    B_N2P: np.ndarray, B_N2N: np.ndarray,
    out_png: Path,
    title: str,
    A_name: str,
    B_name: str,
    legend_show_skew: bool = False,
):
    if (_finite(A_N2P).size + _finite(A_N2N).size + _finite(B_N2P).size + _finite(B_N2N).size) < 4:
        return

    lo, hi = _robust_xlim([A_N2P, A_N2N, B_N2P, B_N2N])
    grid = np.linspace(lo, hi, 600)

    c_blue = "#1f77b4"
    c_orng = "#ff7f0e"

    fig = plt.figure(figsize=(8.2, 4.8), dpi=180)
    ax = fig.add_subplot(111)

    def _plot_if_ok(x, color, ls, label):
        x = _finite(x)
        if x.size < 2:
            return
        y = _kde_gaussian(x, grid)
        if np.max(y) <= 0:
            return
        ax.plot(grid, y, color=color, linewidth=1.7, linestyle=ls, label=label)

    label_A_N2P = _legend_label_with_skew(f"{A_name}:N2P", A_N2P, legend_show_skew)
    label_A_N2N = _legend_label_with_skew(f"{A_name}:N2N", A_N2N, legend_show_skew)
    label_B_N2P = _legend_label_with_skew(f"{B_name}:N2P", B_N2P, legend_show_skew)
    label_B_N2N = _legend_label_with_skew(f"{B_name}:N2N", B_N2N, legend_show_skew)

    _plot_if_ok(A_N2P, c_orng, "-",  label_A_N2P)
    _plot_if_ok(A_N2N, c_orng, "--", label_A_N2N)
    _plot_if_ok(B_N2P, c_blue, "-",  label_B_N2P)
    _plot_if_ok(B_N2N, c_blue, "--", label_B_N2N)

    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_title(title, pad=8)
    ax.set_xlabel("driver (rollout-level)")
    ax.set_ylabel("density (KDE)")
    ax.set_xlim(lo, hi)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.25,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, action="append", default=[],
                    help="repeatable: name=path for rollouts used to compute driver (source)")
    ap.add_argument("--np_base", type=str, action="append", default=[],
                    help="repeatable: name=path for base labels (any_correct per qid)")
    ap.add_argument("--np_post", type=str, action="append", default=[],
                    help="repeatable: name=path for post labels (any_correct per qid)")
    ap.add_argument("--pair", type=str, required=True,
                    help="two names to compare: A,B (must exist in train/np_base/np_post)")
    ap.add_argument("--drivers_py", type=str, default="", help="external drivers.py")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--pad_id", type=int, default=PAD_ID_DEFAULT)
    ap.add_argument("--max_len", type=int, default=3072)

    ap.add_argument("--save_right_skew", action="store_true",
                    help="save right-skew metrics json for each driver")
    ap.add_argument("--legend_show_skew", action="store_true",
                    help="append skew summary into legend labels")

    args = ap.parse_args()

    train_map = parse_name_path_list(args.train, "--train")
    base_map = parse_name_path_list(args.np_base, "--np_base")
    post_map = parse_name_path_list(args.np_post, "--np_post")

    if not train_map:
        raise ValueError("No --train provided.")
    if not base_map:
        raise ValueError("No --np_base provided.")
    if not post_map:
        raise ValueError("No --np_post provided.")

    if "," not in args.pair:
        raise ValueError("--pair expects A,B")
    A_name, B_name = [x.strip() for x in args.pair.split(",", 1)]

    for nm in [A_name, B_name]:
        if nm not in train_map:
            raise ValueError(f"pair name '{nm}' missing in --train")
        if nm not in base_map:
            raise ValueError(f"pair name '{nm}' missing in --np_base")
        if nm not in post_map:
            raise ValueError(f"pair name '{nm}' missing in --np_post")

    outdir = Path(args.outdir) / f"compare_{A_name}_vs_{B_name}"
    outdir.mkdir(parents=True, exist_ok=True)

    drivers: Dict[str, DriverFn] = {}
    if args.drivers_py:
        drivers.update(load_drivers_from_py(Path(args.drivers_py)))
    script_dir = Path(__file__).resolve().parent
    drivers.update(load_drivers_from_py(script_dir / "drivers.py"))
    if not drivers:
        raise RuntimeError("No drivers loaded. Provide --drivers_py or put drivers.py next to this script.")

    base_anyc_A = per_qid_any_correct_from_file(base_map[A_name])
    post_anyc_A = per_qid_any_correct_from_file(post_map[A_name])

    base_anyc_B = per_qid_any_correct_from_file(base_map[B_name])
    post_anyc_B = per_qid_any_correct_from_file(post_map[B_name])

    def build_sets(base_anyc: Dict[str, bool], post_anyc: Dict[str, bool]) -> Tuple[Set[str], Set[str]]:
        baseN = {q for q, v in base_anyc.items() if not bool(v)}
        baseN &= set(post_anyc.keys())
        n2p = {q for q in baseN if bool(post_anyc.get(q, False))}
        n2n = baseN - n2p
        return n2p, n2n

    A_n2p_qids, A_n2n_qids = build_sets(base_anyc_A, post_anyc_A)
    B_n2p_qids, B_n2n_qids = build_sets(base_anyc_B, post_anyc_B)

    A_rollouts = load_rollouts_for_driver(train_map[A_name], pad_id=args.pad_id, max_len=args.max_len)
    B_rollouts = load_rollouts_for_driver(train_map[B_name], pad_id=args.pad_id, max_len=args.max_len)

    for dname, fn in drivers.items():
        try:
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)
        except Exception:
            n_params = 1

        def collect(rollouts: List[Rollout], qset: Set[str]) -> np.ndarray:
            out = []
            for r in rollouts:
                if r.qid not in qset:
                    continue
                try:
                    if n_params >= 2:
                        v = float(fn(r.ent, r.topk_logprobs_per_token))
                    else:
                        v = float(fn(r.ent))
                except Exception:
                    continue
                if np.isfinite(v):
                    out.append(v)
            return np.asarray(out, dtype=np.float64)

        A_N2P = collect(A_rollouts, A_n2p_qids)
        A_N2N = collect(A_rollouts, A_n2n_qids)
        B_N2P = collect(B_rollouts, B_n2p_qids)
        B_N2N = collect(B_rollouts, B_n2n_qids)

        title = f"{A_name} vs {B_name} | {dname}"
        out_png = outdir / f"kde_rollout_{dname}_N2P_N2N.png"
        plot_4kde_npnn(
            A_N2P, A_N2N, B_N2P, B_N2N,
            out_png=out_png,
            title=title,
            A_name=A_name,
            B_name=B_name,
            legend_show_skew=args.legend_show_skew,
        )

        if args.save_right_skew:
            out_json = outdir / f"right_skew_rollout_{dname}.json"
            save_right_skew_metrics(
                out_json=out_json,
                A_name=A_name,
                B_name=B_name,
                A_N2P=A_N2P,
                A_N2N=A_N2N,
                B_N2P=B_N2P,
                B_N2N=B_N2N,
            )

    print(f"[OK] outdir = {outdir}")
    print(f"[OK] drivers = {sorted(list(drivers.keys()))}")
    print(f"[OK] A baseN={len(A_n2p_qids)+len(A_n2n_qids)} (N2P={len(A_n2p_qids)}, N2N={len(A_n2n_qids)})")
    print(f"[OK] B baseN={len(B_n2p_qids)+len(B_n2n_qids)} (N2P={len(B_n2p_qids)}, N2N={len(B_n2n_qids)})")


if __name__ == "__main__":
    main()