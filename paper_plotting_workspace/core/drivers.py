import numpy as np
import math
import math
import numpy as np


def driver_early_mean_ppl_10part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.5 * v.size)))
    vv = np.minimum(10.0, v[:L])
    return float(np.mean(np.exp(vv))/np.mean(np.exp(v)))

def driver_early_mean_ppl_20part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.2 * v.size)))
    vv = np.minimum(50.0, v[:L])
    return float(np.mean(np.exp(vv))/np.mean(np.exp(v)))

def driver_early_mean_ppl_30part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.3 * v.size)))
    vv = np.minimum(50.0, v[:L])
    return float(np.mean(np.exp(vv))/np.mean(np.exp(v)))

def driver_early_mean_ppl_40part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.4 * v.size)))
    vv = np.minimum(50.0, v[:L])
    return float(np.mean(np.exp(vv))/np.mean(np.exp(v)))

def driver_early_mean_ppl_50part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.5 * v.size)))
    vv = np.minimum(50.0, v[:L])
    return float(np.mean(np.exp(vv))/np.mean(np.exp(v)))

def driver_early_mean_entropy_10part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.2 * v.size)))
    return float(np.mean(v[:L])/np.mean(v))

def driver_early_mean_entropy_20part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.2 * v.size)))
    return float(np.mean(v[:L])/np.mean(v))

def driver_early_mean_entropy_30part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.2 * v.size)))
    return float(np.mean(v[:L])/np.mean(v))
    


### new
def driver_late_mean_entropy_20part(ent: np.ndarray) -> float:
    v = ent[np.isfinite(ent)]
    if v.size == 0:
        return float("nan")
    L = int(max(1, math.ceil(0.2 * v.size)))
    return float(np.mean(v[-L:])/np.mean(v))
# def driver_slope(ent: np.ndarray) -> float:
#     pass