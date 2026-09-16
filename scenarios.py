"""Scenario years produced by a seasonally anchored moving-block bootstrap.

A single observed year supplies the chronologies.  Scenario years are
assembled from weekly blocks drawn from a neighbourhood of the same
calendar position, so that seasonality and the joint structure of demand,
irradiance, wind and their day-ahead forecasts are preserved while the
weather realisation is resampled.  Odd and even calendar weeks form two
disjoint pools, which keeps the years used for design strictly separate
from the years used for evaluation.
"""
import numpy as np

BLOCK = 168          # hours per block
N_BLOCK = 52
TAIL = 8760 - BLOCK * N_BLOCK
KEYS = ("load", "pv_cf", "wind_cf", "pv_cf_fc", "wind_cf_fc", "load_fc", "cool")
NEIGH = 2


def _pool(parity):
    return [w for w in range(N_BLOCK) if w % 2 == parity]


def draw_year(d, rng, parity, neigh=NEIGH):
    pool = set(_pool(parity))
    idx = []
    for w in range(N_BLOCK):
        cand = [(w + k) % N_BLOCK for k in range(-neigh, neigh + 1)]
        cand = [c for c in cand if c in pool] or sorted(pool)
        src = cand[rng.integers(len(cand))]
        idx.append(np.arange(src * BLOCK, (src + 1) * BLOCK))
    src = sorted(pool)[rng.integers(len(pool))]
    idx.append(np.arange(src * BLOCK, src * BLOCK + TAIL))
    idx = np.concatenate(idx)
    return {k: d[k][idx].astype(np.float32) for k in KEYS}


def make_set(d, n, seed, parity):
    rng = np.random.default_rng(seed)
    return [draw_year(d, rng, parity) for _ in range(n)]


def stack(years):
    """Stack a list of scenario years into the tuple the simulator expects."""
    return tuple(np.stack([y[k] for y in years]) for k in
                 ("load", "pv_cf", "wind_cf", "pv_cf_fc", "wind_cf_fc", "load_fc", "cool"))


def nominal(d):
    return tuple(np.asarray(d[k][:8760], dtype=np.float32)[None, :] for k in
                 ("load", "pv_cf", "wind_cf", "pv_cf_fc", "wind_cf_fc", "load_fc", "cool"))


def single(y):
    return tuple(np.asarray(y[k], dtype=np.float32) for k in
                 ("load", "pv_cf", "wind_cf", "pv_cf_fc", "wind_cf_fc", "load_fc", "cool"))


def perturb(years, *, fc_noise=0.0, load_growth=0.0, wind_scale=1.0,
            outage=None, seed=0):
    """Robustness transformations applied to evaluation years only."""
    rng = np.random.default_rng(seed)
    out = []
    for y in years:
        z = {k: y[k].copy() for k in KEYS}
        if load_growth:
            z["load"] *= (1.0 + load_growth)
            z["load_fc"] *= (1.0 + load_growth)
            z["cool"] *= (1.0 + load_growth)
        if wind_scale != 1.0:
            z["wind_cf"] = np.clip(z["wind_cf"] * wind_scale, 0.0, 1.0)
            z["wind_cf_fc"] = np.clip(z["wind_cf_fc"] * wind_scale, 0.0, 1.0)
        if fc_noise:
            for a, b, hi in (("pv_cf_fc", "pv_cf", 1.0), ("wind_cf_fc", "wind_cf", 1.0)):
                err = z[a] - z[b]
                z[a] = np.clip(z[b] + err * (1.0 + fc_noise), 0.0, hi)
            err = z["load_fc"] - z["load"]
            z["load_fc"] = np.maximum(z["load"] + err * (1.0 + fc_noise), 0.0)
        if outage is not None:
            start, dur = outage
            z["wind_cf"][start:start + dur] = 0.0
            z["wind_cf_fc"][start:start + dur] = 0.0
            z["pv_cf"][start:start + dur] *= 0.35
            z["pv_cf_fc"][start:start + dur] *= 0.35
        out.append({k: v.astype(np.float32) for k, v in z.items()})
    return out
