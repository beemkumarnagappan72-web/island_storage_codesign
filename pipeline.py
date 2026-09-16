"""Design construction and evaluation protocols."""
import json
import os
import time
import numpy as np

from config import (ISLANDS, build_media, build_system_cost, BAND_TAU_H,
                    energy_caps)
from data_loader import build_island
import scenarios as S
from codesign import make_cfg, train, design_summary
from system import evaluate
from lp import size_lp, dispatch_lp
from heuristics import rule_dispatch, ga_size, pso_size
import mpc

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)

N_TRAIN, N_TEST = 10, 8
TRAIN_SEED, TEST_SEED = 11, 23


def island_data(isl):
    d = build_island(isl)
    tr = S.make_set(d, N_TRAIN, seed=TRAIN_SEED, parity=1)
    te = S.make_set(d, N_TEST, seed=TEST_SEED, parity=0)
    return d, tr, te


def lp_series(y):
    return (np.asarray(y["load"], float), np.asarray(y["pv_cf"], float),
            np.asarray(y["wind_cf"], float), np.asarray(y["cool"], float))


def caps_from_params(p, cfg, media):
    ds = design_summary(p, cfg, media)
    n = len(media)
    return dict(pv=ds["pv"], wind=ds["wind"], diesel=ds["diesel"],
                E=np.array([ds["E"][m.key] for m in media]),
                Pc=np.array([ds["Pc"][m.key] for m in media]),
                Pd=np.array([ds["Pd"][m.key] for m in media])), ds


def caps_to_json(c):
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in c.items()}


# ----------------------------------------------------------------------
def build_designs(isl, media, sc, tr, d, seeds=(0,), quick=False,
                  which=("PF-LP", "SP", "GA-RB", "PSO-RB", "ADR", "DPC")):
    ec = energy_caps(media, isl, d)
    out = {}
    ser_lp = [lp_series(y) for y in tr]
    yrs = [S.single(y) for y in tr]
    steps = 120 if quick else 600

    if "PF-LP" in which:
        t0 = time.time()
        caps, _ = size_lp(ser_lp[:1], media, sc, e_cap=ec)
        out["PF-LP"] = dict(caps=caps, secs=time.time() - t0)
    if "SP" in which:
        t0 = time.time()
        caps, _ = size_lp(ser_lp[:2], media, sc, e_cap=ec)
        out["SP"] = dict(caps=caps, secs=time.time() - t0)
    if "GA-RB" in which:
        t0 = time.time()
        caps, _ = ga_size(yrs[:3], media, sc, isl, ec, pop=40, gens=50, seed=0)
        out["GA-RB"] = dict(caps=caps, secs=time.time() - t0)
    if "PSO-RB" in which:
        t0 = time.time()
        caps, _ = pso_size(yrs[:3], media, sc, isl, ec, swarm=30, iters=60, seed=0)
        out["PSO-RB"] = dict(caps=caps, secs=time.time() - t0)

    trs = S.stack(tr)
    for tag, cfgmod in (("ADR", dict(feat="lag", alloc="free", n_lag=25)),
                        ("DPC", dict())):
        if tag not in which:
            continue
        for sd in seeds:
            cfg = make_cfg(isl, media, len(BAND_TAU_H) + 1, d)
            cfg.update(cfgmod)
            t0 = time.time()
            p, hist = train(trs, media, sc, BAND_TAU_H, cfg, seed=sd, steps=steps,
                            batch=4, lr=0.06)
            caps, ds = caps_from_params(p, cfg, media)
            key = tag if sd == seeds[0] else f"{tag}#s{sd}"
            out[key] = dict(caps=caps, secs=time.time() - t0, params=p, cfg=cfg,
                            summary=ds, hist=hist)
    return out


def ref_trajectory(caps, series_lp, media, sc):
    r = dispatch_lp(caps, series_lp, media, sc)
    E = np.maximum(np.asarray(caps["E"], float), 1e-9)
    frac = np.clip(r["soc"] / E[:, None], 0.0, 1.0)
    return frac, r


def eval_mpc(caps, test_years, media, sc, ref_frac):
    rows = []
    for k, y in enumerate(test_years):
        r = mpc.run(caps, S.single(y), media, sc, ref_frac)
        rows.append({q: float(r[q]) for q in
                     ("tac", "lcoe", "lpsp", "re_share", "fuel_e", "curt_e",
                      "capex", "opex", "penalty", "co2")})
    return rows


def eval_native(tag, caps, test_years, media, sc, design, ref_frac):
    rows = []
    for y in test_years:
        if tag.startswith(("ADR", "DPC")):
            r = evaluate(design["params"], design["cfg"], media, sc, BAND_TAU_H,
                         S.single(y), mu=0.0)
            rows.append({q: float(r[q]) for q in
                         ("tac", "lcoe", "lpsp", "re_share", "fuel_e", "curt_e",
                          "capex", "opex", "penalty", "co2")})
        elif tag.startswith(("GA", "PSO")):
            r = rule_dispatch(caps, S.single(y), media, sc)
            rows.append({q: float(r[q]) for q in
                         ("tac", "lcoe", "lpsp", "re_share", "fuel_e", "curt_e",
                          "capex", "opex", "penalty", "co2")})
        else:
            r = mpc.run(caps, S.single(y), media, sc, ref_frac)
            rows.append({q: float(r[q]) for q in
                         ("tac", "lcoe", "lpsp", "re_share", "fuel_e", "curt_e",
                          "capex", "opex", "penalty", "co2")})
    return rows


def eval_clairvoyant(caps, test_years, media, sc, n=3):
    rows = []
    for y in test_years[:n]:
        r = dispatch_lp(caps, lp_series(y), media, sc)
        rows.append(dict(tac=float(r["tac"]), lcoe=float(r["lcoe"]),
                         lpsp=float(r["lpsp"]), re_share=float(r["re_share"])))
    return rows
