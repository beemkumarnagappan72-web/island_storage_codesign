"""Stages E2-E5 and the foresight decomposition."""
import copy, os, pickle, sys, time
import numpy as np
from config import (ISLANDS, build_media, build_system_cost, BAND_TAU_H,
                    energy_caps, build_media as BM)
import config as C
import pipeline as P
import scenarios as S
from codesign import make_cfg, train
from system import evaluate
from lp import size_lp, dispatch_lp
import mpc

STEPS = 600
ABL_STEPS = 450


def _design(isl, media, sc, tr, d, taus=BAND_TAU_H, seed=0, **cfgmod):
    cfg = make_cfg(isl, media, len(taus) + 1, d)
    cfg.update({k: v for k, v in cfgmod.items()
                if k not in ("rho", "mu0", "mu1", "steps")})
    kw = dict(steps=cfgmod.get("steps", STEPS), batch=4, lr=0.06, seed=seed)
    for k in ("rho", "mu0", "mu1"):
        if k in cfgmod:
            kw[k] = cfgmod[k]
    p, hist = train(S.stack(tr), media, sc, taus, cfg, **kw)
    caps, ds = P.caps_from_params(p, cfg, media)
    return p, cfg, caps, ds, taus


def _native_lcoe(p, cfg, media, sc, taus, years):
    return [float(evaluate(p, cfg, media, sc, taus, S.single(y), mu=0.0)["lcoe"])
            for y in years]


# ---------------------------------------------------------------- E2
VARIANTS = {
    "Full": {},
    "NoBand": dict(feat="lag", alloc="free", n_lag=25),
    "NoSimplex": dict(alloc="free"),
    "NoRestore": dict(use_restore=0.0),
    "NoSeasonal": dict(seasonal=False),
    "NoRisk": dict(rho=0.0),
    "NoAnneal": dict(mu0=0.01, mu1=0.01),
    "NoForecast": dict(use_lam=0.0),
    "SingleBand": dict(_taus=(24.0,)),
}


def ablation(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    media, sc = build_media(), build_system_cost()
    d, tr, te = P.island_data(isl)
    out = {}
    for name, mod in VARIANTS.items():
        mod = dict(mod)
        taus = mod.pop("_taus", BAND_TAU_H)
        t0 = time.time()
        mod.setdefault("steps", ABL_STEPS)
        p, cfg, caps, ds, taus = _design(isl, media, sc, tr, d, taus=taus, **mod)
        rows = [evaluate(p, cfg, media, sc, taus, S.single(y), mu=0.0) for y in te]
        out[name] = dict(caps=P.caps_to_json(caps),
                         native=[float(r["lcoe"]) for r in rows],
                         lpsp=[float(r["lpsp"]) for r in rows],
                         re=[float(r["re_share"]) for r in rows],
                         alpha=np.asarray(ds["alpha"]).tolist(), secs=time.time() - t0)
        print(f"[{key}] ablation {name}: native "
              f"{np.mean(out[name]['native']):.2f}", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"ablation_{key}.pkl"), "wb"))


# ---------------------------------------------------------------- foresight
def foresight(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    media, sc = build_media(), build_system_cost()
    d, tr, te = P.island_data(isl)
    ec = energy_caps(media, isl, d)
    ser_tr = [P.lp_series(y) for y in tr[:1]]
    out = {}
    base = [m for m in media]
    zero = [copy.replace(m, c_energy=1e12) if False else m for m in media]

    # storage-free reference: perfect foresight and causal coincide
    caps0, _ = size_lp(ser_tr, media, sc, e_cap=np.zeros(len(media)))
    c0 = np.mean([dispatch_lp(caps0, P.lp_series(y), media, sc)["lcoe"] for y in te[:3]])
    out["none"] = dict(lcoe_pf=float(c0), lcoe_ca=float(c0))

    for j, m in enumerate(media):
        cap = np.zeros(len(media))
        cap[j] = ec[j] if np.isfinite(ec[j]) else 1e6
        caps, _ = size_lp(ser_tr, media, sc, e_cap=cap)
        pf = np.mean([dispatch_lp(caps, P.lp_series(y), media, sc)["lcoe"] for y in te[:3]])
        cfg_cap = np.array([ec[k] if k == j else 0.0 for k in range(len(media))],
                           dtype=float)
        cfg = make_cfg(isl, media, len(BAND_TAU_H) + 1, d)
        cfg["e_cap"] = np.where(np.arange(len(media)) == j,
                                np.where(np.isfinite(ec), ec, 1e6), 1e-6)
        p, hist = train(S.stack(tr), media, sc, BAND_TAU_H, cfg, steps=STEPS,
                        batch=4, lr=0.06, seed=0)
        ca = np.mean(_native_lcoe(p, cfg, media, sc, BAND_TAU_H, te[:3]))
        capsc, dsc = P.caps_from_params(p, cfg, media)
        dur_pf = caps["E"][j] / max(caps["Pd"][j], 1e-6)
        out[m.key] = dict(lcoe_pf=float(pf), lcoe_ca=float(ca),
                          E_pf=float(caps["E"][j]), P_pf=float(caps["Pd"][j]),
                          duration_pf=float(dur_pf),
                          E_ca=float(capsc["E"][j]), P_ca=float(capsc["Pd"][j]),
                          kappa=float((c0 - ca) / max(c0 - pf, 1e-9)))
        print(f"[{key}] foresight {m.key}: pf {pf:.2f} ca {ca:.2f} "
              f"kappa {out[m.key]['kappa']:.3f} dur {dur_pf:.1f} h", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"foresight_{key}.pkl"), "wb"))


# ---------------------------------------------------------------- E3
def sensitivity(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    d, tr, te = P.island_data(isl)
    out = {}
    grid = ([("fuel", f) for f in (1.0, 1.5, 2.0, 2.5, 3.0)]
            + [("voll", v) for v in (1000.0, 10000.0)]
            + [("rate", r) for r in (0.05, 0.09)]
            + [("windcap", w) for w in (0.7, 1.3)]
            + [("rho", r) for r in (0.0, 0.5)]
            + [("nodiesel", 0.0)])
    for kind, val in grid:
        rate = val if kind == "rate" else C.DISCOUNT_RATE
        media = build_media(rate)
        sc = build_system_cost(rate, val if kind == "fuel" else C.FUEL_DELIVERY_MULT,
                               val if kind == "voll" else C.VOLL)
        if kind == "windcap":
            sc = build_system_cost(rate)
            sc.c_wind *= val
        mod = dict(rho=val) if kind == "rho" else {}
        if kind == "nodiesel":
            mod = dict(dsl_scale=0.0)
        p, cfg, caps, ds, taus = _design(isl, media, sc, tr, d, **mod)
        lc = _native_lcoe(p, cfg, media, sc, taus, te[:4])
        out[f"{kind}={val}"] = dict(caps=P.caps_to_json(caps), lcoe=lc,
                                    alpha=np.asarray(ds["alpha"]).tolist())
        print(f"[{key}] sens {kind}={val}: LCOE {np.mean(lc):.2f} "
              f"E={np.round(caps['E'],1)}", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"sens_{key}.pkl"), "wb"))


# ---------------------------------------------------------------- E4
PERTURB = {
    "nominal": {},
    "fc+50%": dict(fc_noise=0.5),
    "fc+100%": dict(fc_noise=1.0),
    "load+15%": dict(load_growth=0.15),
    "wind-20%": dict(wind_scale=0.8),
    "outage10d": dict(outage=(4200, 240)),
}


def robust(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    media, sc = build_media(), build_system_cost()
    d, tr, te = P.island_data(isl)
    des = pickle.load(open(os.path.join(P.RES, f"designs_{key}.pkl"), "rb"))
    out = {}
    for tag in ("PF-LP", "GA-RB", "ADR", "DPC"):
        if tag not in des:
            continue
        caps, ref = des[tag]["caps"], des[tag]["ref"]
        out[tag] = {}
        for pname, kw in PERTURB.items():
            yrs = S.perturb(te[:3], seed=5, **kw) if kw else te[:3]
            rows = P.eval_mpc(caps, yrs, media, sc, ref)
            out[tag][pname] = dict(lcoe=[r["lcoe"] for r in rows],
                                   lpsp=[r["lpsp"] for r in rows])
            print(f"[{key}] robust {tag} {pname}: "
                  f"{np.mean(out[tag][pname]['lcoe']):.2f} "
                  f"LPSP {np.mean(out[tag][pname]['lpsp']):.5f}", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"robust_{key}.pkl"), "wb"))


# ---------------------------------------------------------------- E5
def transfer(key):
    """Policy trained on one island, capacities re-optimised on another."""
    media, sc = build_media(), build_system_cost()
    src = [i for i in ISLANDS if i.key == key][0]
    d_s, tr_s, _ = P.island_data(src)
    p_s, cfg_s, _, _, taus = _design(src, media, sc, tr_s, d_s)
    out = {}
    for tgt in ISLANDS:
        d_t, tr_t, te_t = P.island_data(tgt)
        cfg = make_cfg(tgt, media, len(taus) + 1, d_t)
        p0 = {"cap": __import__("codesign").init_params(0, media, cfg)["cap"],
              "pol": p_s["pol"], "zeta": p_s["zeta"]}
        p, _ = train(S.stack(tr_t), media, sc, taus, cfg, steps=STEPS, batch=4,
                     lr=0.06, params0=p0, freeze=("logits", "beta", "lam", "seas"))
        lc_t = _native_lcoe(p, cfg, media, sc, taus, te_t[:4])
        p_o, cfg_o, _, _, _ = _design(tgt, media, sc, tr_t, d_t)
        lc_o = _native_lcoe(p_o, cfg_o, media, sc, taus, te_t[:4])
        out[tgt.key] = dict(transferred=lc_t, native=lc_o)
        print(f"[{key}->{tgt.key}] transferred {np.mean(lc_t):.2f} "
              f"native {np.mean(lc_o):.2f}", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"transfer_{key}.pkl"), "wb"))


if __name__ == "__main__":
    {"ablation": ablation, "foresight": foresight, "sensitivity": sensitivity,
     "robust": robust, "transfer": transfer}[sys.argv[1]](sys.argv[2])
