"""How much of the clairvoyant value of storage survives the loss of
foresight.

Two controlled sweeps are performed on a fixed surrounding system.  The
first fixes the power rating of a single generic store and sweeps its
energy capacity, so that duration varies and nothing else does.  The
second fixes the store and sweeps the rating of the dispatchable unit, so
that what varies is the share of peak demand that has to be met by
energy-limited assets.  In both sweeps the operating cost is measured
twice, once under clairvoyant dispatch of the whole year and once under a
causal policy fitted to the same asset, and the ratio of the two savings
is attributable to foresight alone.
"""
import os, pickle, sys, time
import numpy as np
from config import (ISLANDS, MediumSpec, build_system_cost, BAND_TAU_H, TECH,
                    DISCOUNT_RATE, _ann)
import pipeline as P
import scenarios as S
from codesign import make_cfg, train, init_params
from system import evaluate, invert_cap
from lp import dispatch_lp

DURATIONS = (6.0, 24.0, 72.0, 168.0)
RELIANCE = (0.0, 0.25, 0.50, 0.70, 0.85, 1.00)
POWER_FRAC = 0.45
STORE_HOURS = 12.0
N_EVAL, STEPS = 2, 400


def generic():
    eta = 0.922 ** 0.5
    return MediumSpec(name="generic store", key="G",
                      c_energy=_ann(TECH["battery storage"]["investment"], 0.0,
                                    25.0, DISCOUNT_RATE),
                      c_power_ch=_ann(TECH["battery inverter"]["investment"],
                                      TECH["battery inverter"]["FOM"], 10.0,
                                      DISCOUNT_RATE),
                      c_power_dis=0.0, eta_c=eta, eta_d=eta,
                      self_discharge=1.0e-5, c_throughput=2.0)


def main(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    sc = build_system_cost()
    media = [generic()]
    d, tr, te = P.island_data(isl)
    base = pickle.load(open(os.path.join(P.RES, f"designs_{key}.pkl"), "rb"))
    pv = float(base["PF-LP"]["caps"]["pv"])
    wd = float(base["PF-LP"]["caps"]["wind"])
    peak = float(np.max([y["load"].max() for y in te]))
    pw = POWER_FRAC * peak
    cfg = make_cfg(isl, media, len(BAND_TAU_H) + 1, d)
    cfg["e_cap"] = np.array([np.inf])

    def caps_of(E, g):
        return dict(pv=pv, wind=wd, diesel=g, E=np.array([E]),
                    Pc=np.array([pw]), Pd=np.array([pw]))

    def E_of(c):
        return np.asarray(c["E"], dtype=float)

    def causal(E, g):
        c = caps_of(E, g)
        p0 = init_params(0, media, cfg)
        p0["cap"] = invert_cap(c, media, cfg)
        if E > 0:
            p0, _ = train(S.stack(tr), media, sc, BAND_TAU_H, cfg, steps=STEPS,
                          batch=4, lr=0.06, seed=0, params0=p0, freeze=("cap",))
        rows = [evaluate(p0, cfg, media, sc, BAND_TAU_H, S.single(y), mu=0.0,
                         return_traj=True) for y in te[:N_EVAL]]
        # the clairvoyant programme is cyclic in the state of charge, so the
        # causal run is charged for any net energy withdrawn over the year
        vals = []
        for r in rows:
            soc_end = np.asarray(r["soc"])[-1]
            draw = float(np.sum(np.maximum(0.5 - soc_end, 0.0) * np.asarray(E_of(c))))
            vals.append(float(r["opex"]) + float(r["penalty"]) + draw * sc.voll)
        return float(np.mean(vals))

    def clair(E, g):
        rows = [dispatch_lp(caps_of(E, g), P.lp_series(y), media, sc)
                for y in te[:N_EVAL]]
        return float(np.mean([r["opex"] for r in rows]))

    out = {"pv": pv, "wind": wd, "power": pw, "peak": peak,
           "store_hours": STORE_HOURS, "duration": {}, "reliance": {}}

    z_pf, z_ca = clair(0.0, 0.0), causal(0.0, 0.0)
    out["duration"]["none"] = dict(op_pf=z_pf, op_ca=z_ca)
    print(f"[{key}] duration sweep, no dispatchable unit: reference opex "
          f"{z_pf:.0f} / {z_ca:.0f}", flush=True)
    for dur in DURATIONS:
        t0 = time.time()
        E = pw * dur
        a, b = clair(E, 0.0), causal(E, 0.0)
        v_pf, v_ca = z_pf - a, z_ca - b
        out["duration"][dur] = dict(E=float(E), op_pf=a, op_ca=b,
                                    v_pf=float(v_pf), v_ca=float(v_ca),
                                    kappa=float(v_ca / max(v_pf, 1e-9)),
                                    secs=time.time() - t0)
        print(f"[{key}] dur {dur:6.1f} h E {E:8.1f}  value pf {v_pf:12.0f} "
              f"ca {v_ca:12.0f}  kappa {v_ca/max(v_pf,1e-9):6.3f}", flush=True)

    E0 = pw * STORE_HOURS
    for psi in RELIANCE:
        t0 = time.time()
        g = (1.0 - psi) * peak
        a0, b0 = clair(0.0, g), causal(0.0, g)
        a1, b1 = clair(E0, g), causal(E0, g)
        v_pf, v_ca = a0 - a1, b0 - b1
        out["reliance"][psi] = dict(diesel=float(g), op_pf0=a0, op_pf1=a1,
                                    op_ca0=b0, op_ca1=b1, v_pf=float(v_pf),
                                    v_ca=float(v_ca),
                                    kappa=float(v_ca / max(v_pf, 1e-9)),
                                    secs=time.time() - t0)
        print(f"[{key}] reliance {psi:4.2f} (G {g:5.2f} MW)  value pf "
              f"{v_pf:12.0f} ca {v_ca:12.0f}  kappa "
              f"{v_ca/max(v_pf,1e-9):6.3f}", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"foresight_{key}.pkl"), "wb"))


if __name__ == "__main__":
    main(sys.argv[1])
