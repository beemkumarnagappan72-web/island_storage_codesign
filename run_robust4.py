"""Robustness under Protocol IV: every design is perturbed while being
operated by the controller its own planning method delivers."""
import os, pickle, sys
import numpy as np
from config import ISLANDS, build_media, build_system_cost, BAND_TAU_H
import pipeline as P
import scenarios as S
from system import evaluate
from heuristics import rule_dispatch
import mpc

PERTURB = {
    "nominal": {},
    "fc+50%": dict(fc_noise=0.5),
    "fc+100%": dict(fc_noise=1.0),
    "load+15%": dict(load_growth=0.15),
    "wind-20%": dict(wind_scale=0.8),
    "outage10d": dict(outage=(4200, 240)),
}
TAGS = ("PF-LP", "SP", "PF-LP-R", "GA-RB", "PSO-RB", "ADR", "DPC")
N = 4


def main(key):
    media, sc = build_media(), build_system_cost()
    isl = [i for i in ISLANDS if i.key == key][0]
    d, tr, te = P.island_data(isl)
    des = pickle.load(open(os.path.join(P.RES, f"designs_{key}.pkl"), "rb"))
    res = pickle.load(open(os.path.join(P.RES, f"reserve_{key}.pkl"), "rb"))
    ev = pickle.load(open(os.path.join(P.RES, f"eval_{key}.pkl"), "rb"))
    for t, v in res.items():
        des[t] = dict(caps={k: (np.asarray(x) if isinstance(x, list) else x)
                            for k, x in v["caps"].items()}, ref=v["ref"])
    for base in ("ADR", "DPC"):
        des[base] = des[ev["sel"][base]["best"]]

    out = {}
    for tag in TAGS:
        if tag not in des:
            continue
        caps = des[tag]["caps"]
        out[tag] = {}
        for pname, kw in PERTURB.items():
            yrs = S.perturb(te[:N], seed=5, **kw) if kw else te[:N]
            rows = []
            for y in yrs:
                if tag in ("ADR", "DPC"):
                    r = evaluate(des[tag]["params"], des[tag]["cfg"], media, sc,
                                 BAND_TAU_H, S.single(y), mu=0.0)
                    rows.append((float(r["lcoe"]), float(r["lpsp"])))
                elif tag in ("GA-RB", "PSO-RB"):
                    r = rule_dispatch(caps, S.single(y), media, sc)
                    rows.append((float(r["lcoe"]), float(r["lpsp"])))
                else:
                    r = mpc.run(caps, S.single(y), media, sc, des[tag]["ref"],
                                H=48, step=3)
                    rows.append((float(r["lcoe"]), float(r["lpsp"])))
            out[tag][pname] = dict(lcoe=[a for a, _ in rows],
                                   lpsp=[b for _, b in rows])
            print(f"[{key}] {tag} {pname}: "
                  f"{np.mean(out[tag][pname]['lcoe']):.2f} "
                  f"LPSP {np.mean(out[tag][pname]['lpsp']):.5f}", flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"robust4_{key}.pkl"), "wb"))


if __name__ == "__main__":
    main(sys.argv[1])
