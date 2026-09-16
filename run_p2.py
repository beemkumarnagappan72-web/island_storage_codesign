"""Recompute protocol II with the tuned receding-horizon controller."""
import os, pickle, sys
import numpy as np
from config import ISLANDS, build_media, build_system_cost
import pipeline as P, scenarios as S, mpc

QTY = ("tac", "lcoe", "lpsp", "re_share", "fuel_e", "curt_e", "capex",
       "opex", "penalty", "co2")
SWEEP = ((24, 6), (48, 12), (48, 6), (48, 3), (168, 6))


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
    out, sweep = {}, {}
    for tag in ev["P2"]:
        caps = des[tag]["caps"]
        caps = {k: (np.asarray(v) if isinstance(v, (list, np.ndarray)) else v)
                for k, v in caps.items()}
        rows = [mpc.run(caps, S.single(y), media, sc, des[tag]["ref"], H=48, step=3)
                for y in te]
        out[tag] = [{q: float(r[q]) for q in QTY} for r in rows]
        print(f"[{key}] {tag}: P2* {np.mean([r['lcoe'] for r in out[tag]]):.2f} "
              f"LPSP {np.mean([r['lpsp'] for r in out[tag]]):.5f}", flush=True)
    for tag in ("PF-LP", "GA-RB", "DPC"):
        caps = des[tag]["caps"]
        sweep[tag] = {}
        for H, st in SWEEP:
            r = mpc.run(caps, S.single(te[0]), media, sc, des[tag]["ref"],
                        H=H, step=st)
            sweep[tag][f"{H}/{st}"] = dict(lcoe=float(r["lcoe"]),
                                           lpsp=float(r["lpsp"]))
    ev["P2"] = out
    ev["mpc_sweep"] = sweep
    pickle.dump(ev, open(os.path.join(P.RES, f"eval_{key}.pkl"), "wb"))
    print(f"[{key}] P2 updated", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
