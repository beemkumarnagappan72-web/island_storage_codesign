"""Evaluation protocols II-IV with best-of-restarts selection.

Protocol I  clairvoyant dispatch (conventional, not implementable)
Protocol II receding-horizon control common to every design
Protocol III a causal duration-structured policy fitted to each design with
            its capacities held fixed, which isolates the quality of the
            capacity vector from the quality of the policy
Protocol IV each method operated by the controller it actually delivers
"""
import os, pickle, sys, time
import numpy as np
from config import ISLANDS, build_media, build_system_cost, BAND_TAU_H
import pipeline as P
import scenarios as S
from codesign import make_cfg, train, init_params
from system import evaluate, invert_cap
from heuristics import rule_dispatch
import mpc

FIT_STEPS = 400
MAIN = ("PF-LP", "SP", "GA-RB", "PSO-RB", "ADR", "DPC")
QTY = ("tac", "lcoe", "lpsp", "re_share", "fuel_e", "curt_e", "capex",
       "opex", "penalty", "co2")


def _row(r):
    return {q: float(r[q]) for q in QTY}


def best_seed(des, base, media, sc, tr):
    """Pick the restart with the lowest mean training cost."""
    tags = [t for t in des if t == base or t.startswith(base + "#")]
    best, bv = base, np.inf
    for t in tags:
        d = des[t]
        if "params" not in d:
            return base, {}
        v = np.mean([float(evaluate(d["params"], d["cfg"], media, sc, BAND_TAU_H,
                                    S.single(y), mu=0.0)["tac"]) for y in tr[:5]])
        if v < bv:
            best, bv = t, v
    return best, {t: float(np.mean([float(evaluate(des[t]["params"], des[t]["cfg"],
                 media, sc, BAND_TAU_H, S.single(y), mu=0.0)["lcoe"])
                 for y in tr[:5]])) for t in tags}


def main(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    media, sc = build_media(), build_system_cost()
    d, tr, te = P.island_data(isl)
    des = pickle.load(open(os.path.join(P.RES, f"designs_{key}.pkl"), "rb"))
    old = pickle.load(open(os.path.join(P.RES, f"main_{key}.pkl"), "rb"))
    try:
        res = pickle.load(open(os.path.join(P.RES, f"reserve_{key}.pkl"), "rb"))
        for t, v in res.items():
            des[t] = dict(caps={k: (np.asarray(x) if isinstance(x, list) else x)
                                for k, x in v["caps"].items()}, ref=v["ref"])
            old["clair"][t] = v["clair"]
    except FileNotFoundError:
        res = {}
    tags = list(MAIN) + [t for t in ("PF-LP-R", "SP-R") if t in des]

    sel = {}
    for base in ("ADR", "DPC"):
        b, scores = best_seed(des, base, media, sc, tr)
        sel[base] = dict(best=b, scores=scores)
        des[base] = des[b]
        print(f"[{key}] {base}: restart {b} selected  {scores}", flush=True)

    out = {"island": key, "sel": sel, "caps": {}, "P1": old["clair"],
           "P2": {}, "P3": {}, "P4": {}, "seeds": {}, "secs": {}}
    trs = S.stack(tr)
    for tag in tags:
        caps = des[tag]["caps"]
        caps = {k: (np.asarray(v) if isinstance(v, (list, np.ndarray)) else v)
                for k, v in caps.items()}
        out["caps"][tag] = P.caps_to_json(caps)
        ref = des[tag]["ref"]
        t0 = time.time()
        out["P2"][tag] = [_row(mpc.run(caps, S.single(y), media, sc, ref)) for y in te]

        cfg = make_cfg(isl, media, len(BAND_TAU_H) + 1, d)
        p0 = init_params(0, media, cfg)
        p0["cap"] = invert_cap(caps, media, cfg)
        pf, _ = train(trs, media, sc, BAND_TAU_H, cfg, steps=FIT_STEPS, batch=4,
                      lr=0.06, seed=0, params0=p0, freeze=("cap",))
        out["P3"][tag] = [_row(evaluate(pf, cfg, media, sc, BAND_TAU_H,
                                        S.single(y), mu=0.0)) for y in te]
        if tag in ("ADR", "DPC"):
            dd = des[tag]
            out["P4"][tag] = [_row(evaluate(dd["params"], dd["cfg"], media, sc,
                                            BAND_TAU_H, S.single(y), mu=0.0))
                              for y in te]
        elif tag in ("GA-RB", "PSO-RB"):
            out["P4"][tag] = [_row(rule_dispatch(caps, S.single(y), media, sc))
                              for y in te]
        else:
            out["P4"][tag] = out["P2"][tag]
        out["secs"][tag] = des[tag].get("secs", np.nan)
        print(f"[{key}] {tag}: P2 {np.mean([r['lcoe'] for r in out['P2'][tag]]):.2f} "
              f"(LPSP {np.mean([r['lpsp'] for r in out['P2'][tag]]):.5f})  "
              f"P3 {np.mean([r['lcoe'] for r in out['P3'][tag]]):.2f}  "
              f"P4 {np.mean([r['lcoe'] for r in out['P4'][tag]]):.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    for base in ("ADR", "DPC"):
        out["seeds"][base] = {t: [_row(evaluate(des[t]["params"], des[t]["cfg"],
                              media, sc, BAND_TAU_H, S.single(y), mu=0.0))
                              for y in te]
                              for t in des if t == base or t.startswith(base + "#")}
    if "DPC" in des and "params" in des["DPC"]:
        from codesign import design_summary
        ds = design_summary(des["DPC"]["params"], des["DPC"]["cfg"], media)
        out["alpha"] = np.asarray(ds["alpha"]).tolist()
        out["lam"] = float(ds["lam"])
        out["beta"] = np.asarray(ds["beta"]).tolist()
    pickle.dump(out, open(os.path.join(P.RES, f"eval_{key}.pkl"), "wb"))
    print(f"[{key}] eval written", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
