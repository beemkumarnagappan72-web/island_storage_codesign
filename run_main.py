"""Stage E1: designs from every planning method and their evaluation
under the three operating protocols."""
import os, pickle, sys, time
import numpy as np
from config import ISLANDS, build_media, build_system_cost
import pipeline as P
import scenarios as S

DPC_SEEDS = (0, 1, 2, 3, 4)
ADR_SEEDS = (0, 1, 2)


def main(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    media, sc = P.build_media(), P.build_system_cost()
    d, tr, te = P.island_data(isl)
    t_start = time.time()
    out = {"island": key, "designs": {}, "mpc": {}, "native": {}, "clair": {},
           "secs": {}}

    des = P.build_designs(isl, media, sc, tr, d, which=("PF-LP",))
    des.update(P.build_designs(isl, media, sc, tr, d, which=("SP",)))
    des.update(P.build_designs(isl, media, sc, tr[:2], d, which=("GA-RB", "PSO-RB")))
    des.update(P.build_designs(isl, media, sc, tr, d, which=("ADR",), seeds=ADR_SEEDS))
    des.update(P.build_designs(isl, media, sc, tr, d, which=("DPC",), seeds=DPC_SEEDS))
    print(f"[{key}] designs done in {time.time()-t_start:.0f}s", flush=True)

    for tag, v in des.items():
        caps = v["caps"]
        ref, rlp = P.ref_trajectory(caps, P.lp_series(tr[0]), media, sc)
        v["ref"] = ref
        out["designs"][tag] = dict(caps=P.caps_to_json(caps), secs=v["secs"])
        if "summary" in v:
            s = v["summary"]
            out["designs"][tag]["alpha"] = np.asarray(s["alpha"]).tolist()
            out["designs"][tag]["lam"] = float(s["lam"])
            out["designs"][tag]["beta"] = np.asarray(s["beta"]).tolist()
        out["mpc"][tag] = P.eval_mpc(caps, te, media, sc, ref)
        out["native"][tag] = P.eval_native(tag, caps, te, media, sc, v, ref)
        print(f"[{key}] {tag}: MPC LCOE "
              f"{np.mean([r['lcoe'] for r in out['mpc'][tag]]):.2f}  native "
              f"{np.mean([r['lcoe'] for r in out['native'][tag]]):.2f}", flush=True)
    for tag in ("PF-LP", "SP", "GA-RB", "PSO-RB", "ADR", "DPC"):
        if tag in des:
            out["clair"][tag] = P.eval_clairvoyant(des[tag]["caps"], te, media, sc, n=3)
            print(f"[{key}] {tag}: clairvoyant "
                  f"{np.mean([r['lcoe'] for r in out['clair'][tag]]):.2f}", flush=True)

    with open(os.path.join(P.RES, f"main_{key}.pkl"), "wb") as f:
        pickle.dump(out, f)
    with open(os.path.join(P.RES, f"designs_{key}.pkl"), "wb") as f:
        pickle.dump({k: {q: v[q] for q in ("caps", "ref", "secs")
                         if q in v} | ({"params": v["params"], "cfg": v["cfg"]}
                                       if "params" in v else {})
                     for k, v in des.items()}, f)
    print(f"[{key}] TOTAL {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
