"""Reserve-constrained variants of the foresight-based planners.

A planning reserve margin on firm discharge capability is standard
practice for autonomous systems, and imposing it removes the objection
that the perfect-foresight designs fail only because they carry no
margin at all.
"""
import os, pickle, sys, time
import numpy as np
from config import ISLANDS, build_media, build_system_cost, energy_caps
import pipeline as P

MARGIN = 0.10


def main(key):
    isl = [i for i in ISLANDS if i.key == key][0]
    media, sc = build_media(), build_system_cost()
    d, tr, te = P.island_data(isl)
    ec = energy_caps(media, isl, d)
    peak = float(np.max([y["load"].max() for y in tr]))
    ser = [P.lp_series(y) for y in tr]
    from lp import size_lp
    out = {}
    for tag, n_sc in (("PF-LP-R", 1), ("SP-R", 2)):
        t0 = time.time()
        caps, _ = size_lp(ser[:n_sc], media, sc, e_cap=ec, reserve=(MARGIN, peak))
        ref, _ = P.ref_trajectory(caps, ser[0], media, sc)
        rows = P.eval_mpc(caps, te, media, sc, ref)
        out[tag] = dict(caps=P.caps_to_json(caps), ref=ref,
                        mpc=[{q: float(r[q]) for q in
                              ("tac", "lcoe", "lpsp", "re_share", "fuel_e",
                               "curt_e", "capex", "opex", "penalty", "co2")}
                             for r in rows],
                        clair=P.eval_clairvoyant(caps, te, media, sc, n=3),
                        secs=time.time() - t0)
        print(f"[{key}] {tag}: MPC {np.mean([r['lcoe'] for r in rows]):.2f} "
              f"LPSP {np.mean([r['lpsp'] for r in rows]):.5f} "
              f"clair {np.mean([r['lcoe'] for r in out[tag]['clair']]):.2f}",
              flush=True)
    pickle.dump(out, open(os.path.join(P.RES, f"reserve_{key}.pkl"), "wb"))


if __name__ == "__main__":
    main(sys.argv[1])
