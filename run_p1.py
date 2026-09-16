"""Protocol I: clairvoyant dispatch of every selected design."""
import os, pickle, sys
import numpy as np
from config import ISLANDS, build_media, build_system_cost
import pipeline as P
from lp import dispatch_lp

N = 4


def main(key):
    media, sc = build_media(), build_system_cost()
    isl = [i for i in ISLANDS if i.key == key][0]
    d, tr, te = P.island_data(isl)
    ev = pickle.load(open(os.path.join(P.RES, f"eval_{key}.pkl"), "rb"))
    out = {}
    for tag, caps in ev["caps"].items():
        c = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in caps.items()}
        rows = [dispatch_lp(c, P.lp_series(y), media, sc) for y in te[:N]]
        out[tag] = [dict(tac=float(r["tac"]), lcoe=float(r["lcoe"]),
                         lpsp=float(r["lpsp"]), re_share=float(r["re_share"]))
                    for r in rows]
        print(f"[{key}] {tag}: P1 {np.mean([r['lcoe'] for r in out[tag]]):.2f}",
              flush=True)
    ev["P1"] = out
    pickle.dump(ev, open(os.path.join(P.RES, f"eval_{key}.pkl"), "wb"))
    print(f"[{key}] P1 updated", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
