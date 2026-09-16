"""Conventional island planning baselines.

`rule_dispatch` implements the priority-list energy management strategy
that simulation-based island planning studies normally embed: surplus is
absorbed and deficits are covered in a fixed order that runs from the
fastest to the slowest medium, with the dispatchable generator acting
last.  `ga_size` and `pso_size` place that strategy inside the two
population-based searches most frequently used for autonomous-system
sizing.
"""
import numpy as np


def rule_dispatch(caps, series, media, sc, order_dis=None, order_ch=None,
                  reserve=0.15):
    load, pv, wd, _, _, _, cool = [np.asarray(a, dtype=float) for a in series]
    n = len(media)
    E = np.maximum(np.asarray(caps["E"], dtype=float), 1e-9)
    Pc = np.asarray(caps["Pc"], dtype=float)
    Pd = np.asarray(caps["Pd"], dtype=float)
    eta_c = np.array([m.eta_c for m in media])
    eta_d = np.array([m.eta_d for m in media])
    sd = np.array([m.self_discharge for m in media])
    cthr = np.array([m.c_throughput for m in media])
    ll = np.array([m.load_limited for m in media])
    if order_dis is None:
        order_dis = list(range(n))
    if order_ch is None:
        order_ch = list(range(n))

    soc = 0.5 * E
    fuel = unmet = curt = deg = 0.0
    for t in range(load.size):
        r = load[t] - caps["pv"] * pv[t] - caps["wind"] * wd[t]
        if r > 0:
            need = r
            for j in order_dis:
                if need <= 1e-9:
                    break
                cap = min(Pd[j], soc[j] * eta_d[j])
                if ll[j]:
                    cap = min(cap, cool[t])
                # the slowest medium is held back until the others are drawn down
                if j == order_dis[-1] and need < reserve * load[t]:
                    continue
                take = min(cap, need)
                soc[j] -= take / eta_d[j]
                deg += take * cthr[j]
                need -= take
            g = min(need, caps["diesel"])
            fuel += g
            unmet += max(need - g, 0.0)
        else:
            surp = -r
            for j in order_ch:
                if surp <= 1e-9:
                    break
                cap = min(Pc[j], (E[j] - soc[j]) / eta_c[j])
                put = min(cap, surp)
                soc[j] += put * eta_c[j]
                surp -= put
            curt += surp
        soc *= (1.0 - sd)
        soc = np.clip(soc, 0.0, E)

    capex = (sc.c_pv * caps["pv"] + sc.c_wind * caps["wind"] + sc.c_diesel_cap * caps["diesel"]
             + float(np.sum([m.c_energy for m in media] * np.asarray(caps["E"])))
             + float(np.sum([m.c_power_ch for m in media] * Pc))
             + float(np.sum([m.c_power_dis for m in media] * Pd)))
    opex = fuel * (sc.c_fuel + sc.c_vom_diesel) + caps["wind"] * float(wd.sum()) * sc.c_vom_wind + deg
    dem = float(load.sum())
    tac = capex + opex + unmet * sc.voll
    return dict(tac=tac, lcoe=tac / dem, lpsp=unmet / dem, fuel_e=fuel,
                unmet_e=unmet, curt_e=curt, capex=capex, opex=opex,
                penalty=unmet * sc.voll,
                re_share=1.0 - fuel / max(dem - unmet, 1e-9), co2=fuel * sc.co2_rate)


def decode(v, media, island, e_cap):
    n = len(media)
    peak = island.peak_mw
    hi = np.concatenate([[2.5 * peak, 2.5 * peak, 1.2 * peak],
                         np.minimum(np.full(n, 20.0 * peak), e_cap),
                         np.tile([peak, peak], n)])
    x = np.clip(v, 0.0, 1.0) * hi
    E = x[3:3 + n]
    Pc, Pd = [], []
    for j, m in enumerate(media):
        if m.fixed_ep_ratio > 0:
            Pc.append(E[j] / m.fixed_ep_ratio); Pd.append(E[j] / m.fixed_ep_ratio)
        elif m.c_power_dis == 0.0:
            Pc.append(x[3 + n + 2 * j]); Pd.append(x[3 + n + 2 * j])
        else:
            Pc.append(x[3 + n + 2 * j]); Pd.append(x[3 + n + 2 * j + 1])
    return dict(pv=x[0], wind=x[1], diesel=x[2], E=E,
                Pc=np.asarray(Pc), Pd=np.asarray(Pd))


def _fitness(v, years, media, sc, island, e_cap):
    caps = decode(v, media, island, e_cap)
    return float(np.mean([rule_dispatch(caps, y, media, sc)["tac"] for y in years]))


def ga_size(years, media, sc, island, e_cap, pop=40, gens=50, seed=0,
            pc=0.9, pm=0.15, verbose=False):
    rng = np.random.default_rng(seed)
    n = len(media)
    dim = 3 + 3 * n
    P = rng.random((pop, dim))
    f = np.array([_fitness(v, years, media, sc, island, e_cap) for v in P])
    for g in range(gens):
        order = np.argsort(f)
        P, f = P[order], f[order]
        newP = [P[0].copy(), P[1].copy()]
        while len(newP) < pop:
            a, b = (P[min(rng.integers(pop), rng.integers(pop))] for _ in range(2))
            if rng.random() < pc:
                w = rng.uniform(-0.25, 1.25, dim)
                child = np.clip(w * a + (1 - w) * b, 0, 1)
            else:
                child = a.copy()
            m = rng.random(dim) < pm
            child[m] = np.clip(child[m] + rng.normal(0, 0.12, m.sum()), 0, 1)
            newP.append(child)
        P = np.array(newP)
        f = np.array([_fitness(v, years, media, sc, island, e_cap) for v in P])
        if verbose and g % 10 == 0:
            print(f"   GA gen {g:3d} best {f.min():12.0f}")
    i = int(np.argmin(f))
    return decode(P[i], media, island, e_cap), float(f[i])


def pso_size(years, media, sc, island, e_cap, swarm=30, iters=60, seed=0,
             w=0.72, c1=1.5, c2=1.5, verbose=False):
    rng = np.random.default_rng(seed)
    dim = 3 + 3 * len(media)
    X = rng.random((swarm, dim))
    V = rng.normal(0, 0.08, (swarm, dim))
    f = np.array([_fitness(v, years, media, sc, island, e_cap) for v in X])
    pbest, pf = X.copy(), f.copy()
    gi = int(np.argmin(f)); gbest, gf = X[gi].copy(), f[gi]
    for it in range(iters):
        r1, r2 = rng.random((swarm, dim)), rng.random((swarm, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, 0.0, 1.0)
        f = np.array([_fitness(v, years, media, sc, island, e_cap) for v in X])
        imp = f < pf
        pbest[imp], pf[imp] = X[imp], f[imp]
        if f.min() < gf:
            gi = int(np.argmin(f)); gbest, gf = X[gi].copy(), f[gi]
        if verbose and it % 15 == 0:
            print(f"   PSO it {it:3d} best {gf:12.0f}")
    return decode(gbest, media, island, e_cap), float(gf)
