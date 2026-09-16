"""Linear-programming reference models.

`size_lp` solves the joint sizing and dispatch problem with complete
knowledge of the whole horizon, which is the formulation underlying
conventional simulation-based and mathematical-programming island
planning tools.  With several scenario years attached to one shared
capacity vector it becomes the two-stage stochastic variant.
`dispatch_lp` fixes the capacities and returns the clairvoyant operating
cost, which provides the perfect-foresight floor used in the foresight
decomposition.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def _layout(n, T, n_sc):
    nc = 3 + 3 * n
    per = 3 * n * T + 2 * T
    return nc, per, nc + n_sc * per


def _blocks(n, T, media, sc, series, nc, base, rows_eq, rows_ub, c, ub_bounds,
            cap_fixed=None):
    """Vectorised assembly of one scenario block."""
    load, pv, wd, cool = series
    ch0, dis0, soc0 = base, base + n * T, base + 2 * n * T
    dsl0, un0 = base + 3 * n * T, base + 3 * n * T + T
    t = np.arange(T)
    tp = (t - 1) % T

    er, ec, ev = [], [], []
    r0 = rows_eq
    for j, m in enumerate(media):
        rr = r0 + j * T + t
        er += [rr, rr, rr, rr]
        ec += [soc0 + j * T + t, soc0 + j * T + tp, ch0 + j * T + t, dis0 + j * T + t]
        ev += [np.ones(T), np.full(T, -(1.0 - m.self_discharge)),
               np.full(T, -m.eta_c), np.full(T, 1.0 / m.eta_d)]
    rows_eq += n * T

    ar, ac, av, bu = [], [], [], []
    q = rows_ub
    rr = q + t
    ar += [rr, rr]; ac += [dsl0 + t, un0 + t]; av += [-np.ones(T), -np.ones(T)]
    for j in range(n):
        ar += [rr, rr]; ac += [ch0 + j * T + t, dis0 + j * T + t]
        av += [np.ones(T), -np.ones(T)]
    if cap_fixed is None:
        ar += [rr, rr]; ac += [np.zeros(T, int), np.ones(T, int)]
        av += [-pv, -wd]
        bu.append(-load)
    else:
        bu.append(-load + cap_fixed[0] * pv + cap_fixed[1] * wd)
    q += T

    for var0, off, scale in ((soc0, 3, None), (ch0, 3 + n, None), (dis0, 3 + 2 * n, None)):
        for j, m in enumerate(media):
            rr = q + j * T + t
            ar.append(rr); ac.append(var0 + j * T + t); av.append(np.ones(T))
            if cap_fixed is None:
                ar.append(rr); ac.append(np.full(T, off + j)); av.append(-np.ones(T))
                bu.append(np.zeros(T))
            else:
                bu.append(np.full(T, cap_fixed[off + j]))
        q += n * T
    rr = q + t
    ar.append(rr); ac.append(dsl0 + t); av.append(np.ones(T))
    if cap_fixed is None:
        ar.append(rr); ac.append(np.full(T, 2)); av.append(-np.ones(T))
        bu.append(np.zeros(T))
    else:
        bu.append(np.full(T, cap_fixed[2]))
    q += T

    c[dsl0:dsl0 + T] += sc.c_fuel + sc.c_vom_diesel
    c[un0:un0 + T] += sc.voll
    for j, m in enumerate(media):
        c[dis0 + j * T: dis0 + (j + 1) * T] += m.c_throughput
    if cap_fixed is None:
        c[1] += sc.c_vom_wind * float(np.sum(wd))
    for j, m in enumerate(media):
        if m.load_limited:
            ub_bounds[dis0 + j * T: dis0 + (j + 1) * T] = cool

    eq = (np.concatenate(er), np.concatenate(ec), np.concatenate(ev),
          np.zeros(n * T))
    ineq = (np.concatenate(ar), np.concatenate(ac), np.concatenate(av),
            np.concatenate(bu))
    return eq, ineq, rows_eq, q


def _solve(n, T, media, sc, series_list, cap_fixed=None, method="highs-ipm", e_cap=None, reserve=None):
    n_sc = len(series_list)
    nc, per, N = _layout(n, T, n_sc)
    c = np.zeros(N)
    ub = np.full(N, np.inf)
    if e_cap is not None and cap_fixed is None:
        for j in range(n):
            if np.isfinite(e_cap[j]):
                ub[3 + j] = float(e_cap[j])
    if cap_fixed is None:
        c[0] = sc.c_pv
        c[1] = sc.c_wind
        c[2] = sc.c_diesel_cap
        for j, m in enumerate(media):
            c[3 + j] = m.c_energy
            c[3 + n + j] = m.c_power_ch
            c[3 + 2 * n + j] = m.c_power_dis
    w = 1.0 / n_sc
    EQ = [[], [], [], []]
    UB = [[], [], [], []]
    re, ru = 0, 0
    for s, series in enumerate(series_list):
        base = nc + s * per
        ctmp = np.zeros(N)
        eq, ineq, re, ru = _blocks(n, T, media, sc, series, nc, base, re, ru,
                                   ctmp, ub, cap_fixed)
        c += ctmp * w
        for a, b in zip(EQ, eq):
            a.append(b)
        for a, b in zip(UB, ineq):
            a.append(b)
    EQ = [np.concatenate(a) for a in EQ]
    UB = [np.concatenate(a) for a in UB]
    A_eq = sp.csc_matrix((EQ[2], (EQ[0], EQ[1])), shape=(re, N))
    A_ub = sp.csc_matrix((UB[2], (UB[0], UB[1])), shape=(ru, N))
    if cap_fixed is None:
        rows, cols, vals, rhs = [], [], [], []
        r = 0
        for j, m in enumerate(media):
            if m.fixed_ep_ratio > 0:
                for off in (n, 2 * n):
                    rows += [r, r]; cols += [3 + off + j, 3 + j]
                    vals += [1.0, -1.0 / m.fixed_ep_ratio]; rhs.append(0.0); r += 1
            elif m.c_power_dis == 0.0:
                rows += [r, r]; cols += [3 + 2 * n + j, 3 + n + j]
                vals += [1.0, -1.0]; rhs.append(0.0); r += 1
        if r:
            A_eq = sp.vstack([A_eq, sp.csc_matrix((vals, (rows, cols)), shape=(r, N))]).tocsc()
            EQ[3] = np.concatenate([EQ[3], np.asarray(rhs)])
    if reserve is not None and cap_fixed is None:
        m_, peak = reserve
        row = np.zeros((1, N))
        row[0, 2] = -1.0
        for j in range(n):
            row[0, 3 + 2 * n + j] = -1.0
        A_ub = sp.vstack([A_ub, sp.csc_matrix(row)]).tocsc()
        UB[3] = np.concatenate([UB[3], [-(1.0 + m_) * peak]])
    res = linprog(c, A_ub=A_ub, b_ub=np.asarray(UB[3]), A_eq=A_eq,
                  b_eq=np.asarray(EQ[3]), bounds=list(zip(np.zeros(N), ub)),
                  method=method)
    if not res.success:
        raise RuntimeError(res.message)
    return res, c, nc, per


def size_lp(series_list, media, sc, method="highs-ipm", e_cap=None, reserve=None):
    """Perfect-foresight joint sizing and dispatch over one or more years."""
    T = series_list[0][0].size
    n = len(media)
    res, c, nc, per = _solve(n, T, media, sc, series_list, method=method, e_cap=e_cap,
                             reserve=reserve)
    x = res.x
    caps = dict(pv=x[0], wind=x[1], diesel=x[2],
                E=x[3:3 + n].copy(), Pc=x[3 + n:3 + 2 * n].copy(),
                Pd=x[3 + 2 * n:3 + 3 * n].copy())
    return caps, res.fun


def dispatch_lp(caps, series, media, sc, method="highs"):
    """Clairvoyant operating cost for a fixed capacity vector."""
    T = series[0].size
    n = len(media)
    fixed = np.concatenate([[caps["pv"], caps["wind"], caps["diesel"]],
                            caps["E"], caps["Pc"], caps["Pd"]])
    res, c, nc, per = _solve(n, T, media, sc, [series], cap_fixed=fixed, method=method)
    capex = (sc.c_pv * caps["pv"] + sc.c_wind * caps["wind"]
             + sc.c_diesel_cap * caps["diesel"]
             + float(np.sum([m.c_energy for m in media] * caps["E"]))
             + float(np.sum([m.c_power_ch for m in media] * caps["Pc"]))
             + float(np.sum([m.c_power_dis for m in media] * caps["Pd"]))
             + sc.c_vom_wind * caps["wind"] * float(np.sum(series[1])) * 0.0
             + sc.c_vom_wind * caps["wind"] * float(np.sum(series[2])))
    x = res.x
    base = nc
    soc = x[base + 2 * n * T: base + 3 * n * T].reshape(n, T)
    un = x[base + 3 * n * T + T: base + 3 * n * T + 2 * T]
    dsl = x[base + 3 * n * T: base + 3 * n * T + T]
    tac = capex + res.fun
    return dict(tac=tac, opex=res.fun, capex=capex,
                lcoe=tac / float(np.sum(series[0])),
                lpsp=float(np.sum(un)) / float(np.sum(series[0])),
                soc=soc, diesel=dsl,
                re_share=1.0 - float(np.sum(dsl)) / max(float(np.sum(series[0]) - np.sum(un)), 1e-9))
