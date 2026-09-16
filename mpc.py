"""Rolling-horizon evaluation controller.

Every capacity vector is exercised by the same causal controller: a
receding-horizon linear program driven by the published day-ahead
forecasts, closed by a seasonal state-of-charge reference so that the
long-duration media are not penalised by the finite horizon, and
completed by a real-time balancing step that absorbs forecast error.
The controller never sees a future realisation, so its cost is a cost a
system operator could actually incur.  The constraint matrices depend
only on the horizon length, so they are assembled once and reused across
the whole year.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


class HorizonLP:
    """Horizon programme with an operating-energy reserve.

    A deterministic receding-horizon controller that plans exactly to its
    forecast keeps no margin for forecast error.  Island practice is to
    hold a stored-energy reserve covering a stated number of hours of
    demand, and the same requirement is imposed here through a softly
    penalised floor on aggregate available discharge energy, so that the
    baseline controller is not disadvantaged by an omission that any
    operator would correct.
    """

    def __init__(self, n, H, media, sc, caps, reserve_h=2.0):
        self.n, self.H = n, H
        self.media, self.sc, self.caps = media, sc, caps
        ch0, dis0, soc0 = 0, n * H, 2 * n * H
        d0, u0, dv0 = 3 * n * H, 3 * n * H + H, 3 * n * H + 2 * H
        self.off = (ch0, dis0, soc0, d0, u0, dv0)
        self.nv = 3 * n * H + 3 * H + n
        self.rs0 = 3 * n * H + 2 * H + n
        self.reserve_h = reserve_h
        h = np.arange(H)

        c = np.zeros(self.nv)
        c[d0:d0 + H] = sc.c_fuel + sc.c_vom_diesel
        c[u0:u0 + H] = sc.voll
        for j, m in enumerate(media):
            c[dis0 + j * H: dis0 + (j + 1) * H] = m.c_throughput
        c[dv0:dv0 + n] = sc.c_fuel * np.array([m.eta_d for m in media])
        c[self.rs0:self.rs0 + H] = 0.5 * sc.voll
        self.c = c

        r, cc, v, self.b_eq0 = [], [], [], np.zeros(n * H)
        for j, m in enumerate(media):
            keep = 1.0 - m.self_discharge
            rr = j * H + h
            r += [rr, rr, rr]
            cc += [soc0 + j * H + h, ch0 + j * H + h, dis0 + j * H + h]
            v += [np.ones(H), np.full(H, -m.eta_c), np.full(H, 1.0 / m.eta_d)]
            r.append(j * H + h[1:]); cc.append(soc0 + j * H + h[:-1])
            v.append(np.full(H - 1, -keep))
        self.A_eq = sp.csc_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(cc))),
                                  shape=(n * H, self.nv))

        r, cc, v = [], [], []
        r += [h, h]; cc += [d0 + h, u0 + h]; v += [-np.ones(H), -np.ones(H)]
        for j in range(n):
            r += [h, h]; cc += [ch0 + j * H + h, dis0 + j * H + h]
            v += [np.ones(H), -np.ones(H)]
        jj = np.arange(n)
        r += [H + jj, H + jj]; cc += [soc0 + jj * H + H - 1, dv0 + jj]
        v += [-np.ones(n), -np.ones(n)]
        for j, m in enumerate(media):
            r.append(H + n + h); cc.append(soc0 + j * H + h)
            v.append(np.full(H, -m.eta_d))
        r.append(H + n + h); cc.append(self.rs0 + h); v.append(-np.ones(H))
        self.A_ub = sp.csc_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(cc))),
                                  shape=(2 * H + n, self.nv))

        E = np.maximum(np.asarray(caps["E"], float), 1e-9)
        lb = np.zeros(self.nv)
        ub = np.empty(self.nv)
        self.ll = np.array([m.load_limited for m in media])
        for j, m in enumerate(media):
            ub[ch0 + j * H: ch0 + (j + 1) * H] = caps["Pc"][j]
            ub[dis0 + j * H: dis0 + (j + 1) * H] = caps["Pd"][j]
            ub[soc0 + j * H: soc0 + (j + 1) * H] = E[j]
        ub[d0:d0 + H] = caps["diesel"]
        ub[u0:] = np.inf
        self.lb, self.ub = lb, ub
        self.E = E

    def solve(self, soc, fc, ref, cool_fc):
        n, H = self.n, self.H
        ch0, dis0, soc0, d0, u0, dv0 = self.off
        load, pv, wd = fc
        b_eq = self.b_eq0.copy()
        for j, m in enumerate(self.media):
            b_eq[j * H] = (1.0 - m.self_discharge) * soc[j]
        b_ub = np.empty(2 * H + n)
        b_ub[:H] = -load + self.caps["pv"] * pv + self.caps["wind"] * wd
        b_ub[H:H + n] = -ref
        b_ub[H + n:] = -self.reserve_h * np.maximum(
            load - self.caps["diesel"], 0.0)
        ub = self.ub.copy()
        for j in range(n):
            if self.ll[j]:
                ub[dis0 + j * H: dis0 + (j + 1) * H] = np.minimum(
                    self.caps["Pd"][j], cool_fc)
        res = linprog(self.c, A_ub=self.A_ub, b_ub=b_ub, A_eq=self.A_eq, b_eq=b_eq,
                      bounds=np.column_stack([self.lb, ub]), method="highs")
        if not res.success:
            return None
        x = res.x
        return (x[ch0:ch0 + n * H].reshape(n, H), x[dis0:dis0 + n * H].reshape(n, H))


def run(caps, series, media, sc, ref_frac, H=48, step=3):
    load, pv, wd, pv_f, wd_f, load_f, cool = [np.asarray(a, float) for a in series]
    T = load.size
    n = len(media)
    E = np.maximum(np.asarray(caps["E"], float), 1e-9)
    eta_c = np.array([m.eta_c for m in media])
    eta_d = np.array([m.eta_d for m in media])
    sd = np.array([m.self_discharge for m in media])
    cthr = np.array([m.c_throughput for m in media])
    ll = np.array([m.load_limited for m in media])
    Pc = np.asarray(caps["Pc"], float)
    Pd = np.asarray(caps["Pd"], float)

    ref_frac = np.asarray(ref_frac)
    lp_full = HorizonLP(n, H, media, sc, caps)
    lp_tail = {}
    soc = 0.5 * E
    fuel = unmet = curt = deg = 0.0
    traj = np.zeros((n, T))
    socs = np.zeros((n, T))
    for t0 in range(0, T, step):
        Ht = min(H, T - t0)
        lp = lp_full if Ht == H else lp_tail.setdefault(
            Ht, HorizonLP(n, Ht, media, sc, caps))
        sl = slice(t0, t0 + Ht)
        ref = np.clip(ref_frac[:, min(t0 + Ht, T - 1)], 0.0, 1.0) * E
        plan = lp.solve(soc, (load_f[sl], pv_f[sl], wd_f[sl]), ref, cool[sl])
        if plan is None:
            plan = (np.zeros((n, Ht)), np.zeros((n, Ht)))
        ch_p, dis_p = plan
        for h in range(min(step, Ht)):
            t = t0 + h
            u = dis_p[:, h] - ch_p[:, h]
            res_act = load[t] - caps["pv"] * pv[t] - caps["wind"] * wd[t]
            res_plan = load_f[t] - caps["pv"] * pv_f[t] - caps["wind"] * wd_f[t]
            gap = res_act - res_plan
            dis_cap = np.minimum(Pd, soc * eta_d)
            dis_cap = np.where(ll, np.minimum(dis_cap, cool[t]), dis_cap)
            ch_cap = np.minimum(Pc, (E - soc) / eta_c)
            head = np.maximum(np.where(gap > 0, dis_cap - u, u + ch_cap), 0.0)
            tot = head.sum()
            if tot > 1e-9:
                u = u + np.sign(gap) * head / tot * min(abs(gap), tot)
            u = np.clip(u, -ch_cap, dis_cap)
            dis = np.maximum(u, 0.0)
            chg = np.maximum(-u, 0.0)
            soc = np.clip(soc * (1.0 - sd) - (dis / eta_d - eta_c * chg), 0.0, E)
            r = res_act - u.sum()
            g = min(max(r, 0.0), caps["diesel"])
            fuel += g
            unmet += max(r - caps["diesel"], 0.0)
            curt += max(-r, 0.0)
            deg += float(np.sum(dis * cthr))
            traj[:, t] = u
            socs[:, t] = soc / E
    capex = (sc.c_pv * caps["pv"] + sc.c_wind * caps["wind"] + sc.c_diesel_cap * caps["diesel"]
             + float(np.sum([m.c_energy for m in media] * np.asarray(caps["E"])))
             + float(np.sum([m.c_power_ch for m in media] * Pc))
             + float(np.sum([m.c_power_dis for m in media] * Pd)))
    opex = fuel * (sc.c_fuel + sc.c_vom_diesel) + caps["wind"] * float(wd.sum()) * sc.c_vom_wind + deg
    dem = float(load.sum())
    tac = capex + opex + unmet * sc.voll
    return dict(tac=tac, capex=capex, opex=opex, penalty=unmet * sc.voll,
                lcoe=tac / dem, lpsp=unmet / dem, fuel_e=fuel, unmet_e=unmet,
                curt_e=curt, re_share=1.0 - fuel / max(dem - unmet, 1e-9),
                co2=fuel * sc.co2_rate, p=traj, soc=socs)
