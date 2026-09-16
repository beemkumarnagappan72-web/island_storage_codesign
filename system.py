"""Differentiable hourly simulator of an island power system with a
portfolio of storage media operated by a causal, duration-structured
dispatch policy.

Every saturation in the plant model is replaced by a softplus-smoothed
counterpart with width mu, so that the annualised cost is differentiable
with respect to both the capacity vector and the policy parameters.
Setting mu = 0 recovers the exact (non-smooth) plant, which is what the
reported results use.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

DT = 1.0  # hours


# ----------------------------------------------------------------------
# smoothed elementary operations
# ----------------------------------------------------------------------
def sp(z, mu):
    """Softplus of width mu; exact ReLU at mu = 0."""
    return jnp.where(mu > 0.0,
                     mu * jnp.logaddexp(z / jnp.maximum(mu, 1e-12), 0.0),
                     jnp.maximum(z, 0.0))


def clip_mu(z, lo, hi, mu):
    return lo + sp(z - lo, mu) - sp(z - hi, mu)


def min_mu(a, b, mu):
    return a - sp(a - b, mu)


# ----------------------------------------------------------------------
# parameter packing
# ----------------------------------------------------------------------
def n_features(cfg):
    return cfg["n_bands"] if cfg.get("feat", "band") == "band" else cfg.get("n_lag", 25)


def unpack(params, media, cfg):
    """Map unconstrained parameters to physical capacities and policy gains."""
    z = params["cap"]
    scale = cfg["cap_scale"]
    x = jax.nn.softplus(z) * scale
    n = len(media)
    x_pv, x_w, x_d = x[0], x[1], x[2] * cfg.get("dsl_scale", 1.0)
    ecap = cfg["e_cap"]
    capped = jnp.isfinite(ecap)
    safe = jnp.where(capped, ecap, 1.0)
    Eraw = x[3:3 + n]
    E = jnp.where(capped, safe * jax.nn.sigmoid(z[3:3 + n]), Eraw)
    Praw = x[3 + n:3 + 3 * n].reshape(n, 2)

    Pc, Pd = [], []
    for j, m in enumerate(media):
        if m.fixed_ep_ratio > 0:
            p = E[j] / m.fixed_ep_ratio
            Pc.append(p)
            Pd.append(p)
        elif m.c_power_dis == 0.0:          # single bidirectional converter
            Pc.append(Praw[j, 0])
            Pd.append(Praw[j, 0])
        else:
            Pc.append(Praw[j, 0])
            Pd.append(Praw[j, 1])
    Pc = jnp.stack(Pc)
    Pd = jnp.stack(Pd)

    th = params["pol"]
    if cfg.get("alloc", "simplex") == "simplex":
        alpha = jax.nn.softmax(th["logits"], axis=0)[:n, :]   # slack row absorbs the remainder
    else:
        alpha = th["logits"][:n, :]                           # unconstrained coefficients
    beta = jax.nn.softplus(th["beta"]) * cfg.get("use_restore", 1.0)
    lam = jax.nn.sigmoid(th["lam"]) * cfg.get("use_lam", 1.0)
    return dict(x_pv=x_pv, x_w=x_w, x_d=x_d, E=E, Pc=Pc, Pd=Pd,
                alpha=alpha, beta=beta, lam=lam, seas=th["seas"])


def invert_cap(caps_vec, media, cfg):
    """Unconstrained parameters that reproduce a given capacity vector."""
    import numpy as _np
    scale = _np.asarray(cfg["cap_scale"], dtype=float)
    ecap = _np.asarray(cfg["e_cap"], dtype=float)
    n = len(media)
    x = _np.zeros(3 + 3 * n)
    x[0], x[1], x[2] = caps_vec["pv"], caps_vec["wind"], caps_vec["diesel"]
    for j in range(n):
        x[3 + n + 2 * j] = caps_vec["Pc"][j]
        x[3 + n + 2 * j + 1] = caps_vec["Pd"][j]
    z = _np.zeros_like(x)
    for i in range(x.size):
        v = max(float(x[i]) / scale[i], 1e-8)
        z[i] = _np.log(_np.expm1(min(v, 30.0)))
    for j in range(n):
        E = float(caps_vec["E"][j])
        if _np.isfinite(ecap[j]):
            q = min(max(E / max(ecap[j], 1e-9), 1e-6), 1 - 1e-6)
            z[3 + j] = _np.log(q / (1 - q))
        else:
            v = max(E / scale[3 + j], 1e-8)
            z[3 + j] = _np.log(_np.expm1(min(v, 30.0)))
    return jnp.asarray(z, dtype=jnp.float32)


def soc_target(seas, t, n_hours):
    """Seasonal state-of-charge target for each medium (n, 3) -> (n,)."""
    ang = 2.0 * jnp.pi * t / n_hours
    return jax.nn.sigmoid(seas[:, 0] + seas[:, 1] * jnp.cos(ang) + seas[:, 2] * jnp.sin(ang))


# ----------------------------------------------------------------------
# core rollout
# ----------------------------------------------------------------------
def make_simulator(media, sc, taus, n_hours=8760, mu=0.0, return_traj=False,
                   feat="band", n_lag=25, seasonal=True):
    n = len(media)
    a_f = jnp.exp(-DT / jnp.asarray(taus))
    eta_c = jnp.asarray([m.eta_c for m in media])
    eta_d = jnp.asarray([m.eta_d for m in media])
    sdis = jnp.asarray([m.self_discharge for m in media])
    cthr = jnp.asarray([m.c_throughput for m in media])
    load_lim = jnp.asarray([1.0 if m.load_limited else 0.0 for m in media])
    soc0 = jnp.asarray([m.soc0 for m in media])

    def rollout(p, series):
        load, pv, wd, pv_f, wd_f, load_f, cool = series
        E = jnp.maximum(p["E"], 1e-6)

        def step(carry, inp):
            s, soc, t = carry
            ld, g_pv, g_w, f_pv, f_w, f_ld, cl = inp
            res = ld - p["x_pv"] * g_pv - p["x_w"] * g_w
            res_f = f_ld - p["x_pv"] * f_pv - p["x_w"] * f_w
            lt = (1.0 - p["lam"]) * res + p["lam"] * res_f

            if feat == "band":
                # causal cascade of first-order filters -> additive band partition
                s_new, bands, cin = [], [], lt
                for k in range(len(taus)):
                    o = a_f[k] * s[k] + (1.0 - a_f[k]) * cin
                    bands.append(cin - o)
                    s_new.append(o)
                    cin = o
                bands.append(cin)
                b = jnp.stack(bands)
                s_new = jnp.stack(s_new)
            else:
                # dense affine rule on a window of past residuals
                b = jnp.concatenate([jnp.array([lt]), s])
                s_new = jnp.concatenate([jnp.array([lt]), s[:-1]])

            tgt = (soc_target(p["seas"], t, n_hours) if seasonal
                   else jax.nn.sigmoid(p["seas"][:, 0]))
            u = p["alpha"] @ b + p["beta"] * (soc - tgt) * E

            dis_cap = min_mu(p["Pd"], soc * E * eta_d / DT, mu)
            dis_cap = jnp.where(load_lim > 0, min_mu(dis_cap, cl, mu), dis_cap)
            ch_cap = min_mu(p["Pc"], (1.0 - soc) * E / (eta_c * DT), mu)
            pj = clip_mu(u, -ch_cap, dis_cap, mu)

            dis = sp(pj, mu)
            chg = sp(-pj, mu)
            soc_n = soc * (1.0 - sdis) - DT / E * (dis / eta_d - eta_c * chg)
            soc_n = jnp.clip(soc_n, 0.0, 1.0)

            r = res - jnp.sum(pj)
            dsl = clip_mu(r, 0.0, p["x_d"], mu)
            unmet = sp(r - p["x_d"], mu)
            curt = sp(-r, mu)

            acc = jnp.array([dsl, unmet, curt, ld, jnp.sum(dis * cthr),
                             p["x_w"] * g_w])
            out = (s_new, soc_n, t + 1.0), (acc, pj, soc_n, res) if return_traj else (acc,)
            return out

        n_state = len(taus) if feat == "band" else n_lag - 1
        init = (jnp.zeros(n_state), soc0, 0.0)
        (_, _, _), ys = lax.scan(step, init, (load, pv, wd, pv_f, wd_f, load_f, cool))
        acc = ys[0] if return_traj else ys[0]
        tot = jnp.sum(acc, axis=0)
        fuel_e, unmet_e, curt_e, dem_e, deg_c, wind_e = tot

        capex = (sc.c_pv * p["x_pv"] + sc.c_wind * p["x_w"] + sc.c_diesel_cap * p["x_d"]
                 + jnp.sum(jnp.asarray([m.c_energy for m in media]) * p["E"])
                 + jnp.sum(jnp.asarray([m.c_power_ch for m in media]) * p["Pc"])
                 + jnp.sum(jnp.asarray([m.c_power_dis for m in media]) * p["Pd"]))
        opex = (fuel_e * (sc.c_fuel + sc.c_vom_diesel) + wind_e * sc.c_vom_wind
                + deg_c)
        pen = unmet_e * sc.voll
        tac = capex + opex + pen
        served = dem_e - unmet_e
        out = dict(tac=tac, capex=capex, opex=opex, penalty=pen,
                   lcoe=tac / dem_e, lpsp=unmet_e / dem_e,
                   fuel_e=fuel_e, unmet_e=unmet_e, curt_e=curt_e,
                   demand_e=dem_e, served_e=served,
                   re_share=1.0 - fuel_e / jnp.maximum(served, 1e-9),
                   co2=fuel_e * sc.co2_rate)
        if return_traj:
            out["p"] = ys[1]
            out["soc"] = ys[2]
            out["res"] = ys[3]
        return out

    return rollout


def _sim_kwargs(cfg):
    return dict(feat=cfg.get("feat", "band"), n_lag=cfg.get("n_lag", 25),
                seasonal=bool(cfg.get("seasonal", True)))


def evaluate(params, cfg, media, sc, taus, series, mu=0.0, return_traj=False):
    p = unpack(params, media, cfg)
    roll = make_simulator(media, sc, taus, n_hours=series[0].shape[-1], mu=mu,
                          return_traj=return_traj, **_sim_kwargs(cfg))
    return roll(p, series)


def batched_cost(params, cfg, media, sc, taus, batch, mu, rho, eps, zeta):
    """Mean-CVaR objective over a batch of scenario years."""
    p = unpack(params, media, cfg)
    roll = make_simulator(media, sc, taus, n_hours=batch[0].shape[-1], mu=mu,
                          **_sim_kwargs(cfg))
    res = jax.vmap(lambda *s: roll(p, s))(*batch)
    c = res["tac"]
    cvar = zeta + jnp.mean(jax.nn.relu(c - zeta)) / (1.0 - eps)
    return (1.0 - rho) * jnp.mean(c) + rho * cvar, res
