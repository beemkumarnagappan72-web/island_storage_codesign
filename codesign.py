"""Joint capacity and dispatch-policy optimisation on the smoothed plant.

The decision vector contains the investment variables, the parameters of
the duration-structured causal policy and the Rockafellar-Uryasev
auxiliary variable of the conditional value-at-risk term.  All of them are
updated by the same stochastic gradient recursion, while the smoothing
width is annealed geometrically towards zero so that the optimised design
is evaluated on the exact plant.
"""
import numpy as np
import jax
import jax.numpy as jnp
from system import batched_cost, unpack, evaluate

jax.config.update("jax_enable_x64", False)


def make_cfg(island, media, n_bands, data=None):
    n = len(media)
    peak = island.peak_mw
    scale = np.concatenate([
        np.array([peak * 2.0, peak * 2.0, peak * 1.2]),
        np.full(n, peak * 20.0),
        np.tile(np.array([peak * 1.0, peak * 1.0]), n),
    ])
    from config import energy_caps
    return dict(cap_scale=jnp.asarray(scale, dtype=jnp.float32), n_bands=n_bands,
                n_media=n, e_cap=jnp.asarray(energy_caps(media, island, data),
                                             dtype=jnp.float32))


def init_params(seed, media, cfg, dense_lags=0):
    rng = np.random.default_rng(seed)
    from system import n_features
    n = cfg["n_media"]
    k = n_features(cfg)
    cap = jnp.asarray(rng.normal(-1.0, 0.3, size=3 + 3 * n), dtype=jnp.float32)
    pol = dict(
        logits=jnp.asarray(rng.normal(0.0, 0.5 if cfg.get("alloc", "simplex") == "simplex"
                                      else 0.15, size=(n + 1, k)), dtype=jnp.float32),
        beta=jnp.asarray(rng.normal(-3.0, 0.2, size=n), dtype=jnp.float32),
        lam=jnp.asarray(rng.normal(0.0, 0.2, size=()), dtype=jnp.float32),
        seas=jnp.asarray(np.column_stack([rng.normal(0.0, 0.2, n),
                                          rng.normal(0.0, 0.2, n),
                                          rng.normal(0.0, 0.2, n)]), dtype=jnp.float32),
    )
    return dict(cap=cap, pol=pol, zeta=jnp.asarray(0.0, dtype=jnp.float32))


def _adam_init(p):
    return jax.tree_util.tree_map(jnp.zeros_like, p), jax.tree_util.tree_map(jnp.zeros_like, p)


def _adam(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    sc = jnp.minimum(1.0, clip / (gn + 1e-12))
    g = jax.tree_util.tree_map(lambda x: x * sc, g)
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b ** 2, v, g)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
    p = jax.tree_util.tree_map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + eps), p, mh, vh)
    return p, m, v


def train(train_years, media, sc, taus, cfg, *, seed=0, steps=300, batch=4,
          lr=0.05, mu0=0.6, mu1=0.01, rho=0.3, eps=0.8, verbose=False,
          scale_cost=1e-6, params0=None, freeze=()):
    """Return optimised parameters and the optimisation history."""
    params = init_params(seed, media, cfg) if params0 is None else params0
    stacked = tuple(jnp.asarray(a) for a in train_years)
    n_years = stacked[0].shape[0]
    rng = np.random.default_rng(seed + 7919)

    def loss(p, b, mu):
        val, _ = batched_cost(p, cfg, media, sc, taus, b, mu, rho, eps, p["zeta"])
        return val * scale_cost

    gfun = jax.jit(jax.value_and_grad(loss), static_argnums=())
    m, v = _adam_init(params)
    hist = []
    for it in range(1, steps + 1):
        mu = mu0 * (mu1 / mu0) ** ((it - 1) / max(steps - 1, 1))
        idx = rng.choice(n_years, size=min(batch, n_years), replace=False)
        b = tuple(a[idx] for a in stacked)
        val, g = gfun(params, b, mu)
        for f in freeze:
            if f == "cap":
                g["cap"] = jnp.zeros_like(g["cap"])
            else:
                g["pol"][f] = jnp.zeros_like(g["pol"][f])
        lr_t = lr * (0.5 * (1.0 + np.cos(np.pi * (it - 1) / steps)) * 0.9 + 0.1)
        params, m, v = _adam(params, g, m, v, it, lr_t)
        hist.append(float(val))
        if verbose and (it % 25 == 0 or it == 1):
            print(f"  it {it:4d}  mu {mu:6.3f}  obj {val / scale_cost:12.0f}")
    return params, np.array(hist)


def design_summary(params, cfg, media):
    p = unpack(params, media, cfg)
    return dict(pv=float(p["x_pv"]), wind=float(p["x_w"]), diesel=float(p["x_d"]),
                E={m.key: float(p["E"][j]) for j, m in enumerate(media)},
                Pc={m.key: float(p["Pc"][j]) for j, m in enumerate(media)},
                Pd={m.key: float(p["Pd"][j]) for j, m in enumerate(media)},
                alpha=np.asarray(p["alpha"]), lam=float(p["lam"]),
                beta=np.asarray(p["beta"]))
