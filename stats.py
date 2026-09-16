"""Statistical machinery for the comparative analysis.

Comparisons are paired at the level of an individual evaluation year on
an individual island, which is the unit at which every method is exposed
to exactly the same realisation.  Because the paired differences are not
assumed normal and the number of pairs is moderate, the Wilcoxon
signed-rank test is used, effect size is reported as the matched-pairs
rank-biserial correlation, uncertainty as a bias-corrected bootstrap
interval, and the family of pairwise tests is protected by the Holm
step-down correction.  The Friedman test with the Nemenyi critical
difference compares all methods simultaneously.
"""
import numpy as np
from scipy import stats


def paired(a, b, n_boot=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return dict(n=d.size, mean=0.0, median=0.0, p=1.0, rbc=0.0,
                    ci=(0.0, 0.0), rel=0.0)
    st, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    r = stats.rankdata(np.abs(nz))
    rp = r[nz > 0].sum()
    rm = r[nz < 0].sum()
    rbc = (rp - rm) / (rp + rm)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    return dict(n=int(d.size), mean=float(d.mean()), median=float(np.median(d)),
                p=float(p), rbc=float(rbc),
                ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
                rel=float(d.mean() / np.mean(b) * 100.0))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def friedman_nemenyi(matrix, alpha=0.05):
    """matrix: rows = blocks (island-year), columns = methods."""
    X = np.asarray(matrix, float)
    n, k = X.shape
    st, p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    ranks = np.apply_along_axis(stats.rankdata, 1, X)
    mean_rank = ranks.mean(axis=0)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
         8: 3.031, 9: 3.102, 10: 3.164}.get(k, 3.219)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return dict(stat=float(st), p=float(p), mean_rank=mean_rank.tolist(),
                cd=float(cd), n=n, k=k)


def describe(x):
    x = np.asarray(x, float)
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
