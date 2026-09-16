"""Publication figures, restricted to a grayscale palette."""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import figstyle as FS
from config import ISLANDS, BAND_TAU_H, build_media

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIG, exist_ok=True)
MED = ["Battery", "Hydrogen", "Pumped hydro", "Thermal"]
BANDL = ["$b^{(1)}$, $\\tau_1=4$ h", "$b^{(2)}$, $\\tau_2=24$ h",
         "$b^{(3)}$, $\\tau_3=168$ h", "$b^{(4)}$, $\\tau_4=1440$ h",
         "$b^{(5)}$, residual mean"]
BANDX = ["$b^{(1)}$", "$b^{(2)}$", "$b^{(3)}$", "$b^{(4)}$", "$b^{(5)}$"]
COL = 3.42          # single-column width, inches
WIDE = 7.10         # full-width float


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ Fig 2
def fig_bands(path):
    from data_loader import build_island
    import scenarios as S
    FS.use(7.6)
    d = build_island(ISLANDS[0])
    load_ = d["load"][:8760]
    res = load_ - 18.0 * d["pv_cf"][:8760] - 4.0 * d["wind_cf"][:8760]
    a = np.exp(-1.0 / np.asarray(BAND_TAU_H))
    s = np.zeros((len(a), res.size))
    prev = res
    for k in range(len(a)):
        y = np.zeros_like(res)
        acc = 0.0
        for t in range(res.size):
            acc = a[k] * acc + (1 - a[k]) * prev[t]
            y[t] = acc
        s[k] = y
        prev = y
    bands = [res - s[0]] + [s[k - 1] - s[k] for k in range(1, len(a))] + [s[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.45))
    ax = axes[0]
    w = slice(3000, 3000 + 21 * 24)
    t = np.arange(21 * 24) / 24.0
    off = 0.0
    ax.plot(t, res[w], color="black", lw=1.0,
            label="residual $\\tilde{\\ell}_t$")
    step = 0.78 * (np.nanmax(res[w]) - np.nanmin(res[w]))
    for k, b in enumerate(bands):
        off -= step
        ax.plot(t, b[w] + off, color=FS.GREY[min(k, 3)], lw=0.8,
                linestyle=FS.LINES[(k + 1) % len(FS.LINES)], label=BANDL[k])
    ax.set_xlabel("time (days)")
    ax.set_ylabel("power (MW, offset)")
    ax.set_yticks([])
    ax.set_xlim(0, 21)
    FS.outside_legend(ax, ncol=2, y=1.03)

    ax = axes[1]
    f = np.logspace(np.log10(1 / 4000.0), np.log10(0.5), 600)
    w_ = 2 * np.pi * f
    H = np.ones_like(f, dtype=complex)
    Hs = []
    for k in range(len(a)):
        H = H * (1 - a[k]) / (1 - a[k] * np.exp(-1j * w_))
        Hs.append(H.copy())
    prevH = np.ones_like(f, dtype=complex)
    for k in range(len(a)):
        ax.semilogx(1 / f, np.abs(prevH - Hs[k]), color=FS.GREY[min(k, 3)],
                    lw=1.0, linestyle=FS.LINES[k % len(FS.LINES)], label=BANDL[k])
        prevH = Hs[k]
    ax.semilogx(1 / f, np.abs(prevH), color=FS.GREY[3], lw=1.0,
                linestyle=FS.LINES[4], label=BANDL[4])
    ax.set_xlabel("period (h)")
    ax.set_ylabel("band gain $|B_k(\\omega)|$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(2, 4000)
    FS.outside_legend(ax, ncol=2, y=1.03)
    FS.finish(fig, path, pad=0.3, w_pad=1.6)


# ------------------------------------------------------------------ Fig 3
def fig_alpha(path):
    FS.use(7.4)
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.05))
    for ax, isl in zip(axes, ISLANDS):
        m = load(f"eval_{isl.key}.pkl")
        A = np.asarray(m["alpha"])
        im = ax.imshow(A, cmap="Greys", vmin=0, vmax=max(0.6, A.max()),
                       aspect="auto")
        ax.set_xticks(range(A.shape[1]))
        ax.set_xticklabels(BANDX)
        ax.set_yticks(range(len(MED)))
        ax.set_yticklabels(MED if ax is axes[0] else [""] * len(MED))
        ax.set_title(f"Island {isl.key}", pad=3)
        ax.grid(False)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if A[i, j] > 0.35 else "black")
    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.018)
    cb.set_label("allocation share $\\alpha_{j,k}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 4
def fig_foresight(path):
    FS.use(7.2)
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.35))

    ax = axes[0]
    ax2 = ax.twinx(); ax2.grid(False)
    for i, isl in enumerate(ISLANDS):
        f = load(f"foresight_{isl.key}.pkl")
        if not f:
            continue
        du = sorted(k for k in f.get("duration", {}) if k != "none")
        ax.plot(du, [f["duration"][k]["kappa"] for k in du], marker=FS.MARK[i],
                mfc="white", mec="black", color="black", ms=4.5,
                linestyle=FS.LINES[i], lw=0.9, label=isl.key)
        vmax = max(f["duration"][k]["v_pf"] for k in du)
        ax2.plot(du, [f["duration"][k]["v_pf"] / vmax for k in du],
                 marker=FS.MARK[i], mfc=FS.GREY[3], mec=FS.GREY[1],
                 color=FS.GREY[2], ms=3.0, lw=0.7, linestyle=(0, (1, 1.8)))
    ax.axhline(1.0, color=FS.GREY[2], lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("storage duration (h)")
    ax.set_ylabel("capture ratio $\\kappa$")
    ax.set_ylim(0.92, 1.03)
    ax2.set_ylabel("clairvoyant value (normalised)", fontsize=6.4)
    ax2.set_ylim(0.0, 1.25)
    ax2.tick_params(labelsize=6)
    FS.outside_legend(ax, ncol=3, y=1.02, expand=False, title="Island")

    ax = axes[1]
    for i, isl in enumerate(ISLANDS):
        f = load(f"foresight_{isl.key}.pkl")
        if not f:
            continue
        ps = sorted(f.get("reliance", {}))
        ax.plot(ps, [f["reliance"][p]["kappa"] for p in ps], marker=FS.MARK[i],
                mfc="white", mec="black", color="black", ms=4.5,
                linestyle=FS.LINES[i], lw=0.9, label=isl.key)
    ax.axhline(1.0, color=FS.GREY[2], lw=0.6)
    ax.set_xlabel("share of peak demand on energy-limited assets")
    ax.set_ylabel("capture ratio $\\kappa$")
    ax.set_ylim(0.92, 1.03)
    FS.outside_legend(ax, ncol=3, y=1.02, expand=False, title="Island")

    ax = axes[2]
    tags = ["PF-LP", "SP", "PF-LP-R", "GA-RB", "PSO-RB", "ADR", "DPC"]
    for i, isl in enumerate(ISLANDS):
        m = load(f"eval_{isl.key}.pkl")
        vals = []
        for tag in tags:
            c = np.mean([r["lcoe"] for r in m["P1"][tag]])
            a_ = np.mean([r["lcoe"] for r in m["P3"][tag]])
            vals.append(100.0 * (a_ - c) / c)
        ax.bar(np.arange(len(tags)) + (i - 1) * 0.26, vals, 0.26,
               facecolor=FS.GREY[5 - i], edgecolor="black", hatch=FS.HATCH[i],
               label=isl.key)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=40, ha="right")
    ax.set_ylabel("implementability gap $\\Gamma$ (\\%)")
    FS.outside_legend(ax, ncol=3, y=1.02, expand=False, title="Island")
    FS.finish(fig, path, pad=0.3, w_pad=1.9)


# ------------------------------------------------------------------ Fig 5
def fig_protocols(path, tags=("PF-LP", "SP", "PF-LP-R", "SP-R", "GA-RB",
                             "PSO-RB", "ADR", "DPC")):
    FS.use(7.6)
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    ax = axes[0]
    prot = [("P1", "I: clairvoyant"), ("P2", "II: receding horizon"),
            ("P3", "III: fitted policy"), ("P4", "IV: native")]
    w = 0.20
    for i, (key, lab) in enumerate(prot):
        mu, sd = [], []
        for tag in tags:
            v = []
            for isl in ISLANDS:
                m = load(f"eval_{isl.key}.pkl")
                if tag in m[key]:
                    v += [r["lcoe"] for r in m[key][tag]]
            mu.append(np.mean(v) if v else np.nan)
            sd.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(tags)) + (i - 1.5) * w, mu, w, yerr=sd, capsize=1.8,
               facecolor=FS.GREY[5 - i], edgecolor="black",
               hatch=FS.HATCH[i], label=lab,
               error_kw=dict(elinewidth=0.6, capthick=0.6))
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("LCOE (EUR/MWh)")
    FS.outside_legend(ax, ncol=2, y=1.03)

    ax = axes[1]
    for i, isl in enumerate(ISLANDS):
        m = load(f"eval_{isl.key}.pkl")
        base = np.mean([r["lcoe"] for r in m["P4"]["DPC"]])
        vals = [100 * (np.mean([r["lcoe"] for r in m["P4"][t]]) - base) / base
                for t in tags]
        ax.bar(np.arange(len(tags)) + (i - 1) * 0.26, vals, 0.26,
               facecolor=FS.GREY[5 - i], edgecolor="black", hatch=FS.HATCH[i],
               label=f"Island {isl.key}")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("LCOE relative to co-design (%)")
    FS.outside_legend(ax, ncol=3, y=1.03)
    FS.finish(fig, path, pad=0.3, w_pad=1.6)


# ------------------------------------------------------------------ Fig 6
def fig_portfolio(path, tags=("PF-LP", "SP", "PF-LP-R", "GA-RB", "PSO-RB",
                             "ADR", "DPC")):
    FS.use(7.6)
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.4), sharey=True)
    for ax, isl in zip(axes, ISLANDS):
        m = load(f"eval_{isl.key}.pkl")
        bottom = np.zeros(len(tags))
        for j, nm in enumerate(MED):
            v = np.array([m["caps"][t]["E"][j] if t in m["caps"] else 0
                          for t in tags], dtype=float) / isl.peak_mw
            ax.bar(range(len(tags)), v, 0.62, bottom=bottom,
                   facecolor=FS.GREY[4 - j], edgecolor="black",
                   hatch=FS.HATCH[j + 1], label=nm if ax is axes[0] else None)
            bottom += v
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=35, ha="right")
        ax.set_title(f"Island {isl.key}", pad=3)
    axes[0].set_ylabel("storage energy (h of peak demand)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.10),
               frameon=True, edgecolor="black")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 7
def fig_traces(path):
    import scenarios as S
    from system import evaluate
    from config import build_system_cost
    FS.use(7.2)
    media, sc = build_media(), build_system_cost()
    des = load("designs_A.pkl")
    ev = load("eval_A.pkl")
    d, tr, te = __import__("pipeline").island_data(ISLANDS[0])
    dd = des[ev["sel"]["DPC"]["best"]]
    r = evaluate(dd["params"], dd["cfg"], media, sc, BAND_TAU_H,
                 S.single(te[0]), mu=0.0, return_traj=True)
    soc = np.asarray(r["soc"])
    E = np.asarray(ev["caps"]["DPC"]["E"], float)
    fig, axes = plt.subplots(4, 2, figsize=(WIDE, 3.7), sharex="col")
    w0, w1 = 120 * 24, 134 * 24
    t_d = np.arange(w1 - w0) / 24.0
    days = soc[:8760].reshape(365, 24, soc.shape[1])
    dd_ = np.arange(365)
    for j in range(4):
        ax = axes[j, 0]
        ax.fill_between(t_d, 0, soc[w0:w1, j], facecolor=FS.GREY[4],
                        edgecolor="none")
        ax.plot(t_d, soc[w0:w1, j], color="black", lw=0.55)
        ax.set_ylim(-0.03, 1.42)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.02, 0.86, f"{MED[j]}  ({E[j]:.1f} MWh)",
                transform=ax.transAxes, ha="left", va="center", fontsize=6.2,
                bbox=dict(fc="white", ec="black", lw=0.4, pad=1.3))
        ax = axes[j, 1]
        lo, hi = days[:, :, j].min(axis=1), days[:, :, j].max(axis=1)
        ax.fill_between(dd_, lo, hi, facecolor=FS.GREY[4], edgecolor="none")
        ax.plot(dd_, days[:, :, j].mean(axis=1), color="black", lw=0.6)
        ax.set_ylim(-0.03, 1.42)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels([])
        ax.set_xlim(0, 365)
    axes[3, 0].set_xlabel("day of evaluation year (14-day detail)")
    axes[3, 1].set_xlabel("day of evaluation year (daily range)")
    axes[0, 0].set_title("detail", fontsize=7, pad=2)
    axes[0, 1].set_title("whole year", fontsize=7, pad=2)
    fig.tight_layout(pad=0.3, h_pad=0.25, w_pad=0.8)
    fig.subplots_adjust(left=0.115)
    fig.text(0.016, 0.52, "state of charge (per unit)", rotation=90,
             va="center", ha="center", fontsize=7.4)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 8
def fig_ablation_sens(path):
    FS.use(7.6)
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    ax = axes[0]
    names, vals, err = [], [], []
    keys = ["NoBand", "NoSimplex", "NoRestore", "NoSeasonal", "NoRisk",
            "NoAnneal", "NoForecast", "SingleBand"]
    pretty = {"NoBand": "no band basis", "NoSimplex": "no allocation simplex",
              "NoRestore": "no restoring term", "NoSeasonal": "no seasonal target",
              "NoRisk": "no tail-risk term", "NoAnneal": "no annealing",
              "NoForecast": "no anticipation", "SingleBand": "single band"}
    for k in keys:
        dd = []
        for isl in ISLANDS:
            try:
                a = load(f"ablation_{isl.key}.pkl")
            except FileNotFoundError:
                continue
            base = np.mean(a["Full"]["native"])
            dd.append(100 * (np.mean(a[k]["native"]) - base) / base)
        if not dd:
            continue
        names.append(pretty[k]); vals.append(np.mean(dd)); err.append(np.std(dd, ddof=1) if len(dd) > 1 else 0)
    y = np.arange(len(names))
    ax.barh(y, vals, 0.6, xerr=err, facecolor=FS.GREY[4], edgecolor="black",
            hatch=FS.HATCH[2], error_kw=dict(elinewidth=0.6, capthick=0.6),
            capsize=1.8)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("increase in LCOE when removed (%)")

    ax = axes[1]
    try:
        s = load("sens_A.pkl")
    except FileNotFoundError:
        FS.finish(fig, path, pad=0.3, w_pad=1.6)
        return
    fk = sorted([k for k in s if k.startswith("fuel=")],
                key=lambda z: float(z.split("=")[1]))
    x = [float(k.split("=")[1]) for k in fk]
    E = np.array([s[k]["caps"]["E"] for k in fk], dtype=float)
    tot = E.sum(axis=1, keepdims=True)
    share = E / np.maximum(tot, 1e-9) * 100
    bottom = np.zeros(len(x))
    for j, nm in enumerate(MED):
        ax.bar(range(len(x)), share[:, j], 0.6, bottom=bottom,
               facecolor=FS.GREY[4 - j], edgecolor="black",
               hatch=FS.HATCH[j + 1], label=nm)
        bottom += share[:, j]
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([f"{v:.1f}" for v in x])
    ax.set_xlabel("delivered fuel price multiplier")
    ax.set_ylabel("share of energy capacity (%)")
    ax.set_ylim(0, 100)
    FS.outside_legend(ax, ncol=2, y=1.03)
    FS.finish(fig, path, pad=0.3, w_pad=1.6)


# ------------------------------------------------------------------ Fig 9
def fig_robust_cd(path):
    FS.use(7.6)
    import stats as ST
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    ax = axes[0]
    tags = ["PF-LP", "GA-RB", "PSO-RB", "ADR", "DPC"]
    pert = ["nominal", "fc+50%", "fc+100%", "load+15%", "wind-20%", "outage10d"]
    w = 0.20
    for i, tag in enumerate(tags):
        vals = []
        for p in pert:
            v = []
            for isl in ISLANDS:
                r = load(f"robust4_{isl.key}.pkl") or load(f"robust_{isl.key}.pkl")
                if r is None:
                    continue
                if tag in r and p in r[tag]:
                    v += r[tag][p]["lcoe"]
            vals.append(np.mean(v) if v else np.nan)
        ax.bar(np.arange(len(pert)) + (i - 2) * w, vals, w,
               facecolor=FS.GREY[5 - i], edgecolor="black", hatch=FS.HATCH[i],
               label=tag)
    ax.set_xticks(range(len(pert)))
    ax.set_xticklabels(["nominal", "fc $\\times$1.5", "fc $\\times$2",
                        "load $+$15%", "wind $-$20%", "10-day lull"],
                       rotation=25, ha="right")
    ax.set_ylabel("LCOE (EUR/MWh)")
    FS.outside_legend(ax, ncol=3, y=1.03)

    ax = axes[1]
    mats, names = [], ["PF-LP", "SP", "PF-LP-R", "SP-R", "GA-RB",
                       "PSO-RB", "ADR", "DPC"]
    for isl in ISLANDS:
        m = load(f"eval_{isl.key}.pkl")
        rows = np.array([[r["lcoe"] for r in m["P4"][t]] for t in names]).T
        mats.append(rows)
    X = np.vstack(mats)
    fr = ST.friedman_nemenyi(X)
    mr = np.array(fr["mean_rank"])
    cd = fr["cd"]
    order = np.argsort(mr)
    ax.set_xlim(0.7, len(names) + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.72, 1, len(names), color="black", lw=0.8)
    for k in range(1, len(names) + 1):
        ax.vlines(k, 0.70, 0.74, color="black", lw=0.8)
        ax.text(k, 0.63, str(k), ha="center", fontsize=6.6)
    for i, idx in enumerate(order):
        side = -1 if i < len(order) / 2 else 1
        yy = 0.56 - 0.068 * (i if side < 0 else len(order) - 1 - i)
        xx = 1.0 if side < 0 else len(names)
        ax.plot([mr[idx], mr[idx], xx], [0.70, yy, yy], color="black", lw=0.6)
        ax.text(xx + side * 0.10, yy, f"{names[idx]} ({mr[idx]:.2f})",
                ha="right" if side < 0 else "left", va="center", fontsize=6.4)
    ax.hlines(0.90, 1, 1 + cd, color="black", lw=1.4)
    ax.vlines([1, 1 + cd], 0.88, 0.92, color="black", lw=1.0)
    ax.text(1 + cd / 2, 0.94, f"CD $=$ {cd:.2f}", ha="center", fontsize=6.6)
    ax.text(len(names) / 2 + 0.5, 0.55, "", ha="center", fontsize=6.8)
    FS.finish(fig, path, pad=0.3, w_pad=1.6)


ALL = {
    "fig2_bands.png": fig_bands,
    "fig3_allocation.png": fig_alpha,
    "fig4_foresight.png": fig_foresight,
    "fig5_protocols.png": fig_protocols,
    "fig6_portfolio.png": fig_portfolio,
    "fig7_traces.png": fig_traces,
    "fig8_ablation.png": fig_ablation_sens,
    "fig9_robust.png": fig_robust_cd,
}

if __name__ == "__main__":
    import sys
    sel = sys.argv[1:] or list(ALL)
    for name in sel:
        try:
            ALL[name](os.path.join(FIG, name))
            print("wrote", name)
        except Exception as e:
            print("skip", name, type(e).__name__, e)
