"""Figure 1: information flow of the co-design framework."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import figstyle as FS


def box(ax, x, y, w, h, label, fc="white", fs=7.0, lw=0.8, ls="-", bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.30,rounding_size=0.8",
                       linewidth=lw, edgecolor="black", facecolor=fc, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            linespacing=1.35, fontweight="bold" if bold else "normal")
    return p


def arrow(ax, p0, p1, style="-|>", ls="-", lw=0.9, rad=0.0, color="black"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, linestyle=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.0, shrinkB=1.0))


def draw(path):
    FS.use(7.6)
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    box(ax, 2, 30, 17, 12,
        "Observed chronologies\ndemand, irradiance,\nwind speed", fc=FS.GREY[5])
    box(ax, 2, 15, 17, 10, "Published day-ahead\nforecasts", fc=FS.GREY[5])

    box(ax, 25, 22, 16, 13,
        "Residual formation\n$\\ell_t=D_t-\\eta P_{\\mathrm{pv}}g_t-P_wg_t^w$",
        fc="white")
    box(ax, 25, 5, 16, 10, "Anticipation blend\n$\\tilde{\\ell}_t$", fc="white")

    box(ax, 47, 24, 18, 15, "", fc="white")
    ax.text(56, 37.0, "Causal filter cascade", ha="center", va="center",
            fontsize=7.0)
    for i in range(5):
        ax.add_patch(Rectangle((49.2 + i * 3.1, 26.4), 2.3, 2.6 + i * 1.0,
                               facecolor=FS.GREY[5 - i], edgecolor="black",
                               linewidth=0.5))
    ax.text(56, 34.3, "bands $b_t^{(1)}\\ldots b_t^{(K+1)}$", ha="center",
            fontsize=6.8)
    ax.text(56, 25.1, "fast $\\rightarrow$ slow", ha="center", fontsize=6.4,
            style="italic")

    box(ax, 47, 5, 18, 12, "Allocation simplex\n$\\sum_j\\alpha_{j,k}\\leq 1$",
        fc="white")

    box(ax, 71, 22, 26, 17, "", fc="white")
    ax.text(84, 37.0, "Storage portfolio", ha="center", va="center", fontsize=7.0)
    names = ["Battery", "Hydrogen", "Pumped hydro", "Thermal"]
    for i, nm in enumerate(names):
        y0 = 29.8 - (i // 2) * 5.0
        ax.add_patch(Rectangle((73.2 + (i % 2) * 12.2, y0), 11.0, 4.0,
                               facecolor=FS.GREY[5], edgecolor="black",
                               linewidth=0.5))
        ax.text(73.2 + (i % 2) * 12.2 + 5.5, y0 + 2.0, nm, ha="center",
                va="center", fontsize=6.6)
    ax.text(84, 23.4, "state of charge, saturation, degradation", ha="center",
            fontsize=6.3, style="italic")

    box(ax, 71, 5, 26, 12,
        "Dispatchable generator,\ncurtailment, unserved energy\n"
        "$\\rightarrow$ annualised cost, CVaR", fc=FS.GREY[5])

    arrow(ax, (19.6, 37.0), (24.4, 31.0))
    arrow(ax, (19.6, 20.6), (24.4, 11.0))
    arrow(ax, (33.0, 21.4), (33.0, 15.6))
    arrow(ax, (41.6, 10.6), (46.4, 25.0), rad=0.12)
    arrow(ax, (41.6, 29.0), (46.4, 31.4))
    arrow(ax, (56.0, 23.4), (56.0, 17.6))
    arrow(ax, (65.6, 12.0), (70.4, 27.0), rad=0.12)
    arrow(ax, (65.6, 33.0), (70.4, 33.0))
    arrow(ax, (84.0, 21.4), (84.0, 17.6))

    ax.add_patch(FancyArrowPatch((71.0, 8.0), (25.6, 2.2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9,
                                 linestyle=(0, (4, 2)), color="black",
                                 connectionstyle="arc3,rad=0.10"))
    ax.text(48, 0.6, "reverse-mode gradient through the smoothed plant "
                     "$\\rightarrow$ capacities and policy gains",
            ha="center", fontsize=6.8, style="italic")
    FS.finish(fig, path, pad=0.15)


if __name__ == "__main__":
    draw("../figs/fig1_architecture.png")
