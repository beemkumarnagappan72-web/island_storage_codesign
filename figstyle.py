"""Shared grayscale plotting style for journal reproduction.

Figures are restricted to black, white and a small ordered set of greys,
with hatching and marker shape carrying the categorical distinction, so
that nothing is lost when the article is printed without colour.  Legends
are placed outside the data area and axes are padded generously so that
no annotation, legend or label overlaps plotted content.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

GREY = ["#000000", "#4d4d4d", "#7f7f7f", "#a6a6a6", "#c9c9c9", "#e6e6e6"]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "|||", "+++", "ooo"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINES = [(0, ()), (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.2, 1, 1.2)),
         (0, (3, 1, 1, 1, 1, 1)), (0, (7, 2))]


def use(base=8.0):
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "xtick.labelsize": base - 0.8,
        "ytick.labelsize": base - 0.8,
        "legend.fontsize": base - 1.0,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "patch.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linestyle": (0, (1, 2)),
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#000000",
        "legend.borderpad": 0.45,
        "legend.handlelength": 2.1,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "hatch.linewidth": 0.55,
    })


def outside_legend(ax, ncol=3, y=1.02, expand=True, **kw):
    """Legend above the axes, clear of all plotted content."""
    if expand:
        return ax.legend(loc="lower left", bbox_to_anchor=(0.0, y, 1.0, 0.12),
                         mode="expand", ncol=ncol, borderaxespad=0.0, **kw)
    kw.setdefault("handlelength", 1.5)
    kw.setdefault("columnspacing", 0.9)
    kw.setdefault("handletextpad", 0.4)
    return ax.legend(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     borderaxespad=0.0, **kw)


def finish(fig, path, pad=0.55, h_pad=0.9, w_pad=0.9):
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    fig.savefig(path)
    plt.close(fig)
