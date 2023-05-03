"""Utility functions to support plots in other modules"""
from collections import namedtuple

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

COLOR_DEEMPH = "dimgray"
FONTSIZE_FIGLABEL = 12
FONTSIZE_AXLABEL = 10
FONTSIZE_TICKLABEL = 8


FigResult = namedtuple("FigResultAxarr", ["fig", "ax"])
FigResultAxarr = namedtuple("FigResultAxarr", ["fig", "axarr"])


def symlog(arr):
    """Return signed log10 transform of all nonzero values in array"""
    arr = np.array(arr)
    transformed = np.zeros(arr.shape)
    m_nonzero = arr != 0
    transformed[m_nonzero] = np.sign(arr[m_nonzero]) * np.log10(np.abs(arr[m_nonzero]))
    return transformed


def fig_template_axarr(nh, nw, xlabel=None, ylabel=None):
    fig = Figure(constrained_layout=True)
    fig.set_constrained_layout_pads(
        wspace=2 / 72, hspace=2 / 72, w_pad=2 / 72, h_pad=2 / 72
    )
    fig.set_size_inches((2.4 * nw + 0.25, 1.76 * nh + 0.25))
    gs = GridSpec(nh, nw, figure=fig)
    axarr = np.full((nh, nw), None)
    for i in range(nh):  # rows
        for j in range(nw):  # columns
            ax = fig.add_subplot(gs[i, j])
            axarr[i, j] = ax
            ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
            ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
            text = ax.yaxis.get_offset_text()
            text.set_size(FONTSIZE_TICKLABEL)
    if xlabel is not None:
        for j in range(nw):
            axarr[-1, j].set_xlabel(xlabel, fontsize=FONTSIZE_AXLABEL)
    if ylabel is not None:
        for i in range(nh):
            axarr[i, 0].set_ylabel(ylabel, fontsize=FONTSIZE_AXLABEL)
    return FigResultAxarr(fig, axarr)


def plot_eigenvalues_histogram(values, xlabel, ylabel, log=True, cutoff=None):
    """Return bar plot of eigenvalues / singular values"""
    fig = Figure(constrained_layout=True)
    fig.set_size_inches((4, 3))
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color=COLOR_DEEMPH, linewidth=0.5, linestyle="dotted")
    x = 1 + np.arange(len(values))
    if cutoff is not None:
        ax.axhline(cutoff, color=COLOR_DEEMPH, lw=1)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXLABEL)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(x))
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXLABEL)
    bar_colors = np.full(len(values), "C0")
    if log:
        ax.set_yscale("log")
        sign = np.sign(values)
        values = np.abs(values)
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=12))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylim(
            (
                10 ** np.floor(np.log10(np.min(values))),
                10 ** np.ceil(np.log10(np.max(values))),
            )
        )
        bar_colors[sign < 0] = "C3"
        ax.legend(
            handles=[
                Patch(facecolor="C0", label="Positive"),
                Patch(facecolor="C3", label="Negative"),
            ],
            loc="upper right",
        )
    ax.bar(x, values, color=bar_colors)
    for k in ax.spines:
        ax.spines[k].set_visible(False)
    ax.tick_params(
        axis="x",
        color=COLOR_DEEMPH,
        labelsize=FONTSIZE_TICKLABEL,
        labelcolor=COLOR_DEEMPH,
    )
    ax.tick_params(
        axis="y",
        color=COLOR_DEEMPH,
        labelsize=FONTSIZE_TICKLABEL,
        labelcolor=COLOR_DEEMPH,
    )
    return FigResult(fig, ax)


def plot_eigenvalues_pdfs(eigenvalues):
    """Return figure with probability distributions of eigenvalues

    :param eigenvalues: Matrix of eigenvalues with shape (# of eigenvalues,
    # of samples).

    :param xlabel: x-axis label for plot

    :param ylabel: y-axis label for plot
    """
    nv, ns = eigenvalues.shape
    f = fig_template_axarr(nv, 1)
    sz_in = f.fig.get_size_inches()
    f.fig.set_size_inches((2.5 * sz_in[0], sz_in[1]))
    smallest = np.min(np.abs(eigenvalues))
    for i, x in enumerate(eigenvalues[:]):
        ax = f.axarr[i, 0]
        ax.get_shared_x_axes().join(f.axarr[0,0], ax)
        ax.set_xlabel(f"Eigenvalue {i + 1}", fontsize=FONTSIZE_AXLABEL)
        ax.set_xscale("symlog", linthresh=smallest)
        ax.vlines(x, ymin=-1, ymax=1, color="black")
        ax.set_ylim((-1, 1))
        ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
        for k in ax.spines:
            ax.spines[k].set_visible(False)
        # TODO: Can I get a Gaussian KDE plot in here?  The symlog axis makes it
        #  challenging.
    return f


def plot_matrix(mat, tick_labels=None, cbar_label=None, title=None, format_str=".2g"):
    """Plot a square matrix as a heatmap with values written to each cell"""
    nv = mat.shape[0]
    fig = Figure()
    fig.set_tight_layout(False)  # silence warning about tight_layout compatibility
    # FigureCanvas(fig)
    if tick_labels is None:
        tick_labels = 1 + np.arange(nv)
    # Set size to match number of variables
    in_per_var = 0.8
    pad_all = 2 / 72
    mat_w = in_per_var * nv
    mat_h = mat_w
    cbar_lpad = 12 / 72
    cbar_w = 0.3
    cbar_h = mat_h
    cbar_rpad = (24 + FONTSIZE_AXLABEL) / 72
    fig_w = pad_all + mat_w + cbar_lpad + cbar_w + cbar_rpad + pad_all
    fig_h = pad_all + FONTSIZE_FIGLABEL / 72 + FONTSIZE_AXLABEL / 72 + mat_h + pad_all
    fig.set_size_inches(fig_w, fig_h)
    # Plot the matrix itself
    pos_main_in = np.array((pad_all, pad_all, mat_w, mat_h))
    ax = fig.add_axes(pos_main_in / [fig_w, fig_h, fig_w, fig_h])
    cmap = mpl.cm.get_cmap("cividis")
    vextreme = np.max(np.abs(mat))
    cnorm = mpl.colors.Normalize(vmin=-vextreme, vmax=vextreme)
    im = ax.matshow(
        mat,
        cmap=cmap,
        norm=cnorm,
        origin="upper",
        extent=(
            -0.5,
            nv - 0.5,
            -0.5,
            nv - 0.5,
        ),
    )
    # Write the value of each cell as text
    for (i, j), d in np.ndenumerate(np.flipud(mat)):
        ax.text(
            j,
            i,
            ("{:" + format_str + "}").format(d),
            ha="center",
            va="center",
            backgroundcolor=(1, 1, 1, 0.5),
            fontsize=FONTSIZE_TICKLABEL,
        )
    pos_cbar_in = np.array(
        (
            pad_all + mat_w + cbar_lpad,
            pad_all,
            cbar_w,
            cbar_h,
        )
    )
    cax = fig.add_axes(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
    cbar.set_label(cbar_label, fontsize=FONTSIZE_AXLABEL)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
    ax.set_title(title, fontsize=FONTSIZE_FIGLABEL)
    ax.set_xticks(
        [i for i in range(nv)],
    )
    ax.set_yticks([i for i in reversed(range(nv))])
    # ^ reversed b/c origin="upper"
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis="x", labelsize=FONTSIZE_AXLABEL, bottom=False)
    ax.tick_params(axis="y", labelsize=FONTSIZE_AXLABEL)
    #
    # Resize figure to accommodate left axis tick labels, axis title, and colorbar label
    ## Left overflow
    bbox_px = ax.get_tightbbox()
    bbox_in = fig.dpi_scale_trans.inverted().transform(bbox_px)
    Δleft = -bbox_in[0, 0]
    ## Top overflow
    fig_h = bbox_in[1, 1]
    ## Right overflow
    bbox_px = cax.get_tightbbox()
    bbox_in = fig.dpi_scale_trans.inverted().transform(bbox_px)
    Δright = bbox_in[1, 0] - fig_w
    ## Resize the canvas
    fig_w = pad_all + Δleft + fig_w + Δright + pad_all
    fig.set_size_inches(fig_w, fig_h)
    ## Re-apply the axes sizes, which will have changed because they are stored in
    ## figure units
    pos_main_in[0] += Δleft
    pos_cbar_in[0] += Δleft
    ax.set_position(pos_main_in / [fig_w, fig_h, fig_w, fig_h])
    cax.set_position(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    return FigResultAxarr(fig, (ax, cax))
