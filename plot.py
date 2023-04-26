"""Utility functions to support plots in other modules"""
from collections import namedtuple

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

FONTSIZE_FIGLABEL = 12
FONTSIZE_AXLABEL = 10
FONTSIZE_TICKLABEL = 8


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
