"""Utility functions to support plots in other modules"""
from collections import namedtuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

FONTSIZE_FIGLABEL = 12
FONTSIZE_AXLABEL = 10
FONTSIZE_TICKLABEL = 8


FigResultAxarr = namedtuple("FigResultAxarr", ["fig", "axarr"])


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
