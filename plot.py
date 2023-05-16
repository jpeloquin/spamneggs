"""Utility functions to support plots in other modules"""
from collections import namedtuple

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .colors import diverging_bky_60_10_c30_n256
from .core import Parameter, ureg

CMAP_DIVERGE = mpl.colors.LinearSegmentedColormap(
    "div_blue_black_red", diverging_bky_60_10_c30_n256
)
COLOR_DEEMPH = "dimgray"
FONTSIZE_FIGLABEL = 12
FONTSIZE_AXLABEL = 10
FONTSIZE_TICKLABEL = 8
WS_PAD_ALL = 2 / 72

mpl.rcParams["savefig.dpi"] = 300  # we use imshow

FigResult = namedtuple("FigResultAxarr", ["fig", "ax"])
FigAxCbar = namedtuple("FigResultAxarr", ["fig", "ax", "cbar"])
FigResultAxarr = namedtuple("FigResultAxarr", ["fig", "axarr"])


def format_number(v, digits=2):
    """Return formatted number

    Works like "g" format code except only -2 ≤ exponent ≤ 3 are formatted as a
    decimal value.
    """
    m = np.log10(v)
    if (-2 <= m) and (m <= 3):
        return format(v, f".{digits}g")
    else:
        return format(v, f".{digits}e")


def format_parameter_values(definitions, values, digits=2):
    """Return list of pretty-printed parameter values"""
    out = []
    for p, v in zip(definitions, values):
        v = format_number(v, digits)
        if isinstance(p, Parameter):
            nm = p.name
            units = p.units
        else:
            nm, units = p
            units = ureg(units).u
        if isinstance(units, str):
            units = ureg(units).u
        if units is None or units == "1":
            out.append(f"{nm} = {v}")
        else:
            out.append(f"{nm} = {v} {units:~P}")
    return out


def remove_spines(ax):
    for k in ax.spines:
        ax.spines[k].set_visible(False)


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


def plot_eigenvalues_histogram(values, xlabel, ylabel, log=True, errors=None):
    """Return bar plot of eigenvalues / singular values"""
    fig = Figure(constrained_layout=True)
    fig.set_size_inches((4, 3))
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color=COLOR_DEEMPH, linewidth=0.5, linestyle="dotted")
    x = 1 + np.arange(len(values))
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXLABEL)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(x))
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXLABEL)
    bar_colors = np.full(len(values), "C0")
    legend_handles = []
    if log:
        ax.set_yscale("log")
        sign = np.sign(values)
        values = np.abs(values)
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=12))
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ymin = 10 ** np.floor(np.log10(np.min(values)))
        ymax = 10 ** np.ceil(np.log10(np.max(values)))
        if errors is not None:
            ymax = max(ymax, 10 ** np.ceil(np.log10(np.max(errors))))
        ax.set_ylim(ymin, ymax)
        bar_colors[sign < 0] = "C3"
        legend_handles += [
            Patch(facecolor="C0", label="Positive"),
            Patch(facecolor="C3", label="Negative"),
        ]
    bars = ax.bar(x, values, color=bar_colors)
    # Plot bounds
    if errors is not None:
        for c, b in zip(errors, bars):
            width = b.properties()["width"]
            xmin = b.properties()["x"]
            xmax = xmin + width
            ax.hlines(
                c,
                xmin=xmin,
                xmax=xmax,
                color="black",
                lw=1,
            )
        legend_handles += [
            Line2D([0], [0], color="black", linewidth=1, label="Error Bound")
        ]
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
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
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
        ax.get_shared_x_axes().join(f.axarr[0, 0], ax)
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


def plot_matrix(
    mat, scale="linear", tick_labels=None, cbar_label=None, title=None, format_str=".2g"
):
    """Plot a square matrix as a heatmap with values written to each cell"""
    nv = mat.shape[0]
    fig = Figure()
    fig.set_tight_layout(False)  # silence warning about tight_layout compatibility
    # FigureCanvas(fig)
    if tick_labels is None:
        tick_labels = 1 + np.arange(nv)
    # Set size to match number of variables
    in_per_var = 0.8
    mat_w = in_per_var * nv
    mat_h = mat_w
    cbar_lpad = 12 / 72
    cbar_w = 0.3
    cbar_h = mat_h
    cbar_rpad = (24 + FONTSIZE_AXLABEL) / 72
    fig_w = WS_PAD_ALL + mat_w + cbar_lpad + cbar_w + cbar_rpad + WS_PAD_ALL
    fig_h = (
        WS_PAD_ALL + FONTSIZE_FIGLABEL / 72 + FONTSIZE_AXLABEL / 72 + mat_h + WS_PAD_ALL
    )
    fig.set_size_inches(fig_w, fig_h)
    # Plot the matrix itself
    pos_main_in = np.array((WS_PAD_ALL, WS_PAD_ALL, mat_w, mat_h))
    ax = fig.add_axes(pos_main_in / [fig_w, fig_h, fig_w, fig_h])
    cmap = mpl.cm.get_cmap("cividis")
    vextreme = np.max(np.abs(mat))
    if scale == "linear":
        norm = mpl.colors.Normalize(vmin=-vextreme, vmax=vextreme)
    elif scale == "log":
        if np.min(mat) <= 0:
            norm = mpl.colors.SymLogNorm(linthresh=np.min(np.abs(mat)), linscale=0.5)
        else:
            norm = mpl.colors.LogNorm(vmin=-vextreme, vmax=vextreme)
    else:
        raise ValueError("Scale choice '{scale}' not recognized.")
    im = ax.matshow(
        mat,
        cmap=cmap,
        norm=norm,
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
            WS_PAD_ALL + mat_w + cbar_lpad,
            WS_PAD_ALL,
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
    fig_w = WS_PAD_ALL + Δleft + fig_w + Δright + WS_PAD_ALL
    fig.set_size_inches(fig_w, fig_h)
    ## Re-apply the axes sizes, which will have changed because they are stored in
    ## figure units
    pos_main_in[0] += Δleft
    pos_cbar_in[0] += Δleft
    ax.set_position(pos_main_in / [fig_w, fig_h, fig_w, fig_h])
    cax.set_position(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    return FigResultAxarr(fig, (ax, cax))


def plot_neighborhood(
    f,
    θ,
    vectors,
    n,
    relative_extent=((-1, 1), (-1, 1)),
    limits=None,
    θ_label=None,
    xlabel="Vector 1",
    ylabel="Vector 2",
):
    """Return plot of cost function in a 2D plane

    :param f: The function to plot.  Must accept a vector of parameter values of the same shape as `θ` and return a scalar.

    :param θ: Parameter values about which to plot the function `f`.

    :param vectors: List of two vectors [v1, v2] in parameter space.  v1 defines the
    plot's x-axis.  v2 defines the plot's y-axis.  The product of the vector and the
    `relative_extent` values define the extent of the plot.

    :param n: [n1, n2] or [(n1_left, n1_right), (n2_left, n2_right)].  Each value
    defines the number of points to evaluate along the corresponding axis direction
    defined by `vectors`.  If [n1, n2] is given each value will be used twice.

    :param relative_extent: [(v1_left, v1_right), (v2_left, v2_right)].  The left and
    right extents of the plot along v1 and v2, starting at θ.

    :param limits: Array-like of (min, max) limits for each parameter.  Points
    outside these limits will remain blank on the plot.

    :param θ_label: List of labels for each component of the vectors.

    :retuns: FigAxCbar named tuple.

    In intended use, the vectors are unit vectors in scaled parameter space and the
    relative_extent values are distances in the same scaled parameter space.

    """
    # The figure layout could be improved somewhat: (1) The eigenvector legends are
    # prevented from overlapping the central x and y axis labels only by some
    # guesstimated padding.  Constrained layout allowed overlap, I think because the
    # padding is forced by append_axes.  (2) The width of the colorbar cannot be
    # precisely controlled.

    # TODO: This figure is complicated enough to be a class.  Add appropriate functions
    # to set the label.ax.set_aspect("equal")

    # Expand/fixup parameters
    θ = np.array(θ)
    if all([not hasattr(a, "__len__") for a in n]):
        n = [(a, a) for a in n]
    if limits is None:
        limits = np.stack([np.full(len(θ), -np.inf), np.full(len(θ), np.inf)], axis=-1)
    else:
        limits = np.array(limits)
    # Create grid for calculation
    si = [None, None]  # preallocation
    for i, v in enumerate(vectors):
        si[i] = np.hstack(
            [
                np.linspace(relative_extent[i][0], 0, n[i][0] + 1),
                np.linspace(0, relative_extent[i][1], n[i][1] + 1)[1:],
            ]
        )
    # Calculate values
    neighborhood = np.full((len(si[0]), len(si[1])), np.nan)
    for i, sx in enumerate(si[0]):
        for j, sy in enumerate(si[1]):
            θ_pt = θ + sx * vectors[0] + sy * vectors[1]
            if any(θ_pt < limits[:, 0]):
                continue
            if any(θ_pt > limits[:, 1]):
                continue
            neighborhood[i, j] = f(θ_pt)
    # Calculate figure size.  Similar to plot_matrix where possible.  From here on,
    # everything is plot formatting.
    fig = Figure(constrained_layout=True)
    mat_w = 4
    mat_h = (
        mat_w
        * (relative_extent[1][1] - relative_extent[1][0])
        / (relative_extent[0][1] - relative_extent[0][0])
    )
    cbar_lpad = 12 / 72
    cbar_w = 0.2
    lplot_w = 0.5  # width of marginal line plot, apparently relative to main axes
    vbar_w = 1.0  # width of eigenvector legend bar, apparently relative to main axes
    cbar_rpad = (24 + FONTSIZE_AXLABEL) / 72
    fig_w = (
        WS_PAD_ALL
        + 0.35
        + vbar_w * mat_w
        + FONTSIZE_AXLABEL / 72
        + FONTSIZE_TICKLABEL / 72
        + vbar_w * mat_w / 2
        + FONTSIZE_AXLABEL / 72
        + mat_w
        + cbar_lpad
        + lplot_w * mat_w / 2
        + cbar_rpad
        + WS_PAD_ALL
    )
    fig_h = (
        WS_PAD_ALL
        + mat_h
        + FONTSIZE_AXLABEL / 72
        + lplot_w * mat_w / 2
        + FONTSIZE_FIGLABEL / 72
        + FONTSIZE_AXLABEL / 72
        + vbar_w * mat_w / 2
        + 0.35
        + WS_PAD_ALL
    )
    fig.set_size_inches(fig_w, fig_h)
    # Plot the main heatmap showing the local neighborhood
    ax = fig.add_subplot()
    im = ax.imshow(
        neighborhood.T,  # imshow swaps axes
        cmap="magma",
        extent=[si[0][0], si[0][-1], si[1][0], si[1][-1]],
        origin="lower",
    )
    remove_spines(ax)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)
    ax.axvline(linestyle="--", linewidth=0.5, color="k")
    ax.axhline(linestyle="--", linewidth=0.5, color="k")
    div = make_axes_locatable(ax)
    # Add the colorbar
    ax_cbar = div.append_axes("right", size=cbar_w, pad=cbar_lpad)
    ax_cbar.tick_params("y", labelsize=FONTSIZE_TICKLABEL)
    cbar = fig.colorbar(im, cax=ax_cbar)

    def style_lineplot_axes(ax):
        for k in ["top", "bottom", "left", "right"]:
            ax.spines[k].set_linewidth(0.5)
            ax.spines[k].set_color(COLOR_DEEMPH)
        ax.tick_params("x", labelsize=FONTSIZE_TICKLABEL)
        ax.tick_params("y", labelsize=FONTSIZE_TICKLABEL)

    # Add a line plot along vector 1, crossing the origin
    ax_l1 = div.append_axes(
        "bottom", size=vbar_w, pad=FONTSIZE_TICKLABEL / 2 / 72 + WS_PAD_ALL, sharex=ax
    )
    style_lineplot_axes(ax_l1)
    ax_l1.set_xlabel(xlabel, fontsize=FONTSIZE_AXLABEL)
    ax_l1.plot(si[0], neighborhood[:, n[0][0]], color="C0", linewidth=1)
    ax_l1.set_xlabel(xlabel, fontsize=FONTSIZE_AXLABEL)
    ax_l1.tick_params("x", top=True, bottom=True)
    ax_l1.tick_params("y")
    # Add a line plot along vector 2, crossing the origin
    ax_l2 = div.append_axes(
        "left", size=vbar_w, pad=FONTSIZE_TICKLABEL / 2 / 72 + WS_PAD_ALL, sharey=ax
    )
    style_lineplot_axes(ax_l2)
    ax_l2.plot(neighborhood[n[1][0], :], si[1], color="C0", linewidth=1)
    ax_l2.set_ylabel(ylabel, fontsize=FONTSIZE_AXLABEL)
    ax_l2.tick_params("x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_l2.tick_params("y", left=True, right=True)
    # Legend style constants
    cmap_vec = CMAP_DIVERGE
    norm_vec = mpl.colors.Normalize(vmin=-1, vmax=1)
    # Add a legend for vector 1
    ax_vec1 = div.append_axes(
        "bottom",
        vbar_w,
        pad=lplot_w,
    )
    ax_vec1.matshow(
        np.atleast_2d(vectors[0]),
        cmap=cmap_vec,
        norm=norm_vec,
        origin="lower",
        extent=(-0.5, len(θ) - 0.5, 0, 1),
    )
    remove_spines(ax_vec1)
    ax_vec1.tick_params("x", top=False, labeltop=False, labelbottom=True)
    ax_vec1.tick_params("y", left=False, labelleft=False)
    if θ_label is not None:
        ax_vec1.xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(len(θ))))
        ax_vec1.set_xticklabels(θ_label, fontsize=FONTSIZE_TICKLABEL - 1, rotation=90)
    else:
        ax_vec1.tick_params("x", bottom=False, labelbottom=False)
    # Write the value of each component of vector 1 as text
    for i in range(len(θ)):
        ax_vec1.text(
            i,
            0.5,
            format(vectors[0][i], ".2f"),
            ha="center",
            va="center",
            backgroundcolor=(1, 1, 1, 0.3),
            fontsize=FONTSIZE_TICKLABEL - 2,
        )
    # Add a legend for vector 2
    ax_vec2 = div.append_axes(
        "left",
        vbar_w,
        pad=0.35 + (FONTSIZE_TICKLABEL + FONTSIZE_AXLABEL) / 72,
    )
    ax_vec2.matshow(
        np.atleast_2d(vectors[1]).T,
        cmap=cmap_vec,
        norm=norm_vec,
        extent=(0, 1, len(θ) - 0.5, -0.5),  # l r b t
        origin="upper",
    )
    remove_spines(ax_vec2)
    ax_vec2.tick_params(
        "x", top=False, labeltop=False, bottom=False, labelsize=FONTSIZE_TICKLABEL
    )
    ax_vec2.tick_params("y", left=True, labelleft=True, labelsize=FONTSIZE_TICKLABEL)
    if θ_label is not None:
        ax_vec2.yaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(len(θ))))
        ax_vec2.set_yticklabels(θ_label, fontsize=FONTSIZE_TICKLABEL - 1)
    else:
        ax_vec2.tick_params("x", bottom=False, labelbottom=False)
    # Write the value of each component of vector 1 as text
    for i in range(len(θ)):
        ax_vec2.text(
            0.5,
            i,
            format(vectors[1][i], ".2f"),
            ha="center",
            va="center",
            backgroundcolor=(1, 1, 1, 0.3),
            fontsize=FONTSIZE_TICKLABEL - 2,
        )
    return FigAxCbar(fig, ax, cbar)