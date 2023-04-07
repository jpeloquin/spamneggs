"""Numerical analysis of functions"""
from pathlib import Path
from typing import Iterable

from matplotlib.figure import Figure
import numdifftools as nd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

from .plot import FONTSIZE_AXLABEL, FONTSIZE_FIGLABEL, fig_template_axarr


class StepSweep:
    """Step sweep with optimum step size identified as minimum Δ"""

    def __init__(self, f, x, steps, rtol=0.005, atol=1e-7):
        """Run step sweep

        :param f: Finite difference derivative f(x, h) where x is a parameter vector and
        h is the step size.  The derivative can be of any order.

        """
        self.x = x
        self.h = steps
        self.rtol = 0.005
        self.atol = 1e-7
        # Indices corresponding to the function dimensionality are first.  The index
        # across step sizes is last.
        self.v = np.stack([f(x, h) for h in steps], axis=-1)
        self.h_opt = np.full(self.v.shape, np.nan)
        self.v_opt = np.full(self.v.shape, np.nan)
        self.Δv = np.full(self.v.shape, np.nan)
        self.h_filtered = np.full(self.v.shape, np.nan)
        self.v_filtered = np.full(self.v.shape, np.nan)
        self.Δv_filtered = np.full(self.v.shape, np.nan)
        for i in np.ndindex(self.v.shape[:-1]):
            idx_opt, Δv, h_filtered, Δv_filtered = optimum_from_step_sweep(
                self.h, self.v[i]
            )
            self.Δv[i] = Δv
            self.h_filtered[i] = h_filtered
            self.Δv_filtered[i] = Δv_filtered
            self.h_opt[i] = steps[idx_opt]
            self.v_opt[i] = self.v[i, idx_opt]
            self.error = self.v[i] - self.v_opt[i]
            valid_interval = valid_interval_from_step_sweep(self.error, self.tol)
            if valid_interval:
                self.h_valid = (steps[valid_interval[0]], steps[valid_interval[1]])
            else:
                self.h_valid = None

    @property
    def tol(self):
        return max((self.rtol * self.v_opt, self.atol))


def plot_step_sweep_Hessian(sweep: StepSweep):
    fig_err = fig_template_axarr(xlabel="Step size", ylabel="|Error|")
    fig_incr = fig_template_axarr(xlabel="Step size", ylabel="|Incremental Δ|")

    def plot_ax(ax, i, j, h, v, h_opt, h_bounds=None):
        v = np.abs(v)  # log scale cannot cross zero
        ax.set_title(f"$∂f/∂x_{{{i+1}}}∂x_{{{j + 1}}}$|", fontsize=FONTSIZE_AXLABEL)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
        ax.axvline(h_opt[i, j], color="C0", linewidth=2.5)
        ax.plot_ax(h, v, ".", color="deepskyblue")
        if h_bounds is not None:
            ax.axvline(
                h_bounds[0],
                color="C3",
                linestyle="dotted",
                linewidth=1.5,
            )
            ax.axvline(
                h_bounds[1],
                color="C1",
                linestyle="dotted",
                linewidth=1.5,
            )

    for i, j in np.ndindex(sweep.v.shape[:-1]):
        # Incremental change plot
        ax = fig_incr.axarr[i, j]
        plot_ax(ax, i, j, sweep.h, sweep.Δv, sweep.h_opt, sweep.h_valid)
        ax.plot_ax(
            sweep.h_filtered, sweep.Δv_filtered, "-", color="black", linewidth=0.5
        )

        # Error plot
        ax = fig_err.axarr[i, j]
        plot_ax(ax, i, j, sweep.h, sweep.v[i, j], sweep.h_opt, sweep.h_valid)

    return fig_incr, fig_err


def plot_step_sweep_summary(sweeps: Iterable[StepSweep]):
    h = sweeps[0].h
    ni, nj = sweeps[0].v.shape[:2]
    # Put the index over samples last.  The second-to-last index then is over step
    # sizes.
    h_bounds = []
    h_opt = []
    for sweep in sweeps:
        h_bounds.append(sweep.h_valid)
        h_opt.append(sweep.h_opt)
    h_bounds = np.stack(h_bounds, axis=-1)
    h_opt = np.stack(h_opt, axis=-1)

    # Dense x-axis points
    hi = np.logspace(np.log10(h[0]), np.log10(h[-1]), 200)

    def add_pdf(bound_data, ax, color, label=None):
        # Gaussian KDE cannot be constructed if all data values are equal
        if np.all(bound_data == bound_data[0]):
            ax.axvline(bound_data[0], linewidth=2, color=color, label=label)
        else:
            b = gaussian_kde(np.log10(bound_data))
            ax.fill_between(hi, b(np.log10(hi)), color=color, label=label)

    # Plot distribution of bounds of valid step intervals for all components of the
    # Hessian
    fig_combined = Figure(constrained_layout=True)
    fig_combined.set_constrained_layout_pads(
        wspace=2 / 72, hspace=2 / 72, w_pad=2 / 72, h_pad=2 / 72
    )
    ax = fig_combined.add_subplot(111)
    fig_combined.set_size_inches(4, 3)
    add_pdf(h_bounds[0, :].flatten(), ax, color="C3", label="Lower bound")
    add_pdf(h_bounds[1, :].flatten(), ax, color="C1", label="Upper bound")
    add_pdf(h_opt.flatten(), ax, color="C0", label="Optimum")
    ax.set_xscale("log")
    ax.set_xlabel("Step size")
    ax.set_ylabel("Probability density [1]")
    ylim = ax.get_ylim()
    ax.set_ylim((0, ylim[1]))
    ax.legend()
    fig_combined.suptitle("Valid step size interval", fontsize=FONTSIZE_FIGLABEL)

    # Plot distribution of bounds of valid step intervals, one subplot
    # per component of the Hessian
    fig_components, axarr_components = fig_template_axarr(ni, nj)
    for i in range(ni):  # rows
        for j in range(nj):  # columns
            ax = axarr_components[i, j]
            # Bounds of region with acceptable error
            add_pdf(h_bounds[0, i, j], ax, "C3", label="Lower bound")
            add_pdf(h_bounds[1, i, j], ax, "C1", label="Upper bound")
            add_pdf(h_opt[i, j], ax, "C0", label="Optimum")
            ylim = ax.get_ylim()
            ax.set_ylim((0, ylim[1]))
            if i == ni - 1:
                ax.set_xlabel("Step size")
            if j == 0:
                ax.set_ylabel("PDF [1]")
    fig_components.suptitle("Valid step size interval", fontsize=FONTSIZE_FIGLABEL)
    return fig_combined, fig_components


def optimum_from_step_sweep(h, v):
    """Estimate optimal step size by incremental stepwise change

    :param h: Step size values, dim 1 array.

    :param v: Function values for each step in steps, dim 1 array.

    """
    Δv = np.abs(np.diff(v, axis=0))
    # To estimate h_opt as the step size at which the incremental change starts
    # increasing (from the right), need to filter out zeroes and other outliers caused
    # by error cancellation.
    m = Δv != 0
    h_filtered = h[m]
    Δv_filtered = 10 ** savgol_filter(np.log10(Δv[m]), 5, 2)
    idx_opt = np.argmin(Δv_filtered)
    return idx_opt, Δv, h_filtered, Δv_filtered


def run_sweeps_Hessian(dir_out, f, samples, steps, rtol=0.005, atol=1e-7):
    dir_out = Path(dir_out)
    dir_Δ = dir_out / "sample_plots_incremental_Δ"
    dir_Δ.mkdir(exist_ok=True)
    dir_err = dir_out / "sample_plots_error"
    dir_err.mkdir(exist_ok=True)
    eval_hessian = nd.Hessian(f, method="central")
    sweeps = []
    for i, x in enumerate(samples):
        sweep = StepSweep(eval_hessian, x, steps, rtol, atol)
        fig_Δ, fig_err = plot_step_sweep_Hessian(sweep)
        fig_Δ.savefig(dir_Δ / f"{i}_incremental_Δ.svg")
        fig_err.savefig(dir_err / f"{i}_error.svg")
        sweeps.append(sweep)
    fig_combined, fig_components = plot_step_sweep_summary(sweeps)
    fig_combined.savefig(dir_out / "Hessian_valid_step_sizes_-_all_components.svg")
    fig_components.savefig(dir_out / "Hessian_valid_step_sizes_-_by_component.svg")


def valid_interval_from_step_sweep(error, tol):
    """Report valid step size interval from error vs. step size curve

    :param error: Error values, dim 1 array.

    :param tol: Error tolerance, number.

    :returns: Indices of the leftmost and rightmost points for which error ≤ tol.  If
    there are no such valid points, return an empty tuple.  Indices are returned rather
    than values itself to allow selection of other values from corresponding data.

    """
    m = np.abs(error) <= tol
    candidates = np.split(np.arange(len(error)), np.where(m == 0)[0])
    idx_largest_span = candidates[np.argmax([len(a) for a in candidates])] + 1
    if len(idx_largest_span) >= 1:
        return idx_largest_span[0], idx_largest_span[-1]
    else:
        return tuple()
