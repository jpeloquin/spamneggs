"""Numerical analysis of functions"""
import functools
from pathlib import Path
from typing import Iterable

from matplotlib.figure import Figure
import numdifftools as nd
import numpy as np
from pandas import DataFrame
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

from waffleiron.febio import CheckError, FEBioError
from .plot import FONTSIZE_AXLABEL, FONTSIZE_FIGLABEL, fig_template_axarr
from .util import Counter, parameter_to_str


class OptimumStepError(ValueError):
    """Raise when the optimum step size cannot be identified"""

    pass


class StepSweep:
    """Step sweep with optimum step size identified as minimum Δ"""

    def __init__(self, f, x, steps, rtol=0.005, atol=1e-7):
        """Run step sweep

        :param f: Finite difference derivative f(x, h) where x is a parameter vector and
        h is the step size.  The derivative can be of any order.

        :param x: The point at which to evaluate the derivative.

        :param steps: Step sizes.  The derivative will be evaluated once using each
        provided step size.

        """
        self.x = x
        self.h = steps
        self.rtol = 0.005
        self.atol = 1e-7
        # Indices corresponding to the function dimensionality are first.  The index
        # across step sizes is last.
        self.v = np.stack([f(x, h) for h in steps], axis=-1)
        self.error = np.full(self.v.shape, np.nan)
        self.Δv = np.full(self.v.shape[:-1] + (self.v.shape[-1] - 1,), np.nan)
        self.h_opt = np.full(self.v.shape[:-1], np.nan)
        self.v_opt = np.full(self.v.shape[:-1], np.nan)
        # Don't know length of filtered data ahead of time
        self.h_filtered = np.full(self.Δv.shape[:-1], np.nan, dtype=object)
        self.v_filtered = np.full(self.Δv.shape[:-1], np.nan, dtype=object)
        self.Δv_filtered = np.full(self.Δv.shape[:-1], np.nan, dtype=object)
        self.h_valid = np.full(self.Δv.shape[:-1], np.nan, dtype=object)
        # Identify optimum step size and valid bounds
        for i in np.ndindex(self.v.shape[:-1]):
            idx_opt, Δv, h_filtered, Δv_filtered = optimum_from_step_sweep(
                self.h, self.v[i]
            )
            self.Δv[i] = Δv
            self.h_filtered[i] = h_filtered
            self.Δv_filtered[i] = Δv_filtered
            self.h_opt[i] = steps[idx_opt]
            self.v_opt[i] = self.v[i][idx_opt]
            self.error[i] = self.v[i] - self.v_opt[i]
            valid_interval = valid_interval_from_step_sweep(
                self.error[i], tol(self.v_opt[i])
            )
            if valid_interval:
                self.h_valid[i] = (steps[valid_interval[0]], steps[valid_interval[1]])
            else:
                self.h_valid[i] = None


def tol(v, rtol=0.005, atol=1e-7):
    return max((rtol * v, atol))


def eigenvalue_error_bfalt(A, w, V):
    """Return alternate Baur–Fike bound on eigenvalue error

    :param A: Original matrix

    :param w: eigenvalues of A

    :param V: matrix of eigenvectors of A, one eigenvector per column
    """
    κ = np.linalg.cond(V, p=2)
    r = A @ V - w * V
    rnorm = np.linalg.norm(r, ord=2, axis=0)
    vnorm = np.linalg.norm(V, ord=2, axis=0)
    return κ * rnorm / vnorm


def plot_step_sweep(sweep: StepSweep):
    n = sweep.v.shape[0]
    fig_err = fig_template_axarr(n, n, xlabel="Step size", ylabel="|Error|")
    fig_incr = fig_template_axarr(n, n, xlabel="Step size", ylabel="|Incremental Δ|")

    def plot_ax(ax, i, j, h, v, h_opt, h_bounds=None):
        v = np.abs(v)  # log scale cannot cross zero
        ax.set_title(f"|$∂f/∂x_{{{i+1}}}∂x_{{{j + 1}}}$|", fontsize=FONTSIZE_AXLABEL)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.get_minor_locator().set_params(numticks=50)
        ax.yaxis.get_minor_locator().set_params(numticks=50)
        ax.axvline(h_opt, color="C0", linewidth=2.5)
        ax.plot(h, v, ".", color="deepskyblue")
        if h_bounds is not None:
            ax.axvline(
                h_bounds[0],
                color="C3",
                linestyle="dotted",
                linewidth=2.5,
            )
            ax.axvline(
                h_bounds[1],
                color="C1",
                linestyle="dotted",
                linewidth=2.5,
            )

    for i, j in np.ndindex(sweep.v.shape[:-1]):
        # Incremental change plot
        ax = fig_incr.axarr[i, j]
        plot_ax(
            ax,
            i,
            j,
            sweep.h[:-1],
            sweep.Δv[i, j],
            sweep.h_opt[i, j],
            sweep.h_valid[i, j],
        )
        ax.plot(
            sweep.h_filtered[i, j],
            sweep.Δv_filtered[i, j],
            "-",
            color="black",
            linewidth=0.5,
        )

        # Error plot
        ax = fig_err.axarr[i, j]
        plot_ax(
            ax, i, j, sweep.h, sweep.error[i, j], sweep.h_opt[i, j], sweep.h_valid[i, j]
        )

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
        # Gaussian KDE cannot be constructed if all data values are equal.  Even when all data values are nearly equal, the probability density blows up.
        if np.all(bound_data == bound_data[0]):
            ax.axvline(bound_data[0], linewidth=2, color=color, label=label)
            return
        b = gaussian_kde(np.log10(bound_data))
        pdf = b(np.log10(hi))
        if max(pdf) > 10:
            ax.axvline(np.mean(bound_data[0]), linewidth=2, color=color, label=label)
        else:
            ax.fill_between(hi, pdf, color=color, label=label, alpha=0.8)

    # Plot distribution of bounds of valid step intervals for all components of the
    # Hessian
    # TODO: Solving a constrained layout with tens to hundreds of subplots is slow.
    # Takes about a minute per plot.
    fig_combined = Figure(constrained_layout=True)
    fig_combined.set_constrained_layout_pads(
        wspace=2 / 72, hspace=2 / 72, w_pad=2 / 72, h_pad=2 / 72
    )
    ax = fig_combined.add_subplot(111)
    fig_combined.set_size_inches(4, 3)
    add_pdf(np.stack(h_bounds.flatten())[:, 0], ax, color="C3", label="Lower bound")
    add_pdf(np.stack(h_bounds.flatten())[:, 1], ax, color="C1", label="Upper bound")
    add_pdf(h_opt.flatten(), ax, color="C0", label="Optimum")
    ax.set_xscale("log")
    ax.set_xlim(h[0], h[-1])
    ax.set_xlabel("Step size")
    ax.set_ylabel("Probability density")
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
            ax.set_title(
                f"$∂f/∂x_{{{i + 1}}}∂x_{{{j + 1}}}$", fontsize=FONTSIZE_AXLABEL
            )
            ax.set_xscale("log")
            ax.set_xlim(h[0], h[-1])
            ax.xaxis.get_minor_locator().set_params(numticks=50)
            # Bounds of region with acceptable error
            add_pdf(np.stack(h_bounds[i, j])[:, 0], ax, "C3", label="Lower bound")
            add_pdf(np.stack(h_bounds[i, j])[:, 1], ax, "C1", label="Upper bound")
            add_pdf(h_opt[i, j], ax, "C0", label="Optimum")
            ylim = ax.get_ylim()
            ax.set_ylim((0, ylim[1]))
            if i == ni - 1:
                ax.set_xlabel("Step size")
            if j == 0:
                ax.set_ylabel("density")
    fig_components.suptitle("Valid step size interval", fontsize=FONTSIZE_FIGLABEL)
    return fig_combined, fig_components


def optimum_from_step_sweep(h, v):
    """Estimate optimal step size by incremental stepwise change

    :param h: Step size values, dim 1 array, ordered small to large.

    :param v: Function values for each step in steps, dim 1 array.  Ordered to match h.

    :returns:

    idx_opt indexes into the input h and v

    """
    # TODO: Filtering out Δv = 0 is a lousy approach because it breaks simple tests.  I
    # don't like the control flow between this code and plotting of the results.
    Δv = np.abs(np.diff(v[::-1], axis=0))[::-1]
    # To estimate h_opt as the step size at which the incremental change starts
    # increasing (from the right), need to filter out zeroes and other outliers caused
    # by error cancellation.
    m = Δv != 0
    idx_filtered = np.arange(len(Δv))[m]
    h_filtered = h[:-1][m]
    n = np.sum(m)
    if n < 3:
        raise OptimumStepError(
            f"{n} points have non-zero incremental change as step size decreases.  Cannot determine optimal step size with ≤ 2 points."
        )
    elif n == 3:
        Δv_filtered = Δv[m]
    else:
        Δv_filtered = 10 ** savgol_filter(
            np.log10(Δv[m]), window_length=min(n, 5), polyorder=2
        )
    idx_opt = idx_filtered[np.argmin(Δv_filtered)]
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
        fig_Δ, fig_err = plot_step_sweep(sweep)
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
    idx_largest_span = candidates[np.argmax([len(a) for a in candidates])]
    if not m[idx_largest_span[0]]:
        idx_largest_span = idx_largest_span[1:]
    if len(idx_largest_span) >= 1:
        return idx_largest_span[0], idx_largest_span[-1]
    else:
        return tuple()


def add_eval_counter(f):
    eval_counter = Counter()

    def g(*args, **kwargs):
        nonlocal eval_counter
        eval_counter.increment()
        return f(*args, **kwargs)

    return g, eval_counter


def run_step_size_check(
    parameter_defs,
    samples,
    make_f,
    steps,
    scale_parameters,
    unscale_parameters,
    dir_out,
):
    """Run finite difference step size error analysis for the Hessian

    :param parameter_defs: Parameter defintions, as in a case generator.

    :param samples: Sequence of tuple-like parameter values.

    :param make_f: Function of the parameter values for the current sample → a
    function of parameter values that returns a scalar.  This structure allows the
    step size check to evaluate cost functions that depend on both the finite
    difference derivative support points and the sample point.

    :param steps: Sequence of step sizes to sweep across, ordered small to large.

    :param scale_parameters:

    :param unscale_parameters:
    """
    dir_out.mkdir(exist_ok=True)
    dir_Δ = dir_out / "sample_plots_incremental_Δ"
    dir_Δ.mkdir(exist_ok=True)
    dir_err = dir_out / "sample_plots_error"
    dir_err.mkdir(exist_ok=True)

    table = {"Sample": [], "Evals": [], "Result": []} | {
        parameter_to_str(p): [] for p in parameter_defs
    }
    sweeps = []
    for i, x0 in enumerate(samples):
        table["Sample"].append(i)
        for p, v in zip(parameter_defs, x0):
            table[parameter_to_str(p)].append(v)
        fs = scaled_args(make_f(x0), unscale_parameters)
        ψs, evals = add_eval_counter(fs)

        def Hs(xs, h):
            return nd.Hessian(ψs, method="central", step=h)(xs)

        x0s = scale_parameters(x0)
        try:
            sweep = StepSweep(Hs, x0s, steps)
        except (FEBioError, CheckError) as e:
            table["Result"].append(e.__class__.__name__)
            continue
        else:
            table["Result"].append("Success")
        finally:
            table["Evals"].append(evals)
        fig_Δ, fig_err = plot_step_sweep(sweep)
        fig_Δ.fig.savefig(dir_Δ / f"{i}_incremental_Δ.svg")
        fig_err.fig.savefig(dir_err / f"{i}_error.svg")
        sweeps.append(sweep)
    DataFrame(table).to_csv(dir_out / "samples.csv", index=False)
    fig_combined, fig_components = plot_step_sweep_summary(sweeps)
    fig_combined.savefig(dir_out / "Hessian_valid_step_sizes_-_all_components.svg")
    fig_components.savefig(dir_out / "Hessian_valid_step_sizes_-_by_component.svg")
    # TODO: If successful run, clean up the case generation directory.


def scaled_args(fun, unscale):
    """Decorator to make a function of θ accept scaled_args θ"""

    @functools.wraps(fun)
    def fun_s(θ_s, *args, **kwargs):
        θ = unscale(θ_s)
        return fun(θ, *args, **kwargs)

    return fun_s
