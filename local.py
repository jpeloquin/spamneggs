import math

import numpy as np

from .numerics import eigenvalue_error_bfalt
from .plot import (
    FONTSIZE_FIGLABEL,
    plot_sample_eigenvalues_hist,
    plot_matrix,
    plot_sample_eigenvalues_line,
)


class LsqCostFunctionFactory:
    def __init__(self, f, callback=None):
        """Return instance to create cost functions at a parameter point

        :param f: Model function from which to derive a cost function.  Must accept a
        vector of parameter values of length n.

        :param callback: Function with call signature (parameter values, cost). Every
        time the returned function `ψs` is called, it will call the callback with the
        parameter values it was given and the cost it calculated. The callback is
        usually used for logging.

        """
        self.f = f
        self.callback = callback

    def make_ψ(self, f0):
        """Return SSE cost function with respect to "true" observations f0

        :param f0: Vector of "true" values matching the length of the vector returned by
        self.f.

        :returns: The sum-squares error (SSE) function ψ, which accepts a vector of parameters
        of the same length as self.f and returns a scalar (the SSE).

        """

        def ψ_with_callback(x):
            ψ = make_sse_cost(self.f, f0)
            cost = ψ(x)
            if self.callback is not None:
                self.callback(x, cost)
            return cost

        return ψ_with_callback


def make_sse_cost(f, f0):
    """Return SSE cost function with respect to "true" observations f0

    :param f: Function f(θ) to evaluate the model, where θ is the model parameters.

    :param f0: Vector of "true" values matching the length of the vector returned by
    f.

    :returns: The sum-squares error (SSE) function ψ(θ), with θ having the same
    meaning as in f(θ).  Returns a scalar (the SSE).

    """

    def ψ(x):
        # run_case verifies that must points are used and (via run_febio_checked)
        # that FEBio observed them, so we don't need to check for mismatched times
        f_value = f(x)
        se = np.atleast_1d(np.array((f_value - f0) ** 2))  # ensure iterable
        cost = math.fsum(se)
        return cost

    return ψ


def plot_sample(analysis, id_label, Hs, H, Herr=None):
    # Plot the scaled Hessian matrix
    dir_Hs = analysis.directory / "samples_plots_scaled_Hessian"
    dir_Hs.mkdir(exist_ok=True)
    fig = plot_matrix(
        Hs,
        scale="log",
        title=f"Scaled Hessian at sample {id_label}",
        cbar_label="Symmetric log10",
        tick_labels=[p.name for p in analysis.parameters],
    )
    fig.fig.savefig(dir_Hs / f"{id_label}_scaled_Hessian.svg")

    # Plot the Hessian matrix
    dir_H = analysis.directory / "samples_plots_Hessian"
    dir_H.mkdir(exist_ok=True)
    fig = plot_matrix(
        H,
        scale="log",
        title=f"Hessian at sample {id_label}",
        tick_labels=[p.name for p in analysis.parameters],
    )
    fig.fig.savefig(dir_H / f"{id_label}_Hessian.svg")

    # Plot the eigenvalues of the scaled Hessian matrix
    w, v = np.linalg.eig(Hs)
    ## Histogram
    dir_H_eig = analysis.directory / "sample_plots_scaled_Hessian_eigenvalues_hist"
    dir_H_eig.mkdir(exist_ok=True)
    nparams = len(analysis.parameters)
    err = {}
    if Herr is not None:
        err["ΔH"] = np.linalg.norm(np.full((nparams, nparams), 10**-2))
    err["B–F auto"] = eigenvalue_error_bfalt(Hs, w, v)
    fig = plot_sample_eigenvalues_hist(w, "Eigenvector Index", "Eigenvalue", errors=err)
    fig.ax.set_title("Eigenvalues of Scaled Hessian", fontsize=FONTSIZE_FIGLABEL)
    fig.fig.savefig(dir_H_eig / f"{id_label}_scaled_Hessian_eigenvalues_hist.svg")
    ## Line plot
    dir_H_eig = analysis.directory / "sample_plots_scaled_Hessian_eigenvalues_line"
    dir_H_eig.mkdir(exist_ok=True)
    fig = plot_sample_eigenvalues_line(w, errors=err)
    fig.ax.set_title("Eigenvalues of Scaled Hessian", fontsize=FONTSIZE_FIGLABEL)
    fig.fig.savefig(dir_H_eig / f"{id_label}_scaled_Hessian_eigenvalues_line.svg")
