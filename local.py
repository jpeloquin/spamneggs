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

    def make_ψ(self, x0):
        """Return cost function referenced to true parameter values x0

        :param x0: Vector of "true" parameter values on length n.  The returned cost
        function will calculate cost relative to `f(x0)`.

        :returns: Sum-squares cost function, which accepts a vector of parameters with
        length matching x0.

        """

        f0 = self.f(x0)

        def ψ(x):
            # run_case verifies that must points are used and (via run_febio_checked)
            # that FEBio observed them, so we don't need to check for mismatched times
            f_value = self.f(x)
            cost = np.sum((f_value - f0) ** 2)
            if self.callback is not None:
                self.callback(x, cost)
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
    dir_H_eig = analysis.directory / "samples_plots_scaled_Hessian_eigenvalues"
    dir_H_eig.mkdir(exist_ok=True)
    ## Histogram
    nparams = len(analysis.parameters)
    err = {"B–F auto": eigenvalue_error_bfalt(Hs, w, v)}
    if Herr is not None:
        err["ΔH"] = np.linalg.norm(np.full((nparams, nparams), 10**-2))
    fig = plot_sample_eigenvalues_hist(w, "Eigenvector Index", "Eigenvalue", errors=err)
    fig.ax.set_title("Eigenvalues of Scaled Hessian", fontsize=FONTSIZE_FIGLABEL)
    fig.fig.savefig(dir_H_eig / f"{id_label}_scaled_Hessian_eigenvalues_hist.svg")
    ## Line plot
    fig = plot_sample_eigenvalues_line(w, errors=err)
    fig.ax.set_title("Eigenvalues of Scaled Hessian", fontsize=FONTSIZE_FIGLABEL)
    fig.fig.savefig(dir_H_eig / f"{id_label}_scaled_Hessian_eigenvalues_line.svg")
