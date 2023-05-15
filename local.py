import numpy as np

from .numerics import eigenvalue_error_bfalt
from .plot import (
    FONTSIZE_FIGLABEL,
    plot_eigenvalues_histogram,
    plot_matrix,
)


def plot_sample(analysis, id_label, Hs, H):
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
    # w_err = eigenvalue_error_bfalt(Hs, w, v)
    fig = plot_eigenvalues_histogram(
        w,
        "Eigenvector Index",
        "Eigenvalue",
    )
    fig.ax.set_title("Eigenvalues of Scaled Hessian", fontsize=FONTSIZE_FIGLABEL)
    fig.fig.savefig(dir_H_eig / f"{id_label}_scaled_Hessian_eigenvalues.svg")
