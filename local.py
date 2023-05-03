from .plot import plot_matrix, symlog

def plot_sample(analysis, id_, Hs, H):
    dir_Hessian = analysis.directory / "samples_plots_Hessian"
    dir_Hessian.mkdir(exist_ok=True)
    dir_sHessian = analysis.directory / "samples_plots_scaled_Hessian"
    dir_sHessian.mkdir(exist_ok=True)
    fig = plot_matrix(
        symlog(Hs),
        title=f"Scaled Hessian at sample {id_}",
        cbar_label="Symmetric log10",
        tick_labels=[p.name for p in analysis.parameters],
    )
    fig.fig.savefig(dir_sHessian / f"{id_}_scaled_Hessian.svg")
    fig = plot_matrix(
        symlog(H),
        title=f"Hessian at sample {id_}",
        cbar_label="Symmetric log10",
        tick_labels=[p.name for p in analysis.parameters],
    )
    fig.fig.savefig(dir_Hessian / f"{id_}_Hessian.svg")
