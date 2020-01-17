from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import os
from pathlib import Path
import subprocess
# Third-party packages
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker
from matplotlib.colors import DivergingNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import multiprocessing as mp
import numpy as np
import pandas as pd
import psutil
import scipy.cluster
# Same-package modules
from . import febioxml as fx
from . import colors
from .febioxml import tabulate_case
from .febioxml import read_febio_xml as read_xml
from .variables import *


NUM_WORKERS = psutil.cpu_count(logical=False)

class FEBioError(Exception):
    """Raised when an FEBio simulation terminates in an error."""
    pass


def sensitivity_analysis(analysis_file, nlevels, on_febio_error="stop",
                         analysis_dir=None):
    """Run a sensitivity analysis from spam-infused FEBio XML."""
    # Validate input
    on_febio_error_options = ("stop", "hold", "ignore")
    if not on_febio_error in on_febio_error_options:
        raise ValueError(f"on_febio_error = {on_febio_error}; allowed values are {','.join(on_febio_error_options)}")
    # Generate the cases
    tree = read_xml(analysis_file)
    analysis_name = tree.find("preprocessor[@proc='spamneggs']/"
                              "analysis").attrib["name"]
    if analysis_dir is None:
        analysis_dir = Path(analysis_name)
    cases, pth_cases_table = fx.gen_sensitivity_cases(tree, nlevels,
                                                      analysis_dir=analysis_dir)
    # Run the cases
    run_status = [None for i in range(len(cases))]
    febio_error = False
    run_case = partial(run_febio_unchecked, threads=1)
    # Potential improvement: increase OMP_NUM_THREADS for last few jobs
    # as more cores are free.  In most cases this should make little
    # difference, though.
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Submit the first round of cases
        futures = {pool.submit(run_case,
                               analysis_dir / cases["path"].loc[case_id]): case_id
                   for case_id in cases.index[:NUM_WORKERS]}
        pending_cases = set([case_id for case_id in cases.index[NUM_WORKERS:]])
        # Submit remaining cases as workers become available.  We do not
        # submit all cases immediately because we wish to have the
        # option of ending the analysis early if a case ends in error
        # termination.
        while pending_cases:
            future = next(as_completed(futures))
            case_id = futures.pop(future)
            pth_feb = cases["path"].loc[case_id]
            return_code = future.result()
            # print(f"Popped case {case_id} from futures")
            # Log case details
            if return_code == 0:
                run_status[case_id] = "completed"
            else:
                run_status[case_id] = "error"
            # Check if we should continue submitting cases
            if on_febio_error == "stop" and return_code != 0:
                print(f"FEBio returned error code {return_code} while running case {case_id} ({pth_feb}); check {pth_feb.with_suffix('.log')}.  Because `on_febio_error` = {on_febio_error}, spamneggs will finish any currently running cases, then stop.")
                break
            # Submit next case
            next_case = pending_cases.pop()
            # print(f"Submitting case {next_case}")
            futures.update({pool.submit(run_case,
                                        analysis_dir / cases["path"].loc[next_case]):
                            next_case})
        # Finish processing all running cases.
        for future in as_completed(futures):
            case_id = futures[future]
            return_code = future.result()
            if return_code == 0:
                run_status[case_id] = "completed"
            else:
                run_status[case_id] = "error"
    cases["status"] = run_status
    cases.to_csv(pth_cases_table)
    # Check if error terminations prevent analysis of results
    m_error = cases["status"] == "error"
    if np.any(m_error):
        if on_febio_error == "ignore":
            cases = cases[np.logical_not(m_error)]
        elif on_febio_error == "hold":
            raise Exception(f'{np.sum(m_error)} cases terminated in an error.  Because `on_febio_error` = "{on_febio_error}", the sensitivity analysis was stopped prior to data analysis.  The error terminations are listed in `{pth_cases_table}`.  To continue, correct the error terminations and call `make_sensitivity_figures` separately.')
        elif on_febio_error == "stop":
            # Error message was printed above
            exit
    # Tabulate and plot the results
    tabulate_analysis(analysis_file)
    make_sensitivity_figures(analysis_file)


def read_case_data(case_file):
    """Read variables from a single case analysis."""
    case_file = Path(case_file)
    output_dir = case_file.parent / ".." / "case_output"
    pth_record = output_dir / f"{case_file.stem}_vars.json"
    pth_timeseries = output_dir / f"{case_file.stem}_timeseries_vars.csv"
    with open(pth_record, "r") as f:
        record = json.load(f)
    timeseries = pd.read_csv(pth_timeseries, index_col=False)
    return record, timeseries


def tabulate_case_write(analysis_file, case_file, dir_out=None):
    """Tabulate variables for single case analysis & write to disk."""
    analysis_file = Path(analysis_file)
    case_file = Path(case_file)
    # Find/create output directory
    if dir_out is None:
        dir_out = case_file.parent
    else:
        dir_out = Path(dir_out)
    if not dir_out.exists():
        dir_out.mkdir()
    # Tabulate the variables
    record, timeseries = tabulate_case(analysis_file, case_file)
    analysis_tree = read_xml(analysis_file)
    analysis_name = fx.get_analysis_name(analysis_tree)
    with open(dir_out / f"{case_file.stem}_vars.json", "w") as f:
        write_record_to_json(record, f)
    timeseries.to_csv(dir_out / f"{case_file.stem}_timeseries_vars.csv",
                      index=False)
    plot_timeseries_vars(timeseries, dir_out=dir_out, casename=case_file.stem)
    return record, timeseries


def tabulate_analysis_tsvars(analysis_file, cases_file):
    """Tabulate time series variables for all cases in an analysis

    The time series tables for the individual cases must have already
    been written to disk.

    """
    # TODO: It would be beneficial to build up the whole-analysis time
    # series table at the same time as case time series tables are
    # written to disk, instead of re-reading everything from disk.
    tree = read_xml(analysis_file)
    analysis_name = fx.get_analysis_name(tree)
    parameters, parameter_locs = fx.get_parameters(tree)
    pth_cases = Path(cases_file)
    cases = pd.read_csv(cases_file, index_col=0)
    analysis_data = pd.DataFrame()
    for i in cases.index:
        pth_tsvars = pth_cases.parent / "case_output" /\
            f"{analysis_name}_-_case={i}_timeseries_vars.csv"
        case_data = pd.read_csv(pth_tsvars)
        varnames = set(case_data.columns) - set(["Time", "Step"])
        case_data = case_data.rename({k: f"{k} [var]" for k in varnames},
                                     axis=1)
        case_data["Case"] = i
        pcolnames = [f"{p} [param]" for p in parameters]
        for pname, pcolname in zip(parameters, pcolnames):
            case_data[pcolname] = cases[pname].loc[i]
        analysis_data = pd.concat([analysis_data, case_data])
    return analysis_data


def run_febio_checked(pth_feb, threads=psutil.cpu_count(logical=False)):
    """Run FEBio, raising exception on error."""
    proc = _run_febio(pth_feb, threads=threads)
    if proc.returncode != 0:
        raise FEBioError(f"FEBio returned error code {proc.returncode} while running {pth_feb}; check {pth_log}.")
    return proc.returncode


def run_febio_unchecked(pth_feb, threads=psutil.cpu_count(logical=False)):
    """Run FEBio and return its error code."""
    return _run_febio(pth_feb, threads=threads).returncode


def _run_febio(pth_feb, threads=psutil.cpu_count(logical=False)):
    """Run FEBio and return the process object."""
    pth_feb = Path(pth_feb)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": f"{threads}"})
    proc = subprocess.run(['febio', '-i', pth_feb.name],
                          cwd=pth_feb.parent,  # FEBio always writes xplt to current dir
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=env)
    # FEBio doesn't always write a log if it hits a error, but the
    # content that would have been logged is dumped to stdout.  So we'll
    # write the stdout output to disk as a workaround.
    if proc.returncode != 0:
        # FEBio truly does return an error code on "Error Termination"
        pth_log = pth_feb.with_suffix(".log")
        with open(pth_log, "wb") as f:
            f.write(proc.stdout)
    return proc


def tabulate_analysis(analysis_file):
    """Tabulate output from an analysis."""
    tree = read_xml(analysis_file)  # re-read b/c gen_sensitivity_cases modifies tree
    parameters, parameter_locs = fx.get_parameters(tree)
    ivars_table = defaultdict(list)
    analysis_name = fx.get_analysis_name(tree)
    analysis_dir = Path(analysis_name)
    pth_cases = analysis_dir / f"{analysis_name}_-_cases.csv"
    cases = pd.read_csv(pth_cases, index_col=0)
    for i in cases.index:
        record, timeseries = tabulate_case_write(analysis_file,
                                                 analysis_dir / cases["path"].loc[i],
                                                 dir_out=analysis_dir / "case_output")
        ivars_table["case"].append(i)
        for p in parameters:
            k = f"{p} [param]"
            ivars_table[k].append(cases[p].loc[i])
        for v in record["instantaneous variables"]:
            k = f"{v} [var]"
            ivars_table[k].append(record["instantaneous variables"][v]["value"])
    ivars_table = pd.DataFrame(ivars_table).set_index("case")
    ivars_table.to_csv(analysis_dir / f"{analysis_name}_-_inst_vars.csv",
                       index=True)


def make_sensitivity_figures(analysis_file):
    tree = read_xml(analysis_file)
    analysis_name = tree.find("preprocessor[@proc='spamneggs']/"
                              "analysis").attrib["name"]
    analysis_dir = Path(analysis_file).parent / analysis_name
    pth_cases = analysis_dir / f"{analysis_name}_-_cases.csv"
    cases = pd.read_csv(pth_cases, index_col=0)
    cases = cases[cases["status"] == "completed"]
    # TODO: Add a function to get the list of parameters and variables
    # for an analysis up-front.  Extract the relevant code from
    # tabulate_case.
    param_names = [k for k in cases.columns
                   if k not in ["case", "path", "status"]]
    param_values = {k: cases[k] for k in param_names}
    # Make scatter plot matrix for instantaneous variables
    ivar_values = defaultdict(list)
    record, tab_timeseries = read_case_data(analysis_dir /
                                            cases["path"].iloc[0])
    ivar_names = [nm for nm in record["instantaneous variables"]]
    tvar_names = [nm for nm in record["time series variables"]]
    for i in cases.index:
        # TODO: There is an opportunity for improvement here: We could
        # preserve the tree from the first read of the analysis XML and
        # re-use it here instead of re-reading the analysis XML for
        # every case.  However, to do this the case generation must not
        # alter the analysis XML tree.
        record, tab_timeseries = read_case_data(analysis_dir /
                                                cases["path"].loc[i])
        for nm in ivar_names:
            ivar_values[nm].append(record["instantaneous variables"][nm]["value"])

    # Instantaneous variables: Scatter plots of variable vs. parameter
    #
    # Instantaneous variables: Matrix of variable vs. parameter scatter
    # plots
    npanels_w = len(param_names) + 1
    npanels_h = len(ivar_names) + 1
    hist_nbins = 9
    fig, axarr = plt.subplots(npanels_h, npanels_w, sharex="col",
                              sharey="row")
    fig.set_size_inches(1.8*npanels_w + 1, 2*npanels_h + 1)
    # Scatter plots
    for j, param in enumerate(param_names):
        for i, var in enumerate(ivar_names):
            axarr[i+1, j].scatter(param_values[param], ivar_values[var])
    # Marginal distribution plots
    for j, param in enumerate(param_names):
        ax = axarr[0, j]
        ax.hist(param_values[param], bins=10)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if j > 0:
            ax.tick_params(axis="y", left=False)
    axarr[0, 0].set_ylabel("Count")
    for i, var in enumerate(ivar_names):
        ax = axarr[i+1, -1]
        ax.hist(ivar_values[var], bins=hist_nbins,
                            range=axarr[i+1, 0].get_ylim(),
                            orientation="horizontal")
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i < len(ivar_names):
            ax.tick_params(axis="x", bottom=False)
    axarr[-1, -1].set_xlabel("Count")
    axarr[0, -1].axis("off")
    # Set axis labels
    for i, var in enumerate(ivar_names):
        axarr[i+1, 0].set_ylabel(var)
    for j, param in enumerate(param_names):
        axarr[-1, j].set_xlabel(param)
    # Save figure
    fig.tight_layout()
    fig.savefig(analysis_dir /
                f"{analysis_name}_-_inst_var_scatterplots.svg")
    plt.close(fig)
    #
    # Instantaneous variables: Standalone variable vs. parameter scatter plots
    for param in param_names:
        for var in ivar_names:
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.scatter(param_values[param], ivar_values[var])
            ax.set_ylabel(var)
            ax.set_xlabel(param)
            fig.tight_layout()
            fig.savefig(analysis_dir /
                        f"{analysis_name}_-_inst_var_scatterplot_-_{var}_vs_{param}.svg")
    #
    # Instantaneous variables: Standalone variable & parameter histograms
    for data in (param_values, ivar_values):
        for nm in data:
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.hist(data[nm], bins=hist_nbins)
            ax.set_xlabel(nm)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(analysis_dir /
                        f"{analysis_name}_-_distribution_-_{nm}.svg")
    #
    # Time series variables: line plots with parameters coded by weight & color
    #
    # TODO: Find the reference case; the one with all parameters equal
    # to their nominal values.  This means either storing the reference
    # case when the cases are generated or adding a function to
    # calculate and return the nominal values for each variable parameter.
    #
    cen = {}
    levels = {}
    for p in param_values:
        levels[p] = sorted(set(param_values[p]))
        cen[p] = levels[p][len(levels[p]) // 2]
    m_cen = np.ones(len(cases), dtype="bool")
    for p, v in param_values.items():
        m_cen = np.logical_and(m_cen, v == cen[p])
    ind = {}
    for pname, pvalues in param_values.items():
        cmap = mpl.colors.LinearSegmentedColormap("div_blue_black_red",
                                                  colors.diverging_bky_60_10_c30_n256)
        cnorm = DivergingNorm(vmin=min(pvalues), vcenter=cen[pname],
                              vmax=max(pvalues))
        m_ind = np.ones(len(cases), dtype="bool")
        for p_oth in set(param_values.keys()) - set([pname]):
            m_ind = np.logical_and(m_ind, param_values[p_oth] == cen[p_oth])
        for varname in tvar_names:
            fig = Figure()
            fig.set_size_inches((5,4))
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.set_title(f"{varname} time series vs. {pname}")
            ax.set_ylabel(varname)
            ax.set_xlabel("Time")
            # TODO: Plot parameter sensitivity for multiple variation
            # Plot parameter sensitivity for independent variation
            for case_id in cases.index[m_ind]:
                record, tab_timeseries = read_case_data(analysis_dir /
                                                        cases["path"].loc[case_id])
                ax.plot(tab_timeseries["Time"], tab_timeseries[varname],
                        color=cmap(cnorm(pvalues.loc[case_id])))
            # Plot central case
            case_id = cases.index[m_cen][0]
            record, tab_timeseries = read_case_data(analysis_dir /
                                                    cases["path"].loc[case_id])
            ax.plot(tab_timeseries["Time"], tab_timeseries[varname],
                    color=cmap(cnorm(cen[pname])))
            cbar = fig.colorbar(ScalarMappable(norm=cnorm, cmap=cmap))
            cbar.set_label(pname)
            fig.tight_layout()
            fig.savefig(analysis_dir /
                       "_-_".join((analysis_name,
                                   "timeseries_var_lineplot",
                                   f"{varname}_vs_{pname}.svg")))
    #
    # Time series variables: correlation of instantaneous values ~ parameters.
    data = tabulate_analysis_tsvars(analysis_file, pth_cases)
    params = [c for c in data.columns if c.endswith(" [param]")]
    varnames = [c for c in data.columns if c.endswith(" [var]")]
    data = data.drop("Time", axis=1).set_index(["Step", "Case"])
    n = len(data.index.levels[0])
    sensitivity_vectors = np.zeros((len(varnames), n, len(params)))
    for i in range(n):
        corr = data.loc[i][params + varnames].corr()
        for pi, pnm in enumerate(params):
            for vi, vnm in enumerate(varnames):
                sensitivity_vectors[vi][i][pi] = corr[pnm][vnm]
    sensitivity_vectors = np.concatenate(sensitivity_vectors, axis=0)
    # Plot heatmap
    fig = Figure()
    FigureCanvas(fig)
    gs = mpl.gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[1, 5],
                               width_ratios=[25, 1],
                               hspace=0)
    cmap = mpl.colors.LinearSegmentedColormap("div_blue_black_red",
                                              colors.diverging_bky_60_10_c30_n256)
    cnorm = mpl.colors.Normalize(vmin=-1, vmax=1)
    # Plot dendrogram
    dn_ax = fig.add_subplot(gs[0,0])
    dn_ax.axis("off")
    dist = scipy.spatial.distance.pdist(
        sensitivity_vectors[np.all(~np.isnan(sensitivity_vectors), axis=1)].T,
        metric="correlation")
    links = scipy.cluster.hierarchy.linkage(dist, method="average",
                                            metric="correlation")
    dn = scipy.cluster.hierarchy.dendrogram(links, ax=dn_ax,
                                            orientation="top")
    # Plot heatmap
    ax = fig.add_subplot(gs[1,0])
    im = ax.imshow(sensitivity_vectors[:, dn["leaves"]],
                   aspect="auto",
                   origin="upper",
                   interpolation="nearest",
                   cmap=cmap, norm=cnorm,
                   extent=(-0.5, len(params) - 0.5,
                           -0.5, len(varnames) - 0.5))
    ax.set_xticks([i for i in range(len(params))])
    ax.set_xticklabels([params[i].rstrip(" [param]")
                        for i in dn["leaves"]])
    ax.set_xlabel("Parameters")
    ax.set_yticks([i for i in reversed(range(len(varnames)))])
    # ^ reversed b/c origin="upper"
    ax.set_yticklabels([nm.rstrip(" [var]") for nm in varnames],
                       {"rotation": "vertical"})
    ax.set_ylabel("Time series variables")
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[1,1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Correlation coefficient")
    # Write figure to disk
    fig.tight_layout()
    fig.savefig(analysis_dir / f"{analysis_name}_-_sensitivity_vector_heatmap.svg")
    #
    # Plot the distance matrix.  Reorder the parameters to match the
    # sensitivity vector plot.
    dist = scipy.spatial.distance.squareform(dist)[dn["leaves"], :][:, dn["leaves"]]
    cmap = mpl.cm.get_cmap("cividis")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.matshow(dist, cmap=cmap, norm=cnorm, origin="upper",
                    extent=(-0.5, len(params) - 0.5,
                            -0.5, len(params) - 0.5))
    for (i, j), d in np.ndenumerate(np.flipud(dist)):
        ax.text(j, i, '{:0.2f}'.format(d), ha='center', va='center')
    cbar = fig.colorbar(im)
    cbar.set_label("Distance correlation")
    ax.set_title("Sensitivity vector distance matrix")
    ax.set_xticks([i for i in range(len(params))])
    ax.set_yticks([i for i in reversed(range(len(params)))])
    # ^ reversed b/c origin="upper"
    ax.set_xticklabels([params[i].rstrip(" [param]") for i in dn["leaves"]])
    ax.set_yticklabels([params[i].rstrip(" [param]") for i in dn["leaves"]])
    fig.tight_layout()
    fig.savefig(analysis_dir / f"{analysis_name}_-_sensitivity_vector_distance_matrix.svg")


class NDArrayJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.floating):
            return float(o)
        else:
            return super().default(o)


def write_record_to_json(record, f):
    json.dump(record, f, indent=2, ensure_ascii=False,
              cls=NDArrayJSONEncoder)


def plot_timeseries_var(timeseries, varname, casename=None):
    """Return line plot of time series variable."""
    step = timeseries["Step"]
    time = timeseries["Time"]
    values = timeseries[varname]
    # Plot the lone variable in the table on a single axes
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(time, values, marker=".")
    ax.set_xlabel("Time")
    ax.set_ylabel(varname)
    if casename is not None:
        ax.set_title(f"{casename} {varname}")
    else:
        ax.set_title(varname)
    return fig


def plot_timeseries_vars(timeseries, dir_out, casename=None):
    """Plot time series variables and write the plots to disk.

    This function is meant for automated sensitivity analysis.  Plots
    will be written to disk using the standard spamneggs naming
    conventions.

    TODO: Provide a companion function that just returns the plot
    handle, allowing customization.

    """
    dir_out = Path(dir_out)
    if casename is None:
        stem = ""
    else:
        stem = casename + "_"
    if not isinstance(timeseries, pd.DataFrame):
        timeseries = pd.DataFrame(timeseries)
    if len(timeseries) == 0:
        raise ValueError("No values in time series data.")
    timeseries = timeseries.set_index("Step")
    nm_xaxis = "Time"
    nms_yaxis = [nm for nm in timeseries.columns if nm != nm_xaxis]
    # Produce one large plot with all variables
    axarr = timeseries.plot(marker=".", subplots=True, x=nm_xaxis,
                            legend=False)
    for nm, ax in zip(nms_yaxis, axarr):
        ax.set_ylabel(nm)
    fig = axarr[0].figure
    if casename is not None:
        axarr[0].set_title(casename)
    axarr[-1].set_xlabel(nm_xaxis)
    fig.tight_layout()
    fig.savefig(dir_out / f"{stem}timeseries_vars.svg")
    plt.close(fig)
    # Produce one small plot for each variable
    timeseries = timeseries.reset_index()
    for nm in nms_yaxis:
        fig = plot_timeseries_var(timeseries, nm, casename)
        fig.savefig(dir_out / f"{stem}timeseries_var={nm}.svg")
