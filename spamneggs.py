from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
# Third-party packages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import numpy as np
import pandas as pd
import scipy.stats
# Same-package modules
from . import febioxml as fx
from .febioxml import tabulate_case
from .febioxml import read_febio_xml as read_xml
from .variables import *


class FEBioError(Exception):
    """Raised when an FEBio simulation terminates in an error."""
    pass


def sensitivity_analysis(analysis_file, nlevels, on_failure="error",
                         analysis_dir=None):
    """Run a sensitivity analysis from spam-infused FEBio XML."""
    # Validate input
    on_failure_allowed = ("error", "hold", "skip")
    if not on_failure in on_failure_allowed:
        raise ValueError(f"on_failure = {on_failure}; allowed values are {','.join(on_failure_allowed)}")
    # Generate the cases
    tree = read_xml(analysis_file)
    analysis_name = tree.find("preprocessor[@proc='spamneggs']/"
                              "analysis").attrib["name"]
    if analysis_dir is None:
        analysis_dir = Path(analysis_name)
    cases, pth_cases_table = fx.gen_sensitivity_cases(tree, nlevels,
                                                      analysis_dir=analysis_dir)
    # Run the cases
    continue_to_analysis = True
    run_status = [None for i in range(len(cases))]
    for i, (rid, case) in enumerate(cases.iterrows()):
        try:
            run_case(analysis_dir / case["path"])
        except FEBioError as err:
            run_status[i] = "error"
            if on_failure == "error":
                raise err
        run_status[i] = "completed"
    cases["status"] = run_status
    cases.to_csv(pth_cases_table)
    # Check if error terminations prevent continuation
    m_error = cases["status"] == "error"
    if np.sum(m_error):
        if on_failure == "skip":
            cases = cases[np.logical_not(m_error)]
        elif on_failure == "hold":
            raise Exception(f"{np.sum(m_error)} cases terminated in an error.  Because `on_failure` = {on_failure}, the sensitivity analysis was stopped prior to data analysis.  The error terminations are listed in `{pth_cases_table}`.  To continue, correct the error terminations and call `make_sensitivity_figures` separately.")
    # Tabulate and plot the results
    tree = read_xml(analysis_file)  # re-read b/c gen_sensitivity_cases modifies tree
    parameters = fx.get_parameters(tree)
    ivars_table = defaultdict(list)
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


def run_case(pth_feb):
    proc = subprocess.run(['febio', '-i', pth_feb.name],
                      cwd=Path(pth_feb).parent,  # FEBio always writes xplt to current dir
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # FEBio truly does return an error code on "Error Termination";
        # I checked.
        pth_log = Path(pth_feb).with_suffix(".log")
        with open(pth_log, "wb") as f:
            f.write(proc.stdout)  # FEBio doesn't always write a log if it
                                  # hits a error, but the content that would
                                  # have been logged is always dumped to
                                  # stdout.
        raise FEBioError(f"FEBio returned an error (return code = {proc.returncode}) while running {pth_feb}; check {pth_log}.")
    return proc.returncode


def make_sensitivity_figures(analysis_file):
    tree = read_xml(analysis_file)
    analysis_name = tree.find("preprocessor[@proc='spamneggs']/"
                              "analysis").attrib["name"]
    analysis_dir = Path(analysis_name)
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


def sensitivity_loc_ind_curve(solve, cen, incr, dir_out,
                              names, units,
                              output_names, output_units):
    """Local sensitivity for curve output with independently varied params.

    solve := a function that accepts an N-tuple of model parameters and
    returns the corresponding model outputs as a tuple.  The first value
    of the output is considered the independent variable; the others are
    the dependent variables.

    cen := N-tuple of numbers.  Model parameters.  These are the central values
    about which the local sensitivity analysis is done.

    incr := N-tuple of numbers.  Increments specifying how much to change each
    model parameter in the sensitivity analysis.

    dir_out := path to directory into which to write the sensitivity
    analysis plots.

    names := N-tuple of strings naming the parameters.  These will be
    used to label the plots.

    units := N-tuple of strings specifying units for the parameters.  These will be
    used to label the plots.

    """
    linestyles = "solid", "dashed", "dotted"
    for i, (c, Δ, name, unit) in enumerate(zip(cen, incr, names, units)):
        fig, ax = plt.subplots()
        ax.set_title(f"Local model sensitivity to independent variation of {name}")
        ax.set_xlabel(f"{output_names[0]} [{output_units[0]}]")
        ax.set_ylabel(f"{output_names[1]} [{output_units[1]}]")
        # Plot central value
        p = [x for x in cen]
        output = solve(p)
        ax.plot(output[0], output[1], linestyle=linestyles[0],
                color="black",
                label=f"{name} = {c}")
        # Plot ± increments
        for j in (1, 2):
            # High
            p[i] = c + j*Δ
            output = solve(p)
            ax.plot(output[0], output[1], linestyle=linestyles[j],
                    color="black",
                    label=f"{name} = {c} ± {j*Δ}")
            # Low
            p[i] = c - j*Δ
            output = solve(p)
            ax.plot(output[0], output[1], linestyle=linestyles[j],
                    color="black")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(dir_out, f"sensitivity_local_ind_curve_-_{name}.svg"))


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
