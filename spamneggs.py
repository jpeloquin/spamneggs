from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

# Third-party packages
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import multiprocessing as mp
import numpy as np
import pandas as pd
import psutil
import scipy.cluster

# In-house packages
import febtools as feb

# Same-package modules
from . import febioxml as fx
from . import colors
from .febioxml import read_febio_xml as read_xml
from .variables import *


NUM_WORKERS = psutil.cpu_count(logical=False)

CMAP_DIVERGE = mpl.colors.LinearSegmentedColormap(
    "div_blue_black_red", colors.diverging_bky_60_10_c30_n256
)


class FEBioError(Exception):
    """Raised when an FEBio simulation terminates in an error."""

    pass


class CaseGenerationError(Exception):
    """Raised when case generation terminates in an error."""

    pass


class Analysis:
    def __init__(
        self, model, parameters: dict, variables: dict, name=None, directory=None
    ):
        self.model = model
        self.parameters: dict = parameters
        self.variables: dict = variables
        self.name = name
        if directory is None:
            self.directory = Path(name).absolute()
        else:
            self.directory = Path(directory).absolute()

    @classmethod
    def from_xml(cls, pth):
        pth = Path(pth)
        tree = read_xml(pth)
        e_analysis = tree.find("preprocessor[@proc='spamneggs']/analysis")
        if e_analysis is None:
            raise ValueError(
                f"No XML element with path 'preprocessor/analysis' found in file '{analysis_file}'."
            )
        # Analysis name
        e_analysis = tree.find("preprocessor/analysis")
        if "name" in e_analysis.attrib:
            name = e_analysis.attrib["name"]
        else:
            # Use name of file as default name of analysis
            name = pth.stem
        # Analysis directory
        if "path" in e_analysis.attrib:
            analysis_dir = Path(e_analysis["path"]).absolute
        else:
            analysis_dir = pth.parent.absolute() / name
        # Parameters
        parameters, parameter_locations = fx.get_parameters(tree)
        # Variables
        variables = fx.get_variables(tree)
        # Model
        fx.strip_preprocessor_elems(tree, parameters)  # mutates in-place
        model = FEBioXMLModel(tree, parameter_locations)
        return cls(model, parameters, variables, name, analysis_dir)


class Case:
    def __init__(
        self,
        analysis,
        parameters: dict,
        name=None,
        sim_file=None,
        case_dir=None,
        solution=None,
    ):
        self.analysis = analysis
        self.parameters = parameters
        self.name = name
        self.sim_file = sim_file
        self.case_dir = Path(case_dir) if case_dir is not None else None
        self._solution = solution

    @property
    def variables(self):
        return self.analysis.variables

    @property
    def solution(self):
        if self._solution is None:
            self._solution = feb.load_model(self.sim_file)
        return self._solution

    @classmethod
    def from_values(cls, analysis, pvals: dict, name: str):
        """Return a Case for the given parameter values"""
        # For a sensitivity analysis or other large-scale analysis, the
        # cases should go in analysis.directory / "cases".  But in
        # typical interactive use that's a needless level of
        # indirection.  Large-scale analyses should implement their
        # directory structure by calling the initializer directly.
        sim_file = analysis.directory / f"{name}.feb"
        case_dir = analysis.directory / name
        return cls(analysis, pvals, name, sim_file, case_dir)

    @classmethod
    def from_named(cls, analysis, pset: str):
        """Return a Case for the given named parameter set"""
        # Use the parameter set's name as the default name for the case.
        # Don't provide options to adjust the names; it is desirable for
        # the case to match the analysis file.  The user can always call
        # the initializer directly.
        pvals = {p: d[pset] for p, d in analysis.parameters.items()}
        return cls.from_values(analysis, pvals, name=pset.replace(" ", "_"))

    def write_case(self):
        """Write the case's model to disk"""
        if not self.sim_file.parent.exists():
            self.sim_file.parent.mkdir(parents=True)
        tree = self.analysis.model(self)
        with open(self.sim_file, "wb") as f:
            fx.write_febio_xml(tree, f)


class FEBioXMLModel:
    """A model defined by an FEBio XML tree

    FEBioXMLModel is callable.  It accepts a single Case
    instance as an argument and returns a concrete FEBio XML tree, with
    the contents of all parameter elements replaced by the parameter
    values specified in the Case.

    """

    def __init__(self, tree, parameters: dict):
        """Return an FEBioXMLModel instance

        tree := the FEBio XML tree.

        parameters := a dictionary of parameter name → list of
        ElementPath paths to elements in the XML for which the element's
        text content is to be replaced by the parameter's value.

        """
        self.tree = tree
        self.parameters = parameters

    def __call__(self, case: Case):
        # Create a copy of the tree so we don't alter the original tree.  This
        # is necessary in case multiple worker threads are generating models
        # from the same FEBioXMLModel object.
        tree = deepcopy(self.tree)
        # Alter the model parameters to match the current case
        for pname, plocs in self.parameters.items():
            for ploc in plocs:
                e_parameter = tree.find(ploc)
                assert e_parameter is not None
                e_parameter.text = str(case.parameters[pname])
        # Add the needed elements in <Output> to support the requested
        # variables.  We also have to update the file name attribute to match
        # this case.
        logfile_reqs, plotfile_reqs = fx.required_outputs(case.variables)
        fx.insert_output_elem(tree, logfile_reqs, plotfile_reqs, file_stem=case.name)
        return tree


def _validate_on_case_error(value):
    # Validate input
    on_case_error_options = ("stop", "hold", "ignore")
    if not value in on_case_error_options:
        raise ValueError(
            f"on_case_error = {value}; allowed values are {','.join(on_case_error_options)}"
        )


def run_sensitivity(analysis, nlevels, on_case_error="stop"):
    """Run a sensitivity analysis from an analysis object."""
    _validate_on_case_error(on_case_error)
    # Create the cases
    cases, pth_cases_table = gen_sensitivity_cases(
        analysis, nlevels, on_case_error=on_case_error
    )
    good_case_ids = cases.index[cases["status"] == "generation complete"]
    run_case = partial(run_febio_unchecked, threads=1)
    # Potential improvement: increase OMP_NUM_THREADS for last few jobs
    # as more cores are free.  In most cases this would make little
    # difference, though.
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Submit the first round of cases
        futures = {
            pool.submit(
                run_case, analysis.directory / cases.loc[case_id, "path"]
            ): case_id
            for case_id in good_case_ids[:NUM_WORKERS]
        }
        pending_cases = set([case_id for case_id in good_case_ids[NUM_WORKERS:]])
        # Submit remaining cases as workers become available.  We do not
        # submit all cases immediately because we wish to have the
        # option of ending the analysis early if a case ends in error
        # termination.
        while pending_cases:
            future = next(as_completed(futures))
            case_id = futures.pop(future)
            pth_feb = cases.loc[case_id, "path"]
            return_code = future.result()
            # print(f"Popped case {case_id} from futures")
            # Log case details
            if return_code == 0:
                cases.loc[case_id, "status"] = "run complete"
            else:
                cases.loc[case_id, "status"] = "run error"
            # Check if we should continue submitting cases
            if on_case_error == "stop" and return_code != 0:
                print(
                    f"FEBio returned error code {return_code} while running case {case_id} ({pth_feb}); check {pth_feb.with_suffix('.log')}.  Because `on_case_error` = {on_case_error}, spamneggs will finish any currently running cases, then stop."
                )
                break
            # Submit next case
            next_case = pending_cases.pop()
            # print(f"Submitting case {next_case}")
            futures.update(
                {
                    pool.submit(
                        run_case, analysis.directory / cases.loc[next_case, "path"]
                    ): next_case
                }
            )
        # Finish processing all running cases.
        for future in as_completed(futures):
            case_id = futures[future]
            return_code = future.result()
            if return_code == 0:
                cases.loc[case_id, "status"] = "run complete"
            else:
                cases.loc[case_id, "status"] = "run error"
    cases.to_csv(pth_cases_table)
    # Check if error terminations prevent analysis of results
    m_error = np.logical_or(
        cases["status"] == "generation error", cases["status"] == "run error"
    )
    if np.any(m_error):
        if on_case_error == "ignore":
            cases = cases[np.logical_not(m_error)]
        elif on_case_error == "hold":
            raise Exception(
                f'{np.sum(m_error)} cases had generation or simulation errors.  Because `on_case_error` = "{on_case_error}", the sensitivity analysis was stopped prior to data analysis.  The error terminations are listed in `{pth_cases_table}`.  To continue, correct the error terminations and call `plot_sensitivity` separately.'
            )
        elif on_case_error == "stop":
            # Error message was printed above
            sys.exit()
    # Tabulate and plot the results
    tabulate(analysis)
    plot_sensitivity(analysis)


def gen_sensitivity_cases(analysis, nlevels, on_case_error="stop"):
    """Return table of cases for sensitivity analysis."""
    _validate_on_case_error(on_case_error)
    # Check output directory
    if analysis.directory is None:
        raise ValueError(
            "Analysis.directory is None.  gen_sensitivity_cases requries an output directory."
        )
    if not analysis.directory.exists():
        analysis.directory.mkdir()
    # Create output subdirectory for the FEBio files
    cases_dir = analysis.directory / "cases"
    if not cases_dir.exists():
        cases_dir.mkdir()
    # Generate parameter values for each case
    colnames = []
    levels = {}
    for pname, pdata in analysis.parameters.items():
        colnames.append(pname)
        param = pdata.distribution
        # Calculate variable's levels
        if isinstance(param, ContinuousScalar):
            levels[pname] = param.sensitivity_levels(nlevels)
        elif isinstance(param, CategoricalScalar):
            levels[pname] = param.sensitivity_levels()
        else:
            raise ValueError(
                f"Generating levels from a variable of type `{type(var)}` is not yet supported."
            )
    tab_cases = pd.DataFrame(
        {k: v for k, v in zip(colnames, zip(*product(*(levels[k] for k in colnames))))}
    )
    tab_cases["status"] = ""
    tab_cases["path"] = ""
    # Modify the parameters in the XML and write the modified XML to disk
    for i, row in tab_cases.iterrows():
        pvalues = {k: row[k] for k in analysis.parameters}
        case_name = f"case={i}"
        case = Case(
            analysis,
            pvalues,
            case_name,
            sim_file=cases_dir / f"{case_name}.feb",
            case_dir=cases_dir / case_name,
        )
        try:
            gen_case(analysis, case)
        except Exception as err:
            # If the case generation fails, log the error and deal with it
            # according to the on_case_error argument.
            if on_case_error == "stop":
                raise CaseGenerationError(
                    f"Case {i} generation failed.  Because `on_case_error` = {on_case_error}, spamneggs will now exit."
                )
            elif on_case_error in ("hold", "ignore"):
                tab_cases.loc[i, "status"] = "generation error"
        else:
            tab_cases.loc[i, "status"] = "generation complete"
            tab_cases.loc[i, "path"] = case.sim_file.relative_to(analysis.directory)
    # TODO: Figure out same way to demarcate parameters from other
    # metadata so there are no reserved parameter names.  For example, a
    # user should be able to name their parameter "path" without
    # conflicts.
    pth_cases = analysis.directory / f"cases.csv"
    tab_cases.to_csv(pth_cases, index_label="case")
    return tab_cases, pth_cases


def gen_case(analysis, case):
    tree = analysis.model(case)
    # Write the case's FEBio XML tree to disk
    with open(case.sim_file, "wb") as f:
        fx.write_febio_xml(tree, f)


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


def tabulate_case_write(case, dir_out=None):
    """Tabulate variables for single case analysis & write to disk."""
    # Find/create output directory
    if dir_out is None:
        dir_out = case.sim_file.parent
    else:
        dir_out = Path(dir_out)
    if not dir_out.exists():
        dir_out.mkdir()
    # Tabulate the variables
    record, timeseries = fx.tabulate_case(case)
    with open(dir_out / f"{case.sim_file.stem}_vars.json", "w") as f:
        write_record_to_json(record, f)
    timeseries.to_csv(
        dir_out / f"{case.sim_file.stem}_timeseries_vars.csv", index=False
    )
    plot_case_tsvars(timeseries, dir_out=dir_out, casename=case.name)
    return record, timeseries


def tabulate_analysis_tsvars(analysis, cases_file):
    """Tabulate time series variables for all cases in an analysis

    The time series tables for the individual cases must have already
    been written to disk.

    """
    # TODO: It would be beneficial to build up the whole-analysis time
    # series table at the same time as case time series tables are
    # written to disk, instead of re-reading everything from disk.
    pth_cases = Path(cases_file)
    cases = pd.read_csv(cases_file, index_col=0)
    analysis_data = pd.DataFrame()
    for i in cases.index:
        pth_tsvars = pth_cases.parent / "case_output" / f"case={i}_timeseries_vars.csv"
        case_data = pd.read_csv(pth_tsvars)
        varnames = set(case_data.columns) - set(["Time", "Step"])
        case_data = case_data.rename({k: f"{k} [var]" for k in varnames}, axis=1)
        case_data["Case"] = i
        for pname in analysis.parameters:
            pcolname = f"{pname} [param]"
            case_data[pcolname] = cases[pname].loc[i]
        analysis_data = pd.concat([analysis_data, case_data])
    return analysis_data


def run_febio_checked(pth_feb, threads=psutil.cpu_count(logical=False)):
    """Run FEBio, raising exception on error."""
    pth_feb = Path(pth_feb)
    proc = _run_febio(pth_feb, threads=threads)
    if proc.returncode != 0:
        raise FEBioError(
            f"FEBio returned error code {proc.returncode} while running {pth_feb}; check {pth_feb.with_suffix('.log')}."
        )
    return proc.returncode


def run_febio_unchecked(pth_feb, threads=psutil.cpu_count(logical=False)):
    """Run FEBio and return its error code."""
    return _run_febio(pth_feb, threads=threads).returncode


def _run_febio(pth_feb, threads=psutil.cpu_count(logical=False)):
    """Run FEBio and return the process object."""
    # FEBio's error handling is interesting, in a bad way.  XML file
    # read errors are only output to stdout.  If there is a read error,
    # no log file is created and if an old log file exists, it is not
    # updated to reflect the file read error.  Model summary info is
    # only output to the log file.  Time stepper info is output to both
    # stdout and the log file, but the verbosity of the stdout output
    # can be adjusted by the user.  We want to ensure (1) the log file
    # always reflects the last run and (2) all relevant error messages
    # are written to the log file.
    pth_feb = Path(pth_feb)
    pth_log = pth_feb.with_suffix(".log")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": f"{threads}"})
    # Check for the existance of the FEBio XML file ourselves, since if
    # the file doesn't exist FEBio will act as if it was malformed.
    if not pth_feb.exists():
        raise ValueError(f"'{pth_feb}' does not exist or is not accessible.")
    proc = subprocess.run(
        ["febio", "-i", pth_feb.name],
        cwd=pth_feb.parent,  # FEBio always writes xplt to current dir
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    # FEBio does return an error code on "Error Termination"; I checked.
    if proc.returncode != 0:
        # If there is a file read error, we need to write the captured
        # stdout to the log file, because only it has information about
        # the file read error.  Otherwise, we need to leave the log file
        # in place, because it has unique information.
        for ln in proc.stdout.splitlines():
            if ln.startswith("Reading file"):
                if ln.endswith("SUCCESS!"):
                    # No file read error
                    break
                elif ln.endswith("FAILED!"):
                    # File read error; send it to the log file
                    with open(pth_log, "w", encoding="utf-8") as f:
                        f.write(proc.stdout)
                else:
                    raise NotImplementedError(
                        f"spamneggs failed to parse FEBio file read status '{ln}'"
                    )
    return proc


def tabulate(analysis: Analysis):
    """Tabulate output from an analysis."""
    ivars_table = defaultdict(list)
    pth_cases = analysis.directory / f"cases.csv"
    cases = pd.read_csv(pth_cases, index_col=0)
    if len(cases) == 0:
        raise ValueError(f"No cases to tabulate in '{pth_cases}'")
    for i in cases.index:
        casename = Path(cases["path"].loc[i]).stem
        case = Case(
            analysis,
            {p: cases[p].loc[i] for p in analysis.parameters},
            name=casename,
            sim_file=analysis.directory / cases["path"].loc[i],
            case_dir=analysis.directory / "cases" / casename,
            solution=None,
        )
        record, timeseries = tabulate_case_write(
            case, dir_out=analysis.directory / "case_output"
        )
        ivars_table["case"].append(i)
        for p in analysis.parameters:
            k = f"{p} [param]"
            ivars_table[k].append(cases[p].loc[i])
        for v in record["instantaneous variables"]:
            k = f"{v} [var]"
            ivars_table[k].append(record["instantaneous variables"][v]["value"])
    df_ivars = pd.DataFrame(ivars_table).set_index("case")
    df_ivars.to_csv(analysis.directory / f"inst_vars.csv", index=True)


def plot_sensitivity(analysis):
    pth_cases = analysis.directory / f"cases.csv"
    cases = pd.read_csv(pth_cases, index_col=0)
    cases = cases[cases["status"] == "run complete"]

    # Read parameters
    #
    # TODO: Add a function to get the list of parameters and variables
    # for an analysis up-front.  Extract the relevant code from
    # tabulate_case.
    param_names = [k for k in cases.columns if k not in ["case", "path", "status"]]
    param_values = {k: cases[k] for k in param_names}

    # Get variable names
    record, tab_timeseries = read_case_data(analysis.directory / cases["path"].iloc[0])
    ivar_names = [nm for nm in record["instantaneous variables"]]
    tsvar_names = [nm for nm in record["time series variables"]]

    # Plots for instantantaneous variables
    ivar_values = defaultdict(list)
    for i in cases.index:
        # TODO: There is an opportunity for improvement here: We could
        # preserve the tree from the first read of the analysis XML and
        # re-use it here instead of re-reading the analysis XML for
        # every case.  However, to do this the case generation must not
        # alter the analysis XML tree.
        record, tab_timeseries = read_case_data(
            analysis.directory / cases["path"].loc[i]
        )
        for nm in ivar_names:
            ivar_values[nm].append(record["instantaneous variables"][nm]["value"])
    if len(ivar_names) > 0:
        make_sensitivity_ivar_figures(
            analysis, param_names, param_values, ivar_names, ivar_values
        )

    # Plots for time series variables
    if len(tsvar_names) > 0:
        tsdata = tabulate_analysis_tsvars(analysis, pth_cases)
        make_sensitivity_tsvar_figures(
            analysis, param_names, param_values, tsvar_names, tsdata, cases
        )


def make_sensitivity_ivar_figures(
    analysis, param_names, param_values, ivar_names, ivar_values
):
    # Matrix of instantaneous variable vs. parameter scatter plots
    npanels_w = len(param_names) + 1
    npanels_h = len(ivar_names) + 1
    hist_nbins = 9
    fig, axarr = plt.subplots(npanels_h, npanels_w, sharex="col", sharey="row")
    fig.set_size_inches(1.8 * npanels_w + 1, 2 * npanels_h + 1)
    # Scatter plots
    for j, param in enumerate(param_names):
        for i, var in enumerate(ivar_names):
            axarr[i + 1, j].scatter(param_values[param], ivar_values[var])
    # Marginal distribution plots
    for j, param in enumerate(param_names):
        ax = axarr[0, j]
        ax.hist(param_values[param], bins=10)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if j > 0:
            ax.tick_params(axis="y", left=False)
    axarr[0, 0].set_ylabel("Count")
    for i, var in enumerate(ivar_names):
        ax = axarr[i + 1, -1]
        ax.hist(
            ivar_values[var],
            bins=hist_nbins,
            range=axarr[i + 1, 0].get_ylim(),
            orientation="horizontal",
        )
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i < len(ivar_names):
            ax.tick_params(axis="x", bottom=False)
    axarr[-1, -1].set_xlabel("Count")
    axarr[0, -1].axis("off")
    # Set axis labels
    for i, var in enumerate(ivar_names):
        axarr[i + 1, 0].set_ylabel(var)
    for j, param in enumerate(param_names):
        axarr[-1, j].set_xlabel(param)
    # Save figure
    fig.tight_layout()
    fig.savefig(analysis.directory / f"inst_var_scatterplots.svg")
    plt.close(fig)

    # Standalone instantaneous variable vs. parameter scatter plots
    for param in param_names:
        for var in ivar_names:
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.scatter(param_values[param], ivar_values[var])
            ax.set_ylabel(var)
            ax.set_xlabel(param)
            fig.tight_layout()
            fig.savefig(
                analysis.directory / f"inst_var_scatterplot_-_{var}_vs_{param}.svg"
            )

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
            fig.savefig(analysis.directory / f"distribution_-_{nm}.svg")


def make_sensitivity_tsvar_figures(
    analysis, param_names, param_values, tsvar_names, tsdata, cases
):
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
        cnorm = mpl.colors.Normalize(vmin=min(pvalues), vmax=max(pvalues))
        m_ind = np.ones(len(cases), dtype="bool")
        for p_oth in set(param_values.keys()) - set([pname]):
            m_ind = np.logical_and(m_ind, param_values[p_oth] == cen[p_oth])
        for varname in tsvar_names:
            fig = Figure()
            fig.set_size_inches((5, 4))
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.set_title(f"{varname} time series vs. {pname}")
            ax.set_ylabel(varname)
            ax.set_xlabel("Time")
            # TODO: Plot parameter sensitivity for multiple variation
            # Plot parameter sensitivity for independent variation
            for case_id in cases.index[m_ind]:
                record, tab_timeseries = read_case_data(
                    analysis.directory / cases["path"].loc[case_id]
                )
                ax.plot(
                    tab_timeseries["Time"],
                    tab_timeseries[varname],
                    color=CMAP_DIVERGE(cnorm(pvalues.loc[case_id])),
                )
            # Plot central case
            case_id = cases.index[m_cen][0]
            record, tab_timeseries = read_case_data(
                analysis.directory / cases["path"].loc[case_id]
            )
            ax.plot(
                tab_timeseries["Time"],
                tab_timeseries[varname],
                color=CMAP_DIVERGE(cnorm(cen[pname])),
            )
            cbar = fig.colorbar(ScalarMappable(norm=cnorm, cmap=CMAP_DIVERGE))
            cbar.set_label(pname)
            fig.tight_layout()
            nm = f"timeseries_var_lineplot_-_{varname}_vs_{pname}.svg"
            fig.savefig(analysis.directory / nm.replace(" ", "_"))

    plot_tsvars_heat_map(analysis, tsdata, norm="none")
    plot_tsvars_heat_map(analysis, tsdata, norm="all")
    plot_tsvars_heat_map(analysis, tsdata, norm="vector")
    plot_tsvars_heat_map(analysis, tsdata, norm="subvector")


class NDArrayJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.floating):
            return float(o)
        else:
            return super().default(o)


def write_record_to_json(record, f):
    json.dump(record, f, indent=2, ensure_ascii=False, cls=NDArrayJSONEncoder)


def plot_case_tsvar(timeseries, varname, casename=None):
    """Return a line plot of a case's time series variable."""
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


def plot_case_tsvars(timeseries, dir_out, casename=None):
    """Plot a case's time series variables and write the plots to disk.

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
    axarr = timeseries.plot(marker=".", subplots=True, x=nm_xaxis, legend=False)
    for nm, ax in zip(nms_yaxis, axarr):
        ax.set_ylabel(nm)
    fig = axarr[0].figure
    fig.set_size_inches((7, 1.0 + 1.25 * (len(timeseries.columns) - 1)))
    if casename is not None:
        axarr[0].set_title(casename)
    # Format axes
    axarr[-1].set_xlabel(f"{nm_xaxis} [s]")  # assumed unit
    formatter = mpl.ticker.ScalarFormatter()
    formatter.set_powerlimits((-3, 4))
    axarr[-1].xaxis.set_major_formatter(formatter)
    # Write figure to disk
    fig.tight_layout()
    fig.savefig(dir_out / f"{stem}timeseries_vars.svg")
    plt.close(fig)
    # Produce one small plot for each variable
    timeseries = timeseries.reset_index()
    for nm in nms_yaxis:
        fig = plot_case_tsvar(timeseries, nm, casename)
        fig.savefig(dir_out / f"{stem}timeseries_var={nm}.svg")


def plot_tsvars_heat_map(analysis, tsdata, norm="none", corr_threshold=1e-6):
    """Plot times series variable ∝ parameter heat maps.

    norm := "none", "all", "vector", or "individual".  Type of color
    scale normalization to apply.

        - "none" = Use one color scale for the whole plot, with a range
          of ± 1.

        - "all" = Use one color scale for the whole plot, with a range
          of ± absolute maximum across all sensitivity vectors.

        - "vector" = Use a different color scale for parameter's
          sensitivity vector.  Set the range to ± absolute maximum of
          the vector.

        - "subvector" = Use a different color scale for each parameter +
          time series variable pair.  Set the range to ± absolute
          maximum of the subvector corresponding to the pair.

    """
    # Data munging
    params = [c for c in tsdata.columns if c.endswith(" [param]")]
    varnames = [c for c in tsdata.columns if c.endswith(" [var]")]
    # ^ These names are tagged with their type to avoid name collisions.
    # The plain names passed as arguments cannot be used.
    data = tsdata.copy()  # need to mutate the table
    data = data.drop("Time", axis=1).set_index(["Step", "Case"])

    # Calculate sensitivity vectors
    n = len(data.index.levels[0])
    sensitivity_vectors = np.zeros((len(varnames), n, len(params)))
    for i in range(n):
        corr = data.loc[i][params + varnames].corr()
        cov = data.loc[i][params + varnames].cov()
        for pi, pnm in enumerate(params):
            for vi, vnm in enumerate(varnames):
                ρ = corr[pnm][vnm]
                σ = cov[vnm][vnm]
                # Coerce correlation to zero if it is nan only because
                # the output variable has no variance
                if np.isnan(ρ) and σ == 0:
                    ρ = 0
                sensitivity_vectors[vi][i][pi] = ρ

    # Plot the heatmap
    fontsize_axlabel = 11
    fontsize_figlabel = 12
    fontsize_ticklabel = 9
    lh = 1.5  # multiple of font size to use for label height

    # Calculate widths of figure elements.  TODO: It would be better to
    # specify the /figure/ width, then calculate the necessary axes
    # widths.
    dendro_axw = 12 / 72 * len(params)
    fig_llabelw = lh * fontsize_figlabel / 72
    dendro_areaw = fig_llabelw + dendro_axw
    cbar_axw = 12 / 72
    # ^ width of colorbar axes
    cbar_rpad = 36 / 72
    # ^ padding b/w colorbar axes and axes of /next/ heat map
    cbar_lpad = 4 / 72
    # ^ padding b/w heat map axes and its colorbar axes
    cbar_areaw = cbar_lpad + cbar_axw + cbar_rpad
    hmap_lpad = lh * fontsize_axlabel / 72
    hmap_axw = 4
    # ^ width of individual heat map axes object if individual color
    # scales used.  Otherwise heatmap expands to fill space.
    hmap_subw = hmap_lpad + hmap_axw + cbar_lpad + cbar_axw + cbar_rpad
    hmap_subwspace = fontsize_axlabel / 72
    hmap_subw = 4.5
    hmap_areaw = hmap_subw * len(varnames) + hmap_subwspace * (len(varnames) - 1)
    if norm in ("none", "all", "vector"):
        right_cbar = True
        rcbar_areaw = cbar_areaw
        rcbar_wspace = hmap_subwspace
        hmap_axw = hmap_subw - hmap_lpad
    else:  # norm == "individual"
        right_cbar = False
        rcbar_areaw = 0
        rcbar_wspace = 0
        hmap_axw = hmap_subw - hmap_lpad - cbar_areaw
    figw = dendro_areaw + hmap_areaw + rcbar_wspace + rcbar_areaw + rcbar_wspace

    # Calculate heights of figure elements
    fig_tlabelh = lh * fontsize_figlabel / 72  # "Time series variables"
    hmap_tlabelh = lh * fontsize_axlabel / 72  # Variable names
    hmap_blabelh = lh * fontsize_axlabel / 72  # "Step"
    hmap_axh = 0.75
    hmap_vspace = (lh + 0.5) * fontsize_ticklabel / 72  # allocated for x tick labels
    figh = (
        hmap_blabelh
        + (hmap_vspace + hmap_axh) * len(params)
        + hmap_tlabelh
        + fig_tlabelh
    )

    fig = Figure(figsize=(figw, figh))
    FigureCanvas(fig)

    # Top and bottom edges of heat map axes
    hmap_areal = fig_llabelw + dendro_axw
    # ^ left coord of heatmap area
    hmap_areab = hmap_blabelh + 0.5 * hmap_vspace
    # ^ bottom coord of heatmap area, positioned such that vertical
    # center of heatmap axes will line up with the dendrogram ticks
    hmapaxes_b0 = hmap_areab + 0.5 * hmap_vspace
    hmapaxes_t0 = hmapaxes_b0 + (len(params) - 1) * hmap_vspace + len(params) * hmap_axh

    # Plot dendrogram
    b = hmap_blabelh + 0.5 * hmap_vspace
    h = (hmap_vspace + hmap_axh) * len(params)
    dn_ax = fig.add_axes((fig_llabelw / figw, b / figh, dendro_axw / figw, h / figh))
    arr = np.concatenate(sensitivity_vectors, axis=0).T
    # ^ first index over parameters, second over timepoints
    # Compute correlation distance =
    #    1 - (u − u̅) * (v - v̅) / ( 2-norm(u − u̅) * 2-norm(v - v̅) )
    # If the 2-norm of (u − u̅) or (v − v̅) is zero, then the result will be
    # undefined.  Typically this will happen when u or v is all zeroes; in this
    # case, the numerator is also zero.  For this application, it is reasonable
    # to define 0/0 = 0.  Therefore we need to check for (u − u̅) * (v - v̅) = 0
    # and set the correlation distance for those vector pairs to 1.
    dist = scipy.spatial.distance.pdist(arr, metric="correlation")
    n = len(arr)
    numerator = np.empty(n)
    numerator[:] = np.nan  # make indexing errors more obvious
    means = np.mean(arr, axis=1)
    for i in range(n):
        for j in range(i + 1, max(i, n)):
            idx = (
                scipy.special.comb(n, 2, exact=True)
                - scipy.special.comb(n - i, 2, exact=True)
                + (j - i - 1)
            )
            numerator[idx] = (arr[i] - means[i]) @ (arr[j] - means[j])
            # print(f"i = {i}; j = {j}; idx = {idx}; numerator = {numerator[idx]}")
    dist[numerator == 0] = 1
    # Compute the linkages
    links = scipy.cluster.hierarchy.linkage(
        dist, method="average", metric="correlation"
    )
    dn = scipy.cluster.hierarchy.dendrogram(links, ax=dn_ax, orientation="left")
    dn_ax.invert_yaxis()  # to match origin="upper"; 1st param at top
    dn_ax.set_ylabel("Parameters", fontsize=fontsize_figlabel)
    dn_ax.tick_params(
        bottom=False,
        right=False,
        left=False,
        top=False,
        labelbottom=False,
        labelright=False,
        labelleft=False,
        labeltop=False,
    )
    for spine in ["left", "right", "top", "bottom"]:
        dn_ax.spines[spine].set_visible(False)

    # Plot heatmaps
    if norm == "none":
        absmax = 1
        cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
    elif norm == "all":
        absmax = np.max(np.abs(sensitivity_vectors))
        cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
    for irow, iparam in enumerate(reversed(dn["leaves"])):
        if norm == "vector":
            absmax = np.max(np.abs(sensitivity_vectors[:, :, iparam]))
            cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
        for ivar, varname in enumerate(varnames):
            if norm == "subvector":
                absmax = np.max(np.abs(sensitivity_vectors[ivar, :, iparam]))
                cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)

            # Calculate bounding box for heatmap and colorbar axes.
            # Include the left & right labels within the bounding box,
            # but exclude the top and bottom labels.
            bbox = (
                hmap_areal + ivar * (hmap_subw + hmap_subwspace),
                hmap_areab + 0.5 * hmap_vspace + irow * (hmap_vspace + hmap_axh),
                hmap_subw,
                hmap_axh,
            )

            # Draw the heatmap
            #
            # Axes
            l = hmap_areal + ivar * (hmap_subw + hmap_subwspace) + hmap_lpad
            w = hmap_axw
            b = hmapaxes_b0 + irow * (hmap_vspace + hmap_axh)
            h = hmap_axh
            ax = fig.add_axes((l / figw, b / figh, w / figw, h / figh))
            # Image
            ρ = sensitivity_vectors[ivar, :, iparam]
            im = ax.imshow(
                np.atleast_2d(ρ),
                aspect="auto",
                origin="upper",
                interpolation="nearest",
                cmap=CMAP_DIVERGE,
                norm=cnorm,
                extent=(
                    tsdata["Step"].iloc[0] - 0.5,
                    tsdata["Step"].iloc[-1] + 0.5,
                    -0.5,
                    0.5,
                ),
            )
            # Labels
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.tick_params(axis="x", labelsize=fontsize_ticklabel)
            ax.set_ylabel(params[iparam].rstrip(" [param]"), fontsize=fontsize_axlabel)
            ax.tick_params(axis="y", left=False, labelleft=False)
            if irow == 0:
                ax.set_xlabel("Step", fontsize=fontsize_ticklabel)
            if irow == len(params) - 1:
                ax.set_title(varname.rstrip(" [var]"), fontsize=fontsize_figlabel)

            # Draw the heatmap's colorbar
            if norm == "subvector":
                l = l + hmap_axw + cbar_lpad
                cbar_ax = fig.add_axes(
                    (l / figw, b / figh, cbar_axw / figw, hmap_axh / figh)
                )
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
                cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
                cbar.ax.tick_params(labelsize=fontsize_ticklabel)
                cbar.set_label("ρ [1]", fontsize=fontsize_ticklabel)
                cbar.ax.yaxis.set_label_coords(2.7, 0.5)
            elif norm == "vector" and ivar == len(varnames) - 1:
                l = dendro_areaw + hmap_areaw + rcbar_wspace
                b = hmapaxes_b0 + irow * (hmap_vspace + hmap_axh)
                w = cbar_axw
                h = hmap_axh
                ax = fig.add_axes((l / figw, b / figh, w / figw, h / figh))
                cbar = fig.colorbar(im, cax=ax)  # `im` from last imshow
                cbar.ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
                cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
                cbar.ax.tick_params(labelsize=fontsize_ticklabel)
                cbar.set_label("ρ [1]", fontsize=fontsize_ticklabel)

    # Add the whole-plot right-most colorbar, if called for
    if norm in ("none", "all"):
        l = dendro_areaw + hmap_areaw + rcbar_wspace
        b = hmapaxes_b0
        w = cbar_axw
        h = hmapaxes_t0 - hmapaxes_b0
        ax = fig.add_axes((l / figw, b / figh, w / figw, h / figh))
        cbar = fig.colorbar(im, cax=ax)  # `im` from last imshow
        cbar.ax.yaxis.set_major_locator(mpl.ticker.LinearLocator())
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
        cbar.ax.tick_params(labelsize=fontsize_ticklabel)
        cbar.set_label("ρ [1]", fontsize=fontsize_ticklabel)

    # Add whole-plot labels
    fig.suptitle(
        f"Time series variable correlations, norm = {norm}", fontsize=fontsize_figlabel
    )
    # Write figure to disk
    fig.savefig(analysis.directory / f"sensitivity_vector_heatmap_norm={norm}.svg")
    #
    # Plot the distance matrix.  Reorder the parameters to match the
    # sensitivity vector plot.
    dist = scipy.spatial.distance.squareform(dist)[dn["leaves"], :][:, dn["leaves"]]
    cmap = mpl.cm.get_cmap("cividis")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.matshow(
        dist,
        cmap=cmap,
        norm=cnorm,
        origin="upper",
        extent=(-0.5, len(params) - 0.5, -0.5, len(params) - 0.5),
    )
    for (i, j), d in np.ndenumerate(np.flipud(dist)):
        ax.text(
            j,
            i,
            "{:0.2f}".format(d),
            ha="center",
            va="center",
            backgroundcolor=(1, 1, 1, 0.5),
        )
    cbar = fig.colorbar(im)
    cbar.set_label("Distance correlation")
    ax.set_title("Sensitivity vector distance matrix")
    ax.set_xticks([i for i in range(len(params))])
    ax.set_yticks([i for i in reversed(range(len(params)))])
    # ^ reversed b/c origin="upper"
    ax.set_xticklabels([params[i].rstrip(" [param]") for i in dn["leaves"]])
    ax.set_yticklabels([params[i].rstrip(" [param]") for i in dn["leaves"]])
    fig.tight_layout()
    fig.savefig(analysis.directory / f"sensitivity_vector_distance_matrix.svg")
