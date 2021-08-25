import json
import math
import sys
import traceback
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Sequence

# In-house packages
from lxml import etree

import febtools as feb

# Third-party packages
import pandas as pd
import psutil
import scipy.cluster
from febtools.febio import (
    FEBioError,
    CheckError,
    run_febio_checked,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from pathos.pools import ProcessPool

# Same-package modules
from . import colors
from . import febioxml as fx
from .febioxml import read_febio_xml as read_xml
from .variables import *

NUM_WORKERS = psutil.cpu_count(logical=False)

COV_ZERO_THRESH = 1e-15  # Threshold at which a covariance value is treated as zero

CMAP_DIVERGE = mpl.colors.LinearSegmentedColormap(
    "div_blue_black_red", colors.diverging_bky_60_10_c30_n256
)
FONTSIZE_FIGLABEL = 12
FONTSIZE_AXLABEL = 10
FONTSIZE_TICKLABEL = 8


class CaseGenerationError(Exception):
    """Raise when case generation terminates in an error."""

    pass


class Success:
    """Return value for successful execution"""

    def __str__(self):
        return "Success"

    def __eq__(self, other):
        return other.__class__ == self.__class__

    pass


SUCCESS = Success()


class Analysis:
    def __init__(
        self,
        model,
        parameters: dict,
        variables: dict,
        name,
        parentdir=None,
        checks: Sequence[Callable] = tuple(),
    ):
        """Return Analysis object

        name := Name of the analysis. The folder containing the analysis
        files will be created using the analysis name, so it is a
        required parameter.

        parentdir := The directory in which the analysis folder will be
        stored. Defaults to the current working directory.

        :param checks: Sequence of callables that take a febtools Model as their lone
        argument and should raise an exception if the check fails, or return None if
        the check succeeds.  This is meant for user-defined verification of
        simulation output.

        """
        self.model = model
        self.parameters: dict = parameters
        self.variables: dict = variables
        self.checks = checks
        self.name = name
        if parentdir is None:
            self.directory = Path(name).absolute() / self.name
        else:
            self.directory = Path(parentdir).absolute() / self.name

    @classmethod
    def from_xml(cls, pth):
        pth = Path(pth)
        tree = read_xml(pth)
        e_analysis = tree.find("preprocessor[@proc='spamneggs']/analysis")
        if e_analysis is None:
            raise ValueError(
                f"No XML element with path 'preprocessor/analysis' found in file '{pth}'."
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
            analysis_dir = pth.parent.absolute() / pth.stem
        # Parameters
        parameters, parameter_locations = fx.get_parameters(tree)
        # Variables
        variables = fx.get_variables(tree)
        # Model
        fx.strip_preprocessor_elems(tree, parameters)
        # ↑ mutates in-place
        xml = etree.tostring(tree)
        model = FEBioXMLModel(xml, parameter_locations)
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
        pvals = {p: d.levels[pset] for p, d in analysis.parameters.items()}
        return cls.from_values(analysis, pvals, name=pset.replace(" ", "_"))

    def write_model(self):
        """Write the case's model to disk"""
        self.sim_file.parent.mkdir(parents=True, exist_ok=True)
        # ^ exist_ok is needed to make calling this function threadsafe
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

    def __init__(self, xml: str, parameters: dict):
        """Return an FEBioXMLModel instance

        xml := the FEBio XML tree.

        parameters := a dictionary of parameter name → list of
        ElementPath paths to elements in the XML for which the element's
        text content is to be replaced by the parameter's value.

        """
        # Note that lxml objects cannot be serialized using pickle (they have no
        # support for serialization) so we need to store the XML as a string.  This
        # is also helpful because strings are immutable, making the data safe for use
        # in multiprocessing.
        self.xml = xml
        self.parameters = parameters

    def __call__(self, case: Case):
        # Rehydrate the XML
        tree = etree.fromstring(self.xml).getroottree()
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
        fx.insert_output_elem(
            tree, logfile_reqs, plotfile_reqs, file_stem=f"case={case.name}"
        )
        return tree


def _ensure_analysis_directory(analysis):
    # Check output directory
    if analysis.directory is None:
        raise ValueError(
            "Analysis.directory is None.  make_named_cases requires an output directory."
        )
    if not analysis.directory.exists():
        analysis.directory.mkdir(parents=True)


def _trap_err(fun):
    """Wrap case-processing function to catch and return any errors"""

    def wrapped(case):
        try:
            return fun(case)
        except Exception as err:
            print(f"Case {case.name}:")
            # For an error in Python code we want to print the traceback.  Special
            # simulation errors should be trapped elsewhere; only truly exceptional
            # exceptions should hit this trap.
            traceback.print_exc()
            print()
            return case, err

    return wrapped


def _validate_opt_on_case_error(value):
    # Validate input
    on_case_error_options = ("stop", "hold", "ignore")
    if not value in on_case_error_options:
        raise ValueError(
            f"on_case_error = {value}; allowed values are {','.join(on_case_error_options)}"
        )


def cases_table(cases):
    # TODO: Figure out same way to demarcate parameters from other
    # metadata so there are no reserved parameter names.  For example, a
    # user should be able to name their parameter "path" without
    # conflicts.
    data: Dict[str, Dict] = defaultdict(dict)
    for case in cases:
        for p, v in case.parameters.items():
            data[case.name][p] = v
        data[case.name]["status"] = None
        data[case.name]["path"] = case.sim_file
    tab = pd.DataFrame.from_dict(data, orient="index")
    return tab


def do_parallel(cases, fun, on_case_error="stop"):
    _validate_opt_on_case_error(on_case_error)  # TODO: Use enum
    status = {}
    pool = ProcessPool(nodes=NUM_WORKERS)
    results = pool.imap(_trap_err(fun), cases)
    for case, err in results:
        # Log the run outcome
        status[case.name] = err
        # Should we continue submitting cases?
        if isinstance(err, Exception) and on_case_error == "stop":
            print(
                f"While working on case {case.name}, a {err.__class__.__name__} was encountered.  Because `on_case_error` = {on_case_error}, do_parallel will allow any already-running cases to finish, then stop.\n"
            )
            pool.close()
            break
    return status


def gen_case(case):
    case.write_model()
    return case, SUCCESS


def run_case(case):
    """Run a case's simulations and perform all associated checks

    The case may be partially or completely solved.  case.analysis must be defined,
    as the checks are stored on the Analysis object.

    """
    # Having a separate check_case function was considered, but there's no use case
    # for separating the checks from the run.  Some basic checks are done by
    # `run_febio_checked` anyway and these cannot be turned off.
    problems = []
    try:
        proc = run_febio_checked(case.sim_file)
    except FEBioError as err:
        print(f"Case {case.name}: {err!r}")
        problems.append(err)
    except CheckError as err:
        print(f"Case {case.name}: {err!r}")
        problems.append(err)
    # For general exceptions, let the program blow up.  They can be trapped at a
    # higher level if desired.
    for f in case.analysis.checks:
        try:
            f(case)
        except CheckError as err:
            print(f"Case {case.name}: {err!r}")
            problems.append(err)
    if problems:
        status = problems
    else:
        status = SUCCESS
    return case, status


def run_sensitivity(analysis, nlevels, on_case_error="stop"):
    """Run a sensitivity analysis from an analysis object."""
    _validate_opt_on_case_error(on_case_error)
    _ensure_analysis_directory(analysis)
    # Set febtools to run FEBio with only 1 thread, since we'll be
    # running one FEBio process per core
    feb.febio.FEBIO_THREADS = 1

    def replace_status(tab, status, step=None):
        if step is not None:
            prefix = f"{step}: "
        else:
            prefix = ""
        for i, err in status.items():
            if isinstance(err, Exception):
                # Don't include the full exception message, which would make it
                # harder to filter the table
                s = err.__class__.__name__
            elif isinstance(err, Sequence):
                s = ", ".join(e.__class__.__name__ for e in err)
            else:
                s = str(err)
            tab.loc[i, "status"] = f"{prefix}{s}"

    def iter_generated_cases(cases, status: dict):
        for case in cases:
            if status[case.name] == SUCCESS:
                yield case

    # Create the named cases
    ncases = list_named_cases(analysis)
    tab_ncases = cases_table(ncases)
    pth_ncases = analysis.directory / f"named_cases.csv"
    write_cases_table(tab_ncases, pth_ncases)
    status_ncases = do_parallel(ncases, gen_case, on_case_error=on_case_error)
    replace_status(tab_ncases, status_ncases, step="Generate")
    write_cases_table(tab_ncases, pth_ncases)

    # Run the named cases
    status = do_parallel(
        iter_generated_cases(ncases, status_ncases),
        run_case,
        on_case_error=on_case_error,
    )
    replace_status(tab_ncases, status, step="Run")
    write_cases_table(tab_ncases, pth_ncases)

    # Create the sensitivity cases
    scases = list_sensitivity_cases(analysis, nlevels)
    # Write the table before generation so we can see the list of cases
    # even if generation takes a long time.
    tab_scases = cases_table(scases)
    pth_scases = analysis.directory / f"generated_cases.csv"
    write_cases_table(tab_scases, pth_scases)
    status_scases = do_parallel(scases, gen_case, on_case_error=on_case_error)
    replace_status(tab_scases, status_scases, step="Generate")
    write_cases_table(tab_scases, pth_scases)

    # Run the sensitivity cases
    status = do_parallel(
        iter_generated_cases(scases, status_scases),
        run_case,
        on_case_error=on_case_error,
    )
    replace_status(tab_scases, status, step="Run")
    write_cases_table(tab_scases, pth_scases)

    # Check if error terminations prevent analysis of results
    for nm, tab, pth in (
        ("named", tab_ncases, pth_ncases),
        ("sensitivity", tab_scases, pth_scases),
    ):
        m_error = tab["status"] != "Run: Success"
        if np.any(m_error):
            if on_case_error == "ignore":
                tab = tab[np.logical_not(m_error)]
            elif on_case_error == "hold":
                raise Exception(
                    f"Of {len(tab)} sensitivity cases, {np.sum(tab['status'] == 'Run: Success')} cases ran successfully and {np.sum(tab['status'] != 'Run: Success')} did not.  Because `on_case_error` = '{on_case_error}', the sensitivity analysis was stopped prior to data analysis.  The error terminations are listed in `{pth}`.  To continue, correct the error terminations and call `tabulate` and `plot_sensitivity` manually."
                )
            elif on_case_error == "stop":
                # An error message would have been printed earlier when a model
                # generation or run failed and the remaining generations / runs
                # were cancelled.  So we just exit.
                sys.exit()
    # Tabulate and plot the results
    tabulate(analysis)
    plot_sensitivity(analysis)


def list_named_cases(analysis):
    """List an analysis' named cases."""
    # Assemble each case's parameter values
    casedata = defaultdict(dict)
    for pname, param in analysis.parameters.items():
        for cname, value in param.levels.items():
            casedata[cname][pname] = value
    # Create a list of cases
    cases_dir = analysis.directory / "named_cases"
    cases = [
        Case(
            analysis,
            pvals,
            cname,
            sim_file=cases_dir / f"case={cname}.feb",
            case_dir=cases_dir / f"case={cname}",
        )
        for cname, pvals in casedata.items()
    ]
    return cases


def list_sensitivity_cases(analysis, nlevels):
    """List cases for sensitivity analysis."""
    # Generate parameter values for each case
    levels = {}
    for pname, param in analysis.parameters.items():
        dist = param.distribution
        # Calculate variable's levels
        if isinstance(dist, ContinuousScalar):
            levels[pname] = dist.sensitivity_levels(nlevels)
        elif isinstance(dist, CategoricalScalar):
            levels[pname] = dist.sensitivity_levels()
        else:
            raise ValueError(
                f"Generating levels from a variable of type `{type(dist)}` is not yet supported."
            )
    cases_dir = analysis.directory / "generated_cases"
    cases = [
        Case(
            analysis,
            dict(zip(analysis.parameters, pvals)),
            f"{i}",
            sim_file=cases_dir / f"case={i}.feb",
            case_dir=cases_dir / f"case={i}",
        )
        for i, pvals in enumerate(product(*(levels[p] for p in analysis.parameters)))
    ]
    return cases


def make_case_files(analysis, cases, on_case_error="stop"):
    """Write model files for each case and return a table of cases.

    Note that each case's model generation function is allowed to do
    arbitrary computation, so generating the files write for each case
    may do work such as running helper simulations.

    """
    # Generate the case's model files
    case_status = {case.name: None for case in cases}
    for case in cases:
        try:
            case.write_model()
        except Exception as err:
            # If the case generation fails, log the error and deal with it
            # according to the on_case_error argument.
            if on_case_error == "stop":
                raise CaseGenerationError(
                    f"Case generation failed for case ID = '{case.name}' in "
                    "directory '{case.sim_file.parent}'.  Because `on_case_error` "
                    "= {on_case_error}, spamneggs will now exit."
                )
            elif on_case_error in ("hold", "ignore"):
                case_status[case.name] = "generation error"
        else:
            case_status[case.name] = "generation complete"
    return case_status


def read_case_data(model_path):
    """Read variables from a single case analysis."""
    case_file = Path(model_path)
    output_dir = model_path.parent / ".." / "case_output"
    pth_record = output_dir / f"{model_path.stem}_vars.json"
    pth_timeseries = output_dir / f"{model_path.stem}_timeseries_vars.csv"
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


def tabulate_analysis_tsvars(analysis, cases):
    """Tabulate time series variables for all cases in an analysis

    The time series tables for the individual cases must have already
    been written to disk.

    :analysis: Analysis object.

    :cases: DataFrame of cases to tabulate, usually obtained by reading
    generated_cases.csv or named_cases.csv from the analysis directory.

    """
    # TODO: It would be beneficial to build up the whole-analysis time
    # series table at the same time as case time series tables are
    # written to disk, instead of re-reading everything from disk.
    analysis_data = pd.DataFrame()
    for i in cases.index:
        pth_tsvars = analysis.directory / "case_output" / f"case={i}_timeseries_vars.csv"
        tsvars = pd.read_csv(pth_tsvars)
        # Check for missing variables in the on-disk data
        available_vars = set(tsvars.columns)
        missing_vars = set(analysis.variables) - available_vars
        if missing_vars:
            raise ValueError(
                f"Did not find variable(s) {', '.join(missing_vars)} in file: {pth_tsvars}"
            )
        # The presence of extra variables in the data on disk should not be cause for
        # alarm.  One batch of simulations can support multiple analysis, and a given
        # analysis may not use all of the available variables.  But we do not want to
        # collect variables that the analysis did not ask for.
        case_data = {"Case": i, "Step": tsvars["Step"], "Time": tsvars["Time"]}
        case_data.update({f"{p} [param]": cases[p].loc[i] for p in analysis.parameters})
        case_data.update({f"{v} [var]": tsvars[v] for v in analysis.variables})
        case_data = pd.DataFrame(case_data)
        analysis_data = pd.concat([analysis_data, case_data])
    return analysis_data


def tabulate(analysis: Analysis):
    """Tabulate output from an analysis."""
    for nm, fbase in (
        ("named", "named"),
        ("sensitivity", "generated"),
    ):
        ivars_table = defaultdict(list)  # we don't know param names in advance
        ivars_table["case"] = []  # force creation of the index column
        pth_cases = analysis.directory / f"{fbase}_cases.csv"
        cases = pd.read_csv(pth_cases, index_col=0)
        if len(cases) == 0:
            raise ValueError(f"No cases to tabulate in '{pth_cases}'")
        for i in cases.index:
            casename = Path(cases["path"].loc[i]).stem
            status = cases.loc[i, "status"]
            if not status == "Run: Success":
                # Total simulation failure.  There is nothing to
                # tabulate; don't bother trying.
                continue
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
        df_ivars.to_csv(
            analysis.directory / f"{fbase}_cases_-_inst_vars.csv", index=True
        )


def plot_sensitivity(analysis):
    # Named cases
    pth_cases = analysis.directory / f"named_cases.csv"
    named_cases = pd.read_csv(pth_cases, index_col=0)
    # Named cases are manually constructed and so should converge.  If
    # they do not, it's probably a user problem.  But you still need to
    # see what happened when there's an error, so the plotting function
    # should press on.
    named_cases = named_cases[named_cases["status"] == "Run: Success"]

    # Sensitivity cases
    pth_cases = analysis.directory / f"generated_cases.csv"
    cases = pd.read_csv(pth_cases, index_col=0)
    cases = cases[cases["status"] == "Run: Success"]

    # Read parameters
    #
    # TODO: Add a function to get the list of parameters and variables
    # for an analysis up-front.  Extract the relevant code from
    # tabulate_case.
    param_names = [k for k in cases.columns if k not in ["case", "path", "status"]]
    param_values = {k: cases[k] for k in param_names}

    # Get variable names
    ivar_names = [
        nm
        for nm, var in analysis.variables.items()
        if var.temporality == "instantaneous"
    ]
    tsvar_names = [
        nm for nm, var in analysis.variables.items() if var.temporality == "time series"
    ]

    # Plots for instantaneous variables
    ivar_values = defaultdict(list)
    if len(ivar_names) > 0:
        # Skip doing the work of reading each case's data if there are
        # no instantaneous variables to tabulate
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
        make_sensitivity_ivar_figures(
            analysis, param_names, param_values, ivar_names, ivar_values
        )

    # Plots for time series variables
    if len(tsvar_names) > 0:
        tsdata = tabulate_analysis_tsvars(analysis, cases)
        make_sensitivity_tsvar_figures(
            analysis, param_names, param_values, tsvar_names, tsdata, cases, named_cases
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
    analysis, param_names, param_values, tsvar_names, tsdata, cases, named_cases=None
):
    """Plot sensitivity of each time series variable to each parameter"""
    plot_tsvars_line(
        analysis, param_names, param_values, tsvar_names, cases, named_cases
    )
    # TODO: The heat map figure should probably indicate which case is
    # plotting as the time series guide.
    if "nominal" in named_cases.index:
        # Plot nominal case
        pth = analysis.directory / named_cases.loc["nominal", "path"]
        record, ref_ts = read_case_data(pth)
        ref_ts.columns = [
            f"{s} [var]" if not s in ("Time", "Step") else s for s in ref_ts.columns
        ]
    else:
        # Plot median generated case
        median_levels = {
            k: values[len(values) // 2] for k, values in param_values.items()
        }
        m = np.ones(len(cases), dtype="bool")
        for param, med in median_levels.items():
            m = np.logical_and(m, cases[param] == med)
        assert np.sum(m) == 1
        case_id = cases.index[m][0]
        ref_ts = tsdata[tsdata["Case"] == case_id]
    plot_tsvars_heat_map(analysis, tsdata, ref_ts, norm="none")
    plot_tsvars_heat_map(analysis, tsdata, ref_ts, norm="all")
    plot_tsvars_heat_map(analysis, tsdata, ref_ts, norm="vector")
    plot_tsvars_heat_map(analysis, tsdata, ref_ts, norm="subvector")


class NDArrayJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.floating):
            return float(o)
        else:
            return super().default(o)


def write_cases_table(tab, pth):
    """Write a cases table to disk

    File paths are written relative to the target directory.

    """
    tab = tab.copy()  # table might be re-used elsewhere
    tab["path"] = [str(p.relative_to(pth.parent)) for p in tab["path"]]
    tab.to_csv(pth, index_label="ID")


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
    nm_xaxis = "Time"
    varnames = [nm for nm in timeseries.columns if not nm in ("Step", nm_xaxis)]
    # Produce one large plot with all variables
    fig = Figure()
    FigureCanvas(fig)
    axarr = fig.subplots(len(varnames), 1, sharex=True, squeeze=False)
    for varname, ax in zip(varnames, axarr[:, 0]):
        ax.plot(timeseries[nm_xaxis], timeseries[varname], marker=".")
        ax.set_ylabel(varname)
    fig.set_size_inches((7, 1.0 + 1.25 * len(varnames)))
    if casename is not None:
        axarr[0, 0].set_title(casename)
    # Format axes
    axarr[-1, 0].set_xlabel(f"{nm_xaxis} [s]")  # assumed unit
    formatter = mpl.ticker.ScalarFormatter()
    formatter.set_powerlimits((-3, 4))
    axarr[-1, 0].xaxis.set_major_formatter(formatter)
    # Write figure to disk
    fig.tight_layout()
    fig.savefig(dir_out / f"{stem}timeseries_vars.svg")
    plt.close(fig)
    # Produce one small plot for each variable
    for varname in varnames:
        fig = plot_case_tsvar(timeseries, varname, casename)
        fig.savefig(dir_out / f"{stem}timeseries_var={varname}.svg")


def plot_tsvars_heat_map(analysis, tsdata, ref_ts, norm="none", corr_threshold=1e-6):
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
    data = tsdata.copy()  # need to mutate the table
    data = data.drop("Time", axis=1).set_index(["Step", "Case"])

    # Calculate sensitivity vectors
    n = len(data.index.levels[0])
    sensitivity_vectors = np.zeros((len(varnames), n, len(params)))
    for i in range(n):
        values = data.loc[i][params + varnames]
        if np.any(np.isnan(values)):
            raise ValueError(
                "NaNs detected in output variables' values.  Aborting "
                "plots of sensitivity heat maps because the distance "
                "vector calculations will propagate the NaNs."
            )
        corr = values.corr()
        cov = values.cov()
        for pi, pnm in enumerate(params):
            for vi, vnm in enumerate(varnames):
                ρ = corr[pnm][vnm]
                # Coerce correlation to zero if it is nan only because
                # the output variable has practically no variance
                if np.isnan(ρ) and (
                    cov[vnm][pnm] <= COV_ZERO_THRESH
                    and cov[vnm][vnm] <= COV_ZERO_THRESH
                ):
                    ρ = 0
                sensitivity_vectors[vi][i][pi] = ρ

    # Plot the heatmap
    lh = 1.5  # multiple of font size to use for label height

    # Calculate widths of figure elements.  TODO: It would be better to
    # specify the /figure/ width, then calculate the necessary axes
    # widths.
    dendro_axw = 12 / 72 * len(params)
    fig_llabelw = lh * FONTSIZE_FIGLABEL / 72
    dendro_areaw = fig_llabelw + dendro_axw
    cbar_axw = 12 / 72
    # ^ width of colorbar axes
    cbar_rpad = 36 / 72
    # ^ padding b/w colorbar axes and axes of /next/ heat map
    cbar_lpad = 4 / 72
    # ^ padding b/w heat map axes and its colorbar axes
    cbar_areaw = cbar_lpad + cbar_axw + cbar_rpad
    hmap_lpad = lh * FONTSIZE_AXLABEL / 72
    hmap_axw = 4
    # ^ width of individual heat map axes object if individual color
    # scales used.  Otherwise heatmap expands to fill space.
    hmap_subw = hmap_lpad + hmap_axw + cbar_lpad + cbar_axw + cbar_rpad
    hmap_subwspace = FONTSIZE_AXLABEL / 72
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
    fig_tlabelh = lh * FONTSIZE_FIGLABEL / 72  # "Time series variables"
    hmap_tlabelh = lh * FONTSIZE_AXLABEL / 72  # Variable names
    hmap_blabelh = lh * FONTSIZE_AXLABEL / 72  # "Step"
    hmap_axh = 0.75
    # ^ height allocated for each axes
    hmap_vspace = (lh + 0.5) * FONTSIZE_TICKLABEL / 72
    # ^ height allocated for x-axis tick labels, per axes
    nh = len(params) + 1
    # ^ number of axes high; the +1 is for a time series line plot
    figh = hmap_blabelh + (hmap_vspace + hmap_axh) * nh + hmap_tlabelh + fig_tlabelh

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
    #
    # Compute unsigned Pearson correlation distance = 1 - | ρ | where
    # ρ = (u − u̅) * (v - v̅) / ( 2-norm(u − u̅) * 2-norm(v - v̅) ).
    # If the 2-norm of (u − u̅) or (v − v̅) is zero, then the result will be
    # undefined.  Typically this will happen when u or v is all zeroes; in this
    # case, the numerator is also zero.  For this application, it is reasonable
    # to define 0/0 = 0.  Therefore we need to check for (u − u̅) * (v - v̅) = 0
    # and set the correlation distance for those vector pairs to 1.
    dist = (1 - abs(scipy.spatial.distance.pdist(arr, metric="correlation") - 1)) ** 0.5
    n = len(arr)  # number of variables
    numerator = np.empty(len(dist))
    numerator[:] = np.nan  # make indexing errors more obvious
    means = np.mean(arr, axis=1)
    for i in range(n):  # index of u
        for j in range(i + 1, n):  # index of v
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
    dn_ax.set_ylabel("Parameters", fontsize=FONTSIZE_FIGLABEL)
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

    # Create common axis elements
    tick_locator = mpl.ticker.MaxNLocator(integer=True)

    # Draw the time series line plot in the first row
    for ivar, varname in enumerate(varnames):
        l = hmap_areal + ivar * (hmap_subw + hmap_subwspace) + hmap_lpad
        w = hmap_axw
        b = hmapaxes_b0 + len(params) * (hmap_vspace + hmap_axh)
        h = hmap_axh
        ax = fig.add_axes((l / figw, b / figh, w / figw, h / figh), facecolor="#F2F2F2")
        ax.plot(ref_ts["Step"], ref_ts[varname], color="k")
        # Set the axes limits to the min and max of the data, to match
        # the axes limits used for the heatmap images.
        ax.set_xlim(min(ref_ts["Step"]), max(ref_ts["Step"]))
        ax.set_title(
            varname.rpartition(" [var]")[0],
            fontsize=FONTSIZE_FIGLABEL,
            loc="left",
            pad=3.0,
        )
        for spine in ("left", "right", "top", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(tick_locator)
        ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)

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
            ax.xaxis.set_major_locator(tick_locator)
            ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
            ax.set_ylabel(params[iparam].rstrip(" [param]"), fontsize=FONTSIZE_AXLABEL)
            ax.tick_params(axis="y", left=False, labelleft=False)
            if irow == 0:
                ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_TICKLABEL)

            # Draw the heatmap's colorbar
            if norm == "subvector":
                l = l + hmap_axw + cbar_lpad
                cbar_ax = fig.add_axes(
                    (l / figw, b / figh, cbar_axw / figw, hmap_axh / figh)
                )
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
                cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
                cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
                cbar.set_label("ρ [1]", fontsize=FONTSIZE_TICKLABEL)
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
                cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
                cbar.set_label("ρ [1]", fontsize=FONTSIZE_TICKLABEL)

    # Add the whole-plot right-most colorbar, if called for
    if norm in ("none", "all"):
        l = dendro_areaw + hmap_areaw + rcbar_wspace
        b = hmapaxes_b0
        w = cbar_axw
        h = hmapaxes_t0 - hmapaxes_b0
        ax = fig.add_axes((l / figw, b / figh, w / figw, h / figh))
        cbar = fig.colorbar(im, cax=ax)  # `im` from last imshow
        clocator = mpl.ticker.LinearLocator()
        cbar.ax.yaxis.set_major_locator(clocator)
        # Touch up the center tick; it can get a value like -1e-16,
        # which results in a label like "−0.00".  We don't want the
        # minus sign in front of zero; it looks weird.
        ticks = cbar.get_ticks()
        if len(ticks) % 2 == 1:
            ticks[len(ticks) // 2] = 0
            cbar.set_ticks(ticks)
        if absmax >= 0.1:
            fmt = mpl.ticker.StrMethodFormatter("{x:.2f}")
        else:
            fmt = mpl.ticker.ScalarFormatter(useOffset=True)
        cbar.ax.yaxis.set_major_formatter(fmt)
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
        cbar.set_label("ρ [1]", fontsize=FONTSIZE_AXLABEL)

    # Add whole-plot labels
    fig.suptitle(
        f"Time series variable correlations, norm = {norm}",
        y=1.0,
        va="top",
        fontsize=FONTSIZE_FIGLABEL,
    )
    # Write figure to disk
    fig.savefig(analysis.directory / f"sensitivity_vector_heatmap_norm={norm}.svg")

    # Plot the distance matrix.  Reorder the parameters to match the
    # sensitivity vector plot.
    dist = scipy.spatial.distance.squareform(dist)[dn["leaves"], :][:, dn["leaves"]]
    fig = Figure()
    FigureCanvas(fig)
    # Set size to match number of variables
    in_per_var = 0.8
    pad_all = FONTSIZE_TICKLABEL / 2 / 72
    mat_w = in_per_var * len(dist)
    mat_h = mat_w
    cbar_lpad = 12 / 72
    cbar_w = 0.3
    cbar_h = mat_h
    cbar_rpad = (24 + FONTSIZE_AXLABEL) / 72
    fig_w = pad_all + mat_w + cbar_lpad + cbar_w + cbar_rpad + pad_all
    fig_h = (
        pad_all + FONTSIZE_FIGLABEL / 72 + FONTSIZE_AXLABEL / 72 + 0.2 + mat_h + pad_all
    )
    fig.set_size_inches(fig_w, fig_h)
    # Plot the matrix itself
    pos_main_in = np.array((pad_all, pad_all, mat_w, mat_h))
    ax = fig.add_axes(pos_main_in / [fig_w, fig_h, fig_w, fig_h])
    cmap = mpl.cm.get_cmap("cividis")
    cnorm = mpl.colors.Normalize(vmin=0, vmax=np.max(np.abs(dist)))
    im = ax.matshow(
        dist,
        cmap=cmap,
        norm=cnorm,
        origin="upper",
        extent=(-0.5, len(params) - 0.5, -0.5, len(params) - 0.5),
    )
    # Write the value of each cell as text
    for (i, j), d in np.ndenumerate(np.flipud(dist)):
        ax.text(
            j,
            i,
            "{:0.2f}".format(d),
            ha="center",
            va="center",
            backgroundcolor=(1, 1, 1, 0.5),
            fontsize=FONTSIZE_TICKLABEL,
        )
    pos_cbar_in = np.array(
        (
            pad_all + mat_w + cbar_lpad,
            pad_all,
            cbar_w,
            cbar_h,
        )
    )
    cax = fig.add_axes(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Distance correlation", fontsize=FONTSIZE_AXLABEL)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
    ax.set_title("Sensitivity vector distance matrix", fontsize=FONTSIZE_FIGLABEL)
    ax.set_xticks(
        [i for i in range(len(params))],
    )
    ax.set_yticks([i for i in reversed(range(len(params)))])
    # ^ reversed b/c origin="upper"
    ax.set_xticklabels([params[i].rstrip(" [param]") for i in dn["leaves"]])
    ax.set_yticklabels([params[i].rstrip(" [param]") for i in dn["leaves"]])
    ax.tick_params(axis="x", labelsize=FONTSIZE_AXLABEL, bottom=False)
    ax.tick_params(axis="y", labelsize=FONTSIZE_AXLABEL)
    #
    # Resize figure to accomodate left axis tick labels
    ## Calculate left overflow
    bbox_px = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox_in = fig.dpi_scale_trans.inverted().transform(bbox_px)
    Δw_in = -bbox_in[0][0]
    ## Resize the canvas
    fig_w = fig_w + Δw_in
    fig.set_size_inches(fig_w, fig_h)
    ## Re-apply the axes sizes, which will have changed because they are
    ## stored in figure units
    pos_main_in[0] += Δw_in
    pos_cbar_in[0] += Δw_in
    ax.set_position(pos_main_in / [fig_w, fig_h, fig_w, fig_h])
    cax.set_position(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    fig.savefig(analysis.directory / f"sensitivity_vector_distance_matrix.svg", dpi=300)


def plot_tsvars_line(
    analysis, param_names, param_values, tsvar_names, cases, named_cases=None
):
    """Plot time series variables as lines across cases

    For each combination of time series variable + parameter ("subject
    parameter"), use a line plot to show variation of the time series
    variable between levels of the subject parameter.  The lines for the
    levels of the subject parameter are overplotted and differentiated
    by color.  For each i in 1 … # levels, produce one line plot, with
    the non-subject parameters held at their i'th level.

    Additionally, produce a separate plot showing the subject time
    series variable for each named case.

    """
    # TODO: Figure out how to plot parameter sensitivity for multiple
    # variation (parameter interactions)
    levels = {p: sorted(np.unique(param_values[p])) for p in param_names}
    for subject_param, subject_values in param_values.items():
        other_params = [p for p in param_names if p != subject_param]
        n_plots = len(levels[subject_param])
        if named_cases is not None:
            n_plots += 1
        cnorm = mpl.colors.Normalize(
            vmin=min(levels[subject_param]), vmax=max(levels[subject_param])
        )
        # Make a plot for each output variable
        for varname in tsvar_names:
            fig = Figure(constrained_layout=True)
            nh = math.floor(n_plots ** 0.5)
            nw = math.ceil(n_plots / nh)
            fig.set_size_inches((5 * nw + 1, 3 * nh + 0.25))  # TODO: set smart size
            fig.set_constrained_layout_pads(
                wspace=0.04, hspace=0.04, w_pad=2 / 72, h_pad=2 / 72
            )
            gs = GridSpec(nh, nw, figure=fig)
            axs = []
            # Plot the named cases
            if named_cases is not None:
                ax = fig.add_subplot(gs[0, 0])
                axs.append(ax)
                ax.set_title(f"Named cases", fontsize=FONTSIZE_AXLABEL)
                ax.set_ylabel(varname, fontsize=FONTSIZE_AXLABEL)
                ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_AXLABEL)
                ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
                ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
                for i, case_id in enumerate(named_cases.index):
                    record, tab_timeseries = read_case_data(
                        analysis.directory / named_cases.loc[case_id, "path"]
                    )
                    ax.plot(
                        tab_timeseries["Step"],
                        tab_timeseries[varname],
                        label=case_id,
                        color=colors.categorical_n7[i % len(colors.categorical_n7)],
                    )
                ax.legend()
            # Plot the sensitivity levels
            #
            # For i = 1 … # levels, plot the variation in each output
            # variable vs. the subject parameter, holding all other
            # parameters at their i'th level.  Call that latter set of
            # parameter values the "fulcrum".
            cbars = []
            for i in range(len(levels[subject_param])):
                fulcrum = {p: levels[p][i] for p in other_params}
                # Select the cases that belong to the fulcrum
                m = np.ones(len(cases), dtype="bool")  # init
                for p in other_params:
                    m = np.logical_and(m, cases[p] == fulcrum[p])
                # Make the plot panel
                ax = fig.add_subplot(gs[(i + 1) // nw, (i + 1) % nw])
                axs.append(ax)
                ax.set_title(
                    f"Sensitivity levels' index = {i+1}", fontsize=FONTSIZE_AXLABEL
                )
                ax.set_ylabel(varname, fontsize=FONTSIZE_AXLABEL)
                ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_AXLABEL)
                # Plot a line for each sensitivity level of the subject parameter
                for case_id in cases.index[m]:
                    record, tab_timeseries = read_case_data(
                        analysis.directory / cases.loc[case_id, "path"]
                    )
                    ax.plot(
                        tab_timeseries["Step"],
                        tab_timeseries[varname],
                        color=CMAP_DIVERGE(cnorm(cases.loc[case_id, subject_param])),
                    )
                    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
                    ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
                cbar = fig.colorbar(
                    ScalarMappable(norm=cnorm, cmap=CMAP_DIVERGE),
                    ax=ax,
                )
                cbars.append(cbar)
                cbar.set_label(subject_param, fontsize=FONTSIZE_AXLABEL)
            # Link the y axes if each has a similar range (within an
            # order of magnitude) as the others
            ranges = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axs]
            if max(ranges) / min(ranges) < 10:
                # Link the y-axis across axes
                for ax in axs[1:]:
                    axs[0].get_shared_y_axes().join(axs[0], ax)
            fig.suptitle(
                f"{varname} time series vs. {subject_param}", fontsize=FONTSIZE_FIGLABEL
            )
            # Make colorbar width = 12 pt.  The horizontal spacing
            # between subplots will no longer be consistent, but this
            # looks better than fat or skinny colorbars.  It does have
            # the downside of slightly (a few px) breaking alignment
            # between the colorbar and its associated subplot.
            fig.canvas.draw()
            for cbar in cbars:
                old_bbox = cbar.ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                new_bbox = Bbox.from_bounds(
                    old_bbox.x0, old_bbox.y0, 12 / 72, old_bbox.height
                )
                cbar.ax.set_position(
                    new_bbox.transformed(fig.dpi_scale_trans).transformed(
                        fig.transFigure.inverted()
                    )
                )
            nm = f"timeseries_var_lineplot_-_{varname}_vs_{subject_param}.svg"
            fig.savefig(analysis.directory / nm.replace(" ", "_"))
