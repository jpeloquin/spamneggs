from functools import cached_property, partial
import json
import math
import sys
import traceback
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from numbers import Number
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

# Third-party packages
import scipy.cluster
from lxml import etree
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
import pandas as pd
from pandas import CategoricalDtype, DataFrame
from pathos.pools import ProcessPool
from pint import Quantity
import psutil
from sklearn.neighbors import KernelDensity

# In-house packages
import waffleiron as wfl
from waffleiron.febio import (
    FEBioError,
    CheckError,
    run_febio_checked,
)
from waffleiron.input import read_febio_xml as read_xml, textdata_table
from waffleiron.select import find_closest_timestep

# Same-package modules
from . import colors
from .core import Parameter
from . import febioxml as fx
from .febioxml import (
    FunctionVar,
    TAG_FOR_ENTITY_TYPE,
    TextDataSelector,
    TimeSeries,
    XpltDataSelector,
)
from .plot import (
    CMAP_DIVERGE,
    COLOR_DEEMPH,
    FONTSIZE_FIGLABEL,
    FONTSIZE_AXLABEL,
    FONTSIZE_TICKLABEL,
    LABELH_MULT,
    fig_blank_tsvars_by_parameter,
    fig_stacked_line,
    plot_matrix,
    remove_spines,
    plot_reference_tsdata,
)
from .stats import corr_partial, corr_pearson, corr_spearman
from .variables import *


# TODO: This can only be modified through spam.spamneggs.NUM_WORKERS, which is
# nonintuitive
NUM_WORKERS = psutil.cpu_count(logical=False)


class CaseGenerationError(Exception):
    """Raise when case generation terminates in an error."""

    pass


class UnconstrainedTimesError(Exception):
    """Raise when simulation is likely to return values at random times"""

    pass


class Success:
    """Return value for successful execution

    This class exists for consistency with all the WhateverError classes when
    reporting case generation or run status.

    """

    def __str__(self):
        return "Success"

    def __repr__(self):
        return "Success"

    def __eq__(self, other):
        return other.__class__ == self.__class__

    pass


SUCCESS = Success()


class Analysis:
    """Class storing analysis details for use after case generation"""

    # TODO: Write JSON or XML serialization
    def __init__(self, generator, mkdir=True):
        self.name = generator.name
        # Directory containing generated_cases, case_output, etc.
        self.base_directory = generator.directory
        # Directory in which to write files from this analysis
        self.directory = generator.directory
        if mkdir:
            self.directory.mkdir(exist_ok=True)
        self.parameters = generator.parameters
        self.variables = generator.variables

    @cached_property
    def table_named_cases(self):
        pth_cases = self.base_directory / f"named_cases.csv"
        named_cases = pd.read_csv(pth_cases, index_col=0)
        return named_cases

    @cached_property
    def table_auto_cases(self):
        pth_cases = self.base_directory / f"generated_cases.csv"
        cases = pd.read_csv(pth_cases, index_col=0)
        return cases

    def case_data(self, case_id):
        """Read variables from a single case analysis

        Named cases have string IDs; automatically generated cases have integer IDs.

        """
        if isinstance(case_id, str):
            row = self.table_named_cases.loc[case_id]
        elif isinstance(case_id, int):
            row = self.table_auto_cases.loc[case_id]
        else:
            raise TypeError(f"Case ID must be string or integer.  Was {type(case_id)}.")
        # TODO: Unify tables for automatic and named cases
        if row["status"] != "Run: Success":
            raise ValueError(
                f"No data available for case {case_id} with status {row['status']}."
            )
        pth_record = self.base_directory / "case_output" / f"{case_id}_vars.json"
        pth_timeseries = (
            self.base_directory / "case_output" / f"{case_id}_timeseries_vars.csv"
        )
        with open(pth_record, "r") as f:
            record = json.load(f)
        timeseries = pd.read_csv(pth_timeseries, index_col=False)
        return record, timeseries

    def cases_tsdata(self, cases=None):
        """Return table of time series data for multiple succussfully-run cases

        :param cases: Optional DataFrame of cases.  If `cases` is provided,
        `cases_tsdata` will return time series data for the cases tabulated therein.  By
        default, `cases_tsdata` will return time series data for all cases in
        `self.table_auto_cases`.

        The time series tables for the individual cases must have already been written
        to disk.

        """
        if cases is None:
            cases = self.table_auto_cases
            cases = cases[cases["status"] == f"Run: {SUCCESS}"]
        # TODO: It would be beneficial to build up the whole-analysis time
        # series table at the same time as case time series tables are
        # written to disk, instead of re-reading everything from disk.
        tsdata = DataFrame()
        for i in cases.index:
            # It's tempting to store NaN values for cases with incomplete simulations,
            # but then we'd need to choose the time points at which to store those NaN
            # values, which is somewhat complicated.
            pth_tsvars = (
                self.base_directory / "case_output" / f"{i}_timeseries_vars.csv"
            )
            tsvars = pd.read_csv(pth_tsvars)
            # Check for missing variables in the on-disk data
            available_vars = set(tsvars.columns)
            missing_vars = set(self.variables) - available_vars
            if missing_vars:
                raise ValueError(
                    f"Did not find variable(s) {', '.join(missing_vars)} in file: {pth_tsvars}"
                )
            # Extra variables in the data on disk is ok, as one batch of simulations can
            # support multiple analyses using different variables.  But there is no need
            # to collect variables that this analysis did not ask for.
            case_data = {"Case": i, "Step": tsvars["Step"], "Time": tsvars["Time"]}
            case_data.update(
                {f"{p.name} [param]": cases.loc[i, p.name] for p in self.parameters}
            )
            case_data.update({f"{v} [var]": tsvars[v] for v in self.variables})
            case_data = DataFrame(case_data)
            tsdata = pd.concat([tsdata, case_data])
        return tsdata

    def update_table_auto_cases(self, table):
        """Write modified cases table to analysis directory"""
        pth_cases = self.base_directory / f"generated_cases.csv"
        table.reset_index().to_csv(pth_cases, index=False)


class Case:
    def __init__(
        self,
        name: str,
        parameters: Sequence[Parameter],
        parameter_values: Union[
            Mapping[str, Union[Quantity, Number]], Sequence[Union[Quantity, Number]]
        ],
        variables: dict[str, Union[XpltDataSelector, TextDataSelector, FunctionVar]],
        directory: Union[str, Path],
        checks: Iterable[Callable] = tuple(),
    ):
        """Return Case object

        :param name:  Name of the case.  The simulation file should be "{name}.feb",
        and any dependency simulations should be in a companion directory with this
        name.

        :param parameter_values: Parameter values corresponding to this case.  If the
        values are Quantity objects, they will be checked for compatibility with the
        units given in `parameters`.  If the values are plain numbers, they will be assumed to
        have the units given in `parameters`.

        :param variables: Variables to read from this case's solution, provided as
        spamneggs objects.

        :param checks: Check functions.

        :param directory: Parent directory to which the case's files will be written.

        """
        self.parameter_list = tuple(parameters)  # need to store parameter *order*
        # Values are most frequently needed, so they get the shortest name.
        if isinstance(parameter_values, Mapping):
            values = {n: v for n, v in parameter_values.items()}
        else:
            values = {p.name: v for p, v in zip(self.parameter_list, parameter_values)}
        self.parameters_q: Dict[str, Quantity] = cleanup_levels_units(
            values, self.parameter_list
        )
        self.parameters_n: Dict[str, Number] = {
            p.name: self.parameters_q[p.name].to(p.units).m for p in self.parameter_list
        }
        # TODO: Make solution variables available through self.variables
        self.variables_list = variables
        self.checks = checks
        self.name = name
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {self.directory}")

    def __repr__(self):
        return (
            f"Case(name={self.name},"
            f"parameters={self.parameter_list}"
            f"parameter_values={self.parameters_q},"
            f"variables={self.variables_list},"
            f"directory={self.directory},"
            f"checks={self.checks})"
        )

    def __str__(self):
        return f"Case {self.name}"

    @property
    def feb_file(self):
        return self.directory / f"case={self.name}.feb"

    @property
    def log_file(self):
        return self.directory / f"case={self.name}.log"

    @property
    def xplt_file(self):
        return self.directory / f"case={self.name}.xplt"

    def run(self, raise_on_check_fail=True):
        """Run the case's simulation and its checks

        The simulation will not necessarily run to completion, or at all.  Spamneggs
        does not fix invalid or defective simulations.

        """
        # Verify that the simulation file can be loaded at all.  If it cannot,
        # we won't be able to extract data from it later anyway.
        model = wfl.load_model(self.feb_file, read_xplt=False)
        # Verify that simulation file uses must points.  If it does not, the return
        # values from the various cases will not be at the same times, and any
        # sensitivity analysis will be invalid.
        # TODO: This check is specific to batch analysis, so consider moving it
        #  elsewhere.  A user might want to run one-offs without must points.
        if not all(wfl.febio.uses_must_points(model)):
            raise UnconstrainedTimesError(
                f"{self.feb_file} does not use so-called 'must points' in all steps.  To support batch analysis, values must be calculated and stored at the same times in all cases.  FEBio will not do this unless it is forced to through the use of must points."
            )
        # Having a separate check_case function was considered, but there's no use
        # case for separating the checks from the run.  Some basic checks are done by
        # `run_febio_checked` regardless and these cannot be turned off.
        run_febio_checked(self.feb_file)
        check_failures = []
        # TODO: When online tabulation available, make variable values available to checks
        for f in self.checks:
            try:
                f(self)
            except CheckError as err:
                check_failures.append(err)
        # For general exceptions, let the program blow up.  They can be trapped at a
        # higher level if desired.
        if check_failures:
            if raise_on_check_fail:
                raise CheckError(
                    f"Case {self.name} has failed checks: {', '.join([str(e) for e in check_failures])}"
                )
            return check_failures
        return SUCCESS

    def solution(self):
        """Return waffleiron Model object with solution"""
        model = wfl.load_model(self.feb_file)
        if model.solution is None:
            raise ValueError(f"Case {self.name}: Model has no solution")
        return model


class CaseGenerator:
    def __init__(
        self,
        model,
        parameters: Sequence[Union[Parameter, Tuple[str, Union[str, None]]]],
        variables: dict,
        name,
        parentdir=None,
        checks: Sequence[Callable] = tuple(),
    ):
        """Return a CaseGenerator object

        :param name: Name of the case generator.  Used as the name of the directory that
        will contain all generated files.

        :param parentdir: The parent directory for the generator directory.  Defaults to the
        current working directory.

        :param checks: Sequence of callables that take a waffleiron Model as their lone
        argument and should raise an exception if the check fails, or return None if the
        check succeeds.  This is meant for user-defined verification of simulation
        output.

        """
        self.model = model
        self.parameters: Sequence[Parameter] = [
            p if isinstance(p, Parameter) else Parameter(*p) for p in parameters
        ]
        self.variables: dict = variables
        self.checks = checks
        self.name = name
        if parentdir is None:
            self.directory = Path(name).absolute() / self.name
        else:
            self.directory = Path(parentdir).absolute() / self.name
        if not self.directory.exists():
            self.directory.mkdir()

    def generate_case(self, name, parameter_values, sub_dir=None) -> Case:
        """Return a Case and create its backing files"""
        if sub_dir is None:
            directory = self.directory
        else:
            directory = self.directory / sub_dir
            # self.directory should already exist
            if not directory.exists():
                directory.mkdir()
        case = Case(
            name,
            self.parameters,
            parameter_values,
            self.variables,
            directory,
            checks=self.checks,
        )
        tree = self.model(case)
        with open(case.feb_file, "wb") as f:
            wfl.output.write_xml(tree, f)
        return case

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
                e_parameter.text = str(case.parameters_n[pname])
        # Add the needed elements in <Output> to support the requested
        # variables.  We also have to update the file name attribute to match
        # this case.
        logfile_reqs, plotfile_reqs = fx.required_outputs(case.variables_list)
        fx.insert_output_elem(
            tree, logfile_reqs, plotfile_reqs, file_stem=f"case={case.name}"
        )
        return tree


def _trap_err(fun):
    """Wrap case-processing function to catch and return any errors"""

    def wrapped(args):
        name, case = args
        try:
            return fun(args)
        except (FEBioError, CheckError) as err:
            print(f"Case {name}:\n")
            traceback.print_exc()
            print()
            # PROBLEM: If this is being used for case generation, the input "case"
            # object is probably a dictionary of parameter values rather than a case
            # object, so we have an inconsistent return type.
            return case, err

    return wrapped


def cleanup_levels_units(
    levels: Mapping[str, Union[float, int, Quantity]], parameters: Sequence[Parameter]
) -> Dict[str, Quantity]:
    cleaned = {}
    for p in parameters:
        v = levels[p.name]
        if isinstance(v, Quantity):
            if not v.check(p.units):
                raise ValueError(
                    f"Provided value of {v} for parameter '{p.name}' does not have units of '{p.units}'"
                )
            cleaned[p.name] = v
        else:
            cleaned[p.name] = v * p.units
    return cleaned


def corrmap_distances(analysis, correlations):
    parameter_names = [p.name for p in analysis.parameters]
    by_parameter = (
        correlations.set_index(["Parameter", "Variable", "Time Point"])
        .unstack(["Variable", "Time Point"])
        .loc[parameter_names]
    )
    # ^ keep same parameter order
    arr = by_parameter.values
    # ^ first index over parameters, second over variables and time points
    arr_parameters = list(by_parameter.index)
    # Compute unsigned Pearson correlation distance = 1 - | ρ | where
    # ρ = (u − u̅) * (v - v̅) / ( 2-norm(u − u̅) * 2-norm(v - v̅) ).
    # If the 2-norm of (u − u̅) or (v − v̅) is zero, then the result will be
    # undefined.  Typically, this will happen when u or v is all zeroes; in this
    # case, the numerator is also zero.  For this application, it is reasonable
    # to define 0/0 = 0.  Therefore, we need to check for (u − u̅) * (v - v̅) = 0
    # and set the correlation distance for those vector pairs to 1.
    distances = (
        1 - abs(scipy.spatial.distance.pdist(arr, metric="correlation") - 1)
    ) ** 0.5
    n = len(arr)  # number of variables
    numerator = np.empty(len(distances))
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
    distances[numerator == 0] = 1
    return distances


def _validate_opt_on_case_error(value):
    # Validate input
    on_case_error_options = ("stop", "hold", "ignore")
    if not value in on_case_error_options:
        raise ValueError(
            f"on_case_error = {value}; allowed values are {','.join(on_case_error_options)}"
        )


def cases_table(case_generator, parallel_output, step="Generate"):
    # TODO: Figure out same way to demarcate parameters from other
    # metadata so there are no reserved parameter names.  For example, a
    # user should be able to name their parameter "path" without
    # conflicts.
    #
    # Create column names.  Do this explicitly so if there is no output to process we
    # still get a valid (empty) table.
    data = {"ID": []}
    for p in case_generator.parameters:
        data[p.name] = []
    data["status"] = []
    data["path"] = []
    # For each case's output, add a row
    for nm, output in parallel_output.items():
        case = output["Case"]
        # ID
        data["ID"].append(nm)
        # Parameters
        for p in case_generator.parameters:
            data[p.name].append(case.parameters_n[p.name])
        # Status
        status = output["Status"]
        if status == SUCCESS:
            msg = f"{step}: {status}"
        else:
            msg = f"{step}: {', '.join(err for err in status)}"
        data["status"].append(msg)
        # File path
        data["path"].append(case.feb_file)
    tab = DataFrame(data).set_index("ID")
    return tab


def do_parallel(cases: Mapping[str, object], fun, on_case_error="stop"):
    # cases needs to be a mapping because do_parallel is used for both case
    # generation and running cases.  During case generation, the name is separate
    # from the parameter values.
    _validate_opt_on_case_error(on_case_error)  # TODO: Use enum
    output = {}
    pool = ProcessPool(nodes=NUM_WORKERS)
    # TODO: Trapping every error, including Python errors, is a terrible system for
    # reporting case generation and simulation errors.  Python errors should force
    # termination as usual (so that you can use a debugger effectively), or at least
    # this should be toggleable separately.  As-is, case generation must not fail,
    # because then there is no Case object for the subsequent code paths.
    results = pool.map(_trap_err(lambda x: fun(*x)), cases.items())
    for case, status in results:
        # Log the run outcome
        output[case.name] = {"Case": case, "Status": status}
        # Should we continue submitting cases?
        if not status == SUCCESS and on_case_error == "stop":
            print(
                f"While working on case {case.name}, a {status.__class__.__name__} was encountered.  Because `on_case_error` = {on_case_error}, do_parallel will allow each currently running case to finish, then stop.\n"
            )
            pool.close()
            break
    return output


def expand_run_errors(cases: DataFrame):
    """Return cases table with run error columns instead of one status column"""
    errors = defaultdict(lambda: np.zeros(len(cases), dtype=bool))
    run_status = np.full(len(cases), None)
    for irow in range(len(cases)):
        status = cases["status"].iloc[irow]
        i = status.find(":")
        phase = status[:i]
        codes = [s.strip() for s in status[i + 1 :].split(",")]
        if phase != "Run":
            run_status[irow] = "No Run"
        elif "Success" in codes:
            run_status[irow] = "Success"
            if len(codes) > 1:
                raise ValueError(
                    f"Case {irow}: Run was recorded as successful but additional error codes were present.  The codes were {', '.join(codes)}."
                )
        else:
            run_status[irow] = "Error"
        for code in codes:
            if code == "Success":
                continue
            errors[code][irow] = True
    # Return cases table augmented with run status and error columns
    out = cases.copy()
    out["Run Status"] = pd.Series(
        run_status, dtype=CategoricalDtype(["Success", "Error", "No Run"])
    )
    for error in errors:
        out[error] = errors[error]
    return out, tuple(errors.keys())


def import_cases(generator: CaseGenerator):
    """Recheck all auto-generated cases in analysis and update solution status

    This function is meant to be used when you're manually moving simulation files
    between computers.

    """
    # TODO: Handle named cases
    analysis = Analysis(generator)
    cases = analysis.table_auto_cases
    for i in cases.index:
        pth_feb = generator.directory / cases.loc[i, "path"]
        # Check if there is already a run message; do not overwrite it.  Once
        # `import_cases` runs the same suite of checks that the original run did,
        # we can overwrite it.
        if cases.loc[i, "status"].startswith("Run: "):
            continue
        # Check if the model file is even there
        if not pth_feb.exists():
            cases.loc[i, "status"] = "Missing"
            continue
        # Check if the simulation files exist
        pth_log = pth_feb.with_suffix(".log")
        pth_xplt = pth_feb.with_suffix(".xplt")
        if not (pth_log.exists() and pth_xplt.exists()):
            continue
        # Check if the simulation ran successfully
        log = wfl.febio.LogFile(pth_feb.with_suffix(".log"))
        if log.termination is None:
            continue
        elif log.termination == "Normal":
            # TODO: Incorporate generator's checks.  Need to reconstitute the case,
            #  since check functions accept the case as their argument.
            cases.loc[i, "status"] = f"Run: {SUCCESS}"
        else:
            cases.loc[i, "status"] = f"Run: {log.termination} termination"
    analysis.update_table_auto_cases(cases)


def run_case(name, case):
    """Run a case's simulations and check its output as part of a sensitivity analysis

    This function is designed to be used with `do_parallel()`, hence the strange
    return value.

    """
    try:
        status = case.run(raise_on_check_fail=False)
    except FEBioError as err:
        status = [err]
    if status != SUCCESS:
        for err in status:
            print(f"Case {name}: {err!r}")
    return case, status


def run_sensitivity(
    case_generator, distributions, nlevels, named_levels={}, on_case_error="stop"
):
    """Run a sensitivity analysis from an analysis object."""
    named_levels = {
        name: cleanup_levels_units(levels, case_generator.parameters)
        for name, levels in named_levels.items()
    }
    _validate_opt_on_case_error(on_case_error)
    # Set waffleiron to run FEBio with only 1 thread, since we'll be running one
    # FEBio process per core
    wfl.febio.FEBIO_THREADS_DEFAULT = 1

    def replace_status(table, output, step=None):
        if step is not None:
            prefix = f"{step}: "
        else:
            prefix = ""
        for i, info in output.items():
            status = info["Status"]
            if status == SUCCESS:
                msg = f"{prefix}{status}"
            else:
                # We assume status is iterable, but if status came from a captured
                # error instead of the custom checks (which return a list of errors),
                # it probably won't be iterable.  So we need to wrap it.
                try:
                    status = iter(status)
                except TypeError:
                    status = [status]
                msg = f"{prefix}{', '.join(err.__class__.__name__ for err in status)}"
            table.loc[i, "status"] = msg

    def make_f_gen(
        case_generator: CaseGenerator,
        subdir_name,
    ):
        """Generate a case as part of a sensitivity analysis"""

        def f(name, parameter_values):
            return (
                case_generator.generate_case(
                    name, parameter_values, sub_dir=subdir_name
                ),
                SUCCESS,
            )

        return f

    def process_group(case_generator, name, parameter_values):
        pth_table = case_generator.directory / f"{name}_cases.csv"
        f = make_f_gen(case_generator, f"{name}_cases")
        output = do_parallel(parameter_values, f, on_case_error=on_case_error)
        table = cases_table(case_generator, output)
        write_cases_table(table, pth_table)
        # Run the cases
        cases = {nm: e["Case"] for nm, e in output.items()}
        output = do_parallel(cases, run_case, on_case_error=on_case_error)
        replace_status(table, output, step="Run")
        write_cases_table(table, pth_table)
        return table, pth_table

    # Generate the named cases
    pth_named_table = case_generator.directory / f"named_cases.csv"
    groups = {}
    groups["named"] = process_group(case_generator, "named", named_levels)
    sensitivity_values = full_factorial_values(
        case_generator.parameters, distributions, nlevels
    )
    groups["generated"] = process_group(case_generator, "generated", sensitivity_values)

    # Check if error terminations prevent analysis of results
    for nm, (tab, pth) in groups.items():
        n_run_error = np.sum(tab["status"] != "Run: Success")
        n_success = np.sum(tab["status"] == "Run: Success")
        n_not_run = int(np.sum([s.startswith("Generate") for s in tab["status"]]))
        error_report = (
            f"\nOf {len(tab)} {nm} sensitivity cases:\n"
            f"{n_success} cases ran successfully\n"
            f"{n_run_error} cases had a run error\n"
            f"{n_not_run} cases did not have a run attempt\n"
            f"See {pth}"
        )
        print(error_report)
        if n_run_error != 0:
            if on_case_error == "hold":
                print(
                    f"Because there at least one simulation had an error and `on_case_error` = 'hold', simulation was attempted for all cases but the sensitivity analysis was stopped prior to data analysis.  To continue with the sensitivity analysis, manually correct the error terminations (or not) and call `tabulate` and `plot_sensitivity`."
                )
                sys.exit()
            elif on_case_error == "stop":
                print(
                    f"Because there was at least one simulation had an error and `on_case_error` = 'stop', the sensitivity analysis was stopped after the first simulation error, before running all cases."
                )
                sys.exit()
    # Tabulate and plot the results
    tabulate(case_generator)
    makefig_sensitivity_all(case_generator)


def generate_cases_serial(
    generator: CaseGenerator, parameter_values: Mapping[str, Mapping[str, Quantity]]
):
    """Generate simulation definition files for named cases"""
    cases = [
        generator.generate_case(name, levels, sub_dir="named_cases")
        for name, levels in parameter_values.items()
    ]
    return cases


def full_factorial_values(parameters, distributions, nlevels):
    levels = {}
    # Calculate each parameter's levels
    for p in parameters:
        v = distributions[p.name]
        if isinstance(v, ContinuousScalar):
            levels[p.name] = v.sensitivity_levels(nlevels)
        elif isinstance(v, CategoricalScalar):
            levels[p.name] = v.sensitivity_levels()
        else:
            # Perhaps a scalar value?
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    raise ValueError(
                        f"Generating levels from a variable of type `{type(v)}` is not yet supported."
                    )
            levels[p.name] = (v,)
    combinations = product(*(levels[p.name] for p in parameters))
    parameter_values = {
        i: {p.name: v for p, v in zip(parameters, values)}
        for i, values in enumerate(combinations)
    }
    return parameter_values


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
    record, timeseries = tabulate_case(case)
    with open(dir_out / f"{case.name}_vars.json", "w") as f:
        write_record_to_json(record, f)
    timeseries.to_csv(dir_out / f"{case.name}_timeseries_vars.csv", index=False)
    makefig_case_tsvars(timeseries, dir_out=dir_out, casename=case.name)
    return record, timeseries


def tabulate(analysis: CaseGenerator):
    """Tabulate output from an analysis."""
    for nm, fbase in (
        ("named", "named"),
        ("sensitivity", "generated"),
    ):
        ivars_table = defaultdict(list)  # we don't know param names in advance
        ivars_table["case"] = []  # force creation of the index column
        pth_cases = analysis.directory / f"{fbase}_cases.csv"
        cases = pd.read_csv(pth_cases, index_col=0)
        for i in cases.index:
            if not cases.loc[i, "status"] == "Run: Success":
                # Simulation failure.  Don't bother trying to tabulate the results.
                # TODO: Partial tabulations could still be useful for troubleshooting
                continue
            case = Case(
                i,
                analysis.parameters,
                {p.name: cases[p.name].loc[i] for p in analysis.parameters},
                analysis.variables,
                directory=analysis.directory / f"{fbase}_cases",
                checks=analysis.checks,
            )
            record, timeseries = tabulate_case_write(
                case, dir_out=analysis.directory / "case_output"
            )
            ivars_table["case"].append(i)
            for p in analysis.parameters:
                k = f"{p.name} [param]"
                ivars_table[k].append(cases.loc[i, p.name])
            for v in record["instantaneous variables"]:
                k = f"{v} [var]"
                ivars_table[k].append(record["instantaneous variables"][v]["value"])
        df_ivars = DataFrame(ivars_table).set_index("case")
        df_ivars.to_csv(
            analysis.directory / f"{fbase}_cases_-_inst_vars.csv", index=True
        )


def tab_tsvars_corrmap(analysis, tsdata, fun_corr=corr_pearson):
    """Return table of time series vs. parameter correlation vectors"""
    tsdata = tsdata.copy().reset_index().set_index("Step")
    # Calculate sensitivity vectors
    ntimes = len(tsdata.index.unique())  # number of time points
    corr_vectors = {
        (p.name, v): np.zeros(ntimes)
        for p in analysis.parameters
        for v in analysis.variables
    }
    for v in analysis.variables:
        if np.any(np.isnan(tsdata[f"{v} [var]"])):
            raise ValueError(f"NaNs detected in values for {v}.")
    for i in range(ntimes):
        for parameter in analysis.parameters:
            for vnm in analysis.variables:
                ρ = fun_corr(tsdata.loc[i], f"{parameter.name} [param]", f"{vnm} [var]")
                corr_vectors[(parameter.name, vnm)][i] = ρ
    correlations = DataFrame(corr_vectors).stack(dropna=False)
    correlations.index.set_names(["Time Point", "Variable"], inplace=True)
    correlations = correlations.melt(
        ignore_index=False, var_name="Parameter", value_name="Correlation Coefficient"
    ).reset_index()
    return correlations


def makefig_sensitivity_all(analysis, tsfilter=None):
    """Make all sensitivity analysis figures

    :param tsfilter: Function that takes a timeseries data table, modifies it, and
    returns it.

    """
    # For temporary backwards compatibility
    if isinstance(analysis, CaseGenerator):
        analysis = Analysis(analysis)
    # Named cases
    named_cases = analysis.table_named_cases
    # Named cases are manually constructed and so should converge.  If they do not, it's
    # probably a user problem.  But you still  need to see what happened when there's an
    # error, so the plotting function should press on.
    named_cases = named_cases[named_cases["status"] == "Run: Success"]

    # Sensitivity cases
    cases = analysis.table_auto_cases
    cases = cases[cases["status"] == "Run: Success"]

    # Read parameters
    param_values = {p.name: cases[p.name] for p in analysis.parameters}

    # Get variable names
    ivar_names = [
        nm
        for nm, var in analysis.variables.items()
        if var.temporality == "instantaneous"
    ]
    tsvar_names = [
        nm for nm, var in analysis.variables.items() if var.temporality == "time series"
    ]

    # Plots for errors
    makefig_error_counts(analysis)
    makefig_error_pdf_uniparam(analysis)
    makefig_error_pdf_biparam(analysis)

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
            record, tab_timeseries = analysis.case_data(i)
            for nm in ivar_names:
                ivar_values[nm].append(record["instantaneous variables"][nm]["value"])
        makefig_sensitivity_ivar_all(analysis, param_values, ivar_names, ivar_values)

    # Summarize time series variables
    if len(tsvar_names) > 0:
        tsdata = analysis.cases_tsdata(cases)
        if tsfilter is not None:
            tsdata = tsfilter(tsdata)
        # Summarize time series variables' data in a table
        tsvar_summary = (
            tsdata[["Step", "Time"] + [f"{c} [var]" for c in analysis.variables.keys()]]
            .groupby("Step")
            .describe()
        )
        tsvar_summary.columns = [
            f"{c[0].removesuffix(' [var]')}, {c[1]}" for c in tsvar_summary.columns
        ]
        tsvar_summary.reset_index().to_csv(
            analysis.directory / "generated_cases_-_timeseries_var_summary.csv",
            index=False,
        )
        # Plots for time series variables
        makefig_sensitivity_tsvar_all(analysis, tsvar_names, tsdata, cases, named_cases)


def makefig_error_counts(analysis):
    """Plot error proportions and correlations"""
    # Collect error counts from cases table
    cases, error_codes = expand_run_errors(analysis.table_auto_cases)
    outcome_count = dict(cases["Run Status"].value_counts())
    error_count = {code: cases[code].sum() for code in error_codes}
    # Plot outcome & error counts
    figw = 7.5
    figh = 4
    fig = Figure(figsize=(figw, figh))
    FigureCanvas(fig)
    gs0 = GridSpec(
        1, 2, figure=fig, width_ratios=(len(outcome_count), len(error_count))
    )
    # (1) Outcome counts
    ax = fig.add_subplot(gs0[0, 0])
    ax.set_title("Run status", fontsize=FONTSIZE_FIGLABEL)
    ax.set_ylabel("Case Count", fontsize=FONTSIZE_AXLABEL)
    ax.bar("Success", outcome_count["Success"], color="black")
    ax.bar("Error", outcome_count["Error"], color="gray")
    ax.bar("No Run", outcome_count["No Run"], color="white", edgecolor="gray")
    ax.set_ylim(0, len(cases))
    ax.tick_params(axis="x", labelsize=FONTSIZE_AXLABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
    # (2) Error counts
    ax = fig.add_subplot(gs0[0, 1])
    ax.set_title("Error breakdown", fontsize=FONTSIZE_FIGLABEL)
    ax.set_ylabel("Case Count", fontsize=FONTSIZE_AXLABEL)
    for x, c in sorted(error_count.items()):
        if x == "Success":
            continue
        ax.bar(x, c)
    ax.set_ylim(0, outcome_count["Error"])
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")
    # Outcome & error count figure is complete
    fig.tight_layout()
    fig.savefig(analysis.directory / "run_outcomes_and_error_counts.svg")

    # Plot correlation between errors
    # TODO: Clean up the spacing
    figw = 7.5
    figh = 6.0
    fig = Figure(figsize=(figw, figh))
    FigureCanvas(fig)
    marginal_p = np.full((len(error_codes), len(error_codes)), np.nan)
    for i in range(len(error_codes)):
        for j in range(len(error_codes)):
            if i == j:
                continue
            marginal_p[i, j] = np.sum(
                np.logical_and(cases[error_codes[i]], cases[error_codes[j]])
            ) / np.sum(cases[error_codes[j]])
    ax = fig.add_subplot()
    cmap = mpl.cm.get_cmap("cividis")
    im = ax.matshow(marginal_p, cmap=cmap, origin="upper")
    # Write the value of each cell as text
    for (i, j), p in np.ndenumerate(marginal_p):
        if i == j:
            continue
        ax.text(
            j,
            i,
            "{:0.2f}".format(p),
            ha="center",
            va="center",
            backgroundcolor=(1, 1, 1, 0.5),
            fontsize=FONTSIZE_TICKLABEL,
        )
    # x-tick formatting
    ax.xaxis.tick_bottom()
    ax.set_xticks([i for i in range(len(error_codes))])
    ax.set_xticklabels(error_codes)
    ax.tick_params(axis="x", labelsize=FONTSIZE_AXLABEL)
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")
    # y-tick formatting
    ax.set_yticks([i for i in range(len(error_codes))])
    ax.set_yticklabels(error_codes)
    ax.tick_params(axis="y", labelsize=FONTSIZE_AXLABEL)
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$P(Y_{Err} | X_{Err})$", fontsize=FONTSIZE_AXLABEL)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
    fig.tight_layout()
    fig.savefig(analysis.directory / "run_error_co-ocurrence.svg")


def makefig_error_pdf_uniparam(analysis):
    """For each parameter write figure with conditional error PDFs"""
    cases, error_codes = expand_run_errors(analysis.table_auto_cases)
    # TODO: Levels information should probably be stored in the analysis object
    levels = {p: sorted(np.unique(cases[p.name])) for p in analysis.parameters}

    def p_error(cases, parameter, levels, error_code=None):
        p = np.full(len(levels), np.nan)
        for i, level in enumerate(levels):
            # noinspection PyTypeChecker
            m: pd.Series = cases[parameter.name] == level
            if error_code is None:
                # P(any error code)
                p[i] = sum(cases[m]["Run Status"] == "Error") / sum(m)
            else:
                # P(specific error code)
                p[i] = sum(cases[m][error_code]) / sum(m)
        return p

    # Create figure with probability density function plots
    fig = Figure(constrained_layout=True)
    fig.set_constrained_layout_pads(
        wspace=6 / 72, hspace=6 / 72, w_pad=4 / 72, h_pad=4 / 72
    )
    nw = len(analysis.parameters)
    nh = len(error_codes) + 1  # Extra row for union of all error codes
    fig.set_size_inches((3.5 * nw + 0.25, 3.2 * nh + 0.25))  # TODO: set common style
    gs = GridSpec(nh, nw, figure=fig)
    for i, param in enumerate(analysis.parameters):
        for j in range(nh):
            if j == 0:
                e = "Any Error"
                p = p_error(cases, param, levels[param])
                color = "#2e2e2e"
            else:  # plot p(specific error code)
                e = error_codes[j - 1]
                p = p_error(cases, param, levels[param], e)
                color = "#a22222"
            ax = fig.add_subplot(gs[j, i])
            ax.fill_between(levels[param], p, color=color)
            ax.plot(
                levels[param],
                p,
                linestyle="none",
                marker="o",
                color="black",
                markersize=5,
            )
            ax.set_xlabel(param, fontsize=FONTSIZE_AXLABEL)
            ax.tick_params(
                axis="x",
                labelsize=FONTSIZE_TICKLABEL,
                color="dimgray",
                labelcolor="dimgray",
            )
            ax.set_ylabel(f"P( {e} | {param} )", fontsize=FONTSIZE_AXLABEL)
            ax.tick_params(
                axis="y",
                labelsize=FONTSIZE_TICKLABEL,
                color="dimgray",
                labelcolor="dimgray",
            )
            ax.set_ylim([0, 1])
            for k in ax.spines:
                ax.spines[k].set_visible(False)
            ax.set_facecolor("#F6F6F6")
    fig.savefig(analysis.directory / f"run_error_probability_uniparameter.svg")


def makefig_error_pdf_biparam(analysis):
    """For each parameter pair write figure with conditional error PDFs"""
    cases, error_codes = expand_run_errors(analysis.table_auto_cases)
    # TODO: Levels information should probably be stored in the analysis object
    levels = {p: sorted(np.unique(cases[p.name])) for p in analysis.parameters}

    def p_error(
        cases,
        error_code,
        parameter1: Parameter,
        levels1,
        parameter2: Parameter,
        levels2,
    ):
        """Return p(error | parameter1, parameter2)"""
        p = np.full((len(levels1), len(levels2)), np.nan)
        x = np.full((len(levels1), len(levels2), 2), np.nan)
        for i, level1 in enumerate(levels1):
            for j, level2 in enumerate(levels2):
                m = np.logical_and(
                    cases[parameter1.name] == level1, cases[parameter2.name] == level2
                )
                x[i, j, :] = (levels1[i], levels2[i])
                n = np.sum(m)
                if n == 0:
                    p[i, j] = np.nan
                else:
                    p[i, j] = np.sum(cases[error_code][m]) / n
        return x, p

    for e in error_codes:
        # Create figure with probability density function plots
        fig = Figure(constrained_layout=True)
        fig.set_constrained_layout_pads(
            wspace=2 / 72, hspace=2 / 72, w_pad=2 / 72, h_pad=2 / 72
        )
        nw = len(analysis.parameters)
        nh = len(analysis.parameters)
        fig.set_size_inches(
            (3.4 * nw + 0.25, 2.5 * nh + 0.25)
        )  # TODO: set common style
        gs = GridSpec(nh, nw, figure=fig)
        for j, p1 in enumerate(analysis.parameters):  # columns
            for i, p2 in enumerate(analysis.parameters):  # rows
                ax = fig.add_subplot(gs[i, j])
                x, p = p_error(cases, e, p1, levels[p1], p2, levels[p2])
                # warning: pcolormesh maps i → y and j → x
                pcm = ax.pcolormesh(
                    levels[p1],
                    levels[p2],
                    p.T,
                    shading="nearest",
                    cmap="cividis",
                    vmin=0,
                    vmax=1,
                )
                # TODO: Place colorbar manually; it seems that doing it automatically
                #  is slow
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.set_label(f"P( {e} )", fontsize=FONTSIZE_TICKLABEL)
                cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
                ax.set_xlim(levels[p1][0], levels[p1][-1])
                ax.set_xlabel(p1, fontsize=FONTSIZE_AXLABEL)
                ax.tick_params(
                    axis="x",
                    labelsize=FONTSIZE_TICKLABEL,
                    color="dimgray",
                    labelcolor="dimgray",
                )
                ax.set_ylim(levels[p2][0], levels[p2][-1])
                ax.set_ylabel(p2, fontsize=FONTSIZE_AXLABEL)
                ax.tick_params(
                    axis="y",
                    labelsize=FONTSIZE_TICKLABEL,
                    color="dimgray",
                    labelcolor="dimgray",
                )
                # for k in ax.spines:
                #     ax.spines[k].set_visible(False)
        fig.savefig(
            analysis.directory / f"run_error_probability_biparameter_-_error={e}.svg"
        )


def makefig_sensitivity_ivar_all(analysis, param_values, ivar_names, ivar_values):
    param_names = [p.name for p in analysis.parameters]
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


def get_reference_tsdata(analysis, tsdata, cases, named_cases):
    if "nominal" in named_cases.index:
        # Plot nominal case
        record, ref_ts = analysis.case_data("nominal")
        ref_ts.columns = [
            f"{s} [var]" if not s in ("Time", "Step") else s for s in ref_ts.columns
        ]
    else:
        # Return values for median generated case, if possible
        #
        # TODO: It should be easier  to get  the levels of  an analysis, or  the median
        # should be calculated based on the variables rather than the parameters.
        param_values = {k: cases[k.name] for k in analysis.parameters}
        param_nlevels = {k: len(np.unique(v)) for k, v in param_values.items()}
        if any([n % 2 == 0 for n in param_nlevels.values()]):
            # The median parameter values are not represented in the cases
            warn(
                "With an even number of parameter levels, there is no case with median parameter values to serve as a representative case.  A random case will be selected as the representative case."
            )
            candidates = cases[cases["status"] == f"Run: {SUCCESS}"]
            case_id = candidates.index[len(candidates) // 2]
        else:
            median_levels = {
                k.name: np.median(values) for k, values in param_values.items()
            }
            m = np.ones(len(cases), dtype="bool")
            for param, med in median_levels.items():
                m = np.logical_and(m, cases[param] == med)
            assert np.sum(m) == 1
            case_id = cases.index[m][0]
        ref_ts = tsdata[tsdata["Case"] == case_id]
    return ref_ts


def makefig_sensitivity_tsvar_all(
    analysis, tsvar_names, tsdata, cases, named_cases=None
):
    """Plot sensitivity of each time series variable to each parameter"""
    makefig_tsvars_line(analysis, cases, named_cases)
    makefig_tsvars_pdf(analysis, tsdata, cases, named_cases)

    # Obtain reference case for time series variables' correlation heatmap
    # TODO: The heat map figure should probably indicate which case is
    # plotted as the time series guide.
    ref_ts = get_reference_tsdata(analysis, tsdata, cases, named_cases)

    # Sobol analysis
    S1, ST = sobol_analysis_tsvars(analysis, tsdata, cases)
    makefig_sobol_tsvars(analysis, S1, ST, ref_ts)

    # Correlation coefficients
    for nm_corr, fun_corr in (
        ("pearson", corr_pearson),
        ("spearman", corr_spearman),
        ("partial", corr_partial),
    ):
        correlations_table = tab_tsvars_corrmap(analysis, tsdata, fun_corr=fun_corr)
        correlations_table.to_feather(
            analysis.directory / f"tsvar_param_corr={nm_corr}.feather"
        )
        distances = corrmap_distances(analysis, correlations_table)
        for norm in ("none", "all", "vector", "subvector"):
            fig_heatmap, parameter_order = plot_tsvar_param_heatmap(
                analysis,
                correlations_table,
                distances,
                ref_ts,
                norm=norm,
            )
            fig_heatmap.savefig(
                analysis.directory
                / f"tsvar_param_corr={nm_corr}_heatmap_norm={norm}.svg"
            )
            if norm == "none":
                fig_distmat = plot_tsvar_param_distmat(
                    analysis, distances, parameter_order
                )
                fig_distmat.fig.savefig(
                    analysis.directory
                    / f"tsvar_param_corr={nm_corr}_distance_matrix_clustered.svg",
                    dpi=300,
                )
                fig_distmat = plot_tsvar_param_distmat(
                    analysis,
                    distances,
                )
                fig_distmat.fig.savefig(
                    analysis.directory
                    / f"tsvar_param_corr={nm_corr}_distance_matrix_unclustered.svg",
                    dpi=300,
                )
        # Estimate the rank of the sensitivity vectors
        pth_svd_data = analysis.directory / f"tsvar_param_corr={nm_corr}_stats.json"
        pth_svd_fig_s = (
            analysis.directory / f"tsvar_param_corr={nm_corr}_singular_values.svg"
        )
        pth_svd_fig_v = (
            analysis.directory / f"tsvar_param_corr={nm_corr}_principal_axes.svg"
        )
        try:
            # TODO: get the parameter order from the cluster analysis so it can be
            #  re-used for the sensitivity vectors PCA plot
            svd_data = corr_svd(correlations_table)
            with open(pth_svd_data, "w") as f:
                json.dump(svd_data, f)
            fig = fig_corr_singular_values(svd_data)
            fig.savefig(pth_svd_fig_s)
            fig = fig_corr_eigenvectors(svd_data)
            fig.savefig(pth_svd_fig_v)
        except np.linalg.LinAlgError as e:
            warn(f"Correlation matrix SVD failed: {str(e)}")
            # Don't leave files from a previous run (if any); that would confuse the user
            # TODO: Old files should be cleaned up at the beginning of a run
            pth_svd_data.unlink(missing_ok=True)
            pth_svd_fig_s.unlink(missing_ok=True)
            pth_svd_fig_v.unlink(missing_ok=True)


def makefig_sobol_tsvars(analysis, S1, ST, ref_ts):
    S1 = np.array([by_var[v] for by_var in S1.values() for v in by_var])
    ST = np.array([by_var[v] for by_var in ST.values() for v in by_var])
    indices = {"Order 1": S1, "Total": ST}
    variables = [(v, "") for v in analysis.variables]
    ref_values = np.array([ref_ts[f"{v} [var]"] for v in analysis.variables])
    fr = fig_stacked_line(
        indices, analysis.parameters, variables, ref_timeseries=ref_values
    )
    fr.fig.savefig(analysis.directory / "tsvars_sobol_sensitivities_scale=free.svg")
    fr = fig_stacked_line(
        indices,
        analysis.parameters,
        variables,
        ref_timeseries=ref_values,
        ymax="shared",
    )
    fr.fig.savefig(analysis.directory / "tsvars_sobol_sensitivities_scale=shared.svg")


# noinspection PyPep8Naming
def sobol_analysis_tsvars(analysis, tsdata, cases):
    """Write Sobol analysis for time series variables"""
    nsteps = len(np.unique(tsdata["Step"]))
    # TODO: Levels information should probably be stored in the analysis object
    levels = {p: sorted(np.unique(cases[p.name])) for p in analysis.parameters}
    S1 = {
        p: {v: np.zeros(nsteps) for v in analysis.variables}
        for p in analysis.parameters
    }
    ST = deepcopy(S1)
    for v in analysis.variables:
        # Exclude time points with zero variance
        d = tsdata.groupby("Step")[f"{v} [var]"].var()
        skip_steps = d[d == 0].index.values
        data1 = tsdata[~tsdata["Step"].isin(skip_steps)]
        m = np.ones(nsteps, dtype=bool)
        m[skip_steps] = False
        for parameter in analysis.parameters:
            # Exclude cases that are not part of a stratum with at least 2 cases
            idxs = ["Step"] + [
                f"{p.name} [param]" for p in analysis.parameters if p != parameter
            ]
            # TODO: Having the level indices in the data frame itself would be
            #  helpful to guard against floating point imprecision.
            var_by_stratum = data1.groupby(idxs)[f"{v} [var]"].agg(
                [partial(np.var, ddof=0), "count"]
            )
            data2 = pd.merge(
                data1.set_index(idxs),
                var_by_stratum[["count"]],
                left_index=True,
                right_index=True,
            )
            data2 = data2[data2["count"] == len(levels[parameter])]
            # Everything after this should use the data with all exclusion rules applied

            # Important: In this analysis, variances need to be computed with N,
            # not N − 1, in the denominator.  Population variance, not sample
            # variance.  Otherwise S1 and ST may exceed 1.
            var_by_step = data2.groupby("Step")[f"{v} [var]"].var(ddof=0)
            var_by_stratum = var_by_stratum[
                var_by_stratum["count"] == len(levels[parameter])
            ]
            # S1, direct effect
            # With Y as tsvar at time t and θ_i = parameter i
            # S1_i = Var[ E(Y | θ_i) ] / Var[Y]
            # Var[ E(Y | θ_i) ] = E[E(Y|θ_i)^2] - E[E(Y|θ_i)]^2
            # E[E(Y|θ_i)]^2 = E(Y)^2 by law of total expectation
            μ_by_level = data2.groupby(["Step", f"{parameter.name} [param]"])[
                f"{v} [var]"
            ].mean()
            S1[parameter][v][m] = (
                μ_by_level.reset_index().groupby("Step")[f"{v} [var]"].var(ddof=0)
                / var_by_step
            )
            # Total effect
            # With Y as tsvar at time t, V_Ti = Var(Y) - Var_X_≠i[ E_X_i(Y | θ_≠i) ]
            ST[parameter][v][m] = (
                var_by_stratum.groupby("Step")["var"].mean() / var_by_step
            )
            # Cleanup to prevent accidental reuse
            del data2
        del data1
    # Another approach to calculating ST is presented here.  This produced ST < 0,
    # perhaps due to correlations between parameters, or perhaps due to an error in
    # this code.
    # grouped = tsdata.groupby(
    #     ["Step"] + [f"{nm} [param]" for nm in analysis.parameters if nm != p]
    # )
    # E_Y_over_θ_i = grouped[f"{v} [var]"].mean()
    # E_Y_over_θ_i.reset_index(
    #     level=[
    #         i for i, nm in enumerate(E_Y_over_θ_i.index.names) if nm != "Step"
    #     ],
    #     drop=True,
    #     inplace=True,
    # )
    # Var_over_θ_noti = E_Y_over_θ_i.groupby("Step").var().values
    # ST[p][v][m] = 1 - Var_over_θ_noti[m] / Var_Y[v][m]
    return S1, ST


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


def makefig_case_tsvar(timeseries, varname, casename=None):
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


def makefig_case_tsvars(timeseries, dir_out, casename=None):
    """Plot a case's time series variables and write the plots to disk.

    This function is meant for automated sensitivity analysis.  Plots
    will be written to disk using the standard spamneggs naming
    conventions.

    TODO: Provide a companion function that just returns the plot handle, allowing
    customization.

    """
    dir_out = Path(dir_out)
    if casename is None:
        stem = ""
    else:
        stem = f"{casename}_"
    if not isinstance(timeseries, DataFrame):
        timeseries = DataFrame(timeseries)
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
        fig = makefig_case_tsvar(timeseries, varname, casename)
        fig.savefig(dir_out / f"{stem}timeseries_var={varname}.svg")


def plot_tsvar_param_heatmap(
    analysis,
    correlations,
    distances,
    ref_ts,
    norm="none",
):
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
    correlations = correlations.copy().set_index(
        ["Parameter", "Variable", "Time Point"]
    )

    # Widths of plot panels
    base_axw = 5.0

    # Widths of dendrogram axes
    fig_llabelw = LABELH_MULT * FONTSIZE_FIGLABEL / 72
    dendro_axw = 12 / 72 * len(analysis.parameters)

    # Widths of colorbar elements
    cbar_axw = 12 / 72
    # ^ width of colorbar axes
    cbar_lpad = 4 / 72
    commoncbar_lpad = 10 / 72
    # ^ padding b/w heat map axes and its colorbar axes
    cbar_rlabelw = FONTSIZE_AXLABEL / 72
    # ^ height of label text
    cbar_rpad = 36 / 72
    # ^ padding b/w colorbar axes and axes of /next/ heat map

    # Widths of right common colorbar area, if used
    if norm in ("none", "all"):
        # Use a common right-side colorbar
        cbar_areaw = commoncbar_lpad + cbar_axw + cbar_rpad + cbar_rlabelw
        # ^ total width of area needed for a colorbar
        rcbar_areaw = cbar_areaw
        axw = base_axw
    elif norm == "vector":
        # Use a one right-side colorbar for each parameter
        cbar_areaw = cbar_lpad + cbar_axw + cbar_rpad + cbar_rlabelw
        rcbar_areaw = cbar_areaw
        axw = base_axw
    else:  # norm == "individual"
        # No right-side colorbar; a colorbar will be placed alongside each axes
        cbar_areaw = cbar_lpad + cbar_axw + cbar_rpad + cbar_rlabelw
        rcbar_areaw = 0
        axw = base_axw + cbar_areaw

    (
        fig,
        axarr,
    ) = fig_blank_tsvars_by_parameter(
        len(analysis.parameters),
        len(analysis.variables),
        left_blankw=dendro_axw,
        right_blankw=rcbar_areaw,
        axw=axw,
    )
    figw = fig.get_figwidth()

    # Plot dendrogram.  Do this first to get the parameter ordering.
    t = axarr[1, 0].get_position().max[1]
    b = axarr[-1, 0].get_position().min[1]
    dn_ax = fig.add_axes((fig_llabelw / figw, b, dendro_axw / figw, t - b))
    dn_ax.axis("off")
    # Compute the linkages (clusters), unless all the distances are NaN (e.g., in a
    # univariate sensitivity analysis)
    if np.all(np.isnan(distances)):
        warn(
            "Correlation distance matrix is all NaNs.  This is expected if only one parameter is varying."
        )
        ordered_parameter_idx = np.arange(len(analysis.parameters))
    else:
        links = scipy.cluster.hierarchy.linkage(
            distances, method="average", metric="correlation"
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
        ordered_parameter_idx = dn["leaves"]

    # Create common axis elements for time series variable plots
    tick_locator = mpl.ticker.MaxNLocator(integer=True)
    cnorm = None  # if `norm` is valid, always overwritten
    if norm == "none":
        absmax = 1
        cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
    elif norm == "all":
        absmax = np.nanmax(np.abs(correlations.values))
        cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)

    # Draw the time series line plot in the first row
    for i, var in enumerate(analysis.variables):
        plot_reference_tsdata(ref_ts[f"{var} [var]"], axarr[0, i], varname=var)
        axarr[0, i].xaxis.set_major_locator(tick_locator)

    def plot_colorbar(im, ax):
        cbar = fig.colorbar(im, cax=ax)
        cbar.ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
        cbar.set_label("ρ [1]", fontsize=FONTSIZE_TICKLABEL)
        cbar.ax.yaxis.set_label_coords(2.7, 0.5)
        for k in cbar.ax.spines:
            cbar.ax.spines[k].set_visible(False)
        return cbar

    # Plot heatmaps
    by_parameter = correlations.unstack(["Variable", "Time Point"])
    arr_parameters = list(by_parameter.index)
    for j in range(axarr.shape[1]):
        axarr[-1, j].set_xlabel("Time Point Index", fontsize=FONTSIZE_TICKLABEL)
    for irow, iparam in enumerate(ordered_parameter_idx):
        parameter = arr_parameters[iparam]
        if norm == "vector":
            absmax = np.max(np.abs(correlations.loc[parameter].values))
            cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
        for ivar, var in enumerate(analysis.variables):
            if norm == "subvector":
                absmax = np.max(np.abs(correlations.loc[parameter, var].values))
                cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
            ax = axarr[irow + 1, ivar]
            # Image
            ρ = correlations.loc[parameter, var].values.T
            im = ax.imshow(
                ρ,
                aspect="auto",
                origin="upper",
                interpolation="nearest",
                cmap=CMAP_DIVERGE,
                norm=cnorm,
                extent=(
                    ref_ts["Step"].iloc[0] - 0.5,
                    ref_ts["Step"].iloc[-1] + 0.5,
                    -0.5,
                    0.5,
                ),
            )
            # Labels
            ax.xaxis.set_major_locator(tick_locator)
            ax.set_ylabel(
                parameter,
                fontsize=FONTSIZE_AXLABEL,
            )
            ax.tick_params(axis="y", left=False, labelleft=False)

            # Draw the heatmap's colorbar
            if norm == "subvector":
                # Narrow axes to fit cbar
                l, b = ax.get_position().min
                r, t = ax.get_position().max
                ax.set_position((l, b, base_axw / figw, t - b))
                # Create new cbar axes
                r, t = ax.get_position().max
                cbar_ax = fig.add_axes(
                    (r + cbar_lpad / figw, b, cbar_axw / figw, t - b)
                )
                plot_colorbar(im, cbar_ax)
            elif norm == "vector" and ivar == len(analysis.variables) - 1:
                l = ax.get_position().max[0] + commoncbar_lpad / figw
                b = ax.get_position().min[1]
                t = ax.get_position().max[1]
                w = cbar_axw / figw
                cbar_ax = fig.add_axes((l, b, w, t - b))
                plot_colorbar(im, cbar_ax)
    # Adjust the width of the reference line plot
    if norm == "subvector":
        for j in range(axarr.shape[1]):
            l, b = axarr[0, j].get_position().min
            r, t = axarr[0, j].get_position().max
            axarr[0, j].set_position((l, b, base_axw / figw, t - b))

    # Add the whole-plot right-most colorbar, if called for
    if norm in ("none", "all"):
        l = axarr[1, -1].get_position().max[0] + commoncbar_lpad / figw
        b = axarr[-1, -1].get_position().min[1]
        t = axarr[1, -1].get_position().max[1]
        w = cbar_axw / figw
        ax = fig.add_axes((l, b, w, t - b))
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
    return fig, ordered_parameter_idx


def plot_tsvar_param_distmat(analysis, distances, parameter_order=None):
    """Plot distance matrix of tsvar–parameter correlation vectors

    :parameter parameter_order: List of indices into the analysis parameters.  The
    parameters will be plotted in the listed order.  Usually this order is chosen to
    match that in the cluster analysis dendrogram in `plot_tsvar_param_heatmap`.

    """
    if parameter_order is None:
        parameter_order = np.arange(len(analysis.parameters))
    parameter_names = [p.name for p in analysis.parameters]
    distmat = scipy.spatial.distance.squareform(distances)[parameter_order, :][
        :, parameter_order
    ]
    fig = plot_matrix(
        distmat,
        vlim=(0, 1),
        tick_labels=[parameter_names[i] for i in parameter_order],
        cbar_label="Correlation Distance = 1 − |ρ|",
        title="Correlation sensitivity vector distances",
        format_str=".2f",
    )
    return fig


def makefig_tsvars_pdf(analysis, tsdata, cases, named_cases=None):
    tsdata_by_step = tsdata.set_index("Step")
    for variable in analysis.variables:
        for parameter in analysis.parameters:
            # Collect key quantiles.  These will be used for adjusting the range of
            # the probability density plot.
            vmin = tsdata[f"{variable} [var]"].min()
            vmax = tsdata[f"{variable} [var]"].max()
            if vmin == vmax:
                raise ValueError(
                    f"Variable {variable} has a constant value, {vmin}, so it is not appropriate to estimate its probability density function."
                )
            yrange_all = (vmin, vmax)
            idx_steps = tsdata["Step"].unique()
            quantile_levels = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
            quantiles = tuple([] for q in quantile_levels)
            for idx_step in idx_steps:
                v = tsdata_by_step.loc[idx_step, f"{variable} [var]"]
                for i, q in enumerate(v.quantile(quantile_levels).values):
                    quantiles[i].append(q)
            yrange_trim90 = (min(quantiles[1]), max(quantiles[-2]))
            yrange_trim75 = (min(quantiles[2]), max(quantiles[-3]))
            # Make the figures
            nm = f"timeseries_var_pdf_-_range=all_-_{variable}_vs_{parameter.name}.svg".replace(
                " ", "_"
            )
            fig, axs = fig_tsvar_pdf(
                analysis, tsdata, variable, yrange_all, parameter, cases, named_cases
            )
            fig.savefig(analysis.directory / nm)
            nm = f"timeseries_var_pdf_-_range=90percent_-_{variable}_vs_{parameter.name}.svg".replace(
                " ", "_"
            )
            fig, axs = fig_tsvar_pdf(
                analysis, tsdata, variable, yrange_trim90, parameter, cases, named_cases
            )
            fig.savefig(analysis.directory / nm)
            nm = f"timeseries_var_pdf_-_range=75percent_-_{variable}_vs_{parameter.name}.svg".replace(
                " ", "_"
            )
            fig, axs = fig_tsvar_pdf(
                analysis, tsdata, variable, yrange_trim75, parameter, cases, named_cases
            )
            fig.savefig(analysis.directory / nm)


def makefig_tsvar_line(analysis, variable, parameter, cases, named_cases=None):
    """One-at-a-time time series variable sensitivity line plots for one parameter

    The inputs include a set of a parameters, one of which is chosen as the
    sensitivity parameter, and a chosen variable.  Each parameter has n levels.  For
    each level i, plot a one-at-a-time sensitivity line plot for the chosen variable
    with respect to the chosen parameter, centering the one-at-a-time analysis on
    level i of all other parameters.  Lines for the different levels of the
    sensitivity parameter are overplotted and differentiated by color. These plots
    are arrayed as subplots of a single figure.

    Additionally, include a subplot plot showing the subject time series variable for
    each named case, if any are provided.

    """
    CBAR_LEVELS_THRESHOLD = 6
    # Collect parameter names and levels
    subject_parameter = parameter
    other_parameters = [p for p in analysis.parameters if p != subject_parameter]
    # TODO: Levels information should probably be stored in the analysis object
    levels = {p.name: sorted(np.unique(cases[p.name])) for p in analysis.parameters}
    # Assume all parameters have n levels; i.e., full factorial analysis
    n_levels = len(levels[subject_parameter.name])
    # Calculate the number of subplots
    n_plots = n_levels + 1 if named_cases is not None else n_levels
    # Create figure
    fig = Figure(constrained_layout=True)
    nh = math.floor(n_plots**0.5)
    nw = math.ceil(n_plots / nh)
    fig.set_size_inches((5 * nw + 1, 3 * nh + 0.25))  # TODO: set smart size
    fig.set_constrained_layout_pads(
        wspace=0.04, hspace=0.04, w_pad=2 / 72, h_pad=2 / 72
    )
    gs = GridSpec(nh, nw, figure=fig)
    axs = []
    # Plot the named case(s)
    if named_cases is not None:
        ax = fig.add_subplot(gs[0, 0])
        axs.append(ax)
        plot_tsvar_named(analysis, variable, parameter, named_cases, ax)
    # Plot the one-at-a-time sensitivity line plots
    #
    # For each of i = 1 … n levels, plot the variation in the output variable vs. the
    # subject parameter, holding all other parameters at their i'th level (the
    # "fulcrum").
    cbars = []
    cnorm = mpl.colors.Normalize(
        vmin=min(levels[subject_parameter.name]),
        vmax=max(levels[subject_parameter.name]),
    )
    for i in range(n_levels):
        fulcrum = {
            p.name: levels[p.name][i] if i < len(levels[p.name]) else None
            for p in other_parameters
        }
        # Select the cases that belong to the fulcrum
        m = np.ones(len(cases), dtype="bool")  # init
        for nm, v in fulcrum.items():
            if v is None:
                # No case with all parameters at level i
                m[:] = False
                break
            m = np.logical_and(m, cases[nm] == v)
        # Make the plot panel
        ax = fig.add_subplot(gs[(i + 1) // nw, (i + 1) % nw])
        axs.append(ax)
        ax.set_title(
            f"Other parameters set to level index = {i + 1}",
            fontsize=FONTSIZE_AXLABEL,
        )
        # TODO: Units-awareness for variables
        ax.set_ylabel(variable, fontsize=FONTSIZE_AXLABEL)
        ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_AXLABEL)
        # Plot a line for each sensitivity level of the subject parameter
        for case_id in cases.index[m]:
            record, tab_timeseries = analysis.case_data(case_id)
            value = cases.loc[case_id, subject_parameter.name]
            if subject_parameter.units == "1":
                label = f"{value:.3g}"
            else:
                label = f"{value:.3g} {subject_parameter.units:~P}"
            ax.plot(
                tab_timeseries["Step"],
                tab_timeseries[variable],
                color=CMAP_DIVERGE(cnorm(cases.loc[case_id, subject_parameter.name])),
                label=label,
            )
        # Format the plot
        ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
        if len(levels[subject_parameter.name]) >= CBAR_LEVELS_THRESHOLD:
            # There are many levels; use only color-coding to show parameter values
            cbar = fig.colorbar(
                ScalarMappable(norm=cnorm, cmap=CMAP_DIVERGE),
                ax=ax,
            )
            cbars.append(cbar)
            cbar.set_label(subject_parameter.name, fontsize=FONTSIZE_AXLABEL)
        else:
            # There are only a few levels; use a legend as well as color-coding
            if ax.lines:
                ax.legend(title=subject_parameter.name, fontsize=FONTSIZE_AXLABEL)
    # Link the y axes if each has a similar range (within an order of magnitude) as
    # the others
    ranges = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axs]
    if max(ranges) / min(ranges) < 10:
        # Link the y-axis across axes
        for ax in axs[1:]:
            axs[0].get_shared_y_axes().join(axs[0], ax)
    fig.suptitle(
        f"{variable} time series vs. {subject_parameter.name}",
        fontsize=FONTSIZE_FIGLABEL,
    )
    fig.canvas.draw()
    if cbars:
        # Make colorbar width = 12 pt.  The horizontal spacing between subplots will no
        # longer be consistent, but this looks better than fat or skinny colorbars.  It
        # does have the downside of slightly (a few px) breaking alignment between the
        # colorbar and its associated subplot.
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
    nm = f"timeseries_var_lineplot_-_{variable}_vs_{subject_parameter.name}.svg"
    fig.savefig(analysis.directory / nm.replace(" ", "_"))


def makefig_tsvars_line(analysis, cases, named_cases=None):
    """One-at-a-time sensitivity line plots for all variables and parameters"""

    # TODO: Figure out how to plot parameter sensitivity for multiple variation
    # (parameter interactions)
    for variable in analysis.variables:
        for parameter in analysis.parameters:
            makefig_tsvar_line(
                analysis, variable, parameter, cases, named_cases=named_cases
            )


def plot_tsvar_named(analysis, variable, parameter, named_cases, ax):
    """Plot time series variable for named cases into an axes"""
    ax.set_title(f"Named cases", fontsize=FONTSIZE_AXLABEL)
    ax.set_ylabel(variable, fontsize=FONTSIZE_AXLABEL)
    ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_AXLABEL)
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
    for i, case_id in enumerate(named_cases.index):
        record, tab_timeseries = analysis.case_data(case_id)
        value = named_cases.loc[case_id, parameter.name]
        if parameter.units == "1":
            label = f"{case_id}\n{parameter.name} = {value}"
        else:
            label = f"{case_id}\n{parameter.name} = {value} {parameter.units:~P}"
        ax.plot(
            tab_timeseries["Step"],
            tab_timeseries[variable],
            label=label,
            color=colors.categorical_n7[i % len(colors.categorical_n7)],
        )
    if ax.lines:
        ax.legend()


def corr_svd(correlations_table):
    """Calculate singular values and eigenvectors of parameter sensitivities"""
    correlations = correlations_table.set_index(["Parameter", "Variable", "Time Point"])
    arr = correlations.unstack(["Variable", "Time Point"]).values
    u, s, vh = np.linalg.svd(arr.T, full_matrices=False)
    svd_data = {
        "singular values": s.tolist(),
        "parameters": correlations.index.levels[0].values.tolist(),
        "variables": correlations.index.levels[1].values.tolist(),
        "principal axes": [vh[i].tolist() for i in range(len(vh))],
    }
    return svd_data


def fig_corr_singular_values(svd_data, cutoff=0.01):
    """Return figure with singular values from SVD of sensitivity vectors"""
    s = np.array(svd_data["singular values"])
    fig = Figure()
    # Consider using make_axes_locatable so spectrum holds its width
    w_hist = 1 + 0.5 * len(svd_data["parameters"])
    w_spect = 1
    w_fig = w_hist + w_spect
    fig.set_size_inches((w_fig, 3))
    gs = fig.add_gridspec(1, 2, width_ratios=(w_hist, w_spect))
    # Histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    x = 1 + np.arange(len(s))
    ax_hist.axhline(100 * cutoff, color=COLOR_DEEMPH, lw=1, linestyle=":")
    ax_hist.bar(x, 100 * s / np.sum(s))
    ax_hist.set_xlabel("Eigenvector Index", fontsize=FONTSIZE_AXLABEL)
    ax_hist.set_ylabel("Singular Value [%]", fontsize=FONTSIZE_AXLABEL)
    remove_spines(ax_hist)
    ax_hist.xaxis.set_major_locator(mpl.ticker.FixedLocator(x))
    ax_hist.tick_params(
        axis="x",
        color=COLOR_DEEMPH,
        labelsize=FONTSIZE_TICKLABEL,
        labelcolor=COLOR_DEEMPH,
    )
    ax_hist.tick_params(
        axis="y",
        color=COLOR_DEEMPH,
        labelsize=FONTSIZE_TICKLABEL,
        labelcolor=COLOR_DEEMPH,
    )
    # Spectrum
    ax_s = fig.add_subplot(gs[0, 1])
    ax_s.set_yscale("log")
    ax_s.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=99))
    ax_s.yaxis.get_minor_locator().set_params(numticks=50)
    ax_s.xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax_s.set_ylabel("Singular Value", fontsize=FONTSIZE_AXLABEL)
    remove_spines(ax_s)
    for v in svd_data["singular values"]:
        ax_s.axhline(v, lw=1)
    ax_s.axhline(cutoff * np.sum(s), color=COLOR_DEEMPH, lw=1, linestyle=":")
    fig.tight_layout()
    return fig


def fig_corr_eigenvectors(svd_data):
    """Return figure with parameter weights for eigenvectors of sensitivity vectors"""
    n = len(svd_data["parameters"])
    # Calculate figure dimensions
    cell_size = 0.8
    # pad_all is applied on the left and right figure margins at the very end,
    # when the figure is being resized to accommodate text
    pad_all = FONTSIZE_TICKLABEL / 2 / 72
    pad_b = pad_all * 2
    central_w = cell_size * n
    central_h = cell_size * n + pad_all * (n - 1)
    cbar_lpad = 12 / 72
    cbar_rpad = (32 + FONTSIZE_AXLABEL) / 72
    cbar_w = 0.3
    cbar_h = central_h
    pos_central_in = np.array((0, pad_b, central_w, central_h))
    pos_cbar_in = np.array((central_w + cbar_lpad, pad_b, cbar_w, cbar_h))
    fig_w = central_w + cbar_lpad + cbar_w + cbar_rpad
    fig_h = (
        pad_b
        + FONTSIZE_FIGLABEL / 72
        + FONTSIZE_AXLABEL / 72
        + 0.2
        + central_h
        + pad_all
    )
    # Prepare figure
    fig = Figure()
    fig.set_tight_layout(False)  # silence warning about tight_layout compatibility
    FigureCanvas(fig)
    fig.set_size_inches(fig_w, fig_h)
    axarr = []
    for i, parameter in enumerate(svd_data["parameters"]):  # loop is bottom to top
        # Margins will be expanded later to fit axis labels
        l = pad_all
        w = central_w
        b = pad_b + i * (cell_size + pad_all)
        h = cell_size
        ax = fig.add_axes((l / fig_w, b / fig_h, w / fig_w, h / fig_h))
        axarr.append(ax)
        for spine in ("left", "right", "top", "bottom"):
            ax.spines[spine].set_visible(False)
    axarr = axarr[::-1]  # Go top to bottom
    # Plot parameter weights that comprise each eigenvector
    cmap = mpl.cm.get_cmap("cividis")
    cnorm = mpl.colors.Normalize(vmin=-1, vmax=1)
    im = None
    for i in range(n):
        ax = axarr[i]
        v = svd_data["principal axes"][i]
        im = ax.matshow(
            np.atleast_2d(v),
            cmap=cmap,
            norm=cnorm,
            extent=(
                -0.5,
                n - 0.5,
                -0.5,
                0.5,
            ),
        )
        for j, w in enumerate(v):
            ax.text(
                j,
                0,
                "{:0.2f}".format(w),
                ha="center",
                va="center",
                backgroundcolor=(1, 1, 1, 0.5),
                fontsize=FONTSIZE_TICKLABEL,
            )
        ax.tick_params(
            axis="x",
            color=COLOR_DEEMPH,
            labelsize=FONTSIZE_TICKLABEL,
            labelcolor=COLOR_DEEMPH,
            which="both",
            bottom=False,
            top=False,
        )
        ax.tick_params(
            axis="y",
            color=COLOR_DEEMPH,
            labelsize=FONTSIZE_TICKLABEL,
            labelcolor=COLOR_DEEMPH,
        )
        ax.set_yticks([0])
        ax.set_yticklabels([i])
    # Plot colorbar
    cax = fig.add_axes(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    if im is not None:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Parameter weight", fontsize=FONTSIZE_AXLABEL)
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
    # Add labels
    axarr[0].set_title(
        "Sensitivity vectors' principal axes", fontsize=FONTSIZE_FIGLABEL
    )
    axarr[0].set_xticks(
        [i for i in range(n)],
    )
    # ^ reversed b/c origin="upper"
    axarr[0].set_xticklabels(svd_data["parameters"])
    axarr[0].tick_params(axis="x", labelsize=FONTSIZE_AXLABEL, top=True)
    # Resize figure to accommodate left axis tick labels
    ## Calculate left overflow
    bbox_in = [
        fig.dpi_scale_trans.inverted().transform(
            ax.get_tightbbox(fig.canvas.get_renderer())
        )
        for ax in axarr + [cax]
    ]
    Δleft_in = pad_all + 12 / 72 + max((0, -np.min(bbox_in, axis=0)[0][0]))
    Δright_in = max((0, np.max(bbox_in, axis=0)[0][1] - fig_w)) + pad_all
    ## Resize the canvas
    fig_w = fig_w + Δleft_in + Δright_in
    fig.set_size_inches(fig_w, fig_h)
    ## Re-apply the axes sizes, which will have changed because they are stored in
    ## figure units
    pos_central_in[0] += Δleft_in
    pos_cbar_in[0] += Δleft_in
    for ax in axarr:
        bb = ax.get_position().bounds
        bb = (pos_central_in[0] / fig_w, bb[1], central_w / fig_w, bb[3])
        ax.set_position(bb)
    cax.set_position(pos_cbar_in / [fig_w, fig_h, fig_w, fig_h])
    return fig


def fig_tsvar_pdf(
    analysis, tsdata, variable, vrange, parameter, cases, named_cases=None
):
    """Plot probability density of a time series variable into an axes"""
    # TODO: Levels information should probably be stored in the analysis object
    levels = sorted(np.unique(cases[parameter.name]))
    x = np.linspace(vrange[0], vrange[1], 100)
    steps = tsdata["Step"].unique()
    # Calculate the number of subplots
    n_plots = len(levels) + 1 if named_cases is not None else len(levels)
    nh = math.floor(n_plots**0.5)
    nw = math.ceil(n_plots / nh)
    # Create figure
    fig = Figure(constrained_layout=True)
    fig.set_size_inches((5 * nw + 1, 3 * nh + 0.25))  # TODO: set smart size
    fig.set_constrained_layout_pads(
        wspace=0.04, hspace=0.04, w_pad=2 / 72, h_pad=2 / 72
    )
    gs = GridSpec(nh, nw, figure=fig)
    axs = []
    # Plot the named case(s)
    # TODO: This is duplicated with makefig_tsvar_line
    if named_cases is not None:
        ax = fig.add_subplot(gs[-1, -1])
        axs.append(ax)
        plot_tsvar_named(analysis, variable, parameter, named_cases, ax)
        ax.set_ylim([vrange[0], vrange[1]])
    # For each level of the subject parameter, plot the time series variable's
    # probability density
    tsdata_by_case = tsdata.set_index("Case")
    for i, level in enumerate(levels):
        ax = fig.add_subplot(gs[i // nw, i % nw])
        axs.append(ax)
        stratum = (
            tsdata_by_case.loc[
                cases[cases[parameter.name] == level].index
            ].reset_index()
        ).set_index("Step")
        p = np.full((len(x), len(steps)), np.nan)
        for step in steps:
            v = stratum.loc[[step]][f"{variable} [var]"].array
            # ^ wrap index in list literal so we always get a Series object

            # Use sklearn.neighbors.KernelDensity because it's robust to zero
            # variance.  scipy.stats.gaussian_kde is not; it tries to invert a matrix
            # that will be singular if all observations have the same value.  Note
            # that sklearn wants 2D arrays; index 0 across observations and index 1
            # across features.
            kde = KernelDensity(
                kernel="gaussian", bandwidth=(vrange[1] - vrange[0]) / len(x)
            ).fit(v[:, None])
            p[:, step] = np.exp(kde.score_samples(x[:, None]))
        cmap = mpl.cm.get_cmap("cividis")
        im = ax.imshow(
            np.atleast_2d(p),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            extent=(-0.5, len(steps) + 0.5, vrange[0], vrange[1]),
        )
        cbar = fig.colorbar(im)
        cbar.set_label("Probability Density [1]", fontsize=FONTSIZE_TICKLABEL)
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKLABEL)
        # Labels
        s_level = (
            f"{level:3g}"
            if parameter.units == "1"
            else f"{level:3g} {parameter.units:~P}"
        )
        ax.set_title(
            f"{parameter.name} = {s_level}",
            fontsize=FONTSIZE_AXLABEL,
        )
        ax.set_xlabel("Time point [1]")
        ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
        # TODO: Units-awareness for variables
        ax.set_ylabel(variable, fontsize=FONTSIZE_AXLABEL)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
    # Finalize the figure
    fig.suptitle(
        f"{variable} time series vs. {parameter.name}", fontsize=FONTSIZE_FIGLABEL
    )
    return fig, axs


def tabulate_case(case):
    """Tabulate variables for a single analysis case."""
    case_tree = read_xml(case.feb_file)
    # Read the text data files
    # TODO: Finding the text data XML elements should probably be done in Waffleiron.  Need to move this function to spamneggs.py so we can annotate the type of the arguments.
    text_data = {}
    for entity_type in ("node", "element", "body", "connector"):
        tagname = TAG_FOR_ENTITY_TYPE[entity_type]
        fname = case.feb_file.with_suffix("").name + f"_-_{tagname}.txt"
        pth = case.feb_file.parent / fname
        e_textdata = case_tree.find(f"Output/logfile/{tagname}[@file='{fname}']")
        if e_textdata is not None:
            text_data[entity_type] = textdata_table(pth)
        else:
            text_data[entity_type] = None
    # Extract values for each variable based on its <var> element
    record = {"instantaneous variables": {}, "time series variables": {}}
    for varname, var in case.variables_list.items():
        if isinstance(var, XpltDataSelector):
            xplt_data = case.solution().solution
            # Check component selector validity
            dtype = xplt_data.variables[var.variable]["type"]
            oneids = (str(i + 1) for i in var.component)
            if dtype == "float" and len(var.component) != 0:
                msg = f"{var.variable}[{', '.join(oneids)}] requested), but {var.variable} has xplt dtype='{dtype}' and so has 0 dimensions."
                raise ValueError(msg)
            elif dtype == "vec3f" and len(var.component) != 1:
                msg = f"{var.variable}[{', '.join(oneids)}] requested), but {var.variable} has xplt dtype='{dtype}' and so has 1 dimension."
                raise ValueError(msg)
            elif dtype == "mat3fs" and len(var.component) != 2:
                msg = f"{var.variable}[{', '.join(oneids)}] requested), but {var.variable} has xplt dtype='{dtype}' and so has 2 dimensions."
                raise ValueError(msg)
            # Get ID for region selector
            if var.region is None:
                region_id = None
            else:
                region_id = var.region["ID"]
            # Get ID for parent selector
            if var.parent_entity is None:
                parent_id = None
            else:
                parent_id = var.parent_entity["ID"]
            if var.temporality == "instantaneous":
                # Convert time selector to step selector
                if var.time_unit == "time":
                    step = find_closest_timestep(
                        var.time,
                        xplt_data.step_times,
                        np.array(range(len(xplt_data.step_times))),
                    )
                value = xplt_data.value(
                    var.variable, step, var.entity_id, region_id, parent_id
                )
                # Apply component selector
                for c in var.component:
                    value = value[c]
                # Save selected value
                record["instantaneous variables"][varname] = {"value": value}
            elif var.temporality == "time series":
                data = xplt_data.values(
                    var.variable,
                    entity_id=var.entity_id,
                    region_id=region_id,
                    parent_id=parent_id,
                )
                values = np.moveaxis(np.array(data[var.variable]), 0, -1)
                for c in var.component:
                    values = values[c]
                record["time series variables"][varname] = {
                    "times": xplt_data.step_times,
                    "steps": np.array(range(len(xplt_data.step_times))),
                    "values": values,
                }
        elif isinstance(var, TextDataSelector):
            # Apply entity type selector
            tab = text_data[var.entity]
            # Apply entity ID selector
            tab = tab[tab["Item"] == var.entity_id]
            tab = tab.set_index("Step")
            if var.temporality == "instantaneous":
                # Convert time selector to step selector.  The selector
                # has internal validation, so we can assume the time
                # unit is valid.
                if var.time_unit == "time":
                    step = find_closest_timestep(var.time, tab["Time"], tab.index)
                    if step == 0:
                        raise ValueError(
                            "Data for step = 0 requested from an FEBio text data output file, but FEBio does not provide text data output for step = 0 / time = 0."
                        )
                    if step not in tab.index:
                        raise ValueError(
                            f"A value for {var.variable} was requested for step = {step}, but this step is not present in the FEBio text data output."
                        )
                elif var.time_unit == "step":
                    step = var.time
                # Apply variable name and time selector
                value = tab[var.variable].loc[step]
                record["instantaneous variables"][varname] = {"value": value}
            elif var.temporality == "time series":
                record["time series variables"][varname] = {
                    "times": tab["Time"].values,
                    "steps": tab.index.values,
                    "values": tab[var.variable].values,
                }
        elif isinstance(var, FunctionVar):
            v = var(case)
            if not isinstance(v, TimeSeries):
                record["instantaneous variables"][varname] = {"value": v}
            else:
                record["time series variables"][varname] = {
                    "times": v.time,
                    "steps": v.step,
                    "values": v.value,
                }
        else:
            raise ValueError(f"{var} not supported as a variable type.")
    # Assemble the time series table
    tab_timeseries = {"Time": [], "Step": []}
    for var, d in record["time series variables"].items():
        tab_timeseries["Time"] = d["times"]
        tab_timeseries["Step"] = d["steps"]
        tab_timeseries[var] = d["values"]
    tab_timeseries = pd.DataFrame(tab_timeseries)
    return record, tab_timeseries
