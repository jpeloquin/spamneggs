import os
from enum import Enum
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
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Third-party packages
import scipy.cluster
from lxml import etree
import matplotlib as mpl
import matplotlib.pyplot as plt
from lxml.etree import ElementTree
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
import pandas as pd
from pandas import CategoricalDtype, DataFrame, IndexSlice
from pathos.pools import ThreadPool, ProcessPool
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


NUM_WORKERS_DEFAULT = psutil.cpu_count(logical=False)


class OnSimError(Enum):
    """Action when processing a simulation returns an error

    Typical examples of processing are (1) generating a simulation's file and (2)
    running a simulation.

    """

    STOP = 1
    HOLD = 2
    CONTINUE = 3


class CaseGenerationError(Exception):
    """Raise when case generation terminates in an error."""

    pass


class UnconstrainedTimesError(Exception):
    """Raise when simulation does not fully define its output time points"""

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


class EmptySelectionError(Exception):
    """Raise when a selection from a DataFrame would be empty

    Seems more reliable than returning None.  It would be better to return a zero-row
    DataFrame but pandas makes that surprisingly hard.  I haven't figured out to add
    an index to an empty DataFrame.

    """

    pass


class Analysis:
    """Class storing analysis details for use after case generation"""

    # TODO: Write JSON or XML serialization
    def __init__(
        self,
        name,
        parameters: Sequence[Parameter],
        variables: Dict[str, Tuple[str, str]],
        generators: Iterable[Union[str, "SimGenerator"]],
        parentdir: Optional[Union[Path, str]] = None,
        mkdir=False,
    ):
        """Return Analysis object

        :param generators: Simulation generators, which turn samples into concrete
        simulations.  To allow an existing analysis to be loaded, the generators can
        be provided as strings.

        :param parentdir: Parent directory for the analysis.  The analysis' files are
        stored in parentdir / analysis.name.

        """
        self.name = name
        self.parameters = _init_parameters(parameters)
        self.variables = variables

        # Directories
        if parentdir is None:
            parentdir = Path(os.getcwd())
        else:
            parentdir = Path(parentdir)
        # Directory in which to write files from this analysis
        self.directory = parentdir / name
        if mkdir:
            self.directory.mkdir(exist_ok=True)

        # Generators
        self._generators = list(generators)
        for nm, g in zip(self.generator_names, self.generators):
            dir_ = self.directory / "sims" / nm
            if mkdir:
                dir_.mkdir(exist_ok=True, parents=True)
                # TODO: Modifying the generator is not ideal
                if g is not None:
                    g.directory = dir_

    @classmethod
    def from_generators(
        cls,
        name,
        generators: Sequence["SimGenerator"],
        parentdir: Optional[Union[Path, str]] = None,
        mkdir=True,
    ):
        parameters = generators[0].parameters
        variables = {}
        for g in generators:
            if parameters != g.parameters:
                raise ValueError(
                    f"SimGenerator '{g.name}' has parameters inconsistent with at least one other provided generator."
                )
            for vname, v in g.variables.items():
                if vname in variables:
                    raise ValueError(
                        f"{vname} is defined twice.  The second instance was encountered in generator '{g.name}'."
                    )
                # TODO: Create a data type to store a variable definition (units,
                # temporality), but not its implementation, which should remain on the
                # SimGenerator and Sim objects.
                variables[vname] = (g.name, v.temporality)
        return cls(name, parameters, variables, generators, parentdir, mkdir)

    @property
    def generators(self):
        """Return list of SimGenerator objects if available"""
        return [g if not isinstance(g, str) else None for g in self._generators]

    @property
    def generator_names(self):
        """Return list of generator names

        This function is useful because the generators may be provided as strings
        rather than objects.

        """
        return [g if isinstance(g, str) else g.name for g in self._generators]

    def complete_samples(self):
        """Return sample IDs for which all sims were successful"""
        good_sims = self.sims_table[self.sims_table["Status"] == "Run: Success"]
        n = good_sims.groupby(["Group", "Sample"]).size().rename("Size").reset_index()
        good_samples = n[n["Size"] == len(self.generators)].set_index("Group")
        return {g: tab["Sample"].values for g, tab in good_samples.groupby("Group")}

    @cached_property
    def sample_ids(self):
        return self.sims_table.index.unique("Sample")

    @cached_property
    def samples_table(self, group=None):
        """Return table of samples"""
        if group is None:
            group = slice(None)
        samples = self.sims_table.loc[IndexSlice[group, self.generators[0].name, :]]
        samples = samples.rename(
            {p.name: f"{p.name} [param]" for p in self.parameters}, axis=1
        )
        return samples

    def sims(self):
        info = self.sims_table
        generators = {g.name: g for g in self.generators}
        for (group, generator_name, sample_id), r in info.iterrows():
            sim = Sim(
                name=sample_id,
                parameters=self.parameters,
                parameter_values={p.name: r[p.name] for p in self.parameters},
                variables=self.variables,
                directory=self.directory / "sims" / generator_name / group,
                checks=generators[generator_name].checks,
            )
            yield sim

    @cached_property
    def sims_table(self):
        """Return table of simulations"""
        return pd.read_csv(
            self.directory / "simulations.csv", index_col=None
        ).set_index(["Group", "Generator", "Sample"])

    def named_sims_table(self, sample_ids=None):
        if sample_ids is None:
            sample_ids = slice(None)
        if "named" in self.sims_table.index.unique("Group"):
            return self.sims_table.loc[("named", slice(None), sample_ids), :].droplevel(
                "Group"
            )
        else:
            return None

    def sampled_sims_table(self, sample_ids=None):
        if sample_ids is None:
            sample_ids = slice(None)
        if "sampled" in self.sims_table.index.unique("Group"):
            return self.sims_table.loc[
                ("sampled", slice(None), sample_ids), :
            ].droplevel("Group")
        else:
            return None

    @property
    def instantaneous_variables(self):
        # TODO: make temporality not stringly-typed
        return [v for v, (g, temp) in self.variables.items() if temp == "instantaneous"]

    @property
    def timeseries_variables(self):
        # TODO: make temporality not stringly-typed
        return [v for v, (g, temp) in self.variables.items() if temp == "time series"]

    def sim_data(self, generator, sample_id):
        """Read variables from a single simulation

        Named cases have string IDs; automatically generated cases have integer IDs.

        """
        # As of 2023-11-17, sim_data() was only used to access data for named samples.
        if isinstance(sample_id, str):
            row = self.named_sims_table().loc[generator, sample_id]
        elif isinstance(sample_id, int):
            row = self.sampled_sims_table().loc[generator, sample_id]
        else:
            raise TypeError(
                f"Sim ID must be string or integer.  Was {type(sample_id)}."
            )
        if row["Status"] != "Run: Success":
            raise ValueError(f"Simulation {sample_id} has status {row['Status']}.")
        pth_record = self.directory / "output" / generator / f"{sample_id}_vars.json"
        pth_timeseries = (
            self.directory / "output" / generator / f"{sample_id}_timeseries_vars.csv"
        )
        with open(pth_record, "r") as f:
            record = json.load(f)
        timeseries = pd.read_csv(pth_timeseries, index_col=False)
        return record, timeseries

    def samples_tsdata(self, generator: str, group, samples=None):
        """Return table of a generator's time series data for all samples

        `samples_tsdata` combines variables from all of a sample's simulations.

        :param generator: Generator name for which to return data.

        :param group: Group to select samples from.  Required because sample IDs may not
        be unique across groups, and even if they are they may have different types and
        combining them in one pandas DataFrame will cause type coercion.

        :param samples: Optional sequence of sample IDs.  Only data from the listed
        samples will be returned.  Samples with unsuccessful simulations will still
        be dropped as usual though.  If `samples` is None `samples_tsdata` will
        return time series data for all successful simulations.

        The time series table for each simulation needs to already be on disk,
        hence the restriction to use only successful simulations.

        """
        # TODO: It would be beneficial to build up the whole-analysis time series
        #  table at the same time as case time series tables are written to disk,
        #  instead of re-reading everything from disk.  That would also allow the
        #  variables to be accessed individually on demand, which would simplify the
        #  analysis code.
        if group not in self.sims_table.index.unique("Group"):
            raise EmptySelectionError(f"No simulations available from group {group}.")
        sims = self.sims_table.loc[IndexSlice[group, generator]]
        if samples is None:
            samples = sims.index.get_level_values("Sample")[
                sims["Status"] == "Run: Success"
            ]
        if len(samples) == 0:
            raise EmptySelectionError(
                f"With the given inputs, there are no valid simulations available from group {group}."
            )
        all_vars = DataFrame()
        for sample in samples:
            # It's tempting to (optionally) store NaN values for incomplete simulations
            data = pd.read_csv(
                self.directory / "output" / generator / f"{sample}_timeseries_vars.csv"
            )
            # TODO: Different generators can have different time series
            sample_data = {
                "Sample": sample,
                "Step": data["Step"],
                "Time": data["Time"],
            }
            for v, (g, _) in self.variables.items():
                if g == generator:
                    sample_data[v] = data[v].values
            sample_data = DataFrame.from_dict(sample_data)
            all_vars = pd.concat([all_vars, sample_data])
        varnames = set(all_vars.columns) - {"Sample", "Step", "Time"}
        tsdata_by_var = {var: TSVarData(var, all_vars) for var in varnames}
        return tsdata_by_var

    def write_sims_table(self, table):
        """Write simulations table to analysis directory

        File paths are written relative to the analysis directory.

        """
        table = table.copy().reset_index()  # table might be re-used elsewhere
        pth = self.directory / "simulations.csv"
        table["Path"] = [str(p.relative_to(pth.parent)) for p in table["Path"]]
        table.to_csv(pth, index=False)


class Sample:
    """Information for a sample in parameter space

    Sample is intended to work similarly to Sim but with a focus on processing data from
    an already-run analysis, rather than running the analysis.

    """

    def __init__(self):
        raise NotImplementedError


class Sim:
    # TODO: Need a separate class for simulation-as-a-data-record and
    #  simulation-as-a-runnable-object.  Proposing Sample.
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
        """Return an object for access to simulation files and data

        :param name: Name of the case.  The simulation file should be "{name}.feb", and
        any dependency simulations should be in a companion directory with this name.

        :param parameter_values: Parameter values corresponding to this case.  If the
        values are Quantity objects, they will be checked for compatibility with the
        units given in `parameters`.  If the values are plain numbers, they will be assumed to
        have the units given in `parameters`.

        :param variables: Variables to read from this case's solution, provided as
        spamneggs objects.

        :param checks: Check functions.

        :param directory: Parent directory to which the case's files will be written.

        """
        self.parameter_list = _init_parameters(
            parameters
        )  # need to store parameter *order*
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name},"
            f"parameters={self.parameter_list}"
            f"parameter_values={self.parameters_q},"
            f"variables={self.variables_list},"
            f"directory={self.directory},"
            f"checks={self.checks})"
        )

    def __str__(self):
        return f"Sim {self.name}"

    @property
    def feb_file(self):
        return self.directory / f"{self.name}.feb"

    @property
    def log_file(self):
        return self.directory / f"{self.name}.log"

    @property
    def xplt_file(self):
        return self.directory / f"{self.name}.xplt"

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
                    f"Sim {self.name} has failed checks: {', '.join([str(e) for e in check_failures])}"
                )
            return check_failures
        return SUCCESS

    def solution(self):
        """Return waffleiron Model object with solution"""
        model = wfl.load_model(self.feb_file)
        if model.solution is None:
            raise ValueError(f"Sim {self.name}: Model has no solution")
        return model


class SimGenerator:
    def __init__(
        self,
        fun: Callable[[Sim], ElementTree],
        parameters: Sequence[Union[Parameter, Tuple[str, Union[str, None]]]],
        variables: dict,
        name,
        checks: Sequence[Callable] = tuple(),
        parentdir: Optional[Path] = None,
    ):
        """Return a SimGenerator object

        :param fun: Function that accepts a Case object and returns an FEBio XML tree.

        :param name: Name of the case generator.  Used as the name of the directory that
        will contain all generated files.

        :param checks: Sequence of callables that take a waffleiron Model as their lone
        argument and should raise an exception if the check fails, or return None if the
        check succeeds.  This is meant for user-defined verification of simulation
        output.

        :param parentdir: The parent directory for the generator directory.  If the
        SimGenerator will be used with an Analysis, you do not need to set this
        unless you want to send the generated files elsewhere.  If the SimGenerator
        is used outside an Analysis and `parentdir` is None, simulation files will be
        written to the current working directory.

        """
        self.fun = fun
        self.parameters = _init_parameters(parameters)
        self.variables: dict = variables
        self.checks = checks
        self.name = name
        self._directory = None
        if parentdir is not None:
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

    def define_sim(self, name, parameter_values, directory=None) -> Sim:
        """Return a Sim object without generating backing files"""
        if directory is None:
            # self.directory should already exist, created by whomever instantiated
            # the SimGenerator
            directory = self.directory
        sim = Sim(
            name,
            self.parameters,
            parameter_values,
            self.variables,
            directory,
            checks=self.checks,
        )
        return sim

    def generate_sim(self, sim):
        """Generate backing files for a simulation"""
        sim.directory.mkdir(exist_ok=True)
        tree = self.fun(sim)
        with open(sim.feb_file, "wb") as f:
            wfl.output.write_xml(tree, f)

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, dir_):
        self._directory = Path(dir_)


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

    def __call__(self, case: Sim):
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
            tree, logfile_reqs, plotfile_reqs, file_stem=f"{case.name}"
        )
        return tree


class TSVarData:
    """Store timeseries variable values across all samples"""

    # pandas.Series would work if it wasn't necessary to store the time vector

    def __init__(self, var: str, data: DataFrame):
        """Return TSVarData object

        :param var: Variable name.

        :param data: DataFrame with at least the following four columns: Sample,
        Step, Time, and the variable's name.

        """
        self.variable = var
        self.data = (
            data[["Sample", "Step", "Time", var]].copy().set_index(["Sample", "Step"])
        )
        self.samples = sorted(self.data.index.unique("Sample").values)
        self.steps = sorted(self.data.index.unique("Step").values)
        self.time = sorted(pd.unique(self.data["Time"]))
        if np.any(pd.isnull(self.steps)):
            raise ValueError("Some steps are null/NA.")
        if np.any(pd.isnull(self.time)):
            raise ValueError("Some times are null/NA.")
        if not len(self.steps) == len(self.time):
            raise ValueError(
                "Number of unique time points does not match the number of steps."
            )

    def __str__(self):
        return f"TSVarData('{self.variable}')"

    def values(self, sample_id=None):
        if sample_id is None:
            return self.data[self.variable]
        else:
            return self.data.loc[sample_id][self.variable]


def _default_sample_ids(analysis):
    sample_ids = {"sampled": [], "named": []}
    for g, ids in analysis.complete_samples().items():
        sample_ids[g] = ids
    if len(sample_ids["sampled"]) == 0:
        raise Exception("No sample IDs with (successfully) completed simulations.")
    return sample_ids


def _merge_tsdata_params(analysis, tstable):
    """Merge parameter values into time series variable table"""
    data = pd.merge(
        tstable.rename(
            {v: f"{v} [var]" for v in analysis.variables}, axis=1
        ).reset_index("Step"),
        analysis.samples_table.drop(["Status", "Path"], axis=1),
        on="Sample",
    )
    return data


def _update_table_status(table: DataFrame, output, step=None):
    """Transfer sim status from do_parallel to a sims table

    :param table: Table one column per parameter and a "Status" column.  The index
    must be set such that the row IDs match the tags in `output`.

    """
    if step is not None:
        prefix = f"{step}: "
    else:
        prefix = ""
    for id_, err in output:
        if err == SUCCESS:
            msg = f"{prefix}{err}"
        else:
            # The error might be a list of errors if it came from the custom
            # simulation checks.
            try:
                errors = iter(err)
            except TypeError:
                errors = [err]
            msg = f"{prefix}{', '.join(e.__class__.__name__ for e in errors)}"
        table.loc[id_, "Status"] = msg


def _init_parameters(parameters):
    """Return tuple of Parameter objects"""
    return tuple(p if isinstance(p, Parameter) else Parameter(*p) for p in parameters)


def _ordered_parameter_subset(
    use_parameters: Collection, original_parameters: Sequence
):
    return [nm for nm in use_parameters if nm in original_parameters]


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
    # We may be working with a subset of the correlation matrix, so cannot use
    # analysis.parameters as-is
    parameters = _ordered_parameter_subset(
        pd.unique(correlations["Parameter"]), [p.name for p in analysis.parameters]
    )
    by_parameter = (
        correlations.set_index(["Parameter", "Variable", "Time Point"])
        .unstack(["Variable", "Time Point"])
        .loc[parameters]
    )
    # ^ keep same parameter order
    arr = by_parameter.values
    # ^ first index over parameters, second over variables and time points

    # Only consider non-nan values when calculating correlation distances
    m_nonnan = ~np.any(np.isnan(arr), axis=0)
    m_finite = ~np.any(np.isinf(arr), axis=0)
    arr_valid = arr[:, np.logical_and(m_nonnan, m_finite)]

    # Compute unsigned cosine correlation distance = 1 - | ρ | ∈ [0, 1] where ρ = u *
    # v / ( 2-norm(u) * 2-norm(v) ). If the 2-norm of u or v is zero, then the result
    # will be undefined.  Typically, this will happen when u or v is all zeroes; in
    # this case, the numerator is also zero.  For the current application,
    # it is reasonable to define 0/0 = 0.  Therefore, we need to check for u * v = 0
    # and set the distance for those vector pairs to 1.
    distances = 1 - abs(scipy.spatial.distance.pdist(arr_valid, metric="cosine") - 1)
    n = len(arr)  # number of parameters
    numerator = np.empty(len(distances))
    numerator[:] = np.nan  # make indexing errors more obvious
    means = np.mean(arr_valid, axis=1)
    for i in range(n):  # index of u
        for j in range(i + 1, n):  # index of v
            idx = (
                scipy.special.comb(n, 2, exact=True)
                - scipy.special.comb(n - i, 2, exact=True)
                + (j - i - 1)
            )
            numerator[idx] = (arr_valid[i] - means[i]) @ (arr_valid[j] - means[j])
    distances[numerator == 0] = 1
    return distances


def sims_table(analysis: Analysis, sims):
    """Return table of Analysis Sims

    :param analysis: Analysis object
    :param sims: Iterable of (tag, Sim) tuples.

    """
    # TODO: Figure out same way to demarcate parameters from other
    # metadata so there are no reserved parameter names.  For example, a
    # user should be able to name their parameter "path" without
    # conflicts.
    #
    # Create column names.  Do this explicitly so if there is no output to process we
    # still get a valid (empty) table.
    data = {"Group": [], "Generator": [], "Sample": []}
    for p in analysis.parameters:
        data[p.name] = []
    data["Status"] = []
    data["Path"] = []
    # For each sim's output, add a row
    for (group, generator, ix), sim in sims:
        # Sample
        data["Group"].append(group)
        data["Generator"].append(generator)
        data["Sample"].append(ix)
        # Parameters
        for p in analysis.parameters:
            data[p.name].append(sim.parameters_n[p.name])
        # Status
        data["Status"].append("")
        # File path
        data["Path"].append(sim.feb_file)
    tab = DataFrame.from_dict(data).set_index(["Group", "Generator", "Sample"])
    return tab


def do_parallel(
    sims: Iterable[Tuple[object, Sim]], fun: Callable, on_error: OnSimError, pool
):
    """Apply function to simulations in parallel

    :param sims: Iterable of (tag, Sim) tuples.  The tag is some kind of identifier
    to disambiguate Sims when multiple generators are in use and Sim.name may not be
    unique.

    :param fun: Function to apply to each Sim.

        Idiomatically, the function returns a (tag, status) tuple, where the status
        is `spam.Success`, a `spamneggs.CaseGenerationError`,
        a `waffleiron.febio.CheckError`, or a derived class of the above.  The tag is
        included so that the functions may be evaluated out of order without issue.

    :param on_error: Action to take on an exception. OnSimError.STOP = immediately
    stop processing.  OnSimError.HOLD and OnSimError.CONTINUE = continue processing
    and attempt to process all simulations.

    :return: List of (tag, fun return value) tuples.

    """
    output = pool.map(lambda x: fun(*x), sims)
    for tag, status in output:
        # Should we continue submitting cases?
        if status != SUCCESS and on_error == OnSimError.STOP:
            print(
                f"While working on `{tag}`, a {status.__class__.__name__} was returned.  Because `on_error` = OnSimError.{on_error.name}, do_parallel will allow each currently running case to finish, then stop.\n"
            )
            pool.close()
            break
    return output


def expand_run_errors(cases: DataFrame, group="sampled"):
    """Return cases table with run error columns instead of one status column"""
    out = cases.loc[IndexSlice[group, :, :]].copy().reset_index()

    errors = defaultdict(lambda: np.zeros(len(out), dtype=bool))
    run_status = np.full(len(out), None)
    for ix in range(len(out)):
        status = out["Status"].iloc[ix]
        i = status.find(":")
        phase = status[:i]
        codes = [s.strip() for s in status[i + 1 :].split(",")]
        if phase != "Run":
            run_status[ix] = "No Run"
        elif "Success" in codes:
            run_status[ix] = "Success"
            if len(codes) > 1:
                raise ValueError(
                    f"Sim {(out['Sample'].iloc[ix], out['Generator'].iloc[ix])}: Run was recorded as successful but additional error codes were present.  The codes were {', '.join(codes)}."
                )
        else:
            run_status[ix] = "Error"
        for code in codes:
            if code == "Success":
                continue
            errors[code][ix] = True
    # Return cases table augmented with run status and error columns
    out["Run Status"] = pd.Series(
        run_status, dtype=CategoricalDtype(["Success", "Error", "No Run"])
    )
    for error in errors:
        out[error] = errors[error]
    return out, tuple(errors.keys())


def import_cases(generator: SimGenerator):
    """Recheck all auto-generated cases in analysis and update solution status

    This function is meant to be used when you're manually moving simulation files
    between computers.

    """
    # TODO: Handle named cases
    analysis = Analysis(generator)
    table = analysis.table_sampled_sims
    for i in table.index:
        pth_feb = generator.directory / table.loc[i, "path"]
        # Check if there is already a run message; do not overwrite it.  Once
        # `import_cases` runs the same suite of checks that the original run did,
        # we can overwrite it.
        if table.loc[i, "status"].startswith("Run: "):
            continue
        # Check if the model file is even there
        if not pth_feb.exists():
            table.loc[i, "status"] = "Missing"
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
            table.loc[i, "Status"] = f"Run: {SUCCESS}"
        else:
            table.loc[i, "Status"] = f"Run: {log.termination} termination"
    analysis.write_sims_table(table)


def _format_fname(s):
    return s.replace(" ", "_")


def run_sim(tag, sim):
    """Simulate a sample and check its output as part of a sensitivity analysis

    This function is meant to be used with parallel execution in which the results
    may be returned out of order, which is why the tag is included in the return value.

    """
    try:
        status = sim.run(raise_on_check_fail=False)
    except FEBioError as e:
        status = [e]
    if status != SUCCESS:
        for e in status:
            print(f"Sim {tag}: {e!r}")
    return tag, status


def run_sensitivity(
    analysis: Analysis,
    distributions,
    nlevels,
    named_levels={},
    on_error=OnSimError.STOP,
    num_workers=NUM_WORKERS_DEFAULT,
):
    """Run a sensitivity analysis from an analysis object

    :param on_error: Action to take when generating a simulation's files raises a
    CaseGenerationError, or running a simulation raises .

    """
    # ProcessPool allows parallel model generation but is worse than ThreadPool for
    # debugging.  ThreadPool is fine when the parallel work is all in FEBio.
    pool = ProcessPool(nodes=num_workers)
    # Set waffleiron to run FEBio with only 1 thread, since we'll be running one
    # FEBio process per core
    wfl.febio.FEBIO_THREADS_DEFAULT = 1

    # Define samples
    named = {
        name: cleanup_levels_units(levels, analysis.parameters)
        for name, levels in named_levels.items()
    }
    sampled = full_factorial_values(analysis.parameters, distributions, nlevels)

    # Define sims
    generators = {g.name: g for g in analysis.generators}
    sims = [
        (
            (group, g.name, ix),
            g.define_sim(ix, sample, directory=g.directory / group),
        )
        for group, samples in (
            ("named", named),
            ("sampled", sampled),
        )
        for ix, sample in samples.items()
        for g in analysis.generators
    ]

    # Generate named sims
    def f_gen(tag, sim):
        generator = generators[tag[1]]
        try:
            generator.generate_sim(sim)
        except CaseGenerationError as err:
            print(f"Sim {tag}:\n")
            traceback.print_exc()
            print()
            return tag, err
        return tag, SUCCESS

    output = do_parallel(sims, f_gen, on_error=on_error, pool=pool)
    table = sims_table(analysis, sims)
    _update_table_status(table, output, step="Generate")
    analysis.write_sims_table(table)

    # Run the cases
    output = do_parallel(sims, run_sim, on_error=on_error, pool=pool)
    _update_table_status(table, output, step="Run")
    analysis.write_sims_table(table)

    # Check if error terminations prevent analysis of results
    table = table.reset_index()
    for group, tab in table.groupby("Group"):
        n_run_error = np.sum(tab["Status"] != "Run: Success")
        n_success = np.sum(tab["Status"] == "Run: Success")
        n_not_run = int(np.sum([s.startswith("Generate") for s in tab["Status"]]))
        error_report = (
            f"\nOf {len(tab)} {group} simulations:\n"
            f"{n_success} simulations ran successfully\n"
            f"{n_run_error} simulations had a run error\n"
            f"{n_not_run} simulations did not have a run attempt\n"
            f"See {analysis.directory / 'simulations.csv'}"
        )
        print(error_report)
        if n_run_error != 0:
            if on_error == OnSimError.HOLD:
                print(
                    f"Because there at least one simulation had an error and `on_case_error` = 'hold', simulation was attempted for all cases but the sensitivity analysis was stopped prior to data analysis.  To continue with the sensitivity analysis, manually correct the error terminations (or not) and call `tabulate` and `plot_sensitivity`."
                )
                sys.exit()
            elif on_error == OnSimError.STOP:
                print(
                    f"Because there was at least one simulation had an error and `on_case_error` = 'stop', the sensitivity analysis was stopped after the first simulation error, before running all cases."
                )
                sys.exit()
    # Tabulate and plot the results
    tabulate(analysis)
    makefig_sensitivity_all(analysis)


def get_rep_sample(analysis):
    """Return time series variables for a representative specimen"""
    if "nominal" in analysis.sample_ids:
        return "named", "nominal"
    else:
        # Return values for median generated case, if possible
        sims = analysis.sims_table
        # TODO: It should be easier to get the levels of an analysis, or the median
        #  should be calculated based on the variables rather than the parameters.
        param_values = {k: sims[k.name] for k in analysis.parameters}
        param_nlevels = {k: len(np.unique(v)) for k, v in param_values.items()}
        if any([n % 2 == 0 for n in param_nlevels.values()]):
            # The median parameter values are not represented in the cases
            candidates = sims[sims["Status"] == f"Run: {SUCCESS}"]
            sample_id = candidates.index.get_level_values("Sample")[
                len(candidates) // 2
            ]
        else:
            median_levels = {
                k.name: np.median(values) for k, values in param_values.items()
            }
            m = np.ones(len(analysis.samples_table), dtype="bool")
            for param, med in median_levels.items():
                m = np.logical_and(m, analysis.samples_table[f"{param} [param]"] == med)
            assert np.sum(m) == 1
            sample_id = analysis.samples_table.index.get_level_values("Sample")[m][0]
        return "sampled", sample_id


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


def tab_tsvars_corrmap(analysis, tsdata: Dict[str, TSVarData], fun_corr=corr_pearson):
    """Return table of time series vs. parameter correlation vectors"""
    # Calculate sensitivity vectors
    corr_vectors = {
        (p.name, v): np.zeros(len(tsdata[v].steps))
        for p in analysis.parameters
        for v in analysis.variables
    }
    for v in analysis.variables:
        if np.any(np.isnan(tsdata[v].values())):
            raise ValueError(f"NaNs detected in values for {v}.")
    for v in analysis.variables:
        for p in analysis.parameters:
            for i in tsdata[v].steps:
                data = _merge_tsdata_params(analysis, tsdata[v].data).set_index(
                    "Step", append=True
                )
                ρ = fun_corr(
                    data.xs(i, level="Step"), f"{p.name} [param]", f"{v} [var]"
                )
                corr_vectors[(p.name, v)][i] = ρ
    correlations = DataFrame(corr_vectors).stack(future_stack=True)
    correlations.index.set_names(["Time Point", "Variable"], inplace=True)
    correlations = correlations.melt(
        ignore_index=False, var_name="Parameter", value_name="Correlation Coefficient"
    ).reset_index()
    return correlations


def makefig_sensitivity_all(
    analysis: Analysis, sample_ids: Optional[Dict[str, Sequence]] = None, tsfilter=None
):
    """Make all sensitivity analysis figures

    :param analysis: Analysis object, which provides access to simulation data for
    all samples.

    :param sample_ids: Restrict the analysis to the provided groups and sample IDs.
    By default, every sample with a complete set of successful simulations is used.

    :param tsfilter: Function that takes a timeseries data table, modifies it, and
    returns it.

    """
    if sample_ids is None:
        sample_ids = _default_sample_ids(analysis)

    # Sensitivity cases
    cases = analysis.sampled_sims_table().loc[IndexSlice[:, sample_ids["sampled"]], :]

    # Read parameters
    param_values = {p.name: cases[p.name] for p in analysis.parameters}

    # Plots for errors
    makefig_error_counts(analysis)
    makefig_error_pdf_uniparam(analysis)
    makefig_error_pdf_biparam(analysis)

    # Plots for instantaneous variables
    ivar_values = defaultdict(list)
    if len(analysis.instantaneous_variables) > 0:
        # Skip doing the work of reading each case's data if there are
        # no instantaneous variables to tabulate
        for i in sample_ids:
            # TODO: There is an opportunity for improvement here: We could preserve
            #  the tree from the first read of the analysis XML and re-use it here
            #  instead of re-reading the analysis XML for every case.  However,
            #  to do this the case generation must not alter the analysis XML tree.
            for nm in analysis.instantaneous_variables:
                generator = analysis.instantaneous_variables[nm][0]
                record, _ = analysis.sim_data(generator, i)
                ivar_values[nm].append(record["instantaneous variables"][nm]["value"])
        makefig_sensitivity_ivar_all(analysis, param_values, ivar_values)

    # Summarize time series variables and collect time series data.  Unfortunately,
    # if we put both data for both named and sampled samples in the same table,
    # Pandas will coerce the integer sampled IDs to strings, which breaks .loc[].  So
    # we have to keep them separate, or only tabulate the sampled values and access
    # the named values on an ad-hoc basis.  Ad-hoc access is not a good idea because
    # it will be hard to consistently apply `tsfilter`.
    tsvar_data = {group: {} for group in sample_ids}  # group ∈ {"named", "sampled"}
    for group in tsvar_data:
        for g in analysis.generators:
            try:
                tsvar_data[group].update(
                    analysis.samples_tsdata(g.name, group, sample_ids[group])
                )
            except EmptySelectionError:
                continue
    # Filter desired times
    if tsfilter is not None:
        for group in sample_ids:
            for v in analysis.timeseries_variables:
                tsvar_data[group][v] = tsfilter(tsvar_data[v])
    # Summarize sampled time series variable's data in a table
    for v in analysis.timeseries_variables:
        tsvar_summary = tsvar_data["sampled"][v].values().groupby("Step").describe()
        tsvar_summary.to_csv(
            analysis.directory / f"sampled_-_tsvar_{_format_fname(v)}_summary.csv",
            index=True,
        )
    # Plots for time series variables
    makefig_sensitivity_tsvar_all(analysis, tsvar_data, sample_ids=sample_ids)


def makefig_global_correlations_tsvars(
    analysis: Analysis,
    rep_tsvalues: DataFrame,
    correlations: DataFrame,
    directory: Path,
    fname_prefix: str = "",
):
    """Make plots for global correlation vector analysis of time series variables

    The global identifiability analysis should already have been run and their results
    tabulated, usually by `run_sensitivity`.  Optionally, some parameters can be
    excluded to approximate the effect of assuming values for some parameters in an
    attempt to restore identifiability of a subset of parameters for a flawed
    experimental design.

    :param analysis: Analysis object, which provides access to simulation data for
    all samples.

    :param rep_tsvalues: Time series data for a representative sample, with index
    "Step", column "Time", and one column per time series variable.

    :param correlations: Table of correlation coefficient values.  Can be read from
    tsvar_param_corr={corr_type}.feather, where corr_type is "pearson", "spearman",
    or "partial".  Any filtering of samples, time points, or parameters should be
    applied to this table before passing it to `makefig_global_correlations_tsvars`.
    Expected columns: "Time Point" (same as "Step" in `rep_tsvalues`), "Variable",
    "Parameter", "Correlation Coefficient".

    :param directory: Directory for the results.  Must already exist.

    """
    # TODO: Use consistent "Step" or "Time Point", preferable the latter.
    # We may be working with a subset of the correlation matrix, so cannot use
    # analysis.parameters as-is
    parameter_names = _ordered_parameter_subset(
        pd.unique(correlations["Parameter"]), [p.name for p in analysis.parameters]
    )
    # Analysis of instantaneous variables with a reduced parameter set are not supported
    # for now; they don't have much of a role in the identifiability analyses I've run
    # so far.
    distances = corrmap_distances(analysis, correlations)
    for norm in ("none", "all", "vector", "subvector"):
        fig_heatmap, parameter_order = plot_tsvar_param_heatmap(
            analysis,
            correlations,
            distances,
            rep_tsvalues,
            norm=norm,
        )
        fig_heatmap.savefig(directory / f"{fname_prefix}heatmap_norm={norm}.svg")
        if norm == "none":
            fig_distmat = plot_tsvar_param_distmat(
                parameter_names, distances, "Unsigned Cosine Distance", parameter_order
            )
            fig_distmat.fig.savefig(
                analysis.directory / f"{fname_prefix}distance_matrix_clustered.svg",
                dpi=300,
            )
            fig_distmat = plot_tsvar_param_distmat(
                parameter_names,
                distances,
                "Unsigned Cosine Distance",
            )
            fig_distmat.fig.savefig(
                analysis.directory / f"{fname_prefix}distance_matrix_unclustered.svg",
                dpi=300,
            )
    # Estimate the rank of the sensitivity vectors
    pth_svd_data = directory / f"{fname_prefix}svd.json"
    pth_svd_fig_s = directory / f"{fname_prefix}singular_values.svg"
    pth_svd_fig_v = directory / f"{fname_prefix}principal_axes.svg"
    try:
        svd_data = corr_svd(correlations, parameter_order=parameter_names)
        with open(pth_svd_data, "w", encoding="utf8") as f:
            json.dump(svd_data, f, ensure_ascii=False)
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


def makefig_error_counts(analysis):
    """Plot error proportions and correlations"""
    # TODO: split up by generator, probably with a child function
    # Collect error counts from cases table
    cases, error_codes = expand_run_errors(analysis.sims_table, group="sampled")
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
    ax.set_title("Run Status", fontsize=FONTSIZE_FIGLABEL)
    ax.set_ylabel("Simulation Count", fontsize=FONTSIZE_AXLABEL)
    ax.bar("Success", outcome_count["Success"], color="black")
    ax.bar("Error", outcome_count["Error"], color="gray")
    ax.bar("No Run", outcome_count["No Run"], color="white", edgecolor="gray")
    ax.set_ylim(0, len(cases))
    ax.tick_params(axis="x", labelsize=FONTSIZE_AXLABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
    # (2) Error counts
    ax = fig.add_subplot(gs0[0, 1])
    ax.set_title("Error breakdown", fontsize=FONTSIZE_FIGLABEL)
    ax.set_ylabel("Simulation Count", fontsize=FONTSIZE_AXLABEL)
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
    cases, error_codes = expand_run_errors(analysis.sims_table, group="sampled")
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
    cases, error_codes = expand_run_errors(analysis.sims_table, group="sampled")
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


def makefig_sensitivity_ivar_all(analysis, param_values, ivar_values):
    ivar_names = analysis.instantaneous_variables
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


def makefig_sensitivity_tsvar_all(analysis, tsdata, sample_ids=None):
    """Plot sensitivity of each time series variable to each parameter"""
    if sample_ids is None:
        sample_ids = _default_sample_ids(analysis)

    # Obtain reference case for time series variables' correlation heatmap
    # TODO: The heat map figure should probably indicate which case is
    # plotted as the time series guide.
    rep_idx = get_rep_sample(analysis)  # (group, sample ID)
    rep_tsvalues = {
        v: tsdata[rep_idx[0]][v].data.loc[rep_idx[1]]
        for v in analysis.timeseries_variables
    }

    # Sobol analysis
    S1, ST = sobol_analysis_tsvars(
        analysis, analysis.samples_table.xs("sampled"), tsdata["sampled"]
    )
    makefig_sobol_tsvars(analysis, S1, ST, rep_tsvalues)

    # Correlation coefficients
    for nm_corr, fun_corr in (
        ("pearson", corr_pearson),
        ("spearman", corr_spearman),
        ("partial", corr_partial),
    ):
        correlations_table = tab_tsvars_corrmap(
            analysis, tsdata["sampled"], fun_corr=fun_corr
        )
        correlations_table.to_feather(
            analysis.directory / f"tsvar_param_corr={nm_corr}.feather"
        )
        makefig_global_correlations_tsvars(
            analysis,
            rep_tsvalues,
            correlations_table,
            analysis.directory,
            f"tsvar_param_corr={nm_corr}_",
        )

    # One-at-a-time sensitivity plots
    makefig_tsvars_line(analysis, tsdata, sample_ids)

    # PDF plots
    makefig_tsvars_pdf(analysis, tsdata, sample_ids)


def makefig_sobol_tsvars(analysis, S1, ST, rep_tsvalues):
    S1 = np.rollaxis(
        np.array([[by_var[v] for v in by_var] for by_var in S1.values()]), -2
    )
    ST = np.rollaxis(
        np.array([[by_var[v] for v in by_var] for by_var in ST.values()]), -2
    )
    indices = {"Order 1": S1, "Total": ST}
    variables = [(v, "") for v in analysis.variables]
    fr = fig_stacked_line(
        indices, analysis.parameters, variables, rep_tsvalues=rep_tsvalues
    )
    fr.fig.savefig(analysis.directory / "tsvars_sobol_sensitivities_scale=free.svg")
    fr = fig_stacked_line(
        indices,
        analysis.parameters,
        variables,
        rep_tsvalues=rep_tsvalues,
        ymax="shared",
    )
    fr.fig.savefig(analysis.directory / "tsvars_sobol_sensitivities_scale=shared.svg")


# noinspection PyPep8Naming
def sobol_analysis_tsvars(analysis, samples, tsdata):
    """Write Sobol analysis for time series variables"""
    # TODO: Levels information should probably be stored in the analysis object
    levels = {
        p: sorted(np.unique(samples[f"{p.name} [param]"])) for p in analysis.parameters
    }
    S1 = {
        p: {v: np.zeros(len(tsdata[v].steps)) for v in analysis.variables}
        for p in analysis.parameters
    }
    ST = deepcopy(S1)
    for v in analysis.variables:
        tab = tsdata[v].data
        # Exclude time points with zero variance
        d = tab[v].groupby("Step").var()
        include_steps = d[d != 0].index.values
        tab = tab.loc[IndexSlice[:, include_steps], :]
        # data1 = pd.merge(tab.reset_index("Step"), samples, on="Sample")
        data1 = _merge_tsdata_params(analysis, tab)
        m = np.zeros(len(tsdata[v].steps), dtype=bool)
        m[include_steps] = True
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
    rep_tsvalues,
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
    # We may be working with a subset of the correlation matrix, so cannot use
    # analysis.parameters as-is
    parameter_names = _ordered_parameter_subset(
        pd.unique(correlations["Parameter"]), [p.name for p in analysis.parameters]
    )
    correlations = (
        correlations.copy()
        .set_index(["Parameter", "Variable", "Time Point"])
        .sort_index()
    )  # speed up correlations.loc[parameter, var] and avoid warning

    # Widths of plot panels
    base_axw = 5.0

    # Widths of dendrogram axes
    fig_llabelw = LABELH_MULT * FONTSIZE_FIGLABEL / 72
    dendro_axw = 12 / 72 * len(parameter_names)

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
        len(parameter_names),
        len(analysis.variables),
        left_blankw=dendro_axw,
        right_blankw=rcbar_areaw,
        axw=axw,
    )
    figw = fig.get_figwidth()

    # Plot dendrogram.  Do this first to get the parameter ordering.
    # Find the bottom edge of the bottom-most correlation vector plot's x-tick labels
    fig.canvas.draw()
    lowest_ax_bbox = axarr[-1, 0].get_tightbbox(for_layout_only=True)
    fig_size_px = fig.get_size_inches() * fig.dpi
    b_ax = axarr[-1, 0].get_position().min[1]
    b_ticks = lowest_ax_bbox.bounds[1] / fig_size_px[1]
    # To center the dendrogram legs on the axis labels, we need to evenly distribute the space that the tick labels occupy
    b = 0.5 * (b_ticks + b_ax)
    t_ax = axarr[1, 0].get_position().max[1]
    t = t_ax + 0.5 * (b_ax - b_ticks)
    dn_ax = fig.add_axes((fig_llabelw / figw, b, dendro_axw / figw, t - b))
    remove_spines(dn_ax)
    # Compute the linkages (clusters), unless all the distances are NaN (e.g., in a
    # univariate sensitivity analysis)
    if np.all(np.isnan(distances)):
        warn(
            "Correlation distance matrix is all NaNs.  This is expected if only one parameter is varying."
        )
        ordered_parameter_idx = np.arange(len(parameter_names))
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
        plot_reference_tsdata(
            rep_tsvalues[var].index.get_level_values("Step"),
            rep_tsvalues[var][var],
            axarr[0, i],
            varname=var,
        )
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
    for j in range(axarr.shape[1]):
        axarr[-1, j].set_xlabel("Time Point Index", fontsize=FONTSIZE_TICKLABEL)
    for irow, iparam in enumerate(ordered_parameter_idx):
        parameter = parameter_names[iparam]
        if norm == "vector":
            absmax = np.nanmax(np.abs(correlations.loc[parameter].values))
            cnorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
        for ivar, var in enumerate(analysis.variables):
            if norm == "subvector":
                absmax = np.nanmax(np.abs(correlations.loc[parameter, var].values))
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
                    rep_tsvalues[var].index.get_level_values("Step")[0] - 0.5,
                    rep_tsvalues[var].index.get_level_values("Step")[-1] + 0.5,
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


def plot_tsvar_param_distmat(
    parameters: Sequence[str],
    distances,
    metric,
    parameter_order=None,
):
    """Plot distance matrix of tsvar–parameter correlation vectors

    :parameter parameter_order: List of indices into the analysis parameters.  The
    parameters will be plotted in the listed order.  Usually this order is chosen to
    match that in the cluster analysis dendrogram in `plot_tsvar_param_heatmap`.

    """
    if parameter_order is None:
        parameter_order = np.arange(len(parameters))
    distmat = scipy.spatial.distance.squareform(distances)[parameter_order, :][
        :, parameter_order
    ]
    fig = plot_matrix(
        distmat,
        tick_labels=[parameters[i] for i in parameter_order],
        cbar_label=metric,
        title="Correlation sensitivity vector distances",
        format_str=".2f",
    )
    return fig


def makefig_tsvars_pdf(analysis, tsdata, sample_ids=None):
    if sample_ids is None:
        sample_ids = analysis.complete_samples()
    cases = analysis.sampled_sims_table(sample_ids["sampled"])
    named_cases = analysis.named_sims_table(sample_ids["named"])
    for variable in analysis.variables:
        vardata = tsdata["sampled"][variable]
        tsdata_by_step = vardata.data.reset_index().set_index("Step")
        for parameter in analysis.parameters:
            # Collect key quantiles.  These will be used for adjusting the range of
            # the probability density plot.
            vmin = vardata.values().min()
            vmax = vardata.values().max()
            if vmin == vmax:
                raise ValueError(
                    f"Variable {variable} has a constant value, {vmin}, so it is not appropriate to estimate its probability density function."
                )
            yrange_all = (vmin, vmax)
            idx_steps = tsdata_by_step.index.unique("Step")
            quantile_levels = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
            quantiles = tuple([] for q in quantile_levels)
            for idx_step in idx_steps:
                v = tsdata_by_step.loc[idx_step, variable]
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


def makefig_tsvar_line(
    analysis,
    tsdata: Dict[str, Dict[str, TSVarData]],  # group → variable → TSVarData
    variable,
    parameter,
    sample_ids=None,
):
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
    if sample_ids is None:
        sample_ids = _default_sample_ids(analysis)
    cases = analysis.sampled_sims_table().loc[IndexSlice[:, sample_ids["sampled"]], :]
    if sample_ids["named"]:
        named_cases = analysis.named_sims_table().loc[
            IndexSlice[:, sample_ids["named"]], :
        ]
    else:
        named_cases = None
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
        plot_tsvar_named(
            analysis, tsdata["named"][variable], parameter, named_cases, ax
        )
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
    i_offset = (
        1 if named_cases is not None else 0
    )  # offset into axes to skip named cases plot
    for i in range(n_levels):
        fulcrum = {
            p.name: levels[p.name][i] if i < len(levels[p.name]) else None
            for p in other_parameters
        }
        # Select the samples to plot
        m = np.ones(len(cases), dtype="bool")
        for nm, v in fulcrum.items():
            if v is None:
                # No case with all parameters at level i
                m[:] = False
                break
            m = np.logical_and(m, cases[nm] == v)
        ids = pd.unique(cases.index.get_level_values("Sample")[m])
        # Make the plot panel.
        ax = fig.add_subplot(gs[(i + i_offset) // nw, (i + i_offset) % nw])
        axs.append(ax)
        ax.set_title(
            f"Other parameters set to level index = {i + 1}",
            fontsize=FONTSIZE_AXLABEL,
        )
        # TODO: Units-awareness for variables
        ax.set_ylabel(variable, fontsize=FONTSIZE_AXLABEL)
        ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_AXLABEL)
        # Plot a line for each sensitivity level of the subject parameter
        for id_ in ids:
            parameter_value = cases.loc[
                IndexSlice[:, id_], subject_parameter.name
            ].iloc[0]
            if subject_parameter.units == "1":
                label = f"{parameter_value:.3g}"
            else:
                label = f"{parameter_value:.3g} {subject_parameter.units:~P}"
            ax.plot(
                tsdata["sampled"][variable].steps,
                tsdata["sampled"][variable].values(id_),
                color=CMAP_DIVERGE(cnorm(parameter_value)),
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
    ylim = [ax.get_ylim() for ax in axs]
    y1 = np.max(ylim)
    y0 = np.min(ylim)
    spans = np.array([(ymax - ymin) / (y1 - y0) for ymin, ymax in ylim])
    if np.all(spans > 0.1):
        for ax in axs:
            ax.set_ylim((y0, y1))
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


def makefig_tsvars_line(analysis, tsdata, sample_ids=None):
    """One-at-a-time sensitivity line plots for all variables and parameters"""
    if sample_ids is None:
        sample_ids = _default_sample_ids(analysis)
    # TODO: Figure out how to plot parameter sensitivity for multiple variation
    # (parameter interactions)
    for variable in analysis.variables:
        for parameter in analysis.parameters:
            makefig_tsvar_line(
                analysis,
                tsdata,
                variable,
                parameter,
                sample_ids,
            )


def plot_tsvar_named(
    analysis, tsdata: TSVarData, parameter: Parameter, named_cases, ax
):
    """Plot time series variable for named cases into an axes"""
    ax.set_title(f"Named cases", fontsize=FONTSIZE_AXLABEL)
    ax.set_ylabel(tsdata.variable, fontsize=FONTSIZE_AXLABEL)
    ax.set_xlabel("Time point [1]", fontsize=FONTSIZE_AXLABEL)
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL)
    for i, id_ in enumerate(named_cases.index.unique("Sample")):
        generator = analysis.variables[tsdata.variable][0]
        value = named_cases.loc[(generator, id_), parameter.name]
        if parameter.units == "1":
            label = f"{id_}\n{parameter.name} = {value}"
        else:
            label = f"{id_}\n{parameter.name} = {value} {parameter.units:~P}"
        ax.plot(
            tsdata.steps,
            tsdata.values(id_),
            label=label,
            color=colors.categorical_n7[i % len(colors.categorical_n7)],
        )
    if ax.lines:
        ax.legend()


def corr_svd(correlations_table, parameter_order=None):
    """Calculate singular values and eigenvectors of parameter sensitivities"""
    correlations = correlations_table.set_index(["Parameter", "Variable", "Time Point"])
    tab = correlations.unstack(["Variable", "Time Point"])
    parameter_names = tab.index.tolist()
    arr = tab.values
    m_nonnan = ~np.any(np.isnan(arr), axis=0)
    m_finite = ~np.any(np.isinf(arr), axis=0)
    arr_valid = arr[:, np.logical_and(m_nonnan, m_finite)]
    u, s, vh = np.linalg.svd(arr_valid.T, full_matrices=False)
    # ^ vh indices: parameters, then vectors
    singular_values = s.tolist()
    variables = correlations.index.levels[1].values.tolist()
    if parameter_order is not None:
        idx_from_name = {nm: i for i, nm in enumerate(parameter_names)}
        idx = [idx_from_name[nm] for nm in parameter_order]
        assert all(parameter_names[i] == nm for i, nm in zip(idx, parameter_order))
    else:
        parameter_order = parameter_names
        idx = [i for i in range(len(vh))]
    principal_axes = vh[:, idx].tolist()
    svd_data = {
        "singular values": singular_values,
        "parameters": parameter_order,
        "variables": variables,
        "principal axes": principal_axes,
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
    analysis,
    tsdata: Dict[str, Dict[str, TSVarData]],
    variable,
    vrange,
    parameter,
    cases,
    named_cases=None,
):
    """Plot probability density of a time series variable into an axes"""
    sampled_tsdata = tsdata["sampled"][variable]
    # TODO: Levels information should probably be stored in the analysis object
    levels = sorted(np.unique(cases[parameter.name]))
    x = np.linspace(vrange[0], vrange[1], 100)
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
        plot_tsvar_named(
            analysis, tsdata["named"][variable], parameter, named_cases, ax
        )
        ax.set_ylim([vrange[0], vrange[1]])
    # For each level of the subject parameter, plot the time series variable's
    # probability density
    tsdata_by_sample = sampled_tsdata.data.reset_index().set_index("Sample")
    for i, level in enumerate(levels):
        ax = fig.add_subplot(gs[i // nw, i % nw])
        axs.append(ax)
        stratum = (
            tsdata_by_sample.loc[
                cases[cases[parameter.name] == level].index.get_level_values("Sample")
            ].reset_index()
        ).set_index("Step")
        p = np.full((len(x), len(sampled_tsdata.steps)), np.nan)
        for step in sampled_tsdata.steps:
            v = stratum.loc[[step]][f"{variable}"].array
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
            extent=(-0.5, len(sampled_tsdata.steps) + 0.5, vrange[0], vrange[1]),
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


def tabulate(analysis: Analysis):
    """Tabulate output from an analysis"""
    dir_out = analysis.directory / "output"
    dir_out.mkdir(exist_ok=True)
    samples = analysis.samples_table
    sims = analysis.sims_table
    # Iterate over groups first b/c we have group-specific tables to create
    for group, group_samples in samples.groupby("Group"):
        group_samples = group_samples.droplevel("Group")
        ivars_table = {"Sample": []}
        ivars_table.update({f"{p.name} [param]": [] for p in analysis.parameters})
        ivars_table.update({f"{v} [var]": [] for v in analysis.instantaneous_variables})
        for sample_id in group_samples.index.get_level_values("Sample"):
            for generator in analysis.generators:
                status = sims.loc[(group, generator.name, sample_id), "Status"]
                if not status == "Run: Success":
                    # Simulation failure.  Don't bother trying to tabulate the results.
                    # TODO: Partial tabulations could still be useful for troubleshooting
                    continue
                # TODO: Handle IDs for multi-sim samples in a more sophisticated way
                sim = Sim(
                    sample_id,
                    # ^ .feb path uses this, so it cannot change
                    analysis.parameters,
                    {
                        p.name: samples.loc[(group, sample_id), f"{p.name} [param]"]
                        for p in analysis.parameters
                    },
                    generator.variables,
                    directory=(
                        analysis.directory
                        / sims.loc[(group, generator.name, sample_id), "Path"]
                    ).parent,
                    checks=generator.checks,
                )
                record, timeseries = tabulate_sim_write(
                    sim, dir_out=dir_out / generator.name
                )
                ivars_table["Sample"].append(sample_id)
                for p in analysis.parameters:
                    ivars_table[f"{p.name} [param]"].append(
                        samples.loc[(group, sample_id), f"{p.name} [param]"]
                    )
                for v in record["instantaneous variables"]:
                    ivars_table[f"{v} [var]"].append(
                        record["instantaneous variables"][v]["value"]
                    )
        df_ivars = DataFrame(ivars_table)
        df_ivars.to_csv(analysis.directory / f"{group}_-_inst_vars.csv", index=True)


def tabulate_sim(sim):
    """Tabulate variables for a single simulation."""
    case_tree = read_xml(sim.feb_file)
    # Read the text data files
    # TODO: Finding the text data XML elements should probably be done in Waffleiron.  Need to move this function to spamneggs.py so we can annotate the type of the arguments.
    text_data = {}
    for entity_type in ("node", "element", "body", "connector"):
        tagname = TAG_FOR_ENTITY_TYPE[entity_type]
        fname = sim.feb_file.with_suffix("").name + f"_-_{tagname}.txt"
        pth = sim.feb_file.parent / fname
        e_textdata = case_tree.find(f"Output/logfile/{tagname}[@file='{fname}']")
        if e_textdata is not None:
            text_data[entity_type] = textdata_table(pth)
        else:
            text_data[entity_type] = None
    # Extract values for each variable based on its <var> element
    record = {"instantaneous variables": {}, "time series variables": {}}
    for varname, f_var in sim.variables_list.items():
        if isinstance(f_var, XpltDataSelector):
            xplt_data = sim.solution().solution
            # Check component selector validity
            dtype = xplt_data.variables[f_var.variable]["type"]
            oneids = (str(i + 1) for i in f_var.component)
            if dtype == "float" and len(f_var.component) != 0:
                msg = f"{f_var.variable}[{', '.join(oneids)}] requested), but {f_var.variable} has xplt dtype='{dtype}' and so has 0 dimensions."
                raise ValueError(msg)
            elif dtype == "vec3f" and len(f_var.component) != 1:
                msg = f"{f_var.variable}[{', '.join(oneids)}] requested), but {f_var.variable} has xplt dtype='{dtype}' and so has 1 dimension."
                raise ValueError(msg)
            elif dtype == "mat3fs" and len(f_var.component) != 2:
                msg = f"{f_var.variable}[{', '.join(oneids)}] requested), but {f_var.variable} has xplt dtype='{dtype}' and so has 2 dimensions."
                raise ValueError(msg)
            # Get ID for region selector
            if f_var.region is None:
                region_id = None
            else:
                region_id = f_var.region["ID"]
            # Get ID for parent selector
            if f_var.parent_entity is None:
                parent_id = None
            else:
                parent_id = f_var.parent_entity["ID"]
            if f_var.temporality == "instantaneous":
                # Convert time selector to step selector
                if f_var.time_unit == "time":
                    step = find_closest_timestep(
                        f_var.time,
                        xplt_data.step_times,
                        np.array(range(len(xplt_data.step_times))),
                    )
                value = xplt_data.value(
                    f_var.variable, step, f_var.entity_id, region_id, parent_id
                )
                # Apply component selector
                for c in f_var.component:
                    value = value[c]
                # Save selected value
                record["instantaneous variables"][varname] = {"value": value}
            elif f_var.temporality == "time series":
                data = xplt_data.values(
                    f_var.variable,
                    entity_id=f_var.entity_id,
                    region_id=region_id,
                    parent_id=parent_id,
                )
                values = np.moveaxis(np.array(data[f_var.variable]), 0, -1)
                for c in f_var.component:
                    values = values[c]
                record["time series variables"][varname] = {
                    "times": xplt_data.step_times,
                    "steps": np.array(range(len(xplt_data.step_times))),
                    "values": values,
                }
        elif isinstance(f_var, TextDataSelector):
            # Apply entity type selector
            tab = text_data[f_var.entity]
            # Apply entity ID selector
            tab = tab[tab["Item"] == f_var.entity_id]
            tab = tab.set_index("Step")
            if f_var.temporality == "instantaneous":
                # Convert time selector to step selector.  The selector
                # has internal validation, so we can assume the time
                # unit is valid.
                if f_var.time_unit == "time":
                    step = find_closest_timestep(f_var.time, tab["Time"], tab.index)
                    if step == 0:
                        raise ValueError(
                            "Data for step = 0 requested from an FEBio text data output file, but FEBio does not provide text data output for step = 0 / time = 0."
                        )
                    if step not in tab.index:
                        raise ValueError(
                            f"A value for {f_var.variable} was requested for step = {step}, but this step is not present in the FEBio text data output."
                        )
                elif f_var.time_unit == "step":
                    step = f_var.time
                # Apply variable name and time selector
                value = tab[f_var.variable].loc[step]
                record["instantaneous variables"][varname] = {"value": value}
            elif f_var.temporality == "time series":
                record["time series variables"][varname] = {
                    "times": tab["Time"].values,
                    "steps": tab.index.values,
                    "values": tab[f_var.variable].values,
                }
        elif isinstance(f_var, FunctionVar):
            v = f_var(sim)
            if not isinstance(v, TimeSeries):
                record["instantaneous variables"][varname] = {"value": v}
            else:
                record["time series variables"][varname] = {
                    "times": v.time,
                    "steps": v.step,
                    "values": v.value,
                }
        else:
            raise ValueError(f"{f_var} not supported as a variable type.")
    # Assemble the time series table
    tab_timeseries = {"Time": [], "Step": []}
    for f_var, d in record["time series variables"].items():
        tab_timeseries["Time"] = d["times"]
        tab_timeseries["Step"] = d["steps"]
        tab_timeseries[f_var] = d["values"]
    tab_timeseries = pd.DataFrame(tab_timeseries)
    return record, tab_timeseries


def tabulate_sim_write(sim, dir_out=None):
    """Tabulate variables for single case analysis & write to disk."""
    if dir_out is None:
        dir_out = sim.directory
    # Find/create output directory
    dir_out = Path(dir_out)
    if not dir_out.exists():
        dir_out.mkdir()
    # Tabulate the variables
    record, timeseries = tabulate_sim(sim)
    with open(dir_out / f"{sim.name}_vars.json", "w") as f:
        write_record_to_json(record, f)
    timeseries.to_csv(dir_out / f"{sim.name}_timeseries_vars.csv", index=False)
    makefig_case_tsvars(timeseries, dir_out=dir_out, casename=sim.name)
    return record, timeseries
