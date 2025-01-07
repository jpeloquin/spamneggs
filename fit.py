"""Functions related to model fitting (optimization)

"""

import base64
import struct
import weakref
from collections import namedtuple
from math import ceil
from typing import Optional, List, Dict, Union

import numcodecs
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from uuid_utils import uuid7
import zarr

from spamneggs.util import Counter
from spamneggs.plot import symlog_thresh, remove_spines, FONTSIZE_TICKLABEL


class EvaluationDB:
    """Log model evaluations

    An EvaluationDB logs model evaluations for a specific model.  The model evaluations
    must have identifical inputs except for their parameter values, which are provided
    to EvaluationDB when the evaluation is logged.  Multiple analyses can therefore
    store data in the same EvaluationDB instance as long as they use the same model.

    """

    def __init__(self, pth, parameters: List[str], variables: List[str], mode="a"):
        """Return EvaluationDB backed by Zarr DirectoryStore

        :param mode: Passed to zarr.open.  Default = "a", which means read/write,
        create if the file doesn't exist.  Read-only is "r".

        A Zarr DirectoryStore can be written to by multiple threads or processes,
        but there is no mechanism to prevent two writes from modifying the same chunk
        at the same time.

        """
        self.store = zarr.DirectoryStore(pth)
        self.root = zarr.open(self.store, mode=mode)
        if not self.root.read_only:
            # List of parameter names
            self.root["parameter_names"] = parameters
            # List of variable names
            self.root["variable_names"] = variables
            # Map of eval ID → eval record for one model evaluation
            self.root.require_group("eval")
            # Map of parameter value hash → evaluation integer ID.  (One to many.)
            self.root.require_group("eval_id_from_x")
        self._finalizer = weakref.finalize(self, self.store.close)

    @classmethod
    def _encode_x(cls, x):
        """Encode parameter values in byte array

        The byte array is primarily used as a hash/ID to look up the corresponding
        evaluation data.
        """
        bytes_key = struct.pack("<" + "d" * len(x), *x)
        string_key = base64.encodebytes(bytes_key)
        return string_key

    @classmethod
    def _decode_x(cls, s):
        """Decode parameter values from byte array

        The byte array is primarily used as a hash/ID to look up the corresponding
        evaluation data.
        """
        bytes_key = base64.decodebytes(s)
        return struct.unpack(f"<{len(bytes_key) // 8}d", bytes_key)

    def get_eval_ids(self):
        """Return all evaluation IDs"""
        return sorted([int(i) for i in self.root["eval"].keys()])

    def get_evals(self, x):
        """Return evaluation records for parameter values"""
        x_bytes = self._encode_x(x)
        ids = self.root["eval_id_from_x"][x_bytes]
        return {id_: self.root["eval"][id_] for id_ in ids}

    def get_eval_output(self, x):
        """Return output variables' values for parameter values"""
        x_bytes = self._encode_x(x)
        ids = self.root["eval_id_from_x"][x_bytes]
        return {id_: self.root["eval"][id_]["y"] for id_ in ids}

    def _initialize_eval_record(self, x, mdata={}):
        """Initialize a record for a new evaluation

        :param x: Input parameters' values.

        :param mdata: Evaluation metadata, usually file paths or evaluation IDs.

        A primary ID for the evaluation record (UUID7) is generated automatically.
        Adding the option to override the primary ID shouldn't cause problems,
        if it ever becomes necessary.

        """
        id_ = str(uuid7())  # use str since 128-bit int not supported
        self.root["eval"].create_group(id_)
        self.root["eval"][id_]["x"] = x

        # Update map of parameter value hash ID → eval ID (primary key).  This is a
        # one-to-many relationship.
        x_bytes = self._encode_x(x)
        if x_bytes in self.root["eval_id_from_x"]:
            self.root["eval_id_from_x"][x_bytes] = list(
                self.root["eval_id_from_x"][x_bytes]
            ) + [id_]
        else:
            self.root["eval_id_from_x"][x_bytes] = [id_]

        # Write optional metadata
        for k, v in mdata.items():
            self.root["eval"][id_]["metadata"][k] = v

        # Create field to record whether the model evaluation succeeded or failed
        self.root["eval"][id_]["status"] = ""

        return id_, x_bytes

    def write_eval_error(self, x, error: str, mdata={}):
        """Store evaluation error for a list of input parameter values

        :param x: Input parameters' values.

        :param error: Error produced when model evaluation was attempted.

        :param mdata: Evaluation metadata, usually file paths or evaluation IDs.

        """
        id_, _ = self._initialize_eval_record(x, mdata)
        self.root["eval"][id_]["status"] = error
        return id_

    def write_eval_output(
        self, x, y: Union[Dict[str, np.ndarray], np.ndarray], mdata={}
    ):
        """Store input & output of a given model evaluation

        :param x: Input parameters' values.

        :param y: Output variables' values corresponding to `x`.  If y is a
        dictionary, it must have keys matching the variable names provided when the
        EvaluationDB was initialized.

        :param mdata: Evaluation metadata, usually file paths or evaluation IDs.

        """
        id_, x_bytes = self._initialize_eval_record(x, mdata)
        self.root["eval"][id_]["status"] = "Success"

        # Write output variables' values
        if hasattr(y, "__getitem__"):
            y = np.vstack([y[k] for k in self.root["variable_names"]])
        else:
            y = np.atleast_2d(y)
        n_vars = len(self.root["variable_names"])
        if y.shape[0] != n_vars:
            raise ValueError(
                f"EvaluationDB was initialized to store {n_vars}, but the the provided data has cardinality {y.shape[0]} in its first dimension, which is interpreted as the number of variables."
            )
        self.root["eval"][id_]["y"] = y
        # Since a model evaluation might differ from run to run because the
        # evaluation is not fully reproducible (in the deterministic sense), we don't
        # check if the output values for repeated evaluations match.  Better for the
        # caller to check inconsistencies, since it has more context.
        return id_


class ScipyOptimizationDB:
    """Log SciPy optimizer iterations and final result

    To record function evaluations, use EvaluationDB.

    """

    def __init__(self, pth, mode="r"):
        self.store = zarr.DirectoryStore(pth)
        self.root = zarr.open(self.store, mode=mode)
        if not self.root.read_only:
            self.root.create_group("problem")
            self.root.create_group("setup")
            self.root.create_group("iterations")
            self.root.create_group("final")

    def make_scipy_callback(self):
        """Return a callback function for scipy.optimize.minimize"""
        counter = Counter()

        def scipy_callback(intermediate_result: scipy.optimize.OptimizeResult):
            # The name of the argument must be 'intermediate_result' for scipy to
            # provide an OptimizeResult.
            counter.increment()  # first ID will be 1
            return self.write_iteration(counter.count, intermediate_result)

        return scipy_callback

    def _write_result_to_group(self, result: scipy.optimize.OptimizeResult, group):
        group["iteration"] = result["nit"]
        group["fun value"] = result["fun"]
        group["x"] = result["x"]
        if self.algorithm == "trust-constr":
            group["gradient"] = result["grad"]
            group["constraint values"] = result["constr"]
            group["constraint jacobian"] = result["jac"][0].toarray()
            group["barrier parameter"] = result["barrier_parameter"]
            group["barrier tolerance"] = result["barrier_tolerance"]
            group["lagrangian gradient"] = result["lagrangian_grad"]
            group["optimality"] = result["optimality"]
            group["execution time"] = result["execution_time"]

    def write_setup(
        self, parameters, method, y_obs, x_init, x_true=None, cost_true=None
    ):
        self.root["problem"].create_dataset(
            "parameters",
            dtype=object,
            object_codec=numcodecs.VLenUTF8(),
            shape=len(parameters),
        )
        self.root["problem/parameters"] = parameters
        self.root["setup"].create_dataset("method", dtype=str, shape=0)
        self.root["setup/method"] = method
        # ^ have to access value as self.root["setup"]["method"][""]; very odd design
        self.root["setup/f_obs"] = y_obs
        self.root["setup/x_init"] = x_init
        if x_true is not None:
            self.root["setup/x_true"] = x_true
        if cost_true is not None:
            self.root["setup/cost_true"] = cost_true

    @property
    def algorithm(self):
        return self.root["setup"]["method"][""]

    def get_iter_ids(self):
        """Return sorted iteration IDs"""
        return sorted([int(i) for i in self.root["iterations"].keys()])

    def iteration_data(self, keys: Optional[List[str]] = None):
        """Return iteration data as arrays

        :param keys: List of keys for which to retrieve data.  By default, data for
        all available keys will be returned.

        """
        # default
        data = {}
        if len(self.root["iterations"]) == 0:
            return data
        if keys is None:
            id_ = self.get_iter_ids()[0]
            keys = self.root[f"iterations/{id_}"].keys()
        iter_ids = self.get_iter_ids()
        for k in keys:
            data[k] = np.stack(
                [np.array(self.root["iterations"][i][k]) for i in iter_ids]
            )
        # Some of Scipy's algorithms don't report iteration count
        if "iteration" in keys:
            data["iteration"] = np.stack(
                [np.array(self.root["iterations"][i]["iteration"]) for i in iter_ids]
            )
        else:
            data["iteration"] = iter_ids
        return data

    @property
    def parameters(self):
        return np.array(self.root["problem/parameters"])

    # I think appending to an array would be slow, but not sure
    # https://github.com/zarr-developers/zarr-python/issues/583
    # https://zarr.readthedocs.io/en/stable/tutorial.html
    def write_iteration(self, id_, intermediate_result: scipy.optimize.OptimizeResult):
        g = self.root.create_group(f"iterations/{id_}")
        if self.algorithm == "trust-constr":
            self._write_result_to_group(intermediate_result, g)
        else:
            # For Nelder-Mead, only "fun" and "x" are guaranteed to be present
            g["x"] = intermediate_result["x"]
            g["fun value"] = intermediate_result["fun"]

    def write_final_result(self, result: scipy.optimize.OptimizeResult):
        # Write data not part of an intermediate result
        self.root["final/termination_status"] = result["status"]
        self.root["final/termination_message"] = result["message"]
        # Write data that is part of an intermediate result (includes commit)
        self._write_result_to_group(result, self.root["final"])

    def write_final_error(self, status, message):
        self.root["final/termination_status"] = status
        self.root["final/termination_message"] = message

    def close(self):
        self.store.close()


def plot_fit_vs_iteration_1d(
    db: ScipyOptimizationDB,
    parameter_constants={},
    norm_cost=None,
    colors={"fit": "black", "cost": "tab:red", "true": "tab:blue"},
):
    """Plot cost and parameters vs. iteration

    :param parameter_constants: Dictionary of parameter names → parameter values (
    constants), which will be plotted together with the variable parameters in the
    optimization database.  This is useful if some parameters are being fixed in e.g.
    model reduction.

    """
    # This function came from Peloquin & Elliott 2024 and may need some cleanup to
    # make it more general-purpose.

    # TODO: To make this function general-purpose, we will have to pass in parameter
    #  order or more sophisticated data structures must be used
    parameters = list(db.parameters)
    parameters += [p for p in parameter_constants if p not in parameters]
    iterations = db.iteration_data()

    N = len(parameters) + 1  # +1 for cost
    ny = ceil(N / 6)
    nx = ceil(N / ny)
    ax_w = 3.0
    ax_h = 2.4
    fig = Figure(figsize=(nx * ax_w, 2 * ny * ax_h))
    gs = GridSpec(ny, nx, figure=fig)

    PlotCell = namedtuple("PlotCell", ["ax_v", "ax_Δ"])

    def make_axes(gs_cell):
        local_gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell)
        ax_value = plt.Subplot(fig, local_gs[0])
        ax_Δ = plt.Subplot(fig, local_gs[1])
        return PlotCell(ax_value, ax_Δ)

    def plot_local(ax_value, ax_δ, x, y, varname):
        # Value
        fig.add_subplot(ax_value)
        ax_value.set_ylabel(varname)
        ax_value.plot(x, y, "-", color=colors["fit"])
        # Incremental change
        fig.add_subplot(ax_δ)
        ax_δ.set_ylabel(f"δ({varname})")
        Δ = np.diff(y)
        ax_δ.plot(
            x[1:],
            Δ,
            ".",
            color=colors["fit"],
        )
        ax_δ.set_xlabel("Iteration")
        ax_δ.set_yscale("symlog", linthresh=symlog_thresh(Δ)[1])
        # Common styling
        for ax in (ax_value, ax_δ):
            ax.tick_params(axis="x", labelsize=FONTSIZE_TICKLABEL)
            ax.tick_params(axis="y", labelsize=FONTSIZE_TICKLABEL, left=True)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.grid(axis="x", which="major")
            remove_spines(ax)
        return ax_value, ax_δ

    # Create axes
    ax_cost = make_axes(gs[0, 0])
    ax_by_param = {nm: make_axes(gs[i + 1]) for i, nm in enumerate(parameters)}

    # Iteration ID
    iter_idx = iterations["iteration"]
    # Plot Cost
    ψ = iterations["fun value"]
    plot_local(
        *ax_cost,
        iter_idx,
        ψ if norm_cost is None else norm_cost(ψ),
        "Cost" if norm_cost is None else "Norm. Cost",
    )
    # Only apply log scale if we are dealing with multiple orders of magnitude
    ymin, logthresh, ymax = symlog_thresh(iterations["fun value"])
    with np.errstate(divide="ignore"):
        set_log_scale = max(np.log10(-ymin), np.log10(ymax)) - np.log10(logthresh) > 1
    if set_log_scale:
        ax_cost.ax_v.set_yscale("symlog", linthresh=logthresh)
    # Add true parameter values if available
    if "x_true" in db.root["setup"]:
        θ_true = {
            nm: db.root["setup"]["x_true"][i] for i, nm in enumerate(db.parameters)
        } | {nm: v[0] for nm, v in parameter_constants.items()}
        for nm, v in θ_true.items():
            ax = ax_by_param[nm].ax_v
            ax.axhline(
                v,
                linestyle="-",
                color=colors["true"],
            )
    # Plot estimated parameter values by iteration
    θ_est_by_param = {nm: iterations["x"][:, i] for i, nm in enumerate(db.parameters)}
    for nm in db.parameters:
        plot_local(*ax_by_param[nm], iter_idx, θ_est_by_param[nm], nm)
    for nm in parameter_constants:
        plot_local(
            *ax_by_param[nm],
            iter_idx,
            np.repeat(parameter_constants[nm][1], len(iter_idx)),
            nm,
        )
    # link x-axes
    ax_cost.ax_Δ.sharex(ax_cost.ax_v)
    for ax_group in ax_by_param.values():
        for ax in ax_group:
            ax.sharex(ax_cost.ax_v)
    return fig, ax_by_param
