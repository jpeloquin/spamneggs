"""Functions related to model fitting (optimization)

"""

import base64
from collections import namedtuple
from math import ceil
from pathlib import Path
import shutil
import struct
from typing import Optional, List, Dict, Union
import weakref

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

    def __init__(self, store, mode="r"):
        """Return EvaluationDB backed by Zarr storage

        :param store: Store or path to directory.  Passed unmodified to zarr.open.

        :param mode: Passed to zarr.open.  Read-only = "r" (default).  Read/write,
        create if the file doesn't exist = "a".  For other modes see zarr.open.

        The EvaluationDB stores a map of parameter values → output variables' values for
        many model evaluations.

        A Zarr DirectoryStore can be written to by multiple threads or processes,
        but it has no mechanism to prevent two writes from modifying the same chunk
        at the same time.  It contains many small files, which may cause performance
        problems on some file systems.

        A Zarr ZipStore can be written to by multiple threads, but not multiple
        processes.  Existing entries cannot be removed or replaced.  Therefore,
        ZipStore is not currently supported.

        """
        self.root = zarr.open(store, mode=mode)
        self.store = self.root.store
        self._finalizer = weakref.finalize(self, self.root.chunk_store.close)
        self._finalizer = weakref.finalize(self, self.store.close)

    def init(self, parameters: List[str], variables: List[str]):
        # List of parameter names
        self.root["parameter_names"] = parameters
        # List of variable names
        self.root["variable_names"] = variables
        # Map of eval ID → eval record for one model evaluation
        self.root.require_group("eval")
        # Map of parameter value hash → evaluation integer ID.  (One to many.)
        self.root.require_group("eval_id_from_x")
        # TODO: add initialized sentinel
        # Version and compatibility info
        self.root["hash"] = "FNV-1a"
        self.version = 1
        return self

    @property
    def variable_names(self):
        return tuple(self.root.variable_names)

    @classmethod
    def hash_x(cls, x):
        """Compute 64-bit FNV-1a hash for parameter array

        :param x: Parameter vector to be hashed.

        The hash algorithm is FNV-1a with standard parameters.

        """
        x = np.array(x)
        # algorithm ref: http://isthe.com/chongo/tech/comp/fnv/#FNV-1a
        sz = 2**64
        prime = 1099511628211
        k = 14695981039346656037
        for b in x.tobytes():
            k = k ^ b
            k = (k * prime) % sz  # ensure fixed size
        return k

    def get_eval_ids(self):
        """Return all evaluation IDs"""
        return list(self.root["eval"].keys())

    def get_evals(self, x):
        """Return evaluation records for parameter values"""
        if not hasattr(self, "version"):
            return self.get_evals_v0(x)
        x = np.array(x)
        h = f"{self.hash_x(x):x}"
        evals = {}
        for id_ in self.root["eval_id_from_x"][h]:
            x_e = np.array(self.root["eval"][id_]["x"])
            if np.all(x_e == x):
                evals[id_] = self.root["eval"][id_]
        return evals

    def get_evals_v0(self, x):
        """Return evaluation records for parameter values for a version 0 db"""
        bytes_key = struct.pack("<" + "d" * len(x), *x)
        string_key = base64.encodebytes(bytes_key)
        ids = self.root["eval_id_from_x"][string_key]
        return {id_: self.root["eval"][id_] for id_ in ids}

    def get_eval_output(self, x):
        """Return output variables' values for parameter values"""
        evals = self.get_evals(x)
        return {
            id_: {v: e["y"][i] for i, v in enumerate(self.variable_names)}
            for id_, e in evals.items()
        }

    def add_eval(self, x, mdata={}) -> str:
        """Initialize a record for a new evaluation

        :param x: Input parameters' values.

        :param mdata: Evaluation metadata, usually file paths or evaluation IDs.

        :returns: The evaluation's ID.

        A primary ID for the evaluation record (UUID7) is generated automatically.
        Adding an option to override the primary ID shouldn't cause problems,
        if it ever becomes necessary.

        You will need to call `write_eval_output` or `write_eval_error` to add the
        evaluation result.

        """
        id_ = str(uuid7())  # use str since 128-bit int not supported
        eval_: zarr.Group = self.root["eval"].create_group(id_)
        eval_["x"] = x

        # Update hash map of parameter value → eval ID (primary key).  This is a
        # one-to-many relationship.
        h = f"{self.hash_x(x):x}"
        if h in self.root["eval_id_from_x"]:
            self.root["eval_id_from_x"][h] = list(self.root["eval_id_from_x"][h]) + [
                id_
            ]
        else:
            self.root["eval_id_from_x"][h] = [id_]

        # Write optional metadata
        eval_.create_group("metadata")
        for k, v in mdata.items():
            eval_["metadata"][k] = v

        # Create field to record whether the model evaluation succeeded or failed
        eval_["status"] = ""

        return id_

    def write_eval_output(self, id_, y: Union[Dict[str, np.ndarray], np.ndarray]):
        """Store input & output of a given model evaluation

        :param id_: Evaluation ID generated by `write_eval`.

        :param y: Output variables' values corresponding to `x`.  If y is a
        dictionary, it must have keys matching the variable names provided when the
        EvaluationDB was initialized.

        """
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

    def write_eval_error(self, id_, error: str):
        """Store evaluation error for a list of input parameter values

        :param id_: Evaluation ID generated by `write_eval`.

        :param error: Error produced when model evaluation was attempted.

        """
        self.root["eval"][id_]["status"] = error


class ScipyOptimizationDB:
    """Log SciPy optimizer iterations and final result

    To record function evaluations, use EvaluationDB.

    """

    def __init__(self, store, mode="r"):
        """Return ScipyOptimizationDB backed by Zarr storage

        :param store: Store or path to directory.  Passed unmodified to zarr.open.

        :param mode: Passed to zarr.open.  Read-only = "r" (default).  Read/write,
        create if the file doesn't exist = "a".  For other modes see zarr.open.

        A Zarr DirectoryStore can be written to by multiple threads or processes,
        but it has no mechanism to prevent two writes from modifying the same chunk
        at the same time.  It contains many small files, which may cause performance
        problems on some file systems.

        A Zarr ZipStore can be written to by multiple threads, but not multiple
        processes.  Existing entries cannot be removed or replaced.  Therefore,
        ZipStore is not currently supported.

        """
        self.root = zarr.open(store, mode=mode)
        self.store = self.root.store
        self._finalizer = weakref.finalize(self, self.root.chunk_store.close)
        self._finalizer = weakref.finalize(self, self.store.close)

    def init(self):
        self.root.create_group("problem")
        self.root.create_group("setup")
        self.root.create_group("iterations")
        self.root.create_group("final")
        # TODO: add initialized sentinel
        return self

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
    ax_w = 3.2
    ax_h = 2.2
    fig = Figure(figsize=(nx * ax_w, 2 * ny * ax_h))
    gs = GridSpec(ny, nx, wspace=0.46, figure=fig)

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
    ymin, logthresh, ymax, oom = symlog_thresh(iterations["fun value"])
    if oom > 1:
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


def convert_zarr_store(
    src: zarr.storage.MutableMapping, dest: zarr.storage.MutableMapping
):
    """Convert a Zarr store to another store

    The caller is responsible for deleting any pre-existing content at the destination.
    The source store will be closed and deleted after the copy is complete.  (If you
    want to preserve the source store, just zarr.copy_store.)  This function is most
    often used to convert a DirectoryStore to a ZipStore.  A DirectoryStore often
    contains thousands of tiny files, which can degrade system performance in some
    circumstances.

    """
    # Get source path so we can delete it later (this probably won't work for in-memory
    # stores)
    if isinstance(src, str) or isinstance(src, Path):
        pth_src = Path(src)
    else:  # Zarr Store
        pth_src = Path(src.path)
    zarr.copy_store(src, dest)
    if hasattr(src, "close"):
        src.close()
    if pth_src.is_dir():
        shutil.rmtree(pth_src)
    else:
        pth_src.unlink()
