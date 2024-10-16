"""Helper classes and functions

This module is focused on helper code for program execution.  For similarly basic
building blocks to support working with models, see core.py.

"""

import base64
import inspect
import struct
import threading
import weakref
from functools import wraps
from typing import Optional, List, Callable

import numcodecs
import numpy as np
import scipy
import zarr


class Counter:
    """Threadsafe counter"""

    def __init__(self, i=0):
        self.count = i
        self.lock = threading.Lock()

    def __str__(self):
        return str(self.count)

    def __repr__(self):
        return f"{self.__class__}({self.count})"

    def increment(self):
        with self.lock:
            self.count += 1


class EvaluationDB:
    """Log model evaluations

    An EvaluationDB logs model evaluations for a specific model.  The model evaluations
    must have identifical inputs except for their parameter values, which are provided
    to EvaluationDB when the evaluation is logged.  Multiple analyses can therefore
    store data in the same EvaluationDB instance as long as they use the same model.

    """

    def __init__(self, pth, mode="a"):
        """Return EvaluationDB backed by Zarr DirectoryStore

        :param mode: Passed to zarr.open.  Default = "a", which means read/write,
        create if doesn't exist.  Read-only is "r".

        A Zarr DirectoryStore can be written to by multiple threads or processes,
        but there is no mechanism to prevent two writes from modifying the same chunk
        at the same time.

        """
        self.store = zarr.DirectoryStore(pth)
        self.root = zarr.open(self.store, mode=mode)
        if not self.root.read_only:
            # Map of parameter value hash → model evaluation output
            self.root.require_group("eval_values")
            # Metadata for one function evaluation
            self.root.require_group("eval_info")
            # Map of parameter value hash → evaluation integer ID.  (One to many.)
            self.root.require_group("eval_id_from_x")
        self._finalizer = weakref.finalize(self, self.store.close)

    @classmethod
    def _encode_x(cls, x):
        bytes_key = struct.pack("<" + "d" * len(x), *x)
        string_key = base64.encodebytes(bytes_key)
        return string_key

    @classmethod
    def _decode_x(cls, s):
        bytes_key = base64.decodebytes(s)
        return struct.unpack(f"<{len(bytes_key) // 8}d", bytes_key)

    def get_eval_ids(self):
        return sorted([int(i) for i in self.root["eval_info"].keys()])

    def get_output_by_id(self, id_):
        id_ = str(id_)
        x = self.root["eval_info"][id_]["x"]
        return self.get_output_by_x(x)

    def get_eval_by_id(self, id_):
        return self.root["eval_info"][id_]

    def get_eval_by_x(self, x):
        x_hash = self._encode_x(x)
        id_ = self.root["eval_id_from_x"][x_hash]
        return self.get_eval_by_id(id_)

    def get_output_by_x(self, x):
        x_hash = self._encode_x(x)
        return np.array(self.root["eval_values"][x_hash])

    def write_eval(self, id_, x, y, mdata={}):
        """Store input & output of a given model evaluation

        :param id_: Identifier for the model evaluation.  The type can be either str or
        int; it is not checked.

        :param x: Parameter values.

        :param y: Model output values.

        """
        # Consider doing something useful if output already exists but new output
        # differs from the old.  A model evaluation might differ from run to run
        # because the evaluation is not fully reproducible (deterministic), the model
        # definition changed, or the run environment changed.  Only a change in the
        # model definition is unambiguously a problem.  Best to do this check in the
        # caller, which has more context.

        # TODO: Allow repeat evals

        # Call count
        self.root["eval_info"].create_group(id_)
        # Call stack
        outer_frame = inspect.getouterframes(inspect.currentframe())
        callers = [f.function for f in outer_frame]
        self.root["eval_info"][id_].create_dataset(
            "call_chain",
            dtype=str,
            shape=(
                len(
                    callers,
                )
            ),
        )
        self.root["eval_info"][id_]["call_chain"] = callers
        self.root["eval_info"][id_]["x"] = x
        # Evaluation optional metadata (usually file paths or alternate IDs)
        for k, v in mdata.items():
            self.root["eval_info"][id_][k] = v
        # Create hash ID for parameter values
        x_hash = self._encode_x(x)
        # Update map parameter value hash ID → evaluation integer ID.  (One to many.)
        if x_hash in self.root["eval_id_from_x"]:
            self.root["eval_id_from_x"][x_hash] = list(
                self.root["eval_id_from_x"][x_hash]
            ) + [id_]
        else:
            self.root["eval_id_from_x"][x_hash] = [id_]
        # Parameters → output values hashmap
        if hasattr(y, "__getitem__"):
            for k in y:
                self.root["eval_values"].require_group(x_hash, overwrite=True)
                self.root["eval_values"][x_hash][k] = y[k]
        else:
            self.root["eval_values"][x_hash] = y


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
            counter.increment()
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
        if keys is None:
            keys = self.root["iterations/0"].keys()
        iter_ids = self.get_iter_ids()
        data = {
            k: np.stack([np.array(self.root["iterations"][i][k]) for i in iter_ids])
            for k in keys
        }
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
        self.root["final/termination status"] = result["status"]
        self.root["final"].create_dataset("termination message", dtype=str, shape=0)
        self.root["final/termination message"] = result["message"]
        # Write data that is part of an intermediate result (includes commit)
        self._write_result_to_group(result, self.root["final"])

    def close(self):
        self.store.close()


def count_calls(fn: Callable) -> Callable:
    @wraps(fn)
    def g(*args, **kwargs):
        g.call_counter.increment()
        return fn(*args, **kwargs)

    g.call_counter = Counter()
    return g


def parameter_to_str(parameter):
    if parameter.units == "1":
        return f"{parameter.name} [1]"
    else:
        return f"{parameter.name} [{parameter.units:~P}]"
