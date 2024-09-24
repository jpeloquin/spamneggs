"""Helper classes and functions

This module is focused on helper code for program execution.  For similarly basic
building blocks to support working with models, see core.py.

"""

import base64
import inspect
import struct
import threading
import weakref

import numpy as np
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
        # Consider doing something useful if output already exists but new output
        # differs from the old.  A model evaluation might differ from run to run
        # because the evaluation is not fully reproducible (deterministic), the model
        # definition changed, or the run environment changed.  Only a change in the
        # model definition is unambiguously a problem.  Best to do this check in the
        # caller, which has more context.

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
        self.root["eval_values"][x_hash] = y


def parameter_to_str(parameter):
    if parameter.units == "1":
        return f"{parameter.name} [1]"
    else:
        return f"{parameter.name} [{parameter.units:~P}]"
