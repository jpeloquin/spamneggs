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
    must have identifical inputs except for the their parameter values, which are
    provided to EvaluationDB when the evaluation is logged.  Multiple analyses can
    therefore use the same EvaluationDB instance.

    """

    def __init__(self, pth, mode="r+"):
        """Read SQLite store and return EvaluationDB

        :param mode: Passed to zarr.open.  Default = "r+", which means read/write,
        file must exist.  Read-only is "r".

        """
        self.store = zarr.SQLiteStore(pth)
        self.store.db.isolation_level = "DEFERRED"
        self.root = zarr.open(self.store, mode=mode)
        if not self.root.read_only:
            self.root.create_group("eval_values")
            self.root.create_group("eval_info")
        self._finalizer = weakref.finalize(self, self.store.close)

    @classmethod
    def _encode_x(cls, x):
        bytes_key = struct.pack("<" + "d" * len(x), *x)
        string_key = base64.encodebytes(bytes_key)
        return string_key

    @classmethod
    def _decode_x(cls, s):
        bytes_key = base64.decodebytes(s)
        return struct.unpack(f"<{len(bytes_key // 8)}d", bytes_key)

    def get_eval_ids(self):
        return sorted([int(i) for i in self.root["eval_info"].keys()])

    def get_output_by_id(self, id_):
        id_ = str(id_)
        x = self.root["eval_info"][id_]["x"]
        return self.get_output_by_x(x)

    def get_output_by_x(self, x):
        return np.array(self.root["eval_values"][self._encode_x(x)])

    def write_eval(self, id_, x, output):
        k = self._encode_x(x)
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
        # Parameters → output values hashmap
        self.root["eval_values"][k] = output
        self.store.db.commit()


def parameter_to_str(parameter):
    if parameter.units == "1":
        return f"{parameter.name} [1]"
    else:
        return f"{parameter.name} [{parameter.units:~P}]"
