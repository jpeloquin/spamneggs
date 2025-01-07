"""Helper classes and functions

This module contains helper code for program execution.  Similarly basic functions to
support working with models belong in core.py.

"""

import threading
from functools import wraps
from typing import Callable


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
