"""Helper classes and functions

This module is focused on helper code for program execution.  For similarly basic
building blocks to support working with models, see core.py.
"""
import threading

class Counter:
    """Threadsafe counter"""
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def __str__(self):
        return str(self.count)

    def increment(self):
        with self.lock:
            self.count += 1


def parameter_to_str(parameter):
    if parameter.units == "1":
        return f"{parameter.name} [1]"
    else:
        return f"{parameter.name} [{parameter.units:~P}]"
