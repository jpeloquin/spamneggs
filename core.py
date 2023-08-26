"""Basic building blocks for modeling functionality"""
import numpy as np
import pint

ureg = pint.get_application_registry()


class Parameter:
    """A random parameter with optional reference levels."""

    def __init__(self, name, units="1"):
        self.name = name
        if units is None:
            units = "1"
        self.units = ureg.Unit(units)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.units.is_compatible_with(other.units)

    def __hash__(self):
        # We often create parameter → value mappings, so a Parameter needs to be
        # hashed by its value.  Distinct parameters /should/ have distinct names.
        return hash(self.name)

    def __repr__(self):
        return f"Parameter({self.name}, {self.units})"

    def __str__(self):
        if self.units == "1":
            return f"{self.name} [1]"
        else:
            return f"{self.name} [{self.units:~P}]"


def ordered_eig(A):
    """Return ordered eigenvalues of a matrix"""
    w, v = np.linalg.eig(A)
    idx = np.arange(len(w))
    ordered_idx = sorted(idx, key=lambda i: w[i])[::-1]
    return w[ordered_idx], v[:, ordered_idx]


def is_parameter_colname(s):
    return s.endswith("[param]")
