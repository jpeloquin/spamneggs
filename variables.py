from warnings import warn

import numpy as np


class Scalar:
    def __init__(self, name=None, nominal=None):
        self.name = name
        self.nominal = nominal


class ContinuousScalar(Scalar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UniformScalar(ContinuousScalar):
    def __init__(self, lb, ub, **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def sensitivity_levels(self, n):
        if self.nominal is not None:
            # There is a nominal value; include it
            if not n % 2 == 1:
                nm = self.name if self.name is not None else "UniformScalar"
                warn(
                    f"{nm} has a nominal value ({self.nominal}).  To include a nominal value in a sensitivity analysis, the number of levels must be odd.  {n+1} levels will be used instead of the requested {n} levels."
                )
                n = n + 1
            levels = np.hstack(
                [
                    np.linspace(self.lb, self.nominal, n // 2 + 1),
                    np.linspace(self.nominal, self.ub, n // 2 + 1)[1:],
                ]
            )
        else:
            # There is no nominal value, so return evenly spaced levels
            levels = np.linspace(self.lb, self.ub, n)
        return levels


class CategoricalScalar(Scalar):
    def __init__(self, levels, name=None, nominal=None):
        self.levels = tuple(lvl for lvl in levels)
        self.name = name
        self.nominal = nominal
        if (self.nominal is not None) and (not self.nominal in self.levels):
            raise ValueError(
                f"`{self.nominal}` is not in the provided levels for the categorical variable"
            )

    def sensitivity_levels(self):
        return self.levels
