from warnings import warn

import numpy as np
from scipy.stats import uniform


class Scalar:
    def __init__(self, name=None):
        self.name = name


class ContinuousScalar(Scalar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sensitivity_levels(self, n):
        raise NotImplementedError


class UniformScalar(ContinuousScalar):
    def __init__(self, lb, ub, **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def sensitivity_levels(self, n):
        levels = np.linspace(self.lb, self.ub, n)
        return levels

    def sample(self):
        r = uniform.rvs(0, 1)  # uniform on [0, 1]
        return (1 - r) * self.lb + r * self.ub


class CategoricalScalar(Scalar):
    def __init__(self, levels, name=None, nominal=None):
        self.levels = tuple(lvl for lvl in levels)
        self.name = name

    def sensitivity_levels(self):
        return self.levels
