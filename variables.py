from warnings import warn

import numpy as np
from scipy.stats import uniform


class Scalar:
    def __init__(self, name=None):
        self.name = name


class ContinuousScalar(Scalar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_tuple(self):
        return self.__class__.__name__, self.lb, self.ub

    def sensitivity_levels(self, n):
        raise NotImplementedError


class UniformScalar(ContinuousScalar):
    def __init__(self, lb, ub, **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def is_valid(self, value):
        if np.any(value < self.lb):
            return False
        if np.any(value > self.ub):
            return False
        return True

    def sensitivity_levels(self, n):
        levels = np.linspace(self.lb, self.ub, n)
        return levels

    def sample(self, n=1, seed=None):
        r = uniform.rvs(0, 1, n, random_state=seed)  # uniform on [0, 1]
        v = (1 - r) * self.lb + r * self.ub
        assert self.is_valid(v)
        return v

    def scale(self, v):
        """Transform value to [0, 1] uniform distribution"""
        return (v - self.lb) / (self.ub - self.lb)

    def unscale(self, vs):
        """Transform value from [0, 1] uniform distribution"""
        return vs * (self.ub - self.lb) + self.lb


class LogUniformScalar(ContinuousScalar):
    def __init__(self, lb, ub, **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def is_valid(self, value):
        if np.any(value < self.lb):
            return False
        if np.any(value > self.ub):
            return False
        return True

    def sensitivity_levels(self, n):
        levels = 10 ** np.linspace(np.log10(self.lb), np.log10(self.ub), n)
        return levels

    def sample(self, n=1, seed=None):
        r = uniform.rvs(0, 1, n, random_state=seed)  # uniform on [0, 1]
        v = 10 ** ((1 - r) * np.log10(self.lb) + r * np.log10(self.ub))
        assert self.is_valid(v)
        return v

    def scale(self, v):
        """Transform value to [0, 1] uniform distribution"""
        if v == 0:
            return -np.inf
        return (np.log10(v) - np.log10(self.lb)) / (
            np.log10(self.ub) - np.log10(self.lb)
        )

    def unscale(self, vs):
        """Transform value from [0, 1] uniform distribution"""
        return 10 ** (vs * (np.log10(self.ub) - np.log10(self.lb)) + np.log10(self.lb))


class CategoricalScalar(Scalar):
    def __init__(self, levels, name=None, nominal=None):
        self.levels = tuple(lvl for lvl in levels)
        self.name = name

    def sensitivity_levels(self):
        return self.levels
