import os
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn


class Scalar:

    def __init__(self, name=None, nominal=None):
        self.name = name
        self.nominal = nominal


class UniformScalar(Scalar):

    def __init__(self, lb, ub, **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub

    def sensitivity_levels(self, n):
        if self.nominal is not None:
            # There is a nominal value; include it
            if not n % 2 == 1:
                nm = self.name if self.name is not None else "UniformScalar"
                warn(f"{nm} has a nominal value ({self.nominal}).  To include a nominal value in a sensitivity analysis, the number of levels must be odd.  {n+1} levels will be used instead of the requested {n} levels.")
                n = n + 1
            levels = np.hstack([np.linspace(self.lb, self.nominal, n // 2 + 1),
                                np.linspace(self.nominal, self.ub, n // 2 +1)[1:]])
        else:
            # There is no nominal value, so return evenly spaced levels
            levels = np.linspace(self.lb, self.ub, n)
        return levels


def sensitivity_loc_ind_curve(solve, cen, incr, dir_out,
                              names, units,
                              output_names, output_units):
    """Local sensitivity for curve output with independently varied params.

    solve := a function that accepts an N-tuple of model parameters and
    returns the corresponding model outputs as a tuple.  The first value
    of the output is considered the independent variable; the others are
    the dependent variables.

    cen := N-tuple of numbers.  Model parameters.  These are the central values
    about which the local sensitivity analysis is done.

    incr := N-tuple of numbers.  Increments specifying how much to change each
    model parameter in the sentivity analysis.

    dir_out := path to directory into which to write the sensitivity
    analysis plots.

    names := N-tuple of strings naming the parameters.  These will be
    used to label the plots.

    units := N-tuple of strings specifying units for the parameters.  These will be
    used to label the plots.

    """
    linestyles = "solid", "dashed", "dotted"
    for i, (c, Δ, name, unit) in enumerate(zip(cen, incr, names, units)):
        fig, ax = plt.subplots()
        ax.set_title(f"Local model sensitivity to independent variation of {name}")
        ax.set_xlabel(f"{output_names[0]} [{output_units[0]}]")
        ax.set_ylabel(f"{output_names[1]} [{output_units[1]}]")
        # Plot central value
        p = [x for x in cen]
        output = solve(p)
        ax.plot(output[0], output[1], linestyle=linestyles[0],
                color="black",
                label=f"{name} = {c}")
        # Plot ± increments
        for j in (1, 2):
            # High
            p[i] = c + j*Δ
            output = solve(p)
            ax.plot(output[0], output[1], linestyle=linestyles[j],
                    color="black",
                    label=f"{name} = {c} ± {j*Δ}")
            # Low
            p[i] = c - j*Δ
            output = solve(p)
            ax.plot(output[0], output[1], linestyle=linestyles[j],
                    color="black")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(dir_out, f"sensitivity_local_ind_curve_-_{name}.svg"))
