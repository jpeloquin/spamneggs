import json
import os
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import numpy as np
from warnings import warn
# Third-party packages
import pandas as pd
# Same-package modules
from .variables import *


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
    model parameter in the sensitivity analysis.

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


class NDArrayJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


def write_record_to_json(record, f):
    json.dump(record, f, indent=2, ensure_ascii=False,
              cls=NDArrayJSONEncoder)


def plot_timeseries_var(timeseries, varname, dir_out):
    """Plot time series variable and write the plot to disk.

    This function is meant for automated sensitivity analysis.  Plots
    will be written to disk using the standard spamneggs naming
    conventions.

    TODO: Provide a companion function that just returns the plot
    handle, allowing customization.

    """
    step = timeseries["Step"]
    time = timeseries["Time"]
    values = timeseries[varname]
    # Plot the lone variable in the table on a single axes
    fig = plt.figure()
    host = HostAxes(fig, [0.125, 0.2, 0.825, 0.7])
    par1 = ParasiteAxes(host, sharey=host)
    p_time = host.plot(time, values, marker=".")
    p_step = par1.plot(step, values, visible=False)
    fig.add_axes(host)
    host.parasites.append(par1)
    host.set_xlim(0, max(time))
    par1.set_xlim(0, max(step))
    new_axisline = par1.get_grid_helper().new_fixed_axis
    par1.axis["bottom2"] = new_axisline(loc="bottom", axes=par1, offset=(0, -35))
    par1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    host.set_xlabel("Time")
    host.set_ylabel(varname)
    par1.set_xlabel("Step")
    host.set_title(varname)
    fig.savefig(os.path.join(dir_out, f"timeseries_var={varname}.svg"))
    plt.close(fig)


def plot_timeseries_vars(timeseries, dir_out):
    """Plot time series variables and write the plots to disk.

    This function is meant for automated sensitivity analysis.  Plots
    will be written to disk using the standard spamneggs naming
    conventions.

    TODO: Provide a companion function that just returns the plot
    handle, allowing customization.

    """
    if not isinstance(timeseries, pd.DataFrame):
        timeseries = pd.DataFrame(timeseries)
    if len(timeseries) == 0:
        raise ValueError("No values in time series data.")
    timeseries = timeseries.set_index("Step")
    nm_xaxis = "Time"
    nms_yaxis = [nm for nm in timeseries.columns if nm != nm_xaxis]
    # Produce one large plot with all variables
    axarr = timeseries.plot(marker=".", subplots=True, x=nm_xaxis,
                            legend=False)
    for nm, ax in zip(nms_yaxis, axarr):
        ax.set_ylabel(nm)
    # Possible future improvement: Add parasite axis for time step ID:
    # https://stackoverflow.com/questions/50521914/
    fig = axarr[0].figure
    axarr[-1].set_xlabel(nm_xaxis)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_out, "timeseries_vars.svg"))
    plt.close(fig)
    # Produce one small plot for each variable
    timeseries = timeseries.reset_index()
    for nm in nms_yaxis:
        plot_timeseries_var(timeseries, nm, dir_out)
