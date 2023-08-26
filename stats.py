import numpy as np
from pandas import DataFrame
import pingouin
import scipy.stats

from spamneggs.core import is_parameter_colname

COV_ZERO_THRESH = 1e-15  # Threshold at which a covariance value is treated as zero


def corr_partial(data: DataFrame, parameter_col, variable_col):
    """Return partial correlation coeffient

    :parameter data: Table of parameter and variable values for one instant in time.
    Column names ending in "[param]" will be treated as covariate parameters.

    :parameter parameter_col: Name of column in `data` with values for the correlate
    parameter.

    :parameter variable_col: Name of column in `data` with values for the correlate
    output variable.  No other output variables will be considered as covariates.

    """
    other_parameters = [
        c for c in data.columns if is_parameter_colname(c) and c != parameter_col
    ]
    x_var = np.var(data[parameter_col])
    if x_var == 0.0:
        return np.nan
    y_var = np.var(data[variable_col])
    if y_var == 0.0:
        return np.nan
    result = pingouin.partial_corr(
        data, x=parameter_col, y=variable_col, x_covar=other_parameters
    )
    return result.loc["pearson", "r"]


def corr_pearson(
    data: DataFrame, parameter_col, variable_col, cov_zero_thresh=COV_ZERO_THRESH
):
    """Return Pearson product-moment correlation coeffient

    :parameter data: Table of parameter and variable values for one instant in time.
    Column names ending in "[param]" will be treated as covariate parameters.

    :parameter parameter_col: Name of column in `data` with values for the correlate
    parameter.

    :parameter variable_col: Name of column in `data` with values for the correlate
    output variable.  No other output variables will be considered as covariates.

    :parameter cov_zero_thresh: When the variance of the output variable is less than
    `cov_zero_thresh` (i.e., all values are practically the same), the Pearson
    correlation coefficient will be returned as zero rather than NaN.  This is
    convenient and appropriate in a sensitivity analysis context.

    """
    p = data[parameter_col]
    v = data[variable_col]
    with np.errstate(divide="ignore", invalid="ignore"):
        ρ = np.corrcoef(p, v)[0, 1]
    cov = np.cov(p, v)
    if np.isnan(ρ) and (cov[0, 1] <= cov_zero_thresh and cov[1, 1] <= cov_zero_thresh):
        # Coerce correlation to zero if it is nan only because the output variable
        # has practically no variance
        return 0
    else:
        return ρ


def corr_spearman(data: DataFrame, parameter_col, variable_col):
    """Return Spearman rank correlation coefficient.

    :parameter data: Table of parameter and variable values for one instant in time.

    :parameter parameter_col: Name of column in `data` with values for the correlate
    parameter.

    :parameter variable_col: Name of column in `data` with values for the correlate
    output variable.

    """
    p = data[parameter_col]
    v = data[variable_col]
    try:
        ρ, _ = scipy.stats.spearmanr(p, v, nan_policy="propagate")
    except scipy.stats.ConstantInputWarning:
        return 0
    return ρ
