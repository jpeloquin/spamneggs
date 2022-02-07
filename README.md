spamneggs is a Python package to run unsupervised parametric sensitivity analyses using [FEBio](https://github.com/febiosoftware/FEBio) models.

The user provides:
- A template FEBio model and a Python function to update its parameters, or a Python function to construct an FEBio model from a collection of parameter values.
- Names and ranges of each parameter.
- Output variables.  Predefined functions are avaialble for extracting output variables from FEBio text and binary (.xplt) data.
- (Optinal) Python functions to check solved models for errors.  Some checks are done by default.

Spamneggs, without additional user input:
- Generates cases spanning the parameter space using full factorial sampling
- Runs all cases
- Checks the output for errors.  By default, spamneggs checks if (a) FEBio raised an error, (b) FEBio did not write any output, and (c) if the time points in the solution match the requested observation times ("must points", in FEBio terms).  The latter of course can only be checked if all observation times were must points (i.e., `<plot_level>PLOT_MUST_POINTS</plot_level>` was set in the FEBio XML for all simulation steps).
- Extracts all output variables from the FEBio files and tabulates them
- Plots every plot that I've found to be useful in my own work

So far, spamneggs has been tested with analyses of up to 60,000 cases.

# Project status

Spamneggs was released on 2022-01-20 to https://github.com/jpeloquin/spamneggs.  I am working on user documentation.  If you are interested in being a guinea pig, please [raise an issue](https://github.com/jpeloquin/spamneggs/issues) to describe what you're trying to do.

# Using spamneggs

## Terms of use

spamneggs is licensed under the [AGPLv3](LICENSE).  You should read the entire license, but a (non-binding) explanation is given here for your convenience.  The license *allows* you to copy, modify, and use spamneggs, including for commercial use.  By doing so, you *incur obligations* to retain the original copyright and license notifications, state your modifications, and provide your modified version of the source code to your users (even web users).

## Getting started

Spamneggs depends on:
- A working Python ≥ 3.8 environment with the public packages numpy, pandas, sklean, matplotlib, lxml, and pathos; and my packages [waffleiron](https://github.com/jpeloquin/waffleiron) and (optionally; only used for some error checks) [prunetest](https://github.com/jpeloquin/prunetest).
- A working FEBio installation.  By default, waffleiron uses the command `febio` to start an FEBio process, but if the environment variable `FEBIO_CMD` is defined its value will be used to start an FEBio process.

TODO: Minimal example.

## Support

Please feel free to [raise an issue](https://github.com/jpeloquin/spamneggs/issues) to ask for assistance and I will help if able.  Due to the recency of spamneggs' release into the wild there are likely many difficulties to smooth over and I would like to hear about any you encounter.

Note however that responsibility for the accuracy or usefulness of any results you produce lies with you.  (See the license's disclaimer of fitness for any particular purpose.)  If you want me to verify that you are using the software in the correct manner and it is operating correctly, please contact me through the [Delaware Center for Musculoskeletal Research](https://sites.udel.edu/engr-dcmr/) to arrange for collaboration or contract work.

# Contributing

Contributions require a contributor license agreement (CLA).  The main motivation is to allow spamneggs to be re-licensed for use cases that were not originally anticipated.  This decision was made with the assumption that vast majority of contributions will continue to be made by the original authors.  While the CLA process is being worked on, please raise an issue if you wish to contribute.

# Similar packages

Many computer programs exist to perform parametric sensitivity analysis.  Two aspects of Spamneggs make it distinct and possibly worth using:
1. FEBio integration
2. Treats detection and reporting of errors with as having importance equal to the simulation's output variables

A non-comprehensive list of other packages for sensitivity analysis or uncertainty analysis (which can often be used for sensitivity analysis) is:
- [SALib](https://github.com/SALib/SALib)
- [Uncertainpy](https://github.com/simetenn/uncertainpy)
- [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI)
- [UQSA](https://icme.hpc.msstate.edu/mediawiki/index.php/Uncertainty_Quantification_and_Sensitivity_Analysis_Tool.html)
- [IQR Tools](https://www.intiquan.com/iqr-tools/)
- [sensitivity](https://www.rdocumentation.org/packages/caret/versions/3.45/topics/sensitivity)
