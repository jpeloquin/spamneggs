from itertools import product
import os
from pathlib import Path
import pandas as pd
from lxml import etree
from .variables import *


def _to_number(s):
    """Convert numeric string to int or float as appropriate."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def _maybe_to_number(s):
    """Convert string to number if possible, otherwise return string."""
    try:
        return _to_number(s)
    except ValueError:
        return s


def scalar_from_xml(element, nominal=None, **kwargs):
    """Return Scalar object given an FEBio XML element."""
    dist = element.find("distribution")
    if dist.attrib["type"] == "uniform":
        lb = _to_number(dist.find("lb").text)
        ub = _to_number(dist.find("ub").text)
        if nominal is not None:
            nominal = _to_number(nominal)
        return UniformScalar(lb, ub, nominal=nominal, **kwargs)
    elif dist.attrib["type"] == "categorical":
        levels = (e.text for e in dist.findall("levels/level"))
        return CategoricalScalar(levels, nominal=nominal, **kwargs)
    else:
        raise ValueError(f"Distribution type '{dist.attrib['type']}' not yet supported.")


def strip_preprocessor_elems(tree):
    """Remove preprocessor elements from extended FEBio XML.

    The input tree is mutated in-place.

    """
    # Remove the <preprocessor> element b/c FEBio can't handle extra elements
    tree.getroot().remove(tree.find("preprocessor"))
    # Remove the <scalar> elements from the tree.
    for e in tree.findall(".//scalar"):
        parent = e.getparent()
        e_nominal = e.find("nominal")
        if e_nominal is None:
            raise ValueError(f"Element `{tree.getelementpath(parent)}` in file `{tree.base}` has no nominal value defined.")
        else:
            nominal = e_nominal.text.strip()
        parent.remove(e)
        parent.text = nominal


def make_sensitivity_cases(tree, nlevels):
    """Return table of cases for sensitivity analysis."""
    cases = []
    # Find all the variable parameters (at the moment, just <scalar>
    # elements) in the tree, and remember their positions in the tree by
    # storing the parent element of each.
    e_scalars = tree.findall(".//scalar")
    e_parents = {}
    for e_scalar in e_scalars:
        e_parents[e_scalar] = e_scalar.getparent()
    # Remove all the preprocessor elements; FEBio can't handle them
    strip_preprocessor_elems(tree)
    # Generate levels for each variable parameter
    colnames = []
    levels = {}
    for e_scalar in e_scalars:
        # Get / make up name for variable
        try:
            varname = e_scalar.attrib["name"]
        except KeyError:
            varname = tree.getelementpath(e_parents[e_scalar])
        colnames.append(varname)
        # Get nominal value
        e_nominal = e_scalar.find("nominal")
        if e_nominal is None:
            nominal = None
        else:
            nominal = e_nominal.text.strip()
        scalar = scalar_from_xml(e_scalar, nominal=nominal, name=varname)
        # Calculate variable's levels
        if isinstance(scalar, ContinuousScalar):
            levels[varname] = scalar.sensitivity_levels(nlevels)
        elif isinstance(scalar, CategoricalScalar):
            levels[varname] = scalar.sensitivity_levels()
        else:
            raise ValueError(f"Generating levels from a variable of type `{type(scalar)}` is not yet supported.")
    cases = pd.DataFrame({k: v for k, v in zip(colnames,
                                               zip(*product(*(levels[k]
                                                              for k in colnames))))})
    # Process variables
    xml_data = []
    for i, case in cases.iterrows():
        for e_scalar, colname in zip(e_scalars, colnames):
            e_parents[e_scalar].text = str(case[colname])
        xml_data.append(etree.tostring(tree,
                                       pretty_print=True,
                                       xml_declaration=True,
                                       encoding="utf-8"))
    cases["xml"] = xml_data
    cases = pd.DataFrame(cases)
    return cases


def sensitivity_analysis(f, nlevels, path_prefix="."):
    """Return table of cases for sensitivity analysis."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(f, parser)
    analysis_name = tree.find("preprocessor[@proc='spamneggs']/"
                              "analysis[@type='sensitivity']").attrib["name"]
    cases = make_sensitivity_cases(tree, nlevels)
    write_cases(cases, analysis_name, path_prefix=path_prefix)


def write_cases(cases, analysis_name, path_prefix="."):
    """Write tabulated analysis cases to a directory."""
    dir_out = Path(path_prefix) / Path(analysis_name)
    dir_out.mkdir(exist_ok=True)
    analysis_name = dir_out.name
    # Write each file
    case_files = []
    for i, row in cases.iterrows():
        fname = f"{analysis_name}_case={i}.feb"
        with open(dir_out / fname, "wb") as f:
            f.write(row["xml"])
        case_files.append(fname)
    # Replace XML bytes with file paths
    manifest = cases[[c for c in cases.columns if c != "xml"]]
    manifest['path'] = case_files
    manifest.to_csv(dir_out / f"{analysis_name}_manifest.csv",
                    index_label="case")
