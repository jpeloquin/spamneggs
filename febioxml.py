from itertools import product
import os
from pathlib import Path
import pandas as pd
from lxml import etree
from .spamneggs import UniformScalar


def _to_number(s):
    """Convert numeric string to int or float as appropriate."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def scalar_from_xml(element, **kwargs):
    """Return Scalar object given an FEBio XML element."""
    dist = element.find("distribution")
    if dist.attrib["type"] == "uniform":
        lb = _to_number(dist.find("lb").text)
        ub = _to_number(dist.find("ub").text)
        return UniformScalar(lb, ub, **kwargs)
    else:
        raise ValueError(f"Distribution type '{dist.attrib['type']}' not yet supported.")


def make_sensitivity_cases(tree, nlevels):
    """Return table of cases for sensitivity analysis."""
    cases = []
    # Remove the <preprocessor> element b/c FEBio can't handle extra elements
    tree.getroot().remove(tree.find("preprocessor"))
    # Remove the <scalar> elements from the tree and generate levels for
    # each.  We need an FEBio-compatible tree before we can write the
    # XML for each combination of levels, so that must be done later in
    # a separate loop.
    e_scalars = tree.findall(".//scalar")
    e_parents = {}
    colnames = []
    levels = {}
    for e_scalar in e_scalars:
        # Remember the parent element
        e_parents[e_scalar] = e_scalar.getparent()
        # Remove the <scalar> element to leave an FEBio-compatible tree
        e_scalar.getparent().remove(e_scalar)
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
            # A nominal value is defined
            nominal = _to_number(e_nominal.text)
        scalar = scalar_from_xml(e_scalar, nominal=nominal, name=varname)
        # Calculate variable's levels
        levels[varname] = scalar.sensitivity_levels(nlevels)
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
    analysis_name = tree.find("preprocessor[@name='spamneggs']/"
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
