from itertools import product
from math import inf
import os
from pathlib import Path
import pandas as pd
import re
from lxml import etree
# In-house packages
from febtools.input import read_febio_xml
from febtools.output import write_xml as write_febio_xml
# Same-package modules
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


def parse_var_selector(text):
    """Parse the parts of a variable selector for plotfiles and logfiles."""
    var_info = {"entity": None,  # element, node, connector, or body
                "entity_id": None,  # integer
                "variable": "",
                "type": "",  # instantaneous or time series
                "time_enum": None,  # time, step, or (if all time
                                    # points) None
                "time": None}
    groups = text.strip().split("@")
    entity_text = groups[0].rstrip()
    m_entity = re.match("(?P<entity>element|node|connector|body)"
                        "\[(?P<entity_id>\d+)\]"
                        "\[(\'|\"){1}(?P<variable>[^\'\"]+)(\'|\"){1}\]"
                        "(\[(?P<component>\d+(,\d+)?)\])?",
                        text)
    var_info["entity"] = m_entity["entity"]
    var_info["entity_id"] = m_entity["entity_id"]
    var_info["variable"] = m_entity["variable"]
    if len(groups) == 2:
        time_text = groups[1].lstrip()
        m_time = re.match("(?P<time_enum>time|step)"
                          "\s*=\s*"
                          "(?P<range>\d+[\S\s]*)",
                          time_text)
        var_info["time_enum"] = m_time["time_enum"]
        bounds = [v.strip() for v in m_time["range"].split("to")]
        if len(bounds) == 1:
            var_info["type"] = "instantaneous"
            var_info["time"] = _to_number(bounds[0])
        elif len(bounds) == 2:
            var_info["type"] = "time series"
            var_info["time"] = (_to_number(bounds[0]), _to_number(bounds[1]))
    else:
        var_info["type"] = "time series"
        var_info["time"] = (-inf, inf)
    return var_info


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


def gen_single_analysis(pth, dir_out="."):
    """Generate FEBio XML for single case analysis."""
    # Read the relevant analysis parameters and variables
    tree = read_febio_xml(pth)
    e_analysis = tree.find("preprocessor/analysis")
    if "name" in e_analysis.attrib:
        gen_name = e_analysis.attrib["name"]
    else:
        gen_name = Path(pth).stem
    #
    # Generate a clean FEBio XML file with plotfile and text data files
    # for the user's requested variables
    logfile_selections = {"node": {"vars": set(),
                                   "ids": set()},
                          "element": {"vars": set(),
                                      "ids": set()},
                          "body": {"vars": set(),
                                   "ids": set()},
                          "connector": {"vars": set(),
                                        "ids": set()}}
    plotfile_selections = set()
    for e in tree.findall("preprocessor/analysis/var"):
        var_info = parse_var_selector(e.text)
        if e.attrib["source"] == "logfile":
            logfile_selections[var_info["entity"]]["vars"].add(var_info["variable"])
            logfile_selections[var_info["entity"]]["ids"].add(var_info["entity_id"])
        elif e.attrib["source"] == "plotfile":
            plotfile_selections.add(var_info["variable"])
    e_output = tree.find("Output")
    # Insert logfile-related XML elements into FEBio XML
    textdata_file_tag = {"node": "node_data",
                         "element": "element_data",
                         "body": "rigid_body_data",
                         "connector": "rigid_connector_data"}
    # ^ to convert entity names to FEBio XML text data file tag names
    e_logfile = etree.Element("logfile")
    for entity_type, selections in logfile_selections.items():
        if not len(selections["vars"]) > 0:
            continue
        e_tdatafile = etree.SubElement(e_logfile, textdata_file_tag[entity_type],
            file=f"{gen_name}_-_{textdata_file_tag[entity_type]}.txt")
        e_tdatafile.attrib["data"] = ";".join([v for v in selections["vars"]])
        e_tdatafile.text = ", ".join(sorted([v for v in selections["ids"]]))
        # ^ sorting the entity IDs isn't required by FEBio (the text
        # data file will list entities in whatever order it is given),
        # but a sorted list is user-friendly.
    if len(e_logfile.getchildren()) > 0:
        e_output.append(e_logfile)
    # Insert plotfile-related XML elements into FEBio XML
    if len(plotfile_selections) > 0:
        # Re-use the existing plotfile element if the file name matches
        e_plotfile = None
        plotfile_name = f"{gen_name}.xplt"
        for e in e_output.findall("plotfile"):
            if "file" not in e.attrib:
                existing_name = gen_name + ".xplt"
            else:
                existing_name = e.attrib["file"]
            if existing_name == plotfile_name:
                if e_plotfile is None:
                    # Re-use the existing output element
                    e_plotfile = e
                    e_plotfile.attrib["file"] = plotfile_name
                    # ^ if output file name is omitted (default), make it concrete
                else:
                    # More than one <plotfile> element is pointing at
                    # the same file
                    raise ValueError(f"More than one <plotfile> element has `{plotfile_name}` as its output target.  Use a unique file name for each <plotfile> element.")
        # If no suitable existing plotfile element was found, create a
        # new one.
        if e_plotfile is None:
            e_plotfile = etree.SubElement(e_output, "plotfile",
                                          file=plotfile_name)
    for v in plotfile_selections:
        etree.SubElement(e_plotfile, "var", type=v)
    #
    # Write the clean FEBio XML file for FEBio to use
    strip_preprocessor_elems(tree)
    dir_out = Path(dir_out) / gen_name
    if not dir_out.exists():
        dir_out.mkdir(parents=True)
    if not dir_out.is_dir():
        raise ValueError(f"`{dir_out.resolve()}`, given as `{dir_out}`, is not a directory.")
    pth_out = Path(dir_out) / f"{gen_name}.feb"
    with open(pth_out, "wb") as f:
        write_febio_xml(tree, f)
    return pth_out


def sensitivity_analysis(f, nlevels, dir_out="."):
    """Return table of cases for sensitivity analysis."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(f, parser)
    analysis_name = tree.find("preprocessor[@proc='spamneggs']/"
                              "analysis[@type='sensitivity']").attrib["name"]
    cases = make_sensitivity_cases(tree, nlevels)
    write_cases(cases, analysis_name, path_prefix=path_prefix)


def write_cases(cases, analysis_name, dir_out="."):
    """Write tabulated analysis cases to a directory."""
    dir_out = Path(dir_out) / Path(analysis_name)
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
