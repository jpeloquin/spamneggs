from itertools import product
from math import inf
import os
from pathlib import Path
import re
from warnings import warn
# Third-party packages
import pandas as pd
from lxml import etree
# In-house packages
from febtools.input import read_febio_xml, textdata_table
from febtools.xplt import XpltData
from febtools.output import write_xml as write_febio_xml
from febtools.util import find_closest_timestep
# Same-package modules
from .variables import *


# To convert entity names to FEBio XML text data file tag names:
TAG_FOR_ENTITY_TYPE = {"node": "node_data",
                       "element": "element_data",
                       "body": "rigid_body_data",
                       "connector": "rigid_connector_data"}


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


def _parse_selector_part(text):
    """Parse selector component like name[i] or "nm1 nm2"[i]"""
    text = text.strip()
    if text.endswith("]"):
        i = text.rfind("[")
        id_text = text[i+1:-1].strip()
        if id_text.startswith("("):
            # ID part is a canonical tuple
            ids = tuple([tuple(int(a) for a in b.strip().lstrip("(").split(","))
                         for b in id_text.split(")")[:-1]])
        else:
            # ID part is an integer ID or a sequence of integer IDs
            ids = tuple(int(a) for a in id_text.split(","))
        text = text[:i]
    else:
        ids = tuple()
    name = text.strip("'").strip('"')
    return name, ids


def parse_var_selector(text):
    """Parse the parts of a variable selector for plotfiles and logfiles."""
    var_info = {"entity": None,  # element, face, node, connector, or body
                "entity ID": None,  # integer or tuple of integers
                "variable": "",
                "component": tuple(),
                "region": None,
                "parent": None,
                "type": "",  # instantaneous or time series
                "time_enum": None,  # time, step, or (if all time
                                    # points) None
                "time": None}
    groups = text.strip().split("@")
    parts = groups[0].strip().split(".")
    # Variable selector
    var_name, var_components = _parse_selector_part(parts.pop(-1))
    var_info["variable"] = var_name
    var_info["component"] = tuple(a - 1 for a in var_components)
    # Entity selector
    entity_type, entity_id = _parse_selector_part(parts.pop(0))
    var_info["entity"] = entity_type
    if entity_type in ("domain", "surface"):
        # Plotfile regions have 1-indexed IDs because they don't really
        # exist outside the plotfile, so febtools doesn't give them
        # proper IDs
        var_info["entity ID"] = entity_id[0]
    else:
        # Nodes, canonical face tuples, and elements have 0-indexed IDs
        # as far as febtools and spamneggs are concerned
        if hasattr(entity_id[0], "__iter__"):
            var_inf["entity ID"] = tuple(a - 1 for a in entity_id[0])
        else:
            var_info["entity ID"] = entity_id[0] - 1
    # Region selector (optional)
    if parts:
        region_type, region_id = _parse_selector_part(parts.pop(-1))
        var_info["region"] = {"type": region_type,
                              "ID": region_id[0]}
    # Parent selector (optional)
    if parts:
        parent_type, parent_id = _parse_selector_part(parts.pop(0))
        var_info["parent"] = {"type": parent_type,
                              "ID": parent_id[0]}
    # Time selector
    if len(groups) == 2:
        time_text = groups[1].strip()
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


def get_output_reqs(tree):
    """Return the FEBio XML output vars required for <analysis>

    `get_output_reqs` is separate from `insert_output_elem`, which
    actually inserts <Output> child elements, because the XML template
    in which the <Output> child elements are to be inserted usually has
    only FEBio-compatible elements and hence no list of analysis
    variables.

    """
    # user's variables
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
            logfile_selections[var_info["entity"]]["ids"].add(var_info["entity ID"])
        elif e.attrib["source"] == "plotfile":
            plotfile_selections.add(var_info["variable"])
    return logfile_selections, plotfile_selections


def insert_output_elem(tree, logfile_selections, plotfile_selections,
                       file_stem=None):
    """Create <Output> element in tree for variables in <analysis>.

    """
    if file_stem is None:
        file_stem = get_analysis_name(tree)

    # Find or create the <Output> element
    e_output = tree.find("Output")
    if e_output is None:
        e_output = etree.SubElement(tree.getroot(), "Output")
    # Insert logfile-related XML elements into FEBio XML
    e_logfile = etree.Element("logfile")
    for entity_type, selections in logfile_selections.items():
        if not len(selections["vars"]) > 0:
            continue
        e_tdatafile = etree.SubElement(e_logfile, TAG_FOR_ENTITY_TYPE[entity_type],
            file=f"{file_stem}_-_{TAG_FOR_ENTITY_TYPE[entity_type]}.txt")
        e_tdatafile.attrib["data"] = ";".join([v for v in selections["vars"]])
        e_tdatafile.text = ", ".join(sorted([str(v) for v in selections["ids"]]))
        # ^ sorting the entity IDs isn't required by FEBio (the text
        # data file will list entities in whatever order it is given),
        # but a sorted list is user-friendly.
    if len(e_logfile.getchildren()) > 0:
        e_output.append(e_logfile)
    # Insert plotfile-related XML elements into FEBio XML
    # Re-use the existing plotfile element; FEBio doesn't respect
    # the file name attribute and will only ever output one
    # plotfile.
    existing = set(e_output.xpath("plotfile/var/@type"))
    plotfile_selections = existing | set(plotfile_selections)
    for e in e_output.findall("plotfile"):
        e.getparent().remove(e)
    e_plotfile = etree.SubElement(e_output, "plotfile")
    for v in plotfile_selections:
        etree.SubElement(e_plotfile, "var", type=v)


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


def get_analysis_name(tree):
    e_analysis = tree.find("preprocessor/analysis")
    if "name" in e_analysis.attrib:
        name = e_analysis.attrib["name"]
    else:
        name = Path(pth).stem
    return name


def get_parameters(tree):
    # Find all of the variable parameters (at the moment, just <scalar>
    # elements) in the tree.  Note: This function somewhat overlaps in
    # purpose with code in gen_sensitivity_cases, but that code has to
    # keep track of the XML elements for each parameter.  If an Analysis
    # class is ever created, it should subsume this functionality.
    e_scalars = tree.findall(".//scalar")
    parameters = {}
    for e_scalar in e_scalars:
        # Get path of parent element
        parent_path = tree.getelementpath(e_scalar.getparent())
        # Get / make up name for variable
        try:
            pname = e_scalar.attrib["name"]
        except KeyError:
            pname = parent_path
        # Get variable's nominal value
        e_nominal = e_scalar.find("nominal")
        if e_nominal is None:
            nominal = None
        else:
            nominal = e_nominal.text.strip()
        # Store metadata about the variable
        parameters[pname] = {"xml_parent": parent_path,
                             "nominal": nominal,
                             "distribution": scalar_from_xml(e_scalar)}
    return parameters


def gen_single_analysis(pth, dir_out="."):
    """Generate FEBio XML for single case analysis."""
    # Read the relevant analysis parameters and variables
    tree = read_febio_xml(pth)
    analysis_name = get_analysis_name(tree)
    # Generate a clean FEBio XML file that FEBio can run.  Insert
    # appropritae plotfile and text data files for the user's requested
    # variables, and strip all preprocessor-related tags.
    insert_output_elem(tree, *get_output_reqs(tree),
                       file_stem=analysis_name)
    strip_preprocessor_elems(tree)
    # Write the clean FEBio XML file for FEBio to use
    dir_out = Path(dir_out) / analysis_name
    if not dir_out.exists():
        dir_out.mkdir(parents=True)
    if not dir_out.is_dir():
        raise ValueError(f"`{dir_out.resolve()}`, given as `{dir_out}`, is not a directory.")
    pth_out = Path(dir_out) / f"{analysis_name}.feb"
    with open(pth_out, "wb") as f:
        write_febio_xml(tree, f)
    return pth_out


def tabulate_case(analysis_file, case_file):
    """Tabulate variables for single case analysis."""
    analysis_file = Path(analysis_file)
    case_file = Path(case_file)
    analysis_tree = read_febio_xml(analysis_file)
    case_tree = read_febio_xml(case_file)
    analysis_name = get_analysis_name(analysis_tree)
    # Read the plotfile
    pth_xplt = case_file.with_suffix(".xplt")
    with open(pth_xplt, "rb") as f:
        xplt_data = XpltData(f.read())
    # Read the text data files
    text_data = {}
    for entity_type in ("node", "element", "body", "connector"):
        tagname = TAG_FOR_ENTITY_TYPE[entity_type]
        fname = f"{case_file.stem}_-_{tagname}.txt"
        e_textdata = case_tree.find(f"Output/logfile/{tagname}[@file='{fname}']")
        if e_textdata is not None:
            text_data[entity_type] = textdata_table(case_file.parent / fname)
        else:
            text_data[entity_type] = None
    # Extract values for each variable based on its <var> element
    record = {"instantaneous variables": {},
              "time series variables": {}}
    for e in analysis_tree.findall("preprocessor/analysis/var"):
        var_info = parse_var_selector(e.text)
        if e.attrib["source"] == "plotfile":
            # Get ID for region selector
            if var_info["region"] is None:
                region_id = None
            else:
                region_id = var_info["region"]["ID"]
            # Get ID for parent selector
            if var_info["parent"] is None:
                parent_id = None
            else:
                parent_id = var_info["parent"]["ID"]
            if var_info["type"] == "instantaneous":
                # Convert time selector to step selector
                if var_info["time_enum"] == "time":
                    step = find_closest_timestep(var_info["time"],
                                                 xplt_data.step_times,
                                                 np.array(range(len(xplt_data.step_times))))
                value = xplt_data.value(var_info["variable"], step,
                                        var_info["entity ID"],
                                        region_id, parent_id)
                # Apply component selector
                for c in var_info["component"]:
                    value = value[c]
                # Save selected value
                record["instantaneous variables"][e.attrib["name"]] =\
                    {"value": value}
            elif var_info["type"] == "time series":
                data = xplt_data.values(var_info["variable"],
                                        entity_id=var_info["entity ID"],
                                        region_id=region_id,
                                        parent_id=parent_id)
                values = np.moveaxis(np.array(data[var_info["variable"]]), 0, -1)
                for c in var_info["component"]:
                    values = values[c]
                record["time series variables"][e.attrib["name"]] =\
                    {"times": xplt_data.step_times,
                     "steps": np.array(range(len(xplt_data.step_times))),
                     "values": values}
        elif e.attrib["source"] == "logfile":
            # Apply entity type selector
            tab = text_data[var_info["entity"]]
            # Apply entity ID selector
            tab = tab[tab["Item"] == var_info["entity ID"]]
            tab = tab.set_index("Step")
            if var_info["type"] == "instantaneous":
                # Convert time selector to step selector
                if var_info["time_enum"] == "time":
                    step = find_closest_timestep(var_info["time"],
                                                 tab["Time"], tab.index)
                    if step == 0:
                        raise ValueError("Data for step = 0 requested from an FEBio text data output file, but FEBio does not provide text data output for step = 0 / time = 0.")
                    if step not in tab.index:
                        raise ValueError(f"A value for {var_info['variable']} was requested for step = {step}, but this step is not present in the FEBio text data output.")
                elif var_info["time_enum"] == "step":
                    step = var_info["time"]
                else:
                    msg = f"Time selectors of type {var_info['time_enum']} "\
                        "are not supported.  Use 'step' or 'time'."
                    raise ValueError(msg)
                # Apply variable name and time selector
                value = tab[var_info["variable"]].loc[step]
                record["instantaneous variables"][e.attrib["name"]] = {"value": value}
            elif var_info["type"] == "time series":
                record["time series variables"][e.attrib["name"]] =\
                    {"times": tab["Time"].values,
                     "steps": tab.index.values,
                     "values": tab[var_info["variable"]].values}
    # Assemble the time series table
    tab_timeseries = {"Time": [],
                      "Step": []}
    for var, d in record["time series variables"].items():
        tab_timeseries["Time"] = d["times"]
        tab_timeseries["Step"] = d["steps"]
        tab_timeseries[var] = d["values"]
    tab_timeseries = pd.DataFrame(tab_timeseries)
    return record, tab_timeseries


def gen_sensitivity_cases(tree, nlevels, analysis_dir=None):
    """Return table of cases for sensitivity analysis."""
    analysis_name = get_analysis_name(tree)
    # Figure out which directory contains all the files for the analysis
    if analysis_dir is None:
        analysis_dir = Path(analysis_name)
    if not analysis_dir.exists():
        analysis_dir.mkdir()
    # Create a subdirectory for the FEBio files
    cases_dir = analysis_dir / "cases"
    if not cases_dir.exists():
        cases_dir.mkdir()
    # Generate the cases
    analysis = {"name": analysis_name,
                "directory": analysis_dir,
                "FEBio output": get_output_reqs(tree)}
    cases = []
    # Find all the variable parameters (at the moment, just <scalar>
    # elements) in the tree, and remember position of each in the tree by
    # storing the path to its parent element.
    e_scalars = tree.findall(".//scalar")
    parameters = get_parameters(tree)
    # Remove all the preprocessor elements; FEBio can't handle them.
    # This also replaces each variable with its nominal value, but they
    # will be changed later.
    strip_preprocessor_elems(tree)
    # Generate list of cases based on combinations of levels for each
    # variable parameter
    colnames = []
    levels = {}
    for pname, mdata in parameters.items():
        colnames.append(pname)
        param = mdata["distribution"]
        # Calculate variable's levels
        if isinstance(param, ContinuousScalar):
            levels[pname] = param.sensitivity_levels(nlevels)
        elif isinstance(param, CategoricalScalar):
            levels[pname] = param.sensitivity_levels()
        else:
            raise ValueError(f"Generating levels from a variable of type `{type(var)}` is not yet supported.")
    cases = pd.DataFrame({k: v for k, v in zip(colnames,
                                               zip(*product(*(levels[k]
                                                              for k in colnames))))})
    # Modify the parameters in the XML and write the modified XML to disk
    feb_paths = []
    for i, case in cases.iterrows():
        for pname in colnames:
            # Alter the model parameters to match the current case
            e_parameter = tree.find(parameters[pname]["xml_parent"])
            assert(e_parameter is not None)
            e_parameter.text = str(case[pname])
        # Add the needed elements in <Output> to support the requested
        # variables.  We have to do this for each case because the
        # output file names are case-dependent.
        file_stem = f"{analysis['name']}_-_case={i}"
        insert_output_elem(tree, *analysis["FEBio output"],
                           file_stem=file_stem)
        # Write the modified XML to disk
        pth = cases_dir / f"{file_stem}.feb"
        with open(pth, "wb") as f:
            write_febio_xml(tree, f)
        feb_paths.append(pth.relative_to(analysis_dir))
    # TODO: Figure out same way to demarcate parameters from other
    # metadata so there are no reserved parameter names.  For example, a
    # user should be able to name their parameter "path" without
    # conflicts.
    cases["path"] = feb_paths
    cases = pd.DataFrame(cases)
    pth_cases = analysis_dir / f"{analysis_name}_-_cases.csv"
    cases.to_csv(pth_cases, index_label="case")
    return cases, pth_cases
