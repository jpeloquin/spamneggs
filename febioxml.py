from math import inf
import re

# Third-party packages
from lxml import etree

# Same-package modules
from .core import Parameter
from .variables import *


# To convert entity names to FEBio XML text data file tag names:
TAG_FOR_ENTITY_TYPE = {
    "node": "node_data",
    "element": "element_data",
    "body": "rigid_body_data",
    "connector": "rigid_connector_data",
}


# TODO: FunctionVar, XpltDataSelector, and TextDataSelector shouldn't be
# in febioxml.py, since they're fundamental datatypes for spamneggs, but
# they use parse_var_selector, so they're here to avoid a circular
# import.  Organize the files better later.


class FunctionVar:
    def __init__(self, expr, env, temporality="time series"):
        self.expr = expr
        self.env = env
        self.temporality = temporality

    def __call__(self, case):
        return eval(self.expr, self.env, {"case": case, "model": case.solution()})


class TimeSeries:
    def __init__(self, times, steps, values):
        self.time = np.array(times)
        self.step = np.array(steps)
        self.value = np.array(values)
        if __debug__:
            n = len(self.time)
            if len(self.step) != n:
                raise ValueError(
                    f"The number of step indices must equal the number of time points.  {len(self.step)} step indices and {len(self.time)} time points were provided."
                )
            if len(self.value) != n:
                raise ValueError(
                    f"The number of values must equal the number of time points.  {len(self.value)} values and {len(self.time)} time points were provided."
                )


class XpltDataSelector:
    def __init__(
        self,
        variable,
        component,
        temporality,
        time,
        time_unit,
        entity,
        entity_id,
        region=None,
        parent_entity=None,
    ):
        if __debug__:
            _validate_time_selector(temporality, time, time_unit)
        self.variable = variable
        self.component = component
        self.temporality = temporality
        self.time = time
        self.time_unit = time_unit
        self.entity = entity
        self.entity_id = entity_id
        self.region = region
        self.parent_entity = parent_entity

    @classmethod
    def from_expr(cls, expr):
        var = parse_var_selector(expr)
        return cls(
            var["variable"],
            var["component"],
            var["temporality"],
            var["time"],
            var["time_enum"],
            var["entity"],
            var["entity ID"],
            var["region"],
            var["parent"],
        )


class TextDataSelector:
    def __init__(self, variable, temporality, time, time_unit, entity, entity_id):
        if __debug__:
            _validate_time_selector(temporality, time, time_unit)
        self.variable = variable
        self.temporality = temporality
        self.time = time
        self.time_unit = time_unit
        self.entity = entity
        self.entity_id = entity_id

    @classmethod
    def from_expr(cls, expr):
        var = parse_var_selector(expr)
        return cls(
            var["variable"],
            var["temporality"],
            var["time"],
            var["time_enum"],
            var["entity"],
            var["entity ID"],
        )


def _to_number(s, dtype=float):
    """Convert numeric string to int or float as appropriate."""
    try:
        return int(s)
    except ValueError:
        return dtype(s)


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
        id_text = text[i + 1 : -1].strip()
        if id_text.startswith("("):
            # ID part is a canonical tuple
            ids = tuple(
                [
                    tuple(int(a) for a in b.strip().lstrip("(").split(","))
                    for b in id_text.split(")")[:-1]
                ]
            )
        else:
            # ID part is an integer ID or a sequence of integer IDs
            ids = tuple(int(a) for a in id_text.split(","))
        text = text[:i]
    else:
        ids = tuple()
    name = text.strip("'").strip('"')
    return name, ids


def _validate_time_selector(temporality, time, time_unit):
    if not temporality in ("instantaneous", "time series"):
        raise ValueError(
            f"`temporality` must equal 'instantaneous' or 'time series'.  '{time_unit}' was provided."
        )
    if temporality == "time series" and len(time) != 2:
        raise ValueError(
            f"For a time series variable, `time` must be a sequence of two values (lower and upper bound).  '{time}' was provided."
        )
    if not time_unit in ("step", "time"):
        raise ValueError(
            f"`time_unit` must equal 'step' or 'time'.  '{time_unit}' was provided."
        )


def parse_var_selector(text):
    """Parse the parts of a variable selector for plotfiles and logfiles."""
    var_info = {
        "entity": None,  # element, face, node, connector, or body
        "entity ID": None,  # integer or tuple of integers
        "variable": "",
        "component": tuple(),
        "region": None,
        "parent": None,
        "temporality": "",  # instantaneous or time series
        "time_enum": "step",  # time or step
        "time": None,
    }
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
        # exist outside the plotfile, so waffleiron doesn't give them
        # proper IDs
        var_info["entity ID"] = entity_id[0]
    else:
        # Nodes, canonical face tuples, and elements have 0-indexed IDs as far as
        # waffleiron and spamneggs are concerned
        if hasattr(entity_id[0], "__iter__"):
            var_info["entity ID"] = tuple(a - 1 for a in entity_id[0])
        else:
            id_ = entity_id[0]
            if id_ < 1:
                raise ValueError(
                    f"Was given an entity ID of {entity_id[0]}.  Spamneggs' variable selector syntax uses 1-indexed node, face, and element IDs for consistency with FEBio."
                )
            var_info["entity ID"] = id_ - 1
    # Region selector (optional)
    if parts:
        region_type, region_id = _parse_selector_part(parts.pop(-1))
        var_info["region"] = {"type": region_type, "ID": region_id[0]}
    # Parent selector (optional)
    if parts:
        parent_type, parent_id = _parse_selector_part(parts.pop(0))
        var_info["parent"] = {"type": parent_type, "ID": parent_id[0]}
    # Time selector
    if len(groups) == 2:
        time_text = groups[1].strip()
        m_time = re.match(
            "(?P<time_enum>time|step)" "\s*=\s*" "(?P<range>\d+[\S\s]*)", time_text
        )
        var_info["time_enum"] = m_time["time_enum"]
        bounds = [v.strip() for v in m_time["range"].split("to")]
        # Use float32 as the floating point datatype because FEBio uses
        # 32-bit floats, and we want to retain the associated precision
        # so that time point lookups can be done intelligently.
        if len(bounds) == 1:
            var_info["temporality"] = "instantaneous"
            var_info["time"] = _to_number(bounds[0], dtype=np.float32)
        elif len(bounds) == 2:
            var_info["temporality"] = "time series"
            var_info["time"] = (
                _to_number(bounds[0], dtype=np.float32),
                _to_number(bounds[1], dtype=np.float32),
            )
    else:
        var_info["temporality"] = "time series"
        var_info["time"] = (-inf, inf)
    return var_info


def scalar_from_xml(element, **kwargs):
    """Return Scalar distribution object from XML."""
    dist = element.find("distribution")
    if dist.attrib["type"] == "uniform":
        lb = _to_number(dist.find("lb").text)
        ub = _to_number(dist.find("ub").text)
        return UniformScalar(lb, ub, **kwargs)
    elif dist.attrib["type"] == "categorical":
        levels = (e.text for e in dist.findall("levels/level"))
        return CategoricalScalar(levels, **kwargs)
    else:
        raise ValueError(
            f"Distribution type '{dist.attrib['type']}' not yet supported."
        )


def required_outputs(variables):
    """Return the FEBio XML output vars required for <analysis>

    `get_output_reqs` is separate from `insert_output_elem`, which
    actually inserts <Output> child elements, because the XML template
    in which the <Output> child elements are to be inserted usually has
    only FEBio-compatible elements and hence no list of analysis
    variables.

    """
    # user's variables
    logfile_selections = {
        "node": {"vars": set(), "ids": set()},
        "element": {"vars": set(), "ids": set()},
        "body": {"vars": set(), "ids": set()},
        "connector": {"vars": set(), "ids": set()},
    }
    plotfile_selections = set()
    for nm, var in variables.items():
        if isinstance(var, TextDataSelector):
            logfile_selections[var.entity]["vars"].add(var.variable)
            logfile_selections[var.entity]["ids"].add(var.entity_id)
        elif isinstance(var, XpltDataSelector):
            plotfile_selections.add(var.variable)
    return logfile_selections, plotfile_selections


def insert_output_elem(tree, logfile_selections, plotfile_selections, file_stem):
    """Create <Output> element in tree for variables in <analysis>."""
    # Find or create the <Output> element
    e_output = tree.find("Output")
    if e_output is None:
        e_output = etree.SubElement(tree.getroot(), "Output")
    # Insert logfile-related XML elements into FEBio XML
    e_logfile = etree.Element("logfile")
    for entity_type, selections in logfile_selections.items():
        if not len(selections["vars"]) > 0:
            continue
        e_tdatafile = etree.SubElement(
            e_logfile,
            TAG_FOR_ENTITY_TYPE[entity_type],
            file=f"{file_stem}_-_{TAG_FOR_ENTITY_TYPE[entity_type]}.txt",
        )
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


def strip_preprocessor_elems(tree, parameters):
    """Remove preprocessor elements from extended FEBio XML.

    The input tree is mutated in-place.

    """
    # TODO: Separate the task of removing the preprocessor elements &
    # the task of replacing the parameter element targets with concrete
    # values, nominal or otherwise.
    #
    # Remove the <preprocessor> element b/c FEBio can't handle extra elements
    tree.getroot().remove(tree.find("preprocessor"))
    # Remove the <scalar> elements from the tree.
    for e in tree.findall(".//scalar"):
        parent = e.getparent()
        nominal_value = parameters[e.attrib["name"]].levels["nominal"]
        if nominal_value is None:
            raise ValueError(
                f"Element `{tree.getelementpath(parent)}` in file `{tree.base}` has no nominal value defined."
            )
        parent.remove(e)
        parent.text = nominal_value


def get_parameters(tree):
    # Find all of the variable parameters (at the moment, just <scalar>
    # elements) in the tree.  Note: This function somewhat overlaps in
    # purpose with code in gen_sensitivity_cases, but that code has to
    # keep track of the XML elements for each parameter.  If an Analysis
    # class is ever created, it should subsume this functionality.
    e_scalars = tree.findall(".//scalar")
    parameters = {}  # Parameter name → distribution
    parameter_locations = {}  # Parameter name → where it is is used in the XML
    #
    # Handle parameter definitions in <analysis>/<parameters>.  These
    # definitions *must* be named; no automatic name generation is
    # permitted.
    for e_parameter in tree.findall("preprocessor/analysis/parameters/scalar"):
        name = e_parameter.attrib["name"]
        levels = {
            e_level.attrib["name"]: e_level.text.strip()
            for e_level in e_parameter.findall("level")
        }
        dist = scalar_from_xml(e_parameter)
        parameters[name] = Parameter(dist, levels)
    #
    # Handle in-place parameter definitions and parameter references.
    for e_parameter in tree.xpath("*[not(name()='preprocessor')]//scalar"):
        # Get / make up name for variable
        try:
            name = e_parameter.attrib["name"]
        except KeyError:
            name = parent_path
        # Store location of parameter usage
        parent_path = tree.getelementpath(e_parameter.getparent())
        parameter_locations.setdefault(name, []).append(parent_path)
        if e_parameter.getchildren():
            # The parameter element has child elements and is therefore
            # both a parameter use and a parameter definition.
            levels = {
                e_level.attrib["name"]: e_level.text.strip()
                for e_level in e_parameter.findall("level")
            }
            dist = scalar_from_xml(e_parameter)
            parameters[name] = Parameter(dist, levels)
    return parameters, parameter_locations


def get_variables(tree):
    variables = {}
    for e in tree.findall("preprocessor/analysis/variables/var"):
        if e.attrib["source"] == "logfile":
            var = TextDataSelector.from_expr(e.text)
        elif e.attrib["source"] == "plotfile":
            var = XpltDataSelector.from_expr(e.text)
        elif e.attrib["source"] == "function":
            raise NotImplementedError
        else:
            raise ValueError(
                f"Only variables with source = 'logfile', 'plotfile', or 'function' are supported.  '{e.attrib['source']}' was provided."
            )
        variables[e.attrib["name"]] = var
    return variables
