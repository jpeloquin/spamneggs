"""Basic building blocks for modeling functionality"""
import pint

ureg = pint.get_application_registry()



class Parameter:
    """A random parameter with optional reference levels."""

    def __init__(self, name, units="1"):
        self.name = name
        if units is None:
            units = "1"
        self.units = ureg.Unit(units)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.units.is_compatible_with(other.units)

    def __hash__(self):
        # We often create parameter → value mappings, so a Parameter needs to be
        # hashed by its value.  Distinct parameters /should/ have distinct names.
        return hash(self.name)

    def __repr__(self):
        return f"Parameter({self.name}, {self.units})"

    def __str__(self):
        return f"{self.name} [{self.units}]"
