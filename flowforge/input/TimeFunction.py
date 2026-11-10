import re
from typing import Any, Dict, List, Mapping, Optional, Union

import h5py
import numpy as np

# How each parameter of a functional form converts with the units of the value:
#   "scale_and_offset": scales and takes the unit offset (e.g. C to K); the value at t = 0
#   "scale":            scales only
#   None:               does not convert (e.g. the exponential's argument)
# A form without a "scale_and_offset" parameter can not take an offset unit: for example
# magnitude * exp(argument * t) + 273.15 is not an exponential.
FUNCTIONAL_FORMS: Dict[str, Dict[str, Optional[str]]] = {
    "exponential": {"magnitude": "scale", "argument": None},
    "linear": {"intercept": "scale_and_offset", "slope": "scale"},
    "quadratic": {"a": "scale_and_offset", "b": "scale", "c": "scale"},
}


class TimeFunction:
    """A value as a function of time, given by a functional form and its parameters.

    - exponential: magnitude * exp(argument * t)
    - linear:      intercept + slope * t
    - quadratic:   a + b * t + c * t^2

    Attributes:
        functional_form: Name of the functional form, e.g. ``"linear"``.
        parameters: The form's parameters by name.
    """

    def __init__(self, functional_form: str, **parameters: float):
        """
        Args:
            functional_form: One of ``FUNCTIONAL_FORMS``.
            parameters: Every parameter of the form, by name.

        Raises:
            ValueError: If the form is unknown or its parameters don't match.
        """
        if functional_form not in FUNCTIONAL_FORMS:
            raise ValueError(f"Unsupported functional form: {functional_form!r}. "
                             f"Valid forms are {list(FUNCTIONAL_FORMS)}.")
        expected = set(FUNCTIONAL_FORMS[functional_form])
        if set(parameters) != expected:
            raise ValueError(f"The {functional_form} form takes parameters {sorted(expected)}, "
                             f"not {sorted(parameters)}.")
        self.functional_form = functional_form
        self.parameters = {name: float(value) for name, value in parameters.items()}

    @staticmethod
    def parameter_names(functional_form: str) -> List[str]:
        """The parameters of a functional form."""
        return list(FUNCTIONAL_FORMS[functional_form])

    @staticmethod
    def converts_units(functional_form: str, parameter: str) -> bool:
        """Whether a parameter of a functional form converts with the units of the value."""
        return FUNCTIONAL_FORMS[functional_form][parameter] is not None

    @classmethod
    def from_dict(cls, input_dict: Mapping[str, Any]) -> "TimeFunction":
        """Builds a time function from input, e.g. ``{"type": "linear", "intercept": 800, "slope": 10}``."""
        validate_time_function_input(input_dict)
        return cls(input_dict["type"], **{name: input_dict[name] for name in cls.parameter_names(input_dict["type"])})

    def at(self, time: Union[float, np.ndarray]):
        """Evaluates the function at the given time(s)."""
        p = self.parameters
        if self.functional_form == "exponential":
            return p["magnitude"] * np.exp(p["argument"] * time)
        if self.functional_form == "linear":
            return p["intercept"] + p["slope"] * time
        return p["a"] + p["b"] * time + p["c"] * time**2

    def convertUnits(self, scale_factor: float, shift_factor: float = 0.0) -> None:
        """Converts the function's units, so it gives ``scale_factor * f(t) + shift_factor``.

        Args:
            scale_factor: Multiplies every parameter that converts with the value's units.
            shift_factor: Added to the value at t = 0 (e.g. 273.15 for C to K).

        Raises:
            ValueError: If there is an offset but the form can not take one (exponential).
        """
        kinds = FUNCTIONAL_FORMS[self.functional_form]
        if shift_factor != 0.0 and "scale_and_offset" not in kinds.values():
            raise ValueError(f"The {self.functional_form} form can not convert units with an offset "
                             "(e.g. C to K). Give it in units without an offset, or use another form.")
        for name, kind in kinds.items():
            if kind is not None:
                self.parameters[name] *= scale_factor
            if kind == "scale_and_offset":
                self.parameters[name] += shift_factor

    def create_group(self, base_h5_group: h5py.Group, new_group_name: str = "time_dependent_function"):
        """Writes the function to a new HDF5 group (its type and each parameter).

        Args:
            base_h5_group: Parent HDF5 group.
            new_group_name: Name of the group to create.
        """
        new_group = base_h5_group.create_group(new_group_name)
        new_group.create_dataset("type", data=self.functional_form)
        for name, value in self.parameters.items():
            new_group.create_dataset(name, data=value, dtype=float)


def validate_time_function_input(input_dict: Mapping[str, Any]) -> None:
    """Checks a time function input, e.g. ``{"type": "exponential", "magnitude": 800, "argument": 0.01}``.

    Parameters that convert with units may be numbers or (spatial) expressions, but not functions
    of t. The others must be numbers.

    Raises:
        ValueError: If the type is unknown, a parameter is missing or extra, or a value is invalid.
    """
    functional_form = input_dict.get("type")
    if functional_form not in FUNCTIONAL_FORMS:
        raise ValueError(f"Unsupported functional form: {functional_form!r}. Valid forms are {list(FUNCTIONAL_FORMS)}.")
    expected = set(FUNCTIONAL_FORMS[functional_form])
    given = set(input_dict) - {"type"}
    if given != expected:
        raise ValueError(f"The {functional_form} form takes parameters {sorted(expected)}, not {sorted(given)}.")
    for name in expected:
        value = input_dict[name]
        if TimeFunction.converts_units(functional_form, name):
            if not isinstance(value, (int, float, str)):
                raise ValueError(f"Parameter {name!r} must be a number or an expression, not {value!r}.")
            if isinstance(value, str) and "t" in re.findall(r"[A-Za-z_]\w*", value):
                raise ValueError(f"Parameter {name!r} can not be a function of t.")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {name!r} must be a number, not {value!r}.")


def export_time_dependence(group: h5py.Group, time_dependent_values: Union[None, list, TimeFunction]) -> None:
    """Writes time dependence to an HDF5 group, in the form syTH reads it.

    Args:
        group: Group to write to (e.g. a body force or fixed field group).
        time_dependent_values: A list of values at each time step (written as
            ``time_dependent_values``), a ``TimeFunction`` (written as
            ``time_dependent_function``), or None (nothing is written).
    """
    if time_dependent_values is None:
        return
    if isinstance(time_dependent_values, list):
        group.create_dataset("time_dependent_values", data=time_dependent_values)
    elif isinstance(time_dependent_values, TimeFunction):
        time_dependent_values.create_group(group)
    else:
        raise TypeError(f"Invalid time dependent values type: {type(time_dependent_values).__name__}")
