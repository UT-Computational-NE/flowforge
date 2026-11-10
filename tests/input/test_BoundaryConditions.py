import pytest

from flowforge.input.BoundaryConditions import *
from flowforge.input.UnitConverter import UnitConverter


def test_GeneralBC():

    bc_fluid = GeneralBC("inlet", "temperature", "700")
    bc_solid = GeneralBC("bottom", "solid_temperature", "700")

    assert bc_fluid.boundary_type is None
    assert bc_fluid.boundary_value.evaluate() == 700
    assert bc_fluid.variable_name == "temperature"
    assert bc_fluid.surface_name == "inlet"

    assert bc_solid.boundary_type is None
    assert bc_solid.boundary_value.evaluate() == 700
    assert bc_solid.variable_name == "solid_temperature"
    assert bc_solid.surface_name == "bottom"


def test_DirichletBC():

    bc_fluid = DirichletBC("inlet", "temperature", "700")
    assert bc_fluid.boundary_type == "DirichletBC"

def test_BoundaryConditions():

    boundary_conditions = {
        "inlet_mdot" : {"boundary_type": "DirichletBC", "surface": "inlet", "variable": "mass_flow_rate", "value": 25.0},
        "outlet_pressure" : {"boundary_type": "DirichletBC", "surface": "outlet", "variable": "pressure", "value": 1e5},
        "inlet_temp" : {"boundary_type": "DirichletBC", "surface": "inlet", "variable": "temperature", "value": 700},
    }

    BC = BoundaryConditions(**boundary_conditions)
    for bc_name, bc in BC.boundary_conditions.items():
        assert bc.boundary_type == boundary_conditions[bc_name]["boundary_type"]
        assert bc.surface_name == boundary_conditions[bc_name]["surface"]
        assert bc.variable_name == boundary_conditions[bc_name]["variable"]
        assert bc.boundary_value.evaluate()  == boundary_conditions[bc_name]["value"]


def test_BoundaryConditions_value_types():
    # numbers and strings give the same boundary value
    values = {"int": 1, "float": 1.0, "string": "1"}
    BC = BoundaryConditions(**{name: {"boundary_type": "DirichletBC", "surface": "inlet", "variable": "pressure",
                                      "value": value} for name, value in values.items()})
    for bc in BC.boundary_conditions.values():
        assert bc.boundary_value.evaluate() == 1
        assert bc.time_function is None


def test_BoundaryConditions_time_function_converts_units():
    BC = BoundaryConditions(inlet_temp={"boundary_type": "DirichletBC", "surface": "inlet", "variable": "temperature",
                                        "value": {"type": "linear", "intercept": 500.0, "slope": 10.0}})
    BC._convertUnits(UnitConverter({"temperature": "C"}))
    bc = BC.boundary_conditions["inlet_temp"]

    # the boundary value is the time function's value at t = 0, both in K
    assert bc.time_function.at(0.0) == pytest.approx(773.15)
    assert bc.time_function.at(2.0) == pytest.approx(793.15)
    assert bc.boundary_value.evaluate() == pytest.approx(773.15)


if __name__ == "__main__":
    test_GeneralBC()
    test_DirichletBC()

    test_BoundaryConditions()