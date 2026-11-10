import h5py
import numpy as np
import pytest

from flowforge.input.TimeFunction import TimeFunction, export_time_dependence

TIMES = np.array([0.0, 0.5, 2.0])


@pytest.mark.parametrize(
    "input_dict, expected",
    [
        ({"type": "exponential", "magnitude": 2.0, "argument": 0.5}, 2.0 * np.exp(0.5 * TIMES)),
        ({"type": "linear", "intercept": 800.0, "slope": 10.0}, 800.0 + 10.0 * TIMES),
        ({"type": "quadratic", "a": 1.0, "b": 2.0, "c": 3.0}, 1.0 + 2.0 * TIMES + 3.0 * TIMES**2),
    ],
)
def test_evaluates_each_form(input_dict, expected):
    assert np.allclose(TimeFunction.from_dict(input_dict).at(TIMES), expected)


@pytest.mark.parametrize(
    "input_dict",
    [
        {"type": "linear", "intercept": 500.0, "slope": 10.0},
        {"type": "quadratic", "a": 500.0, "b": 10.0, "c": 1.0},
    ],
)
def test_converts_units_with_offset(input_dict):
    # e.g. C to K: the converted function is scale * f(t) + shift at every time
    function = TimeFunction.from_dict(input_dict)
    expected = 1.8 * function.at(TIMES) + 273.15
    function.convertUnits(1.8, 273.15)
    assert np.allclose(function.at(TIMES), expected)


def test_exponential_converts_magnitude_only():
    function = TimeFunction.from_dict({"type": "exponential", "magnitude": 2.0, "argument": 0.5})
    function.convertUnits(1000.0)
    assert function.parameters == {"magnitude": 2000.0, "argument": 0.5}


def test_exponential_rejects_offset():
    function = TimeFunction.from_dict({"type": "exponential", "magnitude": 2.0, "argument": 0.5})
    with pytest.raises(ValueError, match="offset"):
        function.convertUnits(1.0, 273.15)


@pytest.mark.parametrize(
    "input_dict",
    [
        {"type": "sinusoid", "magnitude": 1.0},
        {"type": "linear", "intercept": 1.0},
        {"type": "linear", "intercept": 1.0, "slope": 2.0, "extra": 3.0},
        {"type": "linear", "intercept": "1 + t", "slope": 2.0},
        {"type": "exponential", "magnitude": 1.0, "argument": "0.5"},
    ],
)
def test_rejects_invalid_input(input_dict):
    with pytest.raises(ValueError):
        TimeFunction.from_dict(input_dict)


def test_export_time_dependence(tmp_path):
    with h5py.File(tmp_path / "time.h5", "w") as f:
        export_time_dependence(f.create_group("table"), [1.0, 2.0])
        export_time_dependence(f.create_group("function"),
                               TimeFunction.from_dict({"type": "quadratic", "a": 1.0, "b": 2.0, "c": 3.0}))
        export_time_dependence(f.create_group("none"), None)

        assert list(f["table/time_dependent_values"][()]) == [1.0, 2.0]
        function = f["function/time_dependent_function"]
        assert function["type"][()].decode() == "quadratic"
        assert {name: function[name][()] for name in ("a", "b", "c")} == {"a": 1.0, "b": 2.0, "c": 3.0}
        assert len(f["none"]) == 0
