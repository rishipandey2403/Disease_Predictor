import pytest

from health_assistant.ui import parse_float_inputs


def test_parse_float_inputs_success():
    values = ["1", "2.5", "3"]
    assert parse_float_inputs(values, "Test") == [1.0, 2.5, 3.0]


def test_parse_float_inputs_missing_value():
    with pytest.raises(ValueError, match="Please fill all fields"):
        parse_float_inputs(["1", "", "3"], "Test")


def test_parse_float_inputs_invalid_value():
    with pytest.raises(ValueError, match="Invalid numeric value"):
        parse_float_inputs(["1", "abc", "3"], "Test")
