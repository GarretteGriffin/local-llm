import pytest

from tools.calculator import CalculatorTool


def test_calculator_basic_arithmetic():
    tool = CalculatorTool()
    r = tool.evaluate("(2+3)*4")
    assert r.value == 20


def test_calculator_math_functions_and_constants():
    tool = CalculatorTool()
    r = tool.evaluate("round(sin(pi/2), 6)")
    assert float(r.value) == 1.0


def test_calculator_blocks_attributes_and_imports():
    tool = CalculatorTool()

    with pytest.raises(ValueError):
        tool.evaluate("__import__('os')")

    with pytest.raises(ValueError):
        tool.evaluate("(1).__class__")


def test_calculator_blocks_statements():
    tool = CalculatorTool()
    with pytest.raises(ValueError):
        tool.evaluate("x=1")
