"""Calculator Tool

Provides safe, local evaluation of math expressions.

Security goals:
- No arbitrary code execution
- Explicit allow-list of AST nodes, operators, names, and functions
- No attribute access, indexing, comprehensions, lambdas, imports, etc.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from config import settings


@dataclass
class CalcResult:
    expression: str
    value: Any
    value_type: str


class CalculatorTool:
    def __init__(self):
        self.settings = settings

    @staticmethod
    def _allowed_names() -> Dict[str, Any]:
        # Constants
        names: Dict[str, Any] = {
            "pi": math.pi,
            "e": math.e,
            "tau": getattr(math, "tau", math.pi * 2.0),
        }

        # Safe builtins
        names.update(
            {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
            }
        )

        # Math functions
        for fn in [
            "sqrt",
            "log",
            "log10",
            "exp",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "degrees",
            "radians",
            "floor",
            "ceil",
            "factorial",
        ]:
            if hasattr(math, fn):
                names[fn] = getattr(math, fn)

        # Combinatorics (Python 3.8+)
        if hasattr(math, "comb"):
            names["comb"] = getattr(math, "comb")
        if hasattr(math, "perm"):
            names["perm"] = getattr(math, "perm")

        return names

    @staticmethod
    def _is_safe_expression(tree: ast.AST, allowed_names: Dict[str, Any]) -> Tuple[bool, str]:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Name,
            ast.Call,
            ast.Tuple,
            ast.List,
            ast.keyword,
            ast.Load,
            # Operator nodes appear in ast.walk() in Python 3.12
            ast.operator,
            ast.unaryop,
        )
        allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        allowed_unary = (ast.UAdd, ast.USub)

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False, f"Disallowed syntax: {type(node).__name__}"

            # Block attribute access entirely.
            if isinstance(node, ast.Attribute):
                return False, "Attribute access is not allowed"

            # Block subscripting/indexing (e.g., a[0]).
            if isinstance(node, ast.Subscript):
                return False, "Indexing is not allowed"

            if isinstance(node, ast.BinOp) and not isinstance(node.op, allowed_binops):
                return False, f"Disallowed operator: {type(node.op).__name__}"

            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, allowed_unary):
                return False, f"Disallowed unary operator: {type(node.op).__name__}"

            if isinstance(node, ast.Call):
                # Only allow simple calls like fn(...), not (lambda)() or obj.fn().
                if not isinstance(node.func, ast.Name):
                    return False, "Only direct function calls are allowed"

                func_name = node.func.id
                if func_name.startswith("__"):
                    return False, "Disallowed function name"
                target = allowed_names.get(func_name)
                if target is None or not callable(target):
                    return False, f"Unknown function: {func_name}"

            if isinstance(node, ast.Name):
                name = node.id
                if name.startswith("__"):
                    return False, "Disallowed name"
                if name not in allowed_names:
                    return False, f"Unknown name: {name}"

        return True, "ok"

    def evaluate(self, expression: str) -> CalcResult:
        if not expression or not isinstance(expression, str):
            raise ValueError("Empty expression")

        max_len = int(getattr(self.settings, "calculator_max_expression_length", 512) or 512)
        expr = expression.strip()
        if len(expr) > max_len:
            raise ValueError(f"Expression too long (max {max_len} chars)")

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e.msg}") from e

        env = self._allowed_names()

        ok, reason = self._is_safe_expression(tree, env)
        if not ok:
            raise ValueError(reason)

        code = compile(tree, filename="<calculator>", mode="eval")
        try:
            value = eval(code, {"__builtins__": {}}, env)  # noqa: S307 (guarded by allow-list)
        except NameError as e:
            raise ValueError(str(e)) from e

        # Normalize value type
        value_type = type(value).__name__
        return CalcResult(expression=expr, value=value, value_type=value_type)

    @staticmethod
    def format_result(result: CalcResult) -> str:
        return (
            "[Calculator]\n"
            f"expression: {result.expression}\n"
            f"result: {result.value} ({result.value_type})"
        )


def looks_like_math_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    # If the user pasted a bare arithmetic expression, treat it as math.
    # Keep this conservative to avoid false positives.
    expression_chars = set("0123456789+-*/().%^ ,")
    if len(q) <= 80 and any(c.isdigit() for c in q) and all((c.isdigit() or c in expression_chars or c.isspace()) for c in q):
        return True

    keywords = [
        "calculate",
        "compute",
        "what is ",
        "evaluate",
        "solve",
        "percentage",
        "percent",
        "margin",
        "cagr",
        "compound",
        "interest",
    ]
    return any(k in q for k in keywords)
