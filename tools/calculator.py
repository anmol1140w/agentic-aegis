"""Calculator tool — deterministic arithmetic.

Demonstrates the principle: "Do NOT make the LLM the operating system."
Arithmetic is handled by deterministic code, not by asking a language
model to compute ``125 * 48``.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool


# Allowed operators & functions for safe evaluation
_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "pow": pow,
}


class _SafeEvaluator(ast.NodeVisitor):
    """Walk an AST and evaluate only safe numeric expressions."""

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    # Python < 3.12 compat
    visit_Num = visit_Constant  # type: ignore[assignment]

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_fn = _OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_fn = _OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return op_fn(left, right)

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        if not callable(func):
            raise ValueError(f"Not callable: {func!r}")
        return func(*args)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}"
        )


def safe_eval(expression: str) -> float | int:
    """Evaluate a mathematical expression safely (no exec/eval).

    Raises ``ValueError`` on anything other than numeric math.
    """
    tree = ast.parse(expression.strip(), mode="eval")
    return _SafeEvaluator().visit(tree)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Supports: +, -, *, /, //, %, ** and functions like sqrt, log, sin,
    cos, abs, round, min, max, factorial.

    Examples:
        "125 * 48"
        "sqrt(144) + 3**2"
        "log10(1000)"
    """
    try:
        result = safe_eval(expression)
        return f"{result}"
    except Exception as exc:
        return f"Error: {exc}"
