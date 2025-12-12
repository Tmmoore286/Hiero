from __future__ import annotations

import ast
import operator
from typing import Any

from ..base import ToolProtocol, ToolType


class CalculateTool(ToolProtocol):
    @property
    def name(self) -> ToolType:
        return ToolType.CALCULATE

    @property
    def description(self) -> str:
        return "Safely evaluate basic arithmetic expressions."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }

    async def execute(self, **kwargs) -> Any:
        expression = kwargs.get("expression", "")
        return {"expression": expression, "value": _safe_eval(expression)}


_OPS: dict[type, Any] = {
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


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression, mode="eval")
    return float(_eval_node(node.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        val = _eval_node(node.operand)
        return float(_OPS[type(node.op)](val))
    raise ValueError("Unsupported expression")

