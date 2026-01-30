"""
Expression simplification for symbolic regression.

This module provides algebraic simplification of expression trees
to reduce bloat and improve interpretability.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..expression import Node, NodeType, Expression, make_constant, make_variable
from ..operators import Arity


def simplify(expr: Expression) -> Expression:
    """
    Simplify an expression using algebraic rules.

    Applies rule-based simplifications to reduce expression complexity
    while preserving mathematical equivalence.

    Parameters
    ----------
    expr : Expression
        Expression to simplify.

    Returns
    -------
    Expression
        A new simplified expression (original is not modified).

    Examples
    --------
    >>> # x + 0 → x
    >>> # x * 1 → x
    >>> # x * 0 → 0
    >>> # 2 + 3 → 5 (constant folding)
    """
    # Create a copy
    new_expr = expr.copy()

    # Apply simplifications recursively
    new_expr.root = _simplify_node(new_expr.root)

    # Invalidate cache since structure may have changed
    new_expr.invalidate_cache()

    return new_expr


def _simplify_node(node: Node) -> Node:
    """Recursively simplify a node and its children."""
    # Base case: terminal nodes don't simplify
    if node.is_terminal:
        return node

    # First, simplify children
    if node.left is not None:
        node.left = _simplify_node(node.left)
    if node.right is not None:
        node.right = _simplify_node(node.right)

    # Now apply simplification rules
    if node.is_binary_op:
        return _simplify_binary(node)
    elif node.is_unary_op:
        return _simplify_unary(node)

    return node


def _simplify_binary(node: Node) -> Node:
    """Apply simplification rules to binary operators."""
    left = node.left
    right = node.right
    op_name = node.operator.name if node.operator else ""

    # Extract constant values if both children are constants
    left_val = _get_constant_value(left)
    right_val = _get_constant_value(right)

    # Constant folding: both children are constants
    if left_val is not None and right_val is not None:
        result = _evaluate_binary_op(op_name, left_val, right_val)
        if result is not None and math.isfinite(result):
            return make_constant(result)

    # Addition rules
    if op_name == '+':
        # x + 0 → x
        if right_val is not None and abs(right_val) < 1e-10:
            return left
        # 0 + x → x
        if left_val is not None and abs(left_val) < 1e-10:
            return right

    # Subtraction rules
    elif op_name == '-':
        # x - 0 → x
        if right_val is not None and abs(right_val) < 1e-10:
            return left
        # x - x → 0 (same variable)
        if _nodes_equal(left, right):
            return make_constant(0.0)

    # Multiplication rules
    elif op_name == '*':
        # x * 0 → 0
        if right_val is not None and abs(right_val) < 1e-10:
            return make_constant(0.0)
        # 0 * x → 0
        if left_val is not None and abs(left_val) < 1e-10:
            return make_constant(0.0)
        # x * 1 → x
        if right_val is not None and abs(right_val - 1.0) < 1e-10:
            return left
        # 1 * x → x
        if left_val is not None and abs(left_val - 1.0) < 1e-10:
            return right

    # Division rules
    elif op_name == '/':
        # x / 1 → x
        if right_val is not None and abs(right_val - 1.0) < 1e-10:
            return left
        # x / x → 1 (same variable, non-zero)
        if _nodes_equal(left, right) and not (left.is_constant and abs(_get_constant_value(left) or 0) < 1e-10):
            return make_constant(1.0)

    return node


def _simplify_unary(node: Node) -> Node:
    """Apply simplification rules to unary operators."""
    child = node.left
    op_name = node.operator.name if node.operator else ""

    # Get constant value if child is constant
    child_val = _get_constant_value(child)

    # Constant folding
    if child_val is not None:
        result = _evaluate_unary_op(op_name, child_val)
        if result is not None and math.isfinite(result):
            return make_constant(result)

    # Double negation: neg(neg(x)) → x
    if op_name == 'neg':
        if child.is_unary_op and child.operator and child.operator.name == 'neg':
            return child.left

    # Identity compositions
    # exp(log(x)) → x (approximately, for positive x)
    # log(exp(x)) → x
    if op_name == 'exp' and child.is_unary_op and child.operator and child.operator.name == 'log':
        return child.left
    if op_name == 'log' and child.is_unary_op and child.operator and child.operator.name == 'exp':
        return child.left

    # abs(abs(x)) → abs(x)
    if op_name == 'abs' and child.is_unary_op and child.operator and child.operator.name == 'abs':
        return child

    # sq(sqrt(x)) → x, sqrt(sq(x)) → abs(x)
    if op_name == 'sq' and child.is_unary_op and child.operator and child.operator.name == 'sqrt':
        return child.left

    return node


def _get_constant_value(node: Optional[Node]) -> Optional[float]:
    """Extract the float value if node is a constant."""
    if node is None:
        return None
    if node.is_constant and node.value is not None:
        return node.value.item()
    return None


def _nodes_equal(a: Optional[Node], b: Optional[Node]) -> bool:
    """Check if two nodes are structurally equal."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if a.node_type != b.node_type:
        return False

    if a.is_constant:
        a_val = _get_constant_value(a)
        b_val = _get_constant_value(b)
        if a_val is not None and b_val is not None:
            return abs(a_val - b_val) < 1e-10
        return False

    if a.is_variable:
        return a.var_index == b.var_index

    if a.is_binary_op or a.is_unary_op:
        if a.operator != b.operator:
            return False
        return _nodes_equal(a.left, b.left) and _nodes_equal(a.right, b.right)

    return False


def _evaluate_binary_op(op_name: str, left: float, right: float) -> Optional[float]:
    """Evaluate a binary operation on two constants."""
    try:
        if op_name == '+':
            return left + right
        elif op_name == '-':
            return left - right
        elif op_name == '*':
            return left * right
        elif op_name == '/':
            if abs(right) > 1e-10:
                return left / right
            return None
        else:
            return None
    except Exception:
        return None


def _evaluate_unary_op(op_name: str, value: float) -> Optional[float]:
    """Evaluate a unary operation on a constant."""
    try:
        if op_name == 'neg':
            return -value
        elif op_name == 'abs':
            return abs(value)
        elif op_name == 'sin':
            return math.sin(value)
        elif op_name == 'cos':
            return math.cos(value)
        elif op_name == 'exp':
            if value <= 20:  # Prevent overflow
                return math.exp(value)
            return None
        elif op_name == 'log':
            if value > 0:
                return math.log(value)
            return None
        elif op_name == 'sqrt':
            if value >= 0:
                return math.sqrt(value)
            return None
        elif op_name == 'sq':
            return value * value
        else:
            return None
    except Exception:
        return None


def simplify_population(expressions: list) -> list:
    """
    Simplify a list of expressions.

    Parameters
    ----------
    expressions : List[Expression]
        Expressions to simplify.

    Returns
    -------
    List[Expression]
        Simplified expressions.
    """
    return [simplify(expr) for expr in expressions]
