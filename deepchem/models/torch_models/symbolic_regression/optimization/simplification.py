"""
Rule-based expression simplification for symbolic regression.

This module provides fast algebraic simplification of expression trees
using pure Python rule-based transformations. No external dependencies.

Inspired by PySR's approach: apply algebraic identities iteratively
until no more simplifications are possible.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple

from ..expression import Node, NodeType, Expression, make_constant, make_variable, make_binary, make_unary
from ..operators import DEFAULT_REGISTRY, Arity


# Tolerance for comparing floating point values
_EPSILON = 1e-10


def simplify(expr: Expression, max_iterations: int = 10) -> Expression:
    """
    Simplify an expression using rule-based algebraic transformations.

    Applies simplification rules iteratively until the expression
    no longer changes or max iterations reached.

    Parameters
    ----------
    expr : Expression
        Expression to simplify.
    max_iterations : int
        Maximum simplification passes to prevent infinite loops.

    Returns
    -------
    Expression
        A new simplified expression (original is not modified).

    Examples
    --------
    >>> # x + 0 → x
    >>> # x * 1 → x
    >>> # x * 0 → 0
    >>> # x / x → 1
    >>> # neg(neg(x)) → x
    >>> # exp(log(x)) → x
    >>> # 2 + 3 → 5 (constant folding)
    """
    # Create a copy to avoid modifying the original
    new_expr = expr.copy()

    # Apply simplification iteratively until fixed point
    for _ in range(max_iterations):
        old_str = str(new_expr)
        new_expr.root = _simplify_node(new_expr.root)

        # Check if anything changed
        new_str = str(new_expr)
        if old_str == new_str:
            break

    new_expr.invalidate_cache()
    return new_expr


def _simplify_node(node: Node) -> Node:
    """
    Recursively simplify a node and its children.

    First simplifies children, then applies rules to this node.
    """
    if node is None:
        return node

    # Base case: terminal nodes
    if node.is_terminal:
        return node

    # Recursively simplify children first
    if node.left is not None:
        node.left = _simplify_node(node.left)
    if node.right is not None:
        node.right = _simplify_node(node.right)

    # Apply simplification rules based on node type
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

    # Get constant values if available
    left_val = _get_constant_value(left)
    right_val = _get_constant_value(right)

    # =================================================================
    # CONSTANT FOLDING: Both operands are constants
    # =================================================================
    if left_val is not None and right_val is not None:
        result = _fold_binary(op_name, left_val, right_val)
        if result is not None:
            return make_constant(result)

    # =================================================================
    # ADDITION RULES (a + b)
    # =================================================================
    if op_name == '+':
        # x + 0 → x
        if _is_zero(right):
            return left
        # 0 + x → x
        if _is_zero(left):
            return right
        # x + x → 2*x (same variable)
        if _nodes_structurally_equal(left, right):
            return make_binary(
                DEFAULT_REGISTRY.get_by_name('*'),
                make_constant(2.0),
                _copy_node(left)
            )
        # x + neg(x) → 0
        if _is_negation_of(right, left):
            return make_constant(0.0)
        # neg(x) + x → 0
        if _is_negation_of(left, right):
            return make_constant(0.0)
        # Constant combining: (a + x) + b → (a+b) + x where a,b are constants
        if left.is_binary_op and left.operator.name == '+':
            ll_val = _get_constant_value(left.left)
            if ll_val is not None and right_val is not None:
                return make_binary(
                    DEFAULT_REGISTRY.get_by_name('+'),
                    make_constant(ll_val + right_val),
                    _copy_node(left.right)
                )

    # =================================================================
    # SUBTRACTION RULES (a - b)
    # =================================================================
    elif op_name == '-':
        # x - 0 → x
        if _is_zero(right):
            return left
        # 0 - x → neg(x)
        if _is_zero(left):
            neg_op = DEFAULT_REGISTRY.get_by_name('neg')
            if neg_op:
                return make_unary(neg_op, _copy_node(right))
        # x - x → 0 (same structure)
        if _nodes_structurally_equal(left, right):
            return make_constant(0.0)
        # x - neg(y) → x + y
        if right.is_unary_op and right.operator and right.operator.name == 'neg':
            return make_binary(
                DEFAULT_REGISTRY.get_by_name('+'),
                _copy_node(left),
                _copy_node(right.left)
            )

    # =================================================================
    # MULTIPLICATION RULES (a * b)
    # =================================================================
    elif op_name == '*':
        # x * 0 → 0
        if _is_zero(right):
            return make_constant(0.0)
        # 0 * x → 0
        if _is_zero(left):
            return make_constant(0.0)
        # x * 1 → x
        if _is_one(right):
            return left
        # 1 * x → x
        if _is_one(left):
            return right
        # x * (-1) → neg(x)
        if right_val is not None and abs(right_val + 1.0) < _EPSILON:
            neg_op = DEFAULT_REGISTRY.get_by_name('neg')
            if neg_op:
                return make_unary(neg_op, _copy_node(left))
        # (-1) * x → neg(x)
        if left_val is not None and abs(left_val + 1.0) < _EPSILON:
            neg_op = DEFAULT_REGISTRY.get_by_name('neg')
            if neg_op:
                return make_unary(neg_op, _copy_node(right))
        # x * x → sq(x)
        if _nodes_structurally_equal(left, right):
            sq_op = DEFAULT_REGISTRY.get_by_name('sq')
            if sq_op:
                return make_unary(sq_op, _copy_node(left))
        # neg(x) * neg(y) → x * y
        if (left.is_unary_op and left.operator and left.operator.name == 'neg' and
            right.is_unary_op and right.operator and right.operator.name == 'neg'):
            return make_binary(
                DEFAULT_REGISTRY.get_by_name('*'),
                _copy_node(left.left),
                _copy_node(right.left)
            )
        # Constant combination: (a * x) * b → (a*b) * x
        if left.is_binary_op and left.operator.name == '*':
            ll_val = _get_constant_value(left.left)
            if ll_val is not None and right_val is not None:
                return make_binary(
                    DEFAULT_REGISTRY.get_by_name('*'),
                    make_constant(ll_val * right_val),
                    _copy_node(left.right)
                )

    # =================================================================
    # DIVISION RULES (a / b)
    # =================================================================
    elif op_name == '/':
        # x / 1 → x
        if _is_one(right):
            return left
        # 0 / x → 0 (if x != 0)
        if _is_zero(left) and not _is_zero(right):
            return make_constant(0.0)
        # x / x → 1 (if x != 0 and same structure)
        if _nodes_structurally_equal(left, right) and not _is_zero(left):
            return make_constant(1.0)
        # neg(x) / neg(y) → x / y
        if (left.is_unary_op and left.operator and left.operator.name == 'neg' and
            right.is_unary_op and right.operator and right.operator.name == 'neg'):
            return make_binary(
                DEFAULT_REGISTRY.get_by_name('/'),
                _copy_node(left.left),
                _copy_node(right.left)
            )
        # (a * x) / a → x (where a is constant, non-zero)
        if left.is_binary_op and left.operator.name == '*':
            ll_val = _get_constant_value(left.left)
            if ll_val is not None and right_val is not None and abs(ll_val - right_val) < _EPSILON and abs(right_val) > _EPSILON:
                return _copy_node(left.right)

    # =================================================================
    # POWER RULES (a ^ b)
    # =================================================================
    elif op_name == '^':
        # x ^ 0 → 1
        if _is_zero(right):
            return make_constant(1.0)
        # x ^ 1 → x
        if _is_one(right):
            return left
        # x ^ 2 → sq(x)
        if right_val is not None and abs(right_val - 2.0) < _EPSILON:
            sq_op = DEFAULT_REGISTRY.get_by_name('sq')
            if sq_op:
                return make_unary(sq_op, _copy_node(left))
        # x ^ 0.5 → sqrt(x)
        if right_val is not None and abs(right_val - 0.5) < _EPSILON:
            sqrt_op = DEFAULT_REGISTRY.get_by_name('sqrt')
            if sqrt_op:
                return make_unary(sqrt_op, _copy_node(left))

    return node


def _simplify_unary(node: Node) -> Node:
    """Apply simplification rules to unary operators."""
    child = node.left
    op_name = node.operator.name if node.operator else ""

    # Get constant value if child is constant
    child_val = _get_constant_value(child)

    # =================================================================
    # CONSTANT FOLDING: Operand is constant
    # =================================================================
    if child_val is not None:
        result = _fold_unary(op_name, child_val)
        if result is not None:
            return make_constant(result)

    # =================================================================
    # NEGATION RULES
    # =================================================================
    if op_name == 'neg':
        # neg(neg(x)) → x
        if child.is_unary_op and child.operator and child.operator.name == 'neg':
            return _copy_node(child.left)
        # neg(0) → 0 (already handled by constant folding)

    # =================================================================
    # ABSOLUTE VALUE RULES
    # =================================================================
    elif op_name == 'abs':
        # abs(abs(x)) → abs(x)
        if child.is_unary_op and child.operator and child.operator.name == 'abs':
            return _copy_node(child)
        # abs(sq(x)) → sq(x) (sq is always non-negative)
        if child.is_unary_op and child.operator and child.operator.name == 'sq':
            return _copy_node(child)
        # abs(neg(x)) → abs(x)
        if child.is_unary_op and child.operator and child.operator.name == 'neg':
            abs_op = DEFAULT_REGISTRY.get_by_name('abs')
            return make_unary(abs_op, _copy_node(child.left))

    # =================================================================
    # EXPONENTIAL AND LOGARITHM RULES
    # =================================================================
    elif op_name == 'exp':
        # exp(log(x)) → x
        if child.is_unary_op and child.operator and child.operator.name == 'log':
            return _copy_node(child.left)
        # exp(0) → 1 (constant folding)

    elif op_name == 'log':
        # log(exp(x)) → x
        if child.is_unary_op and child.operator and child.operator.name == 'exp':
            return _copy_node(child.left)
        # log(1) → 0 (constant folding)

    # =================================================================
    # SQUARE AND SQRT RULES
    # =================================================================
    elif op_name == 'sq':
        # sq(sqrt(x)) → x (for x >= 0)
        if child.is_unary_op and child.operator and child.operator.name == 'sqrt':
            return _copy_node(child.left)
        # sq(neg(x)) → sq(x)
        if child.is_unary_op and child.operator and child.operator.name == 'neg':
            sq_op = DEFAULT_REGISTRY.get_by_name('sq')
            return make_unary(sq_op, _copy_node(child.left))

    elif op_name == 'sqrt':
        # sqrt(sq(x)) → abs(x)
        if child.is_unary_op and child.operator and child.operator.name == 'sq':
            abs_op = DEFAULT_REGISTRY.get_by_name('abs')
            if abs_op:
                return make_unary(abs_op, _copy_node(child.left))

    # =================================================================
    # TRIGONOMETRIC RULES
    # =================================================================
    elif op_name == 'sin':
        # sin(0) → 0 (constant folding handles this)
        pass

    elif op_name == 'cos':
        # cos(0) → 1 (constant folding handles this)
        pass

    return node


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_constant_value(node: Optional[Node]) -> Optional[float]:
    """Extract the float value if node is a constant, else None."""
    if node is None:
        return None
    if node.is_constant and node.value is not None:
        return node.value.item()
    return None


def _is_zero(node: Optional[Node]) -> bool:
    """Check if node represents zero."""
    val = _get_constant_value(node)
    return val is not None and abs(val) < _EPSILON


def _is_one(node: Optional[Node]) -> bool:
    """Check if node represents one."""
    val = _get_constant_value(node)
    return val is not None and abs(val - 1.0) < _EPSILON


def _is_negation_of(a: Optional[Node], b: Optional[Node]) -> bool:
    """Check if a == neg(b)."""
    if a is None or b is None:
        return False
    if a.is_unary_op and a.operator and a.operator.name == 'neg':
        return _nodes_structurally_equal(a.left, b)
    return False


def _nodes_structurally_equal(a: Optional[Node], b: Optional[Node]) -> bool:
    """
    Check if two nodes are structurally equal.

    Two nodes are equal if they have the same type, operator,
    variable index, or constant value, and their children are equal.
    """
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
            return abs(a_val - b_val) < _EPSILON
        return False

    if a.is_variable:
        return a.var_index == b.var_index

    if a.is_binary_op or a.is_unary_op:
        if a.operator != b.operator:
            return False
        if not _nodes_structurally_equal(a.left, b.left):
            return False
        if not _nodes_structurally_equal(a.right, b.right):
            return False
        return True

    return False


def _copy_node(node: Node) -> Node:
    """Create a deep copy of a node."""
    if node is None:
        return None

    if node.is_constant:
        return make_constant(_get_constant_value(node) or 0.0)

    if node.is_variable:
        return make_variable(node.var_index)

    if node.is_unary_op:
        return make_unary(node.operator, _copy_node(node.left))

    if node.is_binary_op:
        return make_binary(node.operator, _copy_node(node.left), _copy_node(node.right))

    return node


def _fold_binary(op_name: str, left: float, right: float) -> Optional[float]:
    """Evaluate a binary operation on two constants. Returns None if invalid."""
    try:
        if op_name == '+':
            return left + right
        elif op_name == '-':
            return left - right
        elif op_name == '*':
            return left * right
        elif op_name == '/':
            if abs(right) > _EPSILON:
                result = left / right
                return result if math.isfinite(result) else None
            return None
        elif op_name == '^':
            if left >= 0 or right == int(right):
                result = left ** right
                return result if math.isfinite(result) else None
            return None
        return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def _fold_unary(op_name: str, value: float) -> Optional[float]:
    """Evaluate a unary operation on a constant. Returns None if invalid."""
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
                result = math.exp(value)
                return result if math.isfinite(result) else None
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
        elif op_name == 'cube':
            return value * value * value
        return None
    except (ValueError, OverflowError):
        return None


def simplify_population(expressions: List[Expression]) -> List[Expression]:
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
