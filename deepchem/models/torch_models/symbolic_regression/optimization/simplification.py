"""
Expression simplification using SymPy.

This module provides algebraic simplification of expression trees
using SymPy's powerful symbolic mathematics engine.
"""

from __future__ import annotations

from typing import Dict, Optional, List
import logging

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from ..expression import Node, NodeType, Expression, make_constant, make_variable, make_binary, make_unary
from ..operators import DEFAULT_REGISTRY, Arity

logger = logging.getLogger(__name__)


def simplify(expr: Expression, use_sympy: bool = True) -> Expression:
    """
    Simplify an expression using SymPy.

    Converts the expression tree to a SymPy expression, simplifies it,
    and converts back to our Expression format.

    Parameters
    ----------
    expr : Expression
        Expression to simplify.
    use_sympy : bool
        If True, use SymPy for full algebraic simplification.
        If False or SymPy unavailable, returns a copy unchanged.

    Returns
    -------
    Expression
        A new simplified expression (original is not modified).

    Examples
    --------
    >>> # x + x → 2*x
    >>> # x * 0 → 0
    >>> # ((x0 + x1) + x0) → 2*x0 + x1
    """
    if not use_sympy or not HAS_SYMPY:
        logger.warning("SymPy not available, returning expression unchanged")
        return expr.copy()

    try:
        # Convert to SymPy
        sympy_expr, symbols = _to_sympy(expr.root)

        # Simplify
        simplified_sympy = sp.simplify(sympy_expr)

        # Convert back
        new_root = _from_sympy(simplified_sympy, symbols)
        new_expr = Expression(new_root)
        new_expr.invalidate_cache()

        return new_expr

    except Exception as e:
        # Silently fall back to original - complex numbers or unsupported expressions
        logger.debug(f"Simplification skipped: {e}")
        return expr.copy()


def _to_sympy(node: Node, symbols: Optional[Dict[int, sp.Symbol]] = None) -> tuple:
    """
    Convert an Expression Node to a SymPy expression.

    Returns both the SymPy expression and a mapping of variable indices to symbols.
    """
    if symbols is None:
        symbols = {}

    if node.is_constant:
        value = node.value.item() if node.value is not None else 0.0
        return sp.Float(value), symbols

    elif node.is_variable:
        idx = node.var_index
        if idx not in symbols:
            symbols[idx] = sp.Symbol(f'x{idx}', real=True)
        return symbols[idx], symbols

    elif node.is_binary_op:
        left_sp, symbols = _to_sympy(node.left, symbols)
        right_sp, symbols = _to_sympy(node.right, symbols)

        op_name = node.operator.name if node.operator else '+'

        if op_name == '+':
            return left_sp + right_sp, symbols
        elif op_name == '-':
            return left_sp - right_sp, symbols
        elif op_name == '*':
            return left_sp * right_sp, symbols
        elif op_name == '/':
            return left_sp / right_sp, symbols
        elif op_name == '^':
            return left_sp ** right_sp, symbols
        else:
            raise ValueError(f"Unknown binary operator: {op_name}")

    elif node.is_unary_op:
        child_sp, symbols = _to_sympy(node.left, symbols)

        op_name = node.operator.name if node.operator else 'neg'

        if op_name == 'neg':
            return -child_sp, symbols
        elif op_name == 'abs':
            return sp.Abs(child_sp), symbols
        elif op_name == 'sin':
            return sp.sin(child_sp), symbols
        elif op_name == 'cos':
            return sp.cos(child_sp), symbols
        elif op_name == 'exp':
            return sp.exp(child_sp), symbols
        elif op_name == 'log':
            return sp.log(child_sp), symbols
        elif op_name == 'sqrt':
            return sp.sqrt(child_sp), symbols
        elif op_name == 'sq':
            return child_sp ** 2, symbols
        elif op_name == 'cube':
            return child_sp ** 3, symbols
        else:
            raise ValueError(f"Unknown unary operator: {op_name}")

    raise ValueError(f"Unknown node type: {node.node_type}")


def _from_sympy(sympy_expr, symbols: Dict[int, sp.Symbol]) -> Node:
    """
    Convert a SymPy expression back to our Node format.
    """
    # Reverse symbol mapping
    symbol_to_idx = {sym: idx for idx, sym in symbols.items()}

    return _convert_sympy_node(sympy_expr, symbol_to_idx)


def _convert_sympy_node(expr, symbol_to_idx: Dict[sp.Symbol, int]) -> Node:
    """Recursively convert SymPy expression to Node."""

    # Number (constant)
    if expr.is_number:
        # Handle complex numbers by taking real part if imaginary is negligible
        val = complex(expr)
        if abs(val.imag) < 1e-10:
            return make_constant(float(val.real))
        else:
            raise ValueError(f"Complex number encountered: {expr}")

    # Symbol (variable)
    if expr.is_Symbol:
        if expr in symbol_to_idx:
            return make_variable(symbol_to_idx[expr])
        else:
            # Unknown symbol - extract index from name
            name = str(expr)
            if name.startswith('x') and name[1:].isdigit():
                return make_variable(int(name[1:]))
            raise ValueError(f"Unknown symbol: {expr}")

    # Addition
    if expr.is_Add:
        args = list(expr.args)
        result = _convert_sympy_node(args[0], symbol_to_idx)
        for arg in args[1:]:
            right = _convert_sympy_node(arg, symbol_to_idx)
            result = make_binary(DEFAULT_REGISTRY.get_by_name('+'), result, right)
        return result

    # Multiplication
    if expr.is_Mul:
        args = list(expr.args)
        result = _convert_sympy_node(args[0], symbol_to_idx)
        for arg in args[1:]:
            right = _convert_sympy_node(arg, symbol_to_idx)
            result = make_binary(DEFAULT_REGISTRY.get_by_name('*'), result, right)
        return result

    # Power
    if expr.is_Pow:
        base = _convert_sympy_node(expr.base, symbol_to_idx)
        exp = expr.exp

        # Special cases
        if exp == 2:
            op = DEFAULT_REGISTRY.get_by_name('sq')
            if op:
                return make_unary(op, base)
        elif exp == 3:
            op = DEFAULT_REGISTRY.get_by_name('cube')
            if op:
                return make_unary(op, base)
        elif exp == sp.Rational(1, 2):
            op = DEFAULT_REGISTRY.get_by_name('sqrt')
            if op:
                return make_unary(op, base)

        # General power
        exp_node = _convert_sympy_node(exp, symbol_to_idx)
        pow_op = DEFAULT_REGISTRY.get_by_name('^')
        if pow_op:
            return make_binary(pow_op, base, exp_node)
        # Fallback: no power op, approximate
        return base

    # Function applications
    if isinstance(expr, sp.sin):
        arg = _convert_sympy_node(expr.args[0], symbol_to_idx)
        return make_unary(DEFAULT_REGISTRY.get_by_name('sin'), arg)

    if isinstance(expr, sp.cos):
        arg = _convert_sympy_node(expr.args[0], symbol_to_idx)
        return make_unary(DEFAULT_REGISTRY.get_by_name('cos'), arg)

    if isinstance(expr, sp.exp):
        arg = _convert_sympy_node(expr.args[0], symbol_to_idx)
        return make_unary(DEFAULT_REGISTRY.get_by_name('exp'), arg)

    if isinstance(expr, sp.log):
        arg = _convert_sympy_node(expr.args[0], symbol_to_idx)
        return make_unary(DEFAULT_REGISTRY.get_by_name('log'), arg)

    if isinstance(expr, sp.Abs):
        arg = _convert_sympy_node(expr.args[0], symbol_to_idx)
        return make_unary(DEFAULT_REGISTRY.get_by_name('abs'), arg)

    # Negation (handled as -1 * expr in SymPy, but we check)
    if expr.is_Mul and len(expr.args) == 2 and expr.args[0] == -1:
        arg = _convert_sympy_node(expr.args[1], symbol_to_idx)
        neg_op = DEFAULT_REGISTRY.get_by_name('neg')
        if neg_op:
            return make_unary(neg_op, arg)

    # Fallback: try to evaluate numerically
    try:
        val = complex(expr.evalf())
        if abs(val.imag) < 1e-10:
            return make_constant(float(val.real))
        else:
            raise ValueError(f"Complex result: {expr}")
    except Exception as e:
        raise ValueError(f"Cannot convert SymPy expression: {expr} (type: {type(expr)}, error: {e})")


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
