"""
Symbolic Regression module for DeepChem.

This module provides symbolic regression capabilities using evolutionary
search combined with PyTorch-based constant optimization.
"""

from .operators import (
    Operator,
    Arity,
    OperatorRegistry,
    DEFAULT_REGISTRY,
    # Protected operations
    protected_div,
    protected_log,
    protected_exp,
    protected_sqrt,
    protected_pow,
    # Pre-defined operators
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_NEG,
    OP_SIN,
    OP_COS,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_SQUARE,
    OP_CUBE,
    OP_ABS,
)

from .expression import (
    NodeType,
    Node,
    Expression,
    # Factory functions
    make_constant,
    make_variable,
    make_binary,
    make_unary,
    random_terminal,
    random_expression,
)

from .symbolic_model import SymbolicRegressionModel

# Submodule imports (for advanced use)
from . import search
from . import optimization


__all__ = [
    # Model class
    'SymbolicRegressionModel',
    # Core classes
    'Operator',
    'Arity',
    'OperatorRegistry',
    'DEFAULT_REGISTRY',
    'NodeType',
    'Node',
    'Expression',
    # Factory functions
    'make_constant',
    'make_variable',
    'make_binary',
    'make_unary',
    'random_terminal',
    'random_expression',
    # Pre-defined operators
    'OP_ADD',
    'OP_SUB',
    'OP_MUL',
    'OP_DIV',
    'OP_POW',
    'OP_NEG',
    'OP_SIN',
    'OP_COS',
    'OP_EXP',
    'OP_LOG',
    'OP_SQRT',
    'OP_SQUARE',
    'OP_CUBE',
    'OP_ABS',
    # Protected operations
    'protected_div',
    'protected_log',
    'protected_exp',
    'protected_sqrt',
    'protected_pow',
    # Submodules
    'search',
    'optimization',
]

