"""
Operator definitions for symbolic regression.

This module defines the mathematical operators available for building
expression trees, including protected versions that handle edge cases
(division by zero, overflow, etc.) gracefully.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from enum import Enum, auto


class Arity(Enum):
    """Operator arity (number of operands)."""
    UNARY = auto()
    BINARY = auto()


@dataclass(frozen=True)
class Operator:
    """
    Definition of a mathematical operator.

    Attributes
    ----------
    name : str
        Display name for the operator (e.g., '+', 'sin').
    arity : Arity
        Whether operator takes one or two arguments.
    function : Callable
        The actual PyTorch function to compute the operation.
    complexity : int
        Complexity cost for parsimony pressure (higher = more complex).
    """
    name: str
    arity: Arity
    function: Callable
    complexity: int = 1


# =============================================================================
# Protected Operations (Numerical Stability)
# =============================================================================

_EPSILON = 1e-10
_EXP_CLIP = 20.0
_SQRT_CLIP = 1e-10


def protected_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Division protected against division by zero."""
    # Add small epsilon with same sign as y to avoid changing sign
    safe_y = y + _EPSILON * torch.sign(y + _EPSILON)
    return x / safe_y


def protected_log(x: torch.Tensor) -> torch.Tensor:
    """Logarithm protected against non-positive inputs."""
    return torch.log(torch.abs(x) + _EPSILON)


def protected_exp(x: torch.Tensor) -> torch.Tensor:
    """Exponential with clamping to prevent overflow."""
    return torch.exp(torch.clamp(x, -_EXP_CLIP, _EXP_CLIP))


def protected_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Square root protected against negative inputs."""
    return torch.sqrt(torch.abs(x) + _SQRT_CLIP)


def protected_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Power function with protection against invalid operations."""
    # Use abs(x) to handle negative bases, clamp exponent
    base = torch.abs(x) + _EPSILON
    exp_clamped = torch.clamp(y, -10.0, 10.0)
    return torch.pow(base, exp_clamped)


# =============================================================================
# Standard Binary Operations
# =============================================================================

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Addition."""
    return x + y


def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Subtraction."""
    return x - y


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiplication."""
    return x * y


# =============================================================================
# Standard Unary Operations
# =============================================================================

def neg(x: torch.Tensor) -> torch.Tensor:
    """Negation."""
    return -x


def sin(x: torch.Tensor) -> torch.Tensor:
    """Sine function."""
    return torch.sin(x)


def cos(x: torch.Tensor) -> torch.Tensor:
    """Cosine function."""
    return torch.cos(x)


def square(x: torch.Tensor) -> torch.Tensor:
    """Square (x^2)."""
    return x * x


def cube(x: torch.Tensor) -> torch.Tensor:
    """Cube (x^3)."""
    return x * x * x


def abs_op(x: torch.Tensor) -> torch.Tensor:
    """Absolute value."""
    return torch.abs(x)


# =============================================================================
# Operator Registry
# =============================================================================

# Binary operators
OP_ADD = Operator('+', Arity.BINARY, add, complexity=1)
OP_SUB = Operator('-', Arity.BINARY, sub, complexity=1)
OP_MUL = Operator('*', Arity.BINARY, mul, complexity=1)
OP_DIV = Operator('/', Arity.BINARY, protected_div, complexity=2)
OP_POW = Operator('^', Arity.BINARY, protected_pow, complexity=3)

# Unary operators
OP_NEG = Operator('neg', Arity.UNARY, neg, complexity=1)
OP_SIN = Operator('sin', Arity.UNARY, sin, complexity=3)
OP_COS = Operator('cos', Arity.UNARY, cos, complexity=3)
OP_EXP = Operator('exp', Arity.UNARY, protected_exp, complexity=4)
OP_LOG = Operator('log', Arity.UNARY, protected_log, complexity=4)
OP_SQRT = Operator('sqrt', Arity.UNARY, protected_sqrt, complexity=3)
OP_SQUARE = Operator('sq', Arity.UNARY, square, complexity=2)
OP_CUBE = Operator('cube', Arity.UNARY, cube, complexity=3)
OP_ABS = Operator('abs', Arity.UNARY, abs_op, complexity=2)


class OperatorRegistry:
    """
    Registry of available operators for symbolic regression.

    This class manages which operators are available for building
    expression trees. It supports customization for domain-specific
    applications.

    Parameters
    ----------
    operators : List[Operator], optional
        Custom list of operators. If None, uses default set.

    Examples
    --------
    >>> registry = OperatorRegistry()
    >>> registry.get_binary_operators()
    [OP_ADD, OP_SUB, OP_MUL, OP_DIV]
    >>> registry.get_by_name('+')
    Operator(name='+', arity=Arity.BINARY, ...)
    """

    # Default operator sets
    DEFAULT_BINARY = [OP_ADD, OP_SUB, OP_MUL, OP_DIV]
    DEFAULT_UNARY = [OP_SIN, OP_COS, OP_EXP, OP_LOG, OP_SQRT, OP_SQUARE]

    def __init__(self, operators: Optional[List[Operator]] = None):
        if operators is None:
            self._operators = self.DEFAULT_BINARY + self.DEFAULT_UNARY
        else:
            self._operators = list(operators)

        # Build lookup dictionaries
        self._by_name: Dict[str, Operator] = {
            op.name: op for op in self._operators
        }
        self._binary: List[Operator] = [
            op for op in self._operators if op.arity == Arity.BINARY
        ]
        self._unary: List[Operator] = [
            op for op in self._operators if op.arity == Arity.UNARY
        ]

    def get_by_name(self, name: str) -> Optional[Operator]:
        """Get operator by its name."""
        return self._by_name.get(name)

    def get_binary_operators(self) -> List[Operator]:
        """Get all binary operators."""
        return self._binary.copy()

    def get_unary_operators(self) -> List[Operator]:
        """Get all unary operators."""
        return self._unary.copy()

    def get_all_operators(self) -> List[Operator]:
        """Get all registered operators."""
        return self._operators.copy()

    def add_operator(self, operator: Operator) -> None:
        """Add a new operator to the registry."""
        self._operators.append(operator)
        self._by_name[operator.name] = operator
        if operator.arity == Arity.BINARY:
            self._binary.append(operator)
        else:
            self._unary.append(operator)

    def __len__(self) -> int:
        return len(self._operators)

    def __repr__(self) -> str:
        binary_names = [op.name for op in self._binary]
        unary_names = [op.name for op in self._unary]
        return f"OperatorRegistry(binary={binary_names}, unary={unary_names})"


# Global default registry instance
DEFAULT_REGISTRY = OperatorRegistry()
