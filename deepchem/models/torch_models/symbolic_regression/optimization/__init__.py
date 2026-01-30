"""
Optimization module for symbolic regression.

This module provides:
- Constant optimization via gradient descent (L-BFGS, Adam)
- Expression simplification (algebraic rules, constant folding)
"""

from .constant_optimizer import (
    optimize_constants,
    optimize_population_constants,
    ConstantOptimizer,
)
from .simplification import simplify, simplify_population

__all__ = [
    # Constant optimization
    'optimize_constants',
    'optimize_population_constants',
    'ConstantOptimizer',
    # Simplification
    'simplify',
    'simplify_population',
]
