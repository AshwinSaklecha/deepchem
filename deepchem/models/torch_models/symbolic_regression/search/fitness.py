"""
Fitness evaluation for symbolic regression.

This module provides functions to compute fitness of expressions,
balancing prediction error with expression complexity (parsimony pressure).
"""

from __future__ import annotations

from typing import List, Optional
import torch

from ..expression import Expression


def compute_fitness(
    expr: Expression,
    X: torch.Tensor,
    y: torch.Tensor,
    parsimony_coefficient: float = 0.001,
    error_metric: str = 'mse',
) -> float:
    """
    Compute fitness for an expression (lower is better).

    Fitness combines prediction error with a complexity penalty:
        fitness = error + parsimony_coefficient * complexity

    This creates pressure toward simpler expressions while still
    prioritizing accuracy.

    Parameters
    ----------
    expr : Expression
        Expression to evaluate.
    X : torch.Tensor
        Input data of shape (batch_size, n_features).
    y : torch.Tensor
        Target values of shape (batch_size,).
    parsimony_coefficient : float
        Penalty weight for complexity. Higher values prefer simpler
        expressions. Default: 0.001.
    error_metric : str
        Error metric to use: 'mse' (mean squared error),
        'rmse' (root mean squared error), or 'mae' (mean absolute error).

    Returns
    -------
    float
        Fitness value (lower is better). Returns float('inf') if
        expression produces NaN or Inf values.

    Notes
    -----
    This function updates expr.fitness and expr.error as a side effect.
    """
    try:
        # Evaluate expression
        y_pred = expr.evaluate(X)

        # Handle shape mismatch
        if y_pred.shape != y.shape:
            y_pred = y_pred.view(y.shape)

        # Compute error
        if error_metric == 'mse':
            error = torch.mean((y_pred - y) ** 2).item()
        elif error_metric == 'rmse':
            error = torch.sqrt(torch.mean((y_pred - y) ** 2)).item()
        elif error_metric == 'mae':
            error = torch.mean(torch.abs(y_pred - y)).item()
        else:
            raise ValueError(f"Unknown error metric: {error_metric}")

        # Check for NaN/Inf
        if not torch.isfinite(torch.tensor(error)):
            expr.error = float('inf')
            expr.fitness = float('inf')
            return float('inf')

        # Compute complexity penalty
        complexity = expr.complexity()
        fitness = error + parsimony_coefficient * complexity

        # Cache results
        expr.error = error
        expr.fitness = fitness

        return fitness

    except Exception:
        # Any error (overflow, invalid operation, etc.) -> worst fitness
        expr.error = float('inf')
        expr.fitness = float('inf')
        return float('inf')


def evaluate_batch(
    expressions: List[Expression],
    X: torch.Tensor,
    y: torch.Tensor,
    parsimony_coefficient: float = 0.001,
    error_metric: str = 'mse',
    use_cache: bool = True,
) -> None:
    """
    Evaluate fitness for multiple expressions.

    This function efficiently evaluates a batch of expressions,
    optionally skipping expressions that have cached fitness values.

    Parameters
    ----------
    expressions : List[Expression]
        List of expressions to evaluate.
    X : torch.Tensor
        Input data of shape (batch_size, n_features).
    y : torch.Tensor
        Target values of shape (batch_size,).
    parsimony_coefficient : float
        Penalty weight for complexity.
    error_metric : str
        Error metric: 'mse', 'rmse', or 'mae'.
    use_cache : bool
        If True, skip expressions with valid cached fitness.

    Notes
    -----
    Modifies expressions in-place by setting their fitness and error.
    """
    for expr in expressions:
        # Skip if cached and caching is enabled
        if use_cache and expr.fitness is not None:
            continue

        compute_fitness(
            expr, X, y,
            parsimony_coefficient=parsimony_coefficient,
            error_metric=error_metric,
        )


def rank_by_fitness(expressions: List[Expression]) -> List[Expression]:
    """
    Sort expressions by fitness (best first).

    Parameters
    ----------
    expressions : List[Expression]
        List of expressions to sort.

    Returns
    -------
    List[Expression]
        Sorted list (best fitness first, i.e., lowest values first).
    """
    return sorted(
        expressions,
        key=lambda e: e.fitness if e.fitness is not None else float('inf')
    )
