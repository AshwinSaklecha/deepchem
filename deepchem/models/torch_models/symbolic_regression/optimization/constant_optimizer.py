"""
Constant optimization for symbolic regression.

This module provides gradient-based optimization of constants within
expression trees using PyTorch optimizers.
"""

from __future__ import annotations

from typing import Optional, List, Literal
import torch
import torch.nn as nn

from ..expression import Expression


def optimize_constants(
    expr: Expression,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: Literal['lbfgs', 'adam'] = 'lbfgs',
    steps: int = 50,
    lr: float = 0.1,
    verbose: bool = False,
) -> Expression:
    """
    Optimize constants in an expression using gradient descent.

    This function tunes the numerical constants within an expression tree
    to minimize the prediction error. The expression structure remains
    unchanged; only the constant values are optimized.

    Parameters
    ----------
    expr : Expression
        Expression to optimize. This is NOT modified; a copy is returned.
    X : torch.Tensor
        Input data of shape (batch_size, n_features).
    y : torch.Tensor
        Target values of shape (batch_size,).
    optimizer : str
        Optimizer to use:
        - 'lbfgs': L-BFGS optimizer (fast convergence, recommended)
        - 'adam': Adam optimizer (more robust for noisy gradients)
    steps : int
        Number of optimization steps.
    lr : float
        Learning rate.
    verbose : bool
        If True, print optimization progress.

    Returns
    -------
    Expression
        A new expression with optimized constants.

    Examples
    --------
    >>> # Expression: a * x + b where a=1, b=1
    >>> # Target: y = 3 * x + 2
    >>> # After optimization: a ≈ 3, b ≈ 2
    """
    # Create a copy to avoid modifying the original
    optimized_expr = expr.copy()

    # Get all constant parameters
    params = optimized_expr.get_constants()

    if not params:
        # No constants to optimize
        return optimized_expr

    # Create optimizer
    if optimizer == 'lbfgs':
        optim = torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn='strong_wolfe')
    elif optimizer == 'adam':
        optim = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Ensure y is proper shape
    target = y.view(-1) if y.dim() > 1 else y

    # Optimization loop
    best_loss = float('inf')
    best_values = [p.data.clone() for p in params]

    def closure():
        """Closure for L-BFGS optimizer."""
        optim.zero_grad()
        try:
            y_pred = optimized_expr.evaluate(X)
            loss = torch.mean((y_pred - target) ** 2)

            if torch.isfinite(loss):
                loss.backward()
                return loss
            else:
                return torch.tensor(float('inf'))
        except Exception:
            return torch.tensor(float('inf'))

    for step in range(steps):
        if optimizer == 'lbfgs':
            loss = optim.step(closure)
        else:
            # Adam
            optim.zero_grad()
            try:
                y_pred = optimized_expr.evaluate(X)
                loss = torch.mean((y_pred - target) ** 2)

                if torch.isfinite(loss):
                    loss.backward()
                    optim.step()
                else:
                    break
            except Exception:
                break

        # Track best
        if loss is not None and torch.isfinite(loss):
            current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_values = [p.data.clone() for p in params]

            if verbose and step % 10 == 0:
                print(f"Step {step}: loss = {current_loss:.6f}")

    # Restore best values
    for p, best_val in zip(params, best_values):
        p.data = best_val

    # Invalidate cache since values changed
    optimized_expr.invalidate_cache()

    return optimized_expr


def optimize_population_constants(
    expressions: List[Expression],
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: Literal['lbfgs', 'adam'] = 'lbfgs',
    steps: int = 20,
    lr: float = 0.1,
) -> List[Expression]:
    """
    Optimize constants for a list of expressions.

    This is a convenience function for optimizing an entire population.

    Parameters
    ----------
    expressions : List[Expression]
        Expressions to optimize.
    X : torch.Tensor
        Input data.
    y : torch.Tensor
        Target values.
    optimizer : str
        Optimizer to use ('lbfgs' or 'adam').
    steps : int
        Number of optimization steps per expression.
    lr : float
        Learning rate.

    Returns
    -------
    List[Expression]
        List of expressions with optimized constants.
    """
    optimized = []
    for expr in expressions:
        opt_expr = optimize_constants(
            expr, X, y,
            optimizer=optimizer,
            steps=steps,
            lr=lr,
        )
        optimized.append(opt_expr)
    return optimized


class ConstantOptimizer:
    """
    Reusable constant optimizer with configurable settings.

    This class wraps the optimization function with persistent settings,
    useful when optimizing many expressions with the same configuration.

    Parameters
    ----------
    optimizer : str
        Optimizer type ('lbfgs' or 'adam').
    steps : int
        Number of optimization steps.
    lr : float
        Learning rate.

    Examples
    --------
    >>> optimizer = ConstantOptimizer(optimizer='lbfgs', steps=30)
    >>> optimized_expr = optimizer.optimize(expr, X, y)
    """

    def __init__(
        self,
        optimizer: Literal['lbfgs', 'adam'] = 'lbfgs',
        steps: int = 50,
        lr: float = 0.1,
    ):
        self.optimizer = optimizer
        self.steps = steps
        self.lr = lr

    def optimize(
        self,
        expr: Expression,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Expression:
        """Optimize constants in an expression."""
        return optimize_constants(
            expr, X, y,
            optimizer=self.optimizer,
            steps=self.steps,
            lr=self.lr,
        )

    def optimize_batch(
        self,
        expressions: List[Expression],
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> List[Expression]:
        """Optimize constants for a batch of expressions."""
        return optimize_population_constants(
            expressions, X, y,
            optimizer=self.optimizer,
            steps=self.steps,
            lr=self.lr,
        )
