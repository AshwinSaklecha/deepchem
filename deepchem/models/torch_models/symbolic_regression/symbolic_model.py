"""
Symbolic Regression Model for DeepChem.

This module provides a DeepChem-compatible wrapper for the symbolic
regression evolutionary search, integrating with the TorchModel API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from deepchem.data import Dataset, NumpyDataset
from deepchem.models.models import Model
from deepchem.trans import Transformer, undo_transforms

from .expression import Expression, random_expression
from .operators import OperatorRegistry, DEFAULT_REGISTRY
from .search import Population, ParetoFront, compute_fitness
from .optimization import optimize_constants, simplify

logger = logging.getLogger(__name__)


class ExpressionModule(nn.Module):
    """
    A PyTorch module wrapping an Expression for prediction.

    This allows an Expression to be used like a neural network
    for making predictions.
    """

    def __init__(self, expression: Optional[Expression] = None):
        super().__init__()
        self._expression = expression

    @property
    def expression(self) -> Optional[Expression]:
        """Get the current expression."""
        return self._expression

    @expression.setter
    def expression(self, expr: Expression):
        """Set the expression."""
        self._expression = expr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the expression on input data."""
        if self._expression is None:
            raise RuntimeError("No expression has been fit. Call fit() first.")
        return self._expression.evaluate(x)


class SymbolicRegressionModel(Model):
    """
    Symbolic Regression model using evolutionary search.

    This model discovers mathematical expressions that fit data by
    evolving a population of expression trees. It combines genetic
    programming (mutation, crossover) with gradient-based constant
    optimization.

    The model maintains a Pareto front of solutions trading off
    between accuracy and complexity, allowing users to choose
    expressions based on their preferences.

    Parameters
    ----------
    n_features : int
        Number of input features.
    population_size : int, default=100
        Number of expressions in the population.
    max_depth : int, default=6
        Maximum depth of expression trees.
    generations : int, default=50
        Number of evolutionary generations.
    operators : OperatorRegistry, optional
        Available operators. If None, uses default operators.
    parsimony_coefficient : float, default=0.001
        Weight for complexity penalty in fitness.
    crossover_prob : float, default=0.5
        Probability of crossover vs mutation for generating offspring.
    mutation_prob : float, default=0.5
        Probability of mutation (when not crossover).
    elite_fraction : float, default=0.1
        Fraction of best individuals preserved each generation.
    tournament_size : int, default=3
        Number of individuals competing in tournament selection.
    optimize_constants : bool, default=True
        Whether to optimize constants after each generation.
    constant_optimizer : str, default='lbfgs'
        Optimizer for constants ('lbfgs' or 'adam').
    constant_opt_steps : int, default=20
        Number of optimization steps for constants.
    simplify_expressions : bool, default=True
        Whether to simplify expressions after optimization.
    model_dir : str, optional
        Directory to save model checkpoints.
    device : torch.device, optional
        Device for computation.
    verbose : bool, default=True
        Whether to print progress during fitting.

    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> # Generate data: y = 2*x0 + 3*x1
    >>> X = np.random.randn(100, 2)
    >>> y = 2 * X[:, 0] + 3 * X[:, 1]
    >>> dataset = dc.data.NumpyDataset(X=X, y=y)
    >>> model = SymbolicRegressionModel(n_features=2, generations=20)
    >>> model.fit(dataset)
    >>> print(model.get_best_expression())
    """

    def __init__(
        self,
        n_features: int,
        population_size: int = 100,
        max_depth: int = 6,
        generations: int = 50,
        operators: Optional[OperatorRegistry] = None,
        parsimony_coefficient: float = 0.001,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.5,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        optimize_constants: bool = True,
        constant_optimizer: str = 'lbfgs',
        constant_opt_steps: int = 20,
        simplify_expressions: bool = True,
        model_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        **kwargs,
    ):
        # Create a placeholder module for Model base class
        self._module = ExpressionModule()
        super().__init__(model=self._module, model_dir=model_dir, **kwargs)

        # Store configuration
        self.n_features = n_features
        self.population_size = population_size
        self.max_depth = max_depth
        self.generations = generations
        self.registry = operators or DEFAULT_REGISTRY
        self.parsimony_coefficient = parsimony_coefficient
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.do_optimize_constants = optimize_constants
        self.constant_optimizer = constant_optimizer
        self.constant_opt_steps = constant_opt_steps
        self.do_simplify = simplify_expressions
        self.verbose = verbose

        # Select device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device

        # State
        self._population: Optional[Population] = None
        self._pareto_front = ParetoFront()
        self._best_expression: Optional[Expression] = None
        self._generation_history: List[dict] = []
        self._fitted = False

    def fit(
        self,
        dataset: Dataset,
        nb_epoch: int = 1,
        **kwargs,
    ) -> float:
        """
        Fit the model by running evolutionary search.

        Parameters
        ----------
        dataset : Dataset
            Training data.
        nb_epoch : int
            Number of independent evolutionary runs. Each run
            starts from a new random population.

        Returns
        -------
        float
            Best fitness achieved.
        """
        # Convert dataset to tensors
        X = torch.tensor(dataset.X, dtype=torch.float32, device=self.device)
        y = torch.tensor(dataset.y, dtype=torch.float32, device=self.device)
        if y.dim() > 1:
            y = y.squeeze()

        best_overall_fitness = float('inf')

        for run in range(nb_epoch):
            if self.verbose:
                logger.info(f"Starting evolutionary run {run + 1}/{nb_epoch}")

            # Initialize population
            self._population = Population(
                size=self.population_size,
                max_depth=self.max_depth,
                n_features=self.n_features,
                registry=self.registry,
                elite_fraction=self.elite_fraction,
                tournament_size=self.tournament_size,
                parsimony_coefficient=self.parsimony_coefficient,
            )
            self._population.initialize()

            # Evolution loop
            time_start = time.time()
            for gen in range(self.generations):
                # Evaluate fitness
                self._population.evaluate_fitness(X, y)

                # Optimize constants for top individuals
                if self.do_optimize_constants:
                    self._optimize_top_constants(X, y)

                # Simplify expressions
                if self.do_simplify:
                    self._simplify_population()

                # Re-evaluate after optimization/simplification
                if self.do_optimize_constants or self.do_simplify:
                    self._population.evaluate_fitness(X, y)

                # Update Pareto front
                self._population.update_pareto_front(self._pareto_front)

                # Track best
                best = self._population.get_best()
                
                # Record history
                self._generation_history.append({
                    'run': run,
                    'generation': gen,
                    'best_fitness': best.fitness,
                    'best_error': best.error,
                    'best_complexity': best.complexity(),
                    'pareto_size': len(self._pareto_front),
                })

                if self.verbose and gen % 5 == 0:
                    logger.info(
                        f"Gen {gen}: best_fitness={best.fitness:.4f}, "
                        f"error={best.error:.4f}, complexity={best.complexity()}"
                    )

                # Update best overall
                if best.fitness < best_overall_fitness:
                    best_overall_fitness = best.fitness
                    self._best_expression = best.copy()

                # Evolve to next generation (except last)
                if gen < self.generations - 1:
                    self._population = self._population.evolve(
                        crossover_prob=self.crossover_prob,
                        mutation_prob=self.mutation_prob,
                    )

            time_elapsed = time.time() - time_start
            if self.verbose:
                logger.info(
                    f"Run {run + 1} completed in {time_elapsed:.2f}s. "
                    f"Best: {self._best_expression}"
                )

        # Set the module's expression for prediction
        self._module.expression = self._best_expression
        self._fitted = True

        return best_overall_fitness

    def _optimize_top_constants(self, X: torch.Tensor, y: torch.Tensor):
        """Optimize constants in top expressions."""
        # Optimize top 10% of population
        n_to_optimize = max(1, self.population_size // 10)
        sorted_individuals = sorted(
            self._population.individuals,
            key=lambda e: e.fitness if e.fitness is not None else float('inf')
        )

        for i in range(min(n_to_optimize, len(sorted_individuals))):
            expr = sorted_individuals[i]
            if expr.get_constants():  # Only if has constants
                optimized = optimize_constants(
                    expr, X, y,
                    optimizer=self.constant_optimizer,
                    steps=self.constant_opt_steps,
                )
                # Replace in population
                idx = self._population.individuals.index(expr)
                self._population._individuals[idx] = optimized

    def _simplify_population(self):
        """Apply simplification to all expressions."""
        for i, expr in enumerate(self._population.individuals):
            simplified = simplify(expr)
            self._population._individuals[i] = simplified

    def predict_on_batch(
        self,
        X: np.typing.ArrayLike,
        transformers: List[Transformer] = [],
    ) -> np.ndarray:
        """
        Make predictions for a batch of inputs.

        Parameters
        ----------
        X : array-like
            Input features.
        transformers : list of Transformers
            Transformers to undo on output.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before prediction.")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self._module(X_tensor)
            predictions = predictions.cpu().numpy()

        if len(transformers) > 0:
            predictions = undo_transforms(predictions, transformers)

        return predictions

    def predict(
        self,
        dataset: Dataset,
        transformers: List[Transformer] = [],
    ) -> np.ndarray:
        """
        Make predictions on a dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to predict on.
        transformers : list of Transformers
            Transformers to undo on output.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        return self.predict_on_batch(dataset.X, transformers)

    def get_best_expression(self) -> Optional[Expression]:
        """
        Get the best expression found.

        Returns
        -------
        Expression or None
            The best expression by fitness, or None if not fitted.
        """
        return self._best_expression

    def get_pareto_front(self) -> ParetoFront:
        """
        Get the Pareto front of solutions.

        Returns
        -------
        ParetoFront
            The Pareto front containing non-dominated solutions.
        """
        return self._pareto_front

    def get_expression_by_preference(
        self,
        preference: str = 'balanced',
    ) -> Optional[Expression]:
        """
        Get an expression from the Pareto front based on preference.

        Parameters
        ----------
        preference : str
            Selection preference:
            - 'accuracy': Best accuracy regardless of complexity
            - 'simplicity': Simplest regardless of accuracy
            - 'balanced': Best accuracy among simpler solutions

        Returns
        -------
        Expression or None
            Selected expression, or None if Pareto front is empty.
        """
        return self._pareto_front.get_best(preference)

    def get_all_pareto_expressions(self) -> List[Expression]:
        """
        Get all expressions from the Pareto front.

        Returns
        -------
        List[Expression]
            All Pareto-optimal expressions.
        """
        return [entry.expression for entry in self._pareto_front.get_all()]

    def get_equation_string(self) -> str:
        """
        Get the best expression as a string.

        Returns
        -------
        str
            String representation of the best expression.
        """
        if self._best_expression is None:
            return "No expression found (model not fitted)"
        return str(self._best_expression)

    def get_history(self) -> List[dict]:
        """
        Get the training history.

        Returns
        -------
        List[dict]
            List of dictionaries with generation statistics.
        """
        return self._generation_history

    def save(self):
        """Save the model state."""
        # For symbolic regression, we primarily save the expressions
        # The base Model class handles basic saving
        pass

    def reload(self):
        """Reload saved model state."""
        pass
