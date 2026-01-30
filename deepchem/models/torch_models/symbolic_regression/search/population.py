"""
Population management for evolutionary symbolic regression.

This module provides the Population class for managing a collection
of expressions through evolutionary generations.
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch

from ..expression import Expression, random_expression
from ..operators import OperatorRegistry, DEFAULT_REGISTRY
from .fitness import compute_fitness, evaluate_batch, rank_by_fitness
from .genetic_ops import mutate, crossover
from .pareto import ParetoFront


class Population:
    """
    Manages a population of expressions for evolutionary search.

    This class handles population initialization, fitness evaluation,
    selection, and evolution through genetic operations.

    Parameters
    ----------
    size : int
        Population size.
    max_depth : int
        Maximum depth for expression trees.
    n_features : int
        Number of input features.
    registry : OperatorRegistry
        Available operators for expression generation.
    elite_fraction : float
        Fraction of top individuals preserved unchanged (0.1 = 10%).
    tournament_size : int
        Number of individuals in tournament selection.
    parsimony_coefficient : float
        Penalty weight for complexity in fitness.

    Examples
    --------
    >>> import torch
    >>> pop = Population(size=50, n_features=2)
    >>> pop.initialize()
    >>> X = torch.randn(100, 2)
    >>> y = 2.0 * X[:, 0] + 3.0 * X[:, 1]
    >>> pop.evaluate_fitness(X, y)
    >>> best = pop.get_best()
    >>> print(best)
    """

    def __init__(
        self,
        size: int = 100,
        max_depth: int = 4,
        n_features: int = 1,
        registry: OperatorRegistry = DEFAULT_REGISTRY,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        parsimony_coefficient: float = 0.001,
    ):
        self.size = size
        self.max_depth = max_depth
        self.n_features = n_features
        self.registry = registry
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient

        self._individuals: List[Expression] = []
        self._generation: int = 0

    def initialize(self) -> None:
        """
        Generate initial random population.

        Creates a population of random expression trees using a mix
        of 'grow' and 'full' tree generation methods (ramped half-and-half).
        """
        self._individuals = []

        for i in range(self.size):
            # Ramped half-and-half: alternate between grow and full
            method = 'grow' if i % 2 == 0 else 'full'

            # Vary max depth for diversity
            depth = random.randint(2, self.max_depth)

            expr = random_expression(
                n_features=self.n_features,
                max_depth=depth,
                registry=self.registry,
                method=method,
            )
            self._individuals.append(expr)

        self._generation = 0

    def evaluate_fitness(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        use_cache: bool = True,
    ) -> None:
        """
        Evaluate fitness for all individuals.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (batch_size, n_features).
        y : torch.Tensor
            Target values of shape (batch_size,).
        use_cache : bool
            If True, skip individuals with cached fitness.
        """
        evaluate_batch(
            self._individuals, X, y,
            parsimony_coefficient=self.parsimony_coefficient,
            use_cache=use_cache,
        )

    def select(self, n: int) -> List[Expression]:
        """
        Tournament selection of n individuals.

        Selects n individuals through tournament selection, where
        each tournament picks tournament_size random individuals
        and returns the best one.

        Parameters
        ----------
        n : int
            Number of individuals to select.

        Returns
        -------
        List[Expression]
            Selected individuals.
        """
        selected = []

        for _ in range(n):
            # Random tournament
            tournament = random.sample(
                self._individuals,
                min(self.tournament_size, len(self._individuals))
            )

            # Best in tournament (lowest fitness)
            winner = min(
                tournament,
                key=lambda e: e.fitness if e.fitness is not None else float('inf')
            )
            selected.append(winner)

        return selected

    def evolve(
        self,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.5,
    ) -> 'Population':
        """
        Create next generation via genetic operations.

        1. Preserve elite individuals unchanged
        2. Fill rest via crossover or mutation

        Parameters
        ----------
        crossover_prob : float
            Probability of using crossover vs. mutation.
        mutation_prob : float
            Probability of mutation (used when not doing crossover).

        Returns
        -------
        Population
            A new Population representing the next generation.
        """
        # Create new population with same parameters
        new_pop = Population(
            size=self.size,
            max_depth=self.max_depth,
            n_features=self.n_features,
            registry=self.registry,
            elite_fraction=self.elite_fraction,
            tournament_size=self.tournament_size,
            parsimony_coefficient=self.parsimony_coefficient,
        )

        # Sort by fitness
        sorted_individuals = rank_by_fitness(self._individuals)

        # Elitism: preserve top individuals
        elite_size = max(1, int(self.size * self.elite_fraction))
        elite = [expr.copy() for expr in sorted_individuals[:elite_size]]

        # Generate rest of population
        new_individuals = list(elite)
        remaining = self.size - len(elite)

        for _ in range(remaining):
            if random.random() < crossover_prob:
                # Crossover
                parents = self.select(2)
                child = crossover(parents[0], parents[1], max_depth=self.max_depth)
            else:
                # Mutation
                parent = self.select(1)[0]
                child = mutate(
                    parent,
                    n_features=self.n_features,
                    registry=self.registry,
                    max_depth=self.max_depth,
                )

            new_individuals.append(child)

        new_pop._individuals = new_individuals
        new_pop._generation = self._generation + 1

        return new_pop

    def get_best(self) -> Optional[Expression]:
        """
        Return the best individual by fitness.

        Returns
        -------
        Optional[Expression]
            Best expression, or None if population is empty.
        """
        if not self._individuals:
            return None

        return min(
            self._individuals,
            key=lambda e: e.fitness if e.fitness is not None else float('inf')
        )

    def get_top_n(self, n: int) -> List[Expression]:
        """
        Return the top n individuals by fitness.

        Parameters
        ----------
        n : int
            Number of top individuals to return.

        Returns
        -------
        List[Expression]
            Top n expressions sorted by fitness.
        """
        sorted_individuals = rank_by_fitness(self._individuals)
        return sorted_individuals[:n]

    def update_pareto_front(self, front: ParetoFront) -> int:
        """
        Update a Pareto front with current population.

        Parameters
        ----------
        front : ParetoFront
            Pareto front to update.

        Returns
        -------
        int
            Number of expressions added to the front.
        """
        added = 0
        for expr in self._individuals:
            if front.update(expr):
                added += 1
        return added

    @property
    def individuals(self) -> List[Expression]:
        """List of all expressions in the population."""
        return self._individuals

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    def __len__(self) -> int:
        """Number of individuals in the population."""
        return len(self._individuals)

    def __repr__(self) -> str:
        return f"Population(size={len(self)}, generation={self._generation})"
