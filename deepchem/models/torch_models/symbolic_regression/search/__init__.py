"""
Search module for symbolic regression.

This module provides evolutionary search components:
- Population management
- Genetic operators (mutation, crossover)
- Fitness evaluation
- Pareto front tracking
"""

from .fitness import compute_fitness, evaluate_batch, rank_by_fitness
from .genetic_ops import mutate, crossover
from .pareto import ParetoFront, ParetoEntry
from .population import Population

__all__ = [
    # Fitness
    'compute_fitness',
    'evaluate_batch',
    'rank_by_fitness',
    # Genetic operators
    'mutate',
    'crossover',
    # Pareto front
    'ParetoFront',
    'ParetoEntry',
    # Population
    'Population',
]
