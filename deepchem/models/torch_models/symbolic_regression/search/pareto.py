"""
Pareto front tracking for multi-objective symbolic regression.

This module provides the ParetoFront class for tracking non-dominated
solutions in the error vs. complexity trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..expression import Expression


@dataclass
class ParetoEntry:
    """An entry in the Pareto front."""
    expression: Expression  # COPY of the expression
    error: float
    complexity: int

    def dominates(self, other: 'ParetoEntry') -> bool:
        """
        Check if this entry dominates another.

        An entry dominates another if it is better or equal in all objectives
        and strictly better in at least one.
        """
        better_or_equal = (
            self.error <= other.error and
            self.complexity <= other.complexity
        )
        strictly_better = (
            self.error < other.error or
            self.complexity < other.complexity
        )
        return better_or_equal and strictly_better


class ParetoFront:
    """
    Track non-dominated solutions (error vs complexity trade-off).

    A solution is Pareto-optimal if no other solution is better in BOTH
    error AND complexity. This class maintains a set of such solutions.

    Parameters
    ----------
    max_size : int
        Maximum number of entries to store (prevents memory issues).

    Examples
    --------
    >>> front = ParetoFront(max_size=20)
    >>> expr1 = Expression(...)  # error=0.1, complexity=5
    >>> expr1.error = 0.1
    >>> front.update(expr1)
    True
    >>> len(front)
    1
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._entries: List[ParetoEntry] = []

    def update(self, expr: Expression) -> bool:
        """
        Try to add expression to the Pareto front.

        - Stores a COPY of the expression (not reference)
        - Removes any solutions now dominated by this one
        - Only adds if this solution is non-dominated

        Parameters
        ----------
        expr : Expression
            Expression to potentially add.

        Returns
        -------
        bool
            True if expression was added, False otherwise.
        """
        if expr.error is None:
            return False

        error = expr.error
        complexity = expr.complexity()

        # Create candidate entry (with copy of expression)
        candidate = ParetoEntry(
            expression=expr.copy(),
            error=error,
            complexity=complexity,
        )

        # Check if candidate is dominated by any existing entry
        for entry in self._entries:
            if entry.dominates(candidate):
                return False  # Dominated, don't add

        # Remove entries dominated by candidate
        self._entries = [
            entry for entry in self._entries
            if not candidate.dominates(entry)
        ]

        # Add candidate
        self._entries.append(candidate)

        # Enforce max size by removing most complex entries
        if len(self._entries) > self.max_size:
            self._entries.sort(key=lambda e: e.complexity)
            self._entries = self._entries[:self.max_size]

        return True

    def get_best(self, preference: str = 'balanced') -> Optional[Expression]:
        """
        Get the best expression based on preference.

        Parameters
        ----------
        preference : str
            Trade-off preference:
            - 'accuracy': Lowest error regardless of complexity
            - 'simplicity': Lowest complexity regardless of error
            - 'balanced': Best error among solutions with below-median complexity

        Returns
        -------
        Optional[Expression]
            Best expression according to preference, or None if front is empty.
        """
        if not self._entries:
            return None

        if preference == 'accuracy':
            best = min(self._entries, key=lambda e: e.error)
            return best.expression.copy()

        elif preference == 'simplicity':
            best = min(self._entries, key=lambda e: e.complexity)
            return best.expression.copy()

        elif preference == 'balanced':
            # Get median complexity
            complexities = sorted(e.complexity for e in self._entries)
            median_idx = len(complexities) // 2
            median_complexity = complexities[median_idx]

            # Filter to simpler-than-median entries
            simple_entries = [
                e for e in self._entries
                if e.complexity <= median_complexity
            ]

            if not simple_entries:
                simple_entries = self._entries

            # Best accuracy among simple entries
            best = min(simple_entries, key=lambda e: e.error)
            return best.expression.copy()

        else:
            raise ValueError(f"Unknown preference: {preference}")

    def get_all(self) -> List[ParetoEntry]:
        """
        Return all Pareto-optimal solutions.

        Returns
        -------
        List[ParetoEntry]
            All entries in the Pareto front.
        """
        return list(self._entries)

    def get_expressions(self) -> List[Expression]:
        """
        Return copies of all expressions in the front.

        Returns
        -------
        List[Expression]
            Copies of all Pareto-optimal expressions.
        """
        return [entry.expression.copy() for entry in self._entries]

    def clear(self) -> None:
        """Remove all entries from the front."""
        self._entries = []

    def __len__(self) -> int:
        """Number of solutions in the front."""
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ParetoFront(size={len(self)}, max_size={self.max_size})"
