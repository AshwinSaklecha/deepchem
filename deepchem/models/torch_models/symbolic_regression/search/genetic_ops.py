"""
Genetic operators for symbolic regression.

This module provides mutation and crossover operations for evolving
expression trees in symbolic regression.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ..expression import (
    Node, NodeType, Expression,
    make_constant, make_variable, make_binary, make_unary,
    random_terminal, random_expression,
)
from ..operators import Operator, OperatorRegistry, DEFAULT_REGISTRY, Arity


def mutate(
    expr: Expression,
    n_features: int,
    registry: OperatorRegistry = DEFAULT_REGISTRY,
    max_depth: int = 6,
    mutation_type: str = 'auto',
) -> Expression:
    """
    Apply mutation to an expression.

    Creates a mutated copy of the expression. The original is not modified.

    Parameters
    ----------
    expr : Expression
        Expression to mutate.
    n_features : int
        Number of input features (for variable nodes).
    registry : OperatorRegistry
        Available operators.
    max_depth : int
        Maximum depth for generated subtrees.
    mutation_type : str
        Type of mutation:
        - 'point': Change a single node
        - 'subtree': Replace a random subtree
        - 'auto': Randomly choose (60% point, 40% subtree)

    Returns
    -------
    Expression
        A new mutated expression.
    """
    # Deep copy first
    new_expr = expr.copy()

    # Choose mutation type
    if mutation_type == 'auto':
        mutation_type = 'point' if random.random() < 0.6 else 'subtree'

    if mutation_type == 'point':
        _apply_point_mutation(new_expr.root, n_features, registry)
    elif mutation_type == 'subtree':
        _apply_subtree_mutation(new_expr, n_features, registry, max_depth)
    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    # Invalidate cached fitness
    new_expr.invalidate_cache()

    return new_expr


def _apply_point_mutation(
    root: Node,
    n_features: int,
    registry: OperatorRegistry,
) -> None:
    """
    Apply point mutation: change a single randomly selected node.

    Modifies the tree in-place.
    """
    # Collect all nodes
    nodes = _collect_all_nodes(root)
    if not nodes:
        return

    # Select random node to mutate
    target = random.choice(nodes)

    if target.is_constant:
        # Skip constant mutation - constant optimization (Phase 3) will tune constants
        # via gradient descent, which is more effective than random perturbation
        pass

    elif target.is_variable:
        # Mutate variable: change to different variable
        if n_features > 1:
            new_idx = random.randint(0, n_features - 1)
            target.var_index = new_idx

    elif target.is_binary_op:
        # Mutate operator: change to different binary op
        binary_ops = registry.get_binary_operators()
        if len(binary_ops) > 1:
            new_op = random.choice([op for op in binary_ops if op != target.operator])
            target.operator = new_op

    elif target.is_unary_op:
        # Mutate operator: change to different unary op
        unary_ops = registry.get_unary_operators()
        if len(unary_ops) > 1:
            new_op = random.choice([op for op in unary_ops if op != target.operator])
            target.operator = new_op


def _apply_subtree_mutation(
    expr: Expression,
    n_features: int,
    registry: OperatorRegistry,
    max_depth: int,
) -> None:
    """
    Apply subtree mutation: replace a random subtree with a new one.

    Modifies the expression in-place.
    """
    # Collect all nodes with their parents
    nodes_with_parents = _collect_nodes_with_parents(expr.root)

    if not nodes_with_parents:
        return

    # Select random node (not root) to replace
    # Filter out root since we need a parent
    replaceable = [(node, parent, attr) for node, parent, attr in nodes_with_parents if parent is not None]

    if not replaceable:
        # Only root exists - replace entire tree
        new_root = random_expression(n_features, max_depth=max_depth, registry=registry).root
        expr.root = new_root
        return

    target, parent, attr = random.choice(replaceable)

    # Calculate max depth for new subtree
    depth_at_target = _get_depth_to_root(expr.root, target)
    available_depth = max(1, max_depth - depth_at_target)

    # Generate new random subtree
    new_subtree = random_expression(n_features, max_depth=available_depth, registry=registry).root

    # Replace in parent
    if attr == 'left':
        parent.left = new_subtree
    elif attr == 'right':
        parent.right = new_subtree


def crossover(
    parent1: Expression,
    parent2: Expression,
    max_depth: int = 6,
) -> Expression:
    """
    Produce one child by exchanging subtrees between parents.

    Selects a random crossover point in parent1 and replaces it with
    a random subtree from parent2.

    Parameters
    ----------
    parent1 : Expression
        First parent (provides the base structure).
    parent2 : Expression
        Second parent (provides a subtree).
    max_depth : int
        Maximum depth for the resulting child.

    Returns
    -------
    Expression
        A new child expression.
    """
    # Deep copy parent1 as base
    child = parent1.copy()

    # Get all nodes with parents from child
    child_nodes = _collect_nodes_with_parents(child.root)

    # Get all nodes from parent2
    parent2_nodes = _collect_all_nodes(parent2.root)

    if not child_nodes or not parent2_nodes:
        return child

    # Filter child nodes to those that can be replaced (have parents)
    replaceable = [(n, p, a) for n, p, a in child_nodes if p is not None]

    if not replaceable:
        # Only root - replace entire tree with random subtree from parent2
        subtree = random.choice(parent2_nodes)
        child.root = subtree.copy()
    else:
        # Select crossover point in child
        target, parent, attr = random.choice(replaceable)

        # Select subtree from parent2
        donor_subtree = random.choice(parent2_nodes).copy()

        # Check depth constraint
        depth_at_target = _get_depth_to_root(child.root, target)
        donor_depth = donor_subtree.depth()

        if depth_at_target + donor_depth <= max_depth:
            # Replace
            if attr == 'left':
                parent.left = donor_subtree
            elif attr == 'right':
                parent.right = donor_subtree
        else:
            # Depth exceeded - try with a terminal from donor
            # Find terminals in parent2
            terminals = [n for n in parent2_nodes if n.is_terminal]
            if terminals:
                donor_subtree = random.choice(terminals).copy()
                if attr == 'left':
                    parent.left = donor_subtree
                elif attr == 'right':
                    parent.right = donor_subtree

    # Invalidate cache
    child.invalidate_cache()

    return child


# =============================================================================
# Helper Functions
# =============================================================================

def _collect_all_nodes(root: Node) -> List[Node]:
    """Collect all nodes in the tree."""
    nodes = [root]

    if root.left is not None:
        nodes.extend(_collect_all_nodes(root.left))
    if root.right is not None:
        nodes.extend(_collect_all_nodes(root.right))

    return nodes


def _collect_nodes_with_parents(
    root: Node,
    parent: Optional[Node] = None,
    attr: Optional[str] = None,
) -> List[Tuple[Node, Optional[Node], Optional[str]]]:
    """
    Collect all nodes with their parent references.

    Returns list of (node, parent, attr) where attr is 'left' or 'right'.
    """
    result = [(root, parent, attr)]

    if root.left is not None:
        result.extend(_collect_nodes_with_parents(root.left, root, 'left'))
    if root.right is not None:
        result.extend(_collect_nodes_with_parents(root.right, root, 'right'))

    return result


def _get_depth_to_root(root: Node, target: Node, current_depth: int = 0) -> int:
    """Get the depth of target node from root."""
    if root is target:
        return current_depth

    if root.left is not None:
        left_result = _get_depth_to_root(root.left, target, current_depth + 1)
        if left_result >= 0:
            return left_result

    if root.right is not None:
        right_result = _get_depth_to_root(root.right, target, current_depth + 1)
        if right_result >= 0:
            return right_result

    return -1  # Not found
