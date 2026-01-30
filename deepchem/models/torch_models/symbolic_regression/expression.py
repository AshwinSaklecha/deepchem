"""
Expression tree representation for symbolic regression.

This module defines the Node and Expression classes that form the core
data structures for representing mathematical expressions as trees.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Union, Callable

import torch
import torch.nn as nn

from .operators import Operator, Arity, OperatorRegistry, DEFAULT_REGISTRY


class NodeType(Enum):
    """Type of node in an expression tree."""
    CONSTANT = auto()   # Learnable constant (nn.Parameter)
    VARIABLE = auto()   # Input variable (x0, x1, ...)
    BINARY_OP = auto()  # Binary operator (+, -, *, /)
    UNARY_OP = auto()   # Unary operator (sin, cos, exp, ...)


class Node:
    """
    A node in the expression tree.

    This class represents a single node which can be a constant, variable,
    or an operator with children. Constants are stored as nn.Parameter to
    enable gradient-based optimization.

    Parameters
    ----------
    node_type : NodeType
        The type of this node.
    value : Optional[float]
        For CONSTANT nodes, the initial value.
    var_index : Optional[int]
        For VARIABLE nodes, the index into the input tensor (x0, x1, ...).
    operator : Optional[Operator]
        For operator nodes, the Operator object.
    left : Optional[Node]
        Left child (for binary operators) or only child (for unary).
    right : Optional[Node]
        Right child (for binary operators only).

    Examples
    --------
    >>> # Create a constant node with value 3.14
    >>> const_node = Node(NodeType.CONSTANT, value=3.14)
    >>> # Create a variable node for x0
    >>> var_node = Node(NodeType.VARIABLE, var_index=0)
    >>> # Create a binary operator node: x0 + 3.14
    >>> add_node = Node(NodeType.BINARY_OP, operator=OP_ADD,
    ...                 left=var_node, right=const_node)
    """

    def __init__(
        self,
        node_type: NodeType,
        value: Optional[float] = None,
        var_index: Optional[int] = None,
        operator: Optional[Operator] = None,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
    ):
        self.node_type = node_type
        self.operator = operator
        self.left = left
        self.right = right
        self.var_index = var_index

        # For constants, store as nn.Parameter for gradient optimization
        if node_type == NodeType.CONSTANT:
            if value is None:
                value = random.gauss(0, 1)
            self._param = nn.Parameter(torch.tensor(float(value)))
        else:
            self._param = None

    @property
    def is_constant(self) -> bool:
        return self.node_type == NodeType.CONSTANT

    @property
    def is_variable(self) -> bool:
        return self.node_type == NodeType.VARIABLE

    @property
    def is_binary_op(self) -> bool:
        return self.node_type == NodeType.BINARY_OP

    @property
    def is_unary_op(self) -> bool:
        return self.node_type == NodeType.UNARY_OP

    @property
    def is_terminal(self) -> bool:
        """Check if this is a leaf node (constant or variable)."""
        return self.node_type in (NodeType.CONSTANT, NodeType.VARIABLE)

    @property
    def value(self) -> Optional[torch.Tensor]:
        """Get the value for constant nodes."""
        if self._param is not None:
            return self._param.data
        return None

    @value.setter
    def value(self, val: float) -> None:
        """Set the value for constant nodes."""
        if self._param is not None:
            self._param.data = torch.tensor(float(val))

    def get_parameter(self) -> Optional[nn.Parameter]:
        """Get the nn.Parameter if this is a constant node."""
        return self._param

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate this node on input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output of shape (batch_size,).
        """
        if self.is_constant:
            # Broadcast constant to batch size
            return self._param.expand(X.shape[0])

        elif self.is_variable:
            # Extract the appropriate column
            return X[:, self.var_index]

        elif self.is_binary_op:
            left_val = self.left.evaluate(X)
            right_val = self.right.evaluate(X)
            return self.operator.function(left_val, right_val)

        elif self.is_unary_op:
            child_val = self.left.evaluate(X)  # Unary uses 'left' for child
            return self.operator.function(child_val)

        else:
            raise ValueError(f"Unknown node type: {self.node_type}")

    def complexity(self) -> int:
        """
        Compute the complexity (cost) of this subtree.

        Returns
        -------
        int
            Total complexity including this node and all descendants.
        """
        if self.is_terminal:
            return 1

        op_cost = self.operator.complexity if self.operator else 1
        left_cost = self.left.complexity() if self.left else 0
        right_cost = self.right.complexity() if self.right else 0

        return op_cost + left_cost + right_cost

    def depth(self) -> int:
        """
        Compute the depth of this subtree.

        Returns
        -------
        int
            Maximum depth from this node to any leaf.
        """
        if self.is_terminal:
            return 1

        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0

        return 1 + max(left_depth, right_depth)

    def count_nodes(self) -> int:
        """Count total number of nodes in this subtree."""
        if self.is_terminal:
            return 1

        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0

        return 1 + left_count + right_count

    def get_all_constants(self) -> List[nn.Parameter]:
        """
        Collect all constant parameters from this subtree.

        Returns
        -------
        List[nn.Parameter]
            All learnable constant parameters in this subtree.
        """
        params = []

        if self._param is not None:
            params.append(self._param)

        if self.left is not None:
            params.extend(self.left.get_all_constants())

        if self.right is not None:
            params.extend(self.right.get_all_constants())

        return params

    def copy(self) -> Node:
        """
        Create a deep copy of this node and its subtree.

        Returns
        -------
        Node
            A new Node that is a deep copy of this one.
        """
        new_node = Node(
            node_type=self.node_type,
            value=self.value.item() if self.value is not None else None,
            var_index=self.var_index,
            operator=self.operator,  # Operators are immutable, safe to share
            left=self.left.copy() if self.left else None,
            right=self.right.copy() if self.right else None,
        )
        return new_node

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """
        Convert this subtree to a human-readable string.

        Parameters
        ----------
        var_names : List[str], optional
            Names for variables. If None, uses x0, x1, etc.

        Returns
        -------
        str
            String representation of the expression.
        """
        if self.is_constant:
            val = self._param.data.item()
            # Format nicely
            if abs(val - round(val)) < 1e-6:
                return str(int(round(val)))
            return f"{val:.4f}"

        elif self.is_variable:
            if var_names and self.var_index < len(var_names):
                return var_names[self.var_index]
            return f"x{self.var_index}"

        elif self.is_binary_op:
            left_str = self.left.to_string(var_names)
            right_str = self.right.to_string(var_names)
            op_name = self.operator.name
            return f"({left_str} {op_name} {right_str})"

        elif self.is_unary_op:
            child_str = self.left.to_string(var_names)
            op_name = self.operator.name
            return f"{op_name}({child_str})"

        return "?"

    def __repr__(self) -> str:
        return self.to_string()


class Expression:
    """
    A complete expression tree for symbolic regression.

    This is the main class for working with mathematical expressions.
    It wraps a root Node and provides high-level methods for evaluation,
    optimization, and manipulation.

    Parameters
    ----------
    root : Node
        The root node of the expression tree.

    Examples
    --------
    >>> # Build expression: 2*x0 + x1
    >>> x0 = Node(NodeType.VARIABLE, var_index=0)
    >>> x1 = Node(NodeType.VARIABLE, var_index=1)
    >>> two = Node(NodeType.CONSTANT, value=2.0)
    >>> mul_node = Node(NodeType.BINARY_OP, operator=OP_MUL, left=two, right=x0)
    >>> add_node = Node(NodeType.BINARY_OP, operator=OP_ADD, left=mul_node, right=x1)
    >>> expr = Expression(add_node)
    >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> expr.evaluate(X)  # Returns [4.0, 10.0]
    """

    def __init__(self, root: Node):
        self.root = root
        self._fitness: Optional[float] = None
        self._error: Optional[float] = None

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the expression on input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output of shape (batch_size,).
        """
        return self.root.evaluate(X)

    def complexity(self) -> int:
        """Get total complexity of the expression."""
        return self.root.complexity()

    def depth(self) -> int:
        """Get maximum depth of the expression tree."""
        return self.root.depth()

    def count_nodes(self) -> int:
        """Count total number of nodes."""
        return self.root.count_nodes()

    def get_constants(self) -> List[nn.Parameter]:
        """Get all learnable constant parameters."""
        return self.root.get_all_constants()

    def copy(self) -> Expression:
        """Create a deep copy of this expression."""
        return Expression(self.root.copy())

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """Convert to human-readable string."""
        return self.root.to_string(var_names)

    @property
    def fitness(self) -> Optional[float]:
        """Cached fitness value (lower is better)."""
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = value

    @property
    def error(self) -> Optional[float]:
        """Cached error value (MSE or similar)."""
        return self._error

    @error.setter
    def error(self, value: float) -> None:
        self._error = value

    def invalidate_cache(self) -> None:
        """Clear cached fitness/error values after modification."""
        self._fitness = None
        self._error = None

    def __repr__(self) -> str:
        return f"Expression({self.to_string()})"

    def __str__(self) -> str:
        return self.to_string()


# =============================================================================
# Factory Functions for Creating Expressions
# =============================================================================

def make_constant(value: float) -> Node:
    """Create a constant node."""
    return Node(NodeType.CONSTANT, value=value)


def make_variable(index: int) -> Node:
    """Create a variable node (x0, x1, ...)."""
    return Node(NodeType.VARIABLE, var_index=index)


def make_binary(op: Operator, left: Node, right: Node) -> Node:
    """Create a binary operator node."""
    if op.arity != Arity.BINARY:
        raise ValueError(f"Operator {op.name} is not binary")
    return Node(NodeType.BINARY_OP, operator=op, left=left, right=right)


def make_unary(op: Operator, child: Node) -> Node:
    """Create a unary operator node."""
    if op.arity != Arity.UNARY:
        raise ValueError(f"Operator {op.name} is not unary")
    return Node(NodeType.UNARY_OP, operator=op, left=child)


def random_terminal(
    n_features: int,
    const_range: tuple = (-5.0, 5.0),
    const_prob: float = 0.5
) -> Node:
    """
    Create a random terminal node (constant or variable).

    Parameters
    ----------
    n_features : int
        Number of input features (determines variable indices).
    const_range : tuple
        Range for random constant values.
    const_prob : float
        Probability of creating a constant vs. a variable.
    """
    if random.random() < const_prob:
        value = random.uniform(const_range[0], const_range[1])
        return make_constant(value)
    else:
        idx = random.randint(0, n_features - 1)
        return make_variable(idx)


def random_expression(
    n_features: int,
    max_depth: int = 4,
    registry: OperatorRegistry = DEFAULT_REGISTRY,
    method: str = 'grow'
) -> Expression:
    """
    Generate a random expression tree.

    Parameters
    ----------
    n_features : int
        Number of input features.
    max_depth : int
        Maximum depth of the tree.
    registry : OperatorRegistry
        Registry of available operators.
    method : str
        Tree generation method: 'grow' (variable depth) or 'full' (max depth).

    Returns
    -------
    Expression
        A randomly generated expression.
    """
    root = _random_tree(n_features, max_depth, registry, method, current_depth=0)
    return Expression(root)


def _random_tree(
    n_features: int,
    max_depth: int,
    registry: OperatorRegistry,
    method: str,
    current_depth: int
) -> Node:
    """Recursive helper for random tree generation."""
    binary_ops = registry.get_binary_operators()
    unary_ops = registry.get_unary_operators()

    # At max depth, must return terminal
    if current_depth >= max_depth:
        return random_terminal(n_features)

    # At depth 0, must return operator (avoid single-node trees)
    if current_depth == 0:
        force_operator = True
    elif method == 'grow':
        # Grow: randomly choose terminal or operator
        force_operator = random.random() > 0.3
    else:
        # Full: always choose operator until max depth
        force_operator = True

    if not force_operator:
        return random_terminal(n_features)

    # Choose operator type
    if random.random() < 0.7 and binary_ops:  # Prefer binary ops
        op = random.choice(binary_ops)
        left = _random_tree(n_features, max_depth, registry, method, current_depth + 1)
        right = _random_tree(n_features, max_depth, registry, method, current_depth + 1)
        return make_binary(op, left, right)
    elif unary_ops:
        op = random.choice(unary_ops)
        child = _random_tree(n_features, max_depth, registry, method, current_depth + 1)
        return make_unary(op, child)
    else:
        # Fallback to terminal if no operators available
        return random_terminal(n_features)
