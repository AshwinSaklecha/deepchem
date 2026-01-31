"""
NGUYEN Benchmark Test Suite for Symbolic Regression

Tests the algorithm on standard NGUYEN benchmarks with varying difficulty.

Run with:
    python test_nguyen.py
"""

import torch
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('deepchem').setLevel(logging.ERROR)

from deepchem.models.torch_models.symbolic_regression import SymbolicRegressionModel
from deepchem.data import NumpyDataset

np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# NGUYEN BENCHMARK DEFINITIONS
# =============================================================================

BENCHMARKS = {
    # Easy: We have the primitive
    "Nguyen-8": {
        "equation": "sqrt(x)",
        "func": lambda X: np.sqrt(np.abs(X[:, 0])),  # abs for safety
        "x_range": (0, 4),
        "n_features": 1,
        "difficulty": "Easy",
    },
    
    # Easy-Medium: Simple polynomial
    "Nguyen-1": {
        "equation": "x³ + x² + x",
        "func": lambda X: X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
        "x_range": (-1, 1),
        "n_features": 1,
        "difficulty": "Easy-Medium",
    },
    
    # Medium: Single trig
    "Simple-Sin": {
        "equation": "sin(x)",
        "func": lambda X: np.sin(X[:, 0]),
        "x_range": (-3.14, 3.14),
        "n_features": 1,
        "difficulty": "Easy",
    },
    
    # Medium: Linear combination (2 variables)
    "Linear-2D": {
        "equation": "2*x + 3*y",
        "func": lambda X: 2 * X[:, 0] + 3 * X[:, 1],
        "x_range": (-2, 2),
        "n_features": 2,
        "difficulty": "Easy",
    },
    
    # Hard: Nested composition
    "Nguyen-5": {
        "equation": "sin(x²)·cos(x) - 1",
        "func": lambda X: np.sin(X[:, 0]**2) * np.cos(X[:, 0]) - 1,
        "x_range": (-1, 1),
        "n_features": 1,
        "difficulty": "Hard",
    },
    
    # Hard: Log composition
    "Nguyen-7": {
        "equation": "log(x+1) + log(x²+1)",
        "func": lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0]**2 + 1),
        "x_range": (0, 2),
        "n_features": 1,
        "difficulty": "Hard",
    },
}

# Choose which benchmarks to run
TESTS_TO_RUN = [
    "Simple-Sin",    # Should pass easily
    "Linear-2D",     # Should pass easily
    "Nguyen-8",      # Should pass (we have sqrt)
    "Nguyen-1",      # Medium difficulty
    # "Nguyen-5",    # Hard - uncomment to test
    # "Nguyen-7",    # Hard - uncomment to test
]

# =============================================================================
# SETTINGS
# =============================================================================

POPULATION_SIZE = 100
GENERATIONS = 30  # Moderate - increase for harder problems
MAX_DEPTH = 5
N_SAMPLES = 200

# Success thresholds
RMSE_SUCCESS = 0.05
RMSE_PARTIAL = 0.2

# =============================================================================
# RUN TESTS
# =============================================================================

print("=" * 70)
print("NGUYEN BENCHMARK TEST SUITE")
print("=" * 70)
print(f"\nSettings: pop={POPULATION_SIZE}, gen={GENERATIONS}, depth={MAX_DEPTH}")
print(f"Success threshold: RMSE < {RMSE_SUCCESS}")
print()

results = []

for test_name in TESTS_TO_RUN:
    bench = BENCHMARKS[test_name]
    print("-" * 70)
    print(f"TEST: {test_name}")
    print(f"Target: {bench['equation']}")
    print(f"Difficulty: {bench['difficulty']}")
    print("-" * 70)
    
    # Generate data
    n_feat = bench["n_features"]
    lo, hi = bench["x_range"]
    X = np.random.uniform(lo, hi, size=(N_SAMPLES, n_feat)).astype(np.float32)
    y = bench["func"](X).astype(np.float32)
    
    dataset = NumpyDataset(X=X, y=y)
    
    # Create and fit model
    model = SymbolicRegressionModel(
        n_features=n_feat,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        max_depth=MAX_DEPTH,
        simplify_expressions=True,
        constant_opt_steps=15,
        parsimony_coefficient=0.001,
        verbose=False,  # Quiet mode for cleaner output
    )
    
    start = time.time()
    model.fit(dataset)
    elapsed = time.time() - start
    
    # Evaluate
    best = model.get_best_expression()
    preds = model.predict(dataset).flatten()
    
    rmse = np.sqrt(np.mean((preds - y)**2))
    r2 = 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
    
    # Determine success
    if rmse < RMSE_SUCCESS:
        status = "✅ PASS"
    elif rmse < RMSE_PARTIAL:
        status = "⚠️ PARTIAL"
    else:
        status = "❌ FAIL"
    
    print(f"Time: {elapsed:.1f}s")
    print(f"Found: {best}")
    print(f"RMSE: {rmse:.6f} | R²: {r2:.4f}")
    print(f"Result: {status}")
    print()
    
    results.append({
        "name": test_name,
        "equation": bench["equation"],
        "difficulty": bench["difficulty"],
        "found": str(best),
        "rmse": rmse,
        "r2": r2,
        "time": elapsed,
        "status": status,
    })

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"{'Benchmark':<15} {'Difficulty':<12} {'RMSE':<12} {'Status':<10}")
print("-" * 50)
for r in results:
    print(f"{r['name']:<15} {r['difficulty']:<12} {r['rmse']:<12.6f} {r['status']}")

passed = sum(1 for r in results if "PASS" in r["status"])
partial = sum(1 for r in results if "PARTIAL" in r["status"])
failed = sum(1 for r in results if "FAIL" in r["status"])

print()
print(f"Total: {len(results)} tests")
print(f"Passed: {passed} | Partial: {partial} | Failed: {failed}")
print()

# Show best expressions found
print("=" * 70)
print("EXPRESSIONS FOUND")
print("=" * 70)
for r in results:
    expr_str = r["found"]
    if len(expr_str) > 60:
        expr_str = expr_str[:60] + "..."
    print(f"{r['name']}: {expr_str}")

print()
print("TEST COMPLETE")
