"""
Benchmark script to compare running time of sample_proba vs sample_proba_new
for BayesianNeuralNetwork with random context.
"""

import time
from typing import List, Tuple

import numpy as np

from pybandits.model import BayesianNeuralNetwork


def benchmark_sample_proba(
    n_features: int = 10,
    hidden_dim_list: List[int] = None,
    n_samples_list: List[int] = None,
    n_iterations: int = 10,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Benchmark sample_proba vs sample_proba_new for BayesianNeuralNetwork.

    Parameters
    ----------
    n_features : int
        Number of features in the context matrix.
    hidden_dim_list : List[int], optional
        List of hidden layer dimensions. If None, uses [32, 16].
    n_samples_list : List[int], optional
        List of sample sizes to test. If None, uses [10, 100, 500, 1000, 5000].
    n_iterations : int
        Number of iterations to run for each sample size (for averaging).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[dict, dict]
        Two dictionaries containing timing results for sample_proba and sample_proba_new.
    """
    if hidden_dim_list is None:
        hidden_dim_list = [32, 16]

    if n_samples_list is None:
        n_samples_list = [10, 100, 500, 1000, 5000]

    np.random.seed(seed)

    # Initialize the BayesianNeuralNetwork
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method="VI",
    )

    print("=" * 70)
    print("Benchmark: sample_proba vs sample_proba_new")
    print("=" * 70)
    print(f"Network architecture: {n_features} -> {hidden_dim_list} -> 1")
    print(f"Number of iterations per sample size: {n_iterations}")
    print("=" * 70)

    results_old = {}
    results_new = {}

    for n_samples in n_samples_list:
        # Generate random context
        context = np.random.randn(n_samples, n_features)

        # Benchmark sample_proba (original)
        times_old = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = bnn.sample_proba(context)
            end = time.perf_counter()
            times_old.append(end - start)

        # Benchmark sample_proba_new
        times_new = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = bnn.sample_proba_new(context)
            end = time.perf_counter()
            times_new.append(end - start)

        avg_old = np.mean(times_old)
        std_old = np.std(times_old)
        avg_new = np.mean(times_new)
        std_new = np.std(times_new)

        results_old[n_samples] = {"mean": avg_old, "std": std_old, "all": times_old}
        results_new[n_samples] = {"mean": avg_new, "std": std_new, "all": times_new}

        speedup = avg_old / avg_new if avg_new > 0 else float("inf")

        print(f"\nSample size: {n_samples}")
        print(f"  sample_proba:     {avg_old * 1000:.4f} ms ± {std_old * 1000:.4f} ms")
        print(f"  sample_proba_new: {avg_new * 1000:.4f} ms ± {std_new * 1000:.4f} ms")
        print(f"  Speedup (old/new): {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'n_samples':>10} | {'sample_proba (ms)':>20} | {'sample_proba_new (ms)':>22} | {'Speedup':>10}")
    print("-" * 70)
    for n_samples in n_samples_list:
        old_mean = results_old[n_samples]["mean"] * 1000
        new_mean = results_new[n_samples]["mean"] * 1000
        speedup = results_old[n_samples]["mean"] / results_new[n_samples]["mean"]
        print(f"{n_samples:>10} | {old_mean:>20.4f} | {new_mean:>22.4f} | {speedup:>10.2f}x")

    return results_old, results_new


def verify_outputs_match(
    n_features: int = 10,
    hidden_dim_list: List[int] = None,
    n_samples: int = 100,
    seed: int = 42,
) -> bool:
    """
    Verify that sample_proba and sample_proba_new produce statistically similar outputs.

    Note: Due to the stochastic nature of sampling, outputs won't be identical,
    but their statistical properties should be similar.

    Parameters
    ----------
    n_features : int
        Number of features in the context matrix.
    hidden_dim_list : List[int], optional
        List of hidden layer dimensions.
    n_samples : int
        Number of samples to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    bool
        True if outputs are statistically similar.
    """
    if hidden_dim_list is None:
        hidden_dim_list = [32, 16]

    np.random.seed(seed)

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method="VI",
    )

    context = np.random.randn(n_samples, n_features)

    # Run multiple times to get distributions
    n_runs = 100
    probs_old = []
    probs_new = []

    for _ in range(n_runs):
        result_old = bnn.sample_proba(context)
        result_new = bnn.sample_proba_new(context)
        probs_old.append([r[0] for r in result_old])
        probs_new.append([r[0] for r in result_new])

    probs_old = np.array(probs_old)
    probs_new = np.array(probs_new)

    # Compare means and stds across runs
    mean_old = np.mean(probs_old)
    mean_new = np.mean(probs_new)
    std_old = np.std(probs_old)
    std_new = np.std(probs_new)

    print("\n" + "=" * 70)
    print("Output Verification (Statistical Comparison)")
    print("=" * 70)
    print(f"Mean probability (sample_proba):     {mean_old:.6f}")
    print(f"Mean probability (sample_proba_new): {mean_new:.6f}")
    print(f"Std probability (sample_proba):      {std_old:.6f}")
    print(f"Std probability (sample_proba_new):  {std_new:.6f}")

    # Check if means are close (within tolerance)
    mean_diff = abs(mean_old - mean_new)
    std_diff = abs(std_old - std_new)
    tolerance = 0.1

    is_similar = mean_diff < tolerance and std_diff < tolerance
    print(f"\nMean difference: {mean_diff:.6f} (tolerance: {tolerance})")
    print(f"Std difference: {std_diff:.6f} (tolerance: {tolerance})")
    print(f"Outputs are statistically similar: {is_similar}")

    return is_similar


if __name__ == "__main__":
    # Run benchmark with different configurations
    print("\n" + "#" * 70)
    print("# Configuration 1: Small network (10 features, hidden=[32, 16])")
    print("#" * 70)
    benchmark_sample_proba(
        n_features=10,
        hidden_dim_list=[32, 16],
        n_samples_list=[10, 100, 500, 1000, 5000],
        n_iterations=10,
    )

    print("\n" + "#" * 70)
    print("# Configuration 2: Larger network (50 features, hidden=[64, 32, 16])")
    print("#" * 70)
    benchmark_sample_proba(
        n_features=50,
        hidden_dim_list=[64, 32, 16],
        n_samples_list=[10, 100, 500, 1000, 5000],
        n_iterations=10,
    )

    print("\n" + "#" * 70)
    print("# Configuration 3: Simple logistic regression (20 features, no hidden)")
    print("#" * 70)
    benchmark_sample_proba(
        n_features=20,
        hidden_dim_list=None,
        n_samples_list=[10, 100, 500, 1000, 5000],
        n_iterations=10,
    )

    # Verify outputs are statistically similar
    print("\n" + "#" * 70)
    print("# Verifying output consistency")
    print("#" * 70)
    verify_outputs_match(n_features=10, hidden_dim_list=[32, 16], n_samples=100)
