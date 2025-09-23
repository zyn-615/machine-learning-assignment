#!/usr/bin/env python3
"""Multivariate linear regression via gradient descent and normal equation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("ex2Data") / "ex2Data"
OUTPUT_DIR = Path("outputs")


@dataclass
class GradientDescentResult:
    theta: np.ndarray
    cost_history: np.ndarray


@dataclass
class ExperimentResult:
    alpha: float
    iterations: int
    final_cost: float
    theta: List[float]


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load raw feature matrix X and target vector y."""
    x_path = DATA_DIR / "ex2x.dat"
    y_path = DATA_DIR / "ex2y.dat"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Data files not found. Ensure ex2Data/ex2Data contains ex2x.dat and ex2y.dat.")
    X = np.loadtxt(x_path)
    y = np.loadtxt(y_path)
    return X, y


def feature_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns to zero mean and unit variance."""
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    # Avoid division by zero for constant columns
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds
    return X_norm, means, stds


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X."""
    m = X.shape[0]
    return np.column_stack([np.ones(m), X])


def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    diff = X @ theta - y
    return float((diff @ diff) / (2.0 * len(y)))


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int,
    initial_theta: np.ndarray | None = None,
) -> GradientDescentResult:
    m, n = X.shape
    theta = np.zeros(n) if initial_theta is None else initial_theta.copy()
    costs = np.zeros(iterations)
    for i in range(iterations):
        error = X @ theta - y
        gradient = (X.T @ error) / m
        theta -= alpha * gradient
        costs[i] = compute_cost(X, y, theta)
    return GradientDescentResult(theta=theta, cost_history=costs)


def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    return XtX_inv @ X.T @ y


def run_learning_rate_sweep(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Iterable[float],
    iterations: int,
) -> Dict[float, GradientDescentResult]:
    results: Dict[float, GradientDescentResult] = {}
    initial_theta = np.zeros(X.shape[1])
    for alpha in alphas:
        results[alpha] = gradient_descent(X, y, alpha, iterations, initial_theta)
    return results


def save_cost_histories(cost_histories: Dict[float, np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "learning_curves.csv"
    with out_path.open("w", encoding="ascii") as f:
        f.write("iteration,alpha,cost\n")
        for alpha, costs in sorted(cost_histories.items()):
            for iteration, cost in enumerate(costs, start=1):
                f.write(f"{iteration},{alpha},{cost}\n")


def plot_learning_curves(cost_histories: Dict[float, np.ndarray]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "learning_curves.png"
    plt.figure(figsize=(7, 4))
    for alpha, costs in sorted(cost_histories.items()):
        iterations = np.arange(1, len(costs) + 1)
        plt.plot(iterations, costs, label=f"alpha={alpha:g}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(θ)")
    plt.title("Gradient Descent Learning Rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_prediction_parity(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "prediction_parity.png"
    plt.figure(figsize=(6, 6))
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", label="Ideal")
    for label, preds in predictions.items():
        plt.scatter(y_true, preds, s=25, alpha=0.7, label=label)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Prediction Parity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_summary(summary: Dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "summary.json"
    with out_path.open("w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)


def predict_price(
    size_sqft: float,
    bedrooms: int,
    means: np.ndarray,
    stds: np.ndarray,
    theta: np.ndarray,
) -> float:
    features = np.array([size_sqft, bedrooms], dtype=float)
    normalized = (features - means) / stds
    x_vec = np.hstack([1.0, normalized])
    return float(x_vec @ theta)


def predict_price_no_scaling(
    size_sqft: float,
    bedrooms: int,
    theta: np.ndarray,
) -> float:
    x_vec = np.array([1.0, size_sqft, bedrooms], dtype=float)
    return float(x_vec @ theta)


def rescale_theta(theta_scaled: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    theta_unscaled = np.empty_like(theta_scaled)
    theta_unscaled[1:] = theta_scaled[1:] / stds
    theta_unscaled[0] = theta_scaled[0] - np.sum((means / stds) * theta_scaled[1:])
    return theta_unscaled


def main() -> None:
    X_raw, y = load_dataset()
    m = len(y)
    X_norm, means, stds = feature_normalize(X_raw)
    X_gd = add_intercept(X_norm)

    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    sweep_iters = 50
    sweep_results = run_learning_rate_sweep(X_gd, y, learning_rates, sweep_iters)

    cost_histories = {alpha: res.cost_history for alpha, res in sweep_results.items()}
    save_cost_histories(cost_histories)
    learning_curve_path = plot_learning_curves(cost_histories)

    # Select best alpha by final cost after sweep iterations
    best_alpha = min(
        learning_rates,
        key=lambda a: sweep_results[a].cost_history[-1],
    )

    convergence_iters = 400
    gd_full = gradient_descent(
        X_gd,
        y,
        alpha=best_alpha,
        iterations=convergence_iters,
        initial_theta=np.zeros(X_gd.shape[1]),
    )

    price_prediction = predict_price(1650.0, 3, means, stds, gd_full.theta)
    theta_gd_rescaled = rescale_theta(gd_full.theta, means, stds)

    X_ne = add_intercept(X_raw)
    theta_ne = normal_equation(X_ne, y)
    price_ne = predict_price_no_scaling(1650.0, 3, theta_ne)

    predictions_gd = X_gd @ gd_full.theta
    predictions_ne = X_ne @ theta_ne
    parity_path = plot_prediction_parity(
        y,
        {
            "Gradient descent": predictions_gd,
            "Normal equation": predictions_ne,
        },
    )

    rmse_gd = float(np.sqrt(np.mean((predictions_gd - y) ** 2)))
    rmse_ne = float(np.sqrt(np.mean((predictions_ne - y) ** 2)))

    summary = {
        "m": m,
        "learning_rate_results": [
            ExperimentResult(
                alpha=alpha,
                iterations=sweep_iters,
                final_cost=float(result.cost_history[-1]),
                theta=result.theta.tolist(),
            ).__dict__
            for alpha, result in sorted(sweep_results.items())
        ],
        "selected_alpha": best_alpha,
        "gradient_descent": {
            "iterations": convergence_iters,
            "theta": gd_full.theta.tolist(),
            "theta_rescaled": theta_gd_rescaled.tolist(),
            "final_cost": float(gd_full.cost_history[-1]),
            "prediction": price_prediction,
            "rmse": rmse_gd,
        },
        "normal_equation": {
            "theta": theta_ne.tolist(),
            "prediction": price_ne,
            "rmse": rmse_ne,
        },
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "artifacts": {
            "learning_curves": str(learning_curve_path),
            "prediction_parity": str(parity_path),
        },
    }

    save_summary(summary)

    print("Training examples:", m)
    print("Learning rate sweep (50 iterations):")
    for alpha in sorted(learning_rates):
        final_cost = sweep_results[alpha].cost_history[-1]
        print(f"  alpha={alpha:.2g}, cost={final_cost:.4f}")
    print(f"Selected learning rate: {best_alpha}")
    print("Gradient descent after 400 iterations:")
    print("  theta=", gd_full.theta)
    print(f"  final cost={gd_full.cost_history[-1]:.4f}")
    print(f"  predicted price (1650 sqft, 3 br)={price_prediction:.2f}")
    print("  theta (original scale)=", theta_gd_rescaled)
    print(f"  RMSE={rmse_gd:.2f}")
    print("Normal equation solution:")
    print("  theta=", theta_ne)
    print(f"  predicted price (1650 sqft, 3 br)={price_ne:.2f}")
    print(f"  RMSE={rmse_ne:.2f}")


if __name__ == "__main__":
    main()
