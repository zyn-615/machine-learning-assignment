#!/usr/bin/env python3
"""Regularized linear and logistic regression experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_linear_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load 1-D input and target values for polynomial regression."""
    x = np.loadtxt(DATA_DIR / "ex5Linx.dat")
    y = np.loadtxt(DATA_DIR / "ex5Liny.dat")
    return x.reshape(-1, 1), y.reshape(-1, 1)


def load_logistic_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load 2-D input and binary labels for classification."""
    uv = np.loadtxt(DATA_DIR / "ex5Logx.dat", delimiter=",")
    y = np.loadtxt(DATA_DIR / "ex5Logy.dat")
    return uv, y


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features up to supplied degree (inclusive)."""
    powers = [np.ones_like(x)]
    for exp in range(1, degree + 1):
        powers.append(x ** exp)
    return np.hstack(powers)


def regularized_normal_equation(
    X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """Solve (X^T X + λL)θ = X^T y for θ with L excluding the bias term."""
    n_features = X.shape[1]
    L = np.eye(n_features)
    L[0, 0] = 0.0
    left = X.T @ X + lambda_ * L
    right = X.T @ y
    try:
        theta = np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(left) @ right
    return theta


def plot_linear_fit(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float, degree: int
) -> Path:
    """Plot polynomial regression fit for a given λ."""
    x_range = np.linspace(x.min() - 0.2, x.max() + 0.2, 400).reshape(-1, 1)
    X_range = polynomial_features(x_range, degree)
    y_pred = X_range @ theta

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, c="tab:blue", label="Training data")
    plt.plot(x_range, y_pred, c="tab:red", label=f"Polynomial fit (λ={lambda_})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Regularized Polynomial Regression (degree={degree}, λ={lambda_})")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"linear_lambda_{lambda_}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def map_feature(u: np.ndarray, v: np.ndarray, degree: int = 6) -> np.ndarray:
    """Map features to all polynomial terms up to the provided degree."""
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    features = [np.ones_like(u)]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            features.append((u ** (i - j)) * (v ** j))
    return np.hstack(features)


def logistic_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> float:
    m = y.size
    h = sigmoid(X @ theta)
    eps = 1e-10
    cost = (
        -np.dot(y, np.log(h + eps))
        - np.dot(1 - y, np.log(1 - h + eps))
    ) / m
    cost += (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return float(cost)


def logistic_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    m = y.size
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    reg = np.concatenate(([0.0], theta[1:])) * (lambda_ / m)
    return grad + reg


def logistic_hessian(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    m = y.size
    h = sigmoid(X @ theta)
    weights = h * (1 - h)
    weighted_X = X * weights[:, None]
    H = (X.T @ weighted_X) / m
    L = np.eye(theta.size)
    L[0, 0] = 0.0
    H += (lambda_ / m) * L
    return H


@dataclass
class NewtonResult:
    theta: np.ndarray
    costs: List[float]
    iterations: int


def newton_method(
    X: np.ndarray, y: np.ndarray, lambda_: float, max_iter: int = 30, tol: float = 1e-6
) -> NewtonResult:
    """Run Newton's method for regularized logistic regression."""
    theta = np.zeros(X.shape[1])
    costs: List[float] = []

    for iteration in range(1, max_iter + 1):
        cost = logistic_cost(theta, X, y, lambda_)
        costs.append(cost)

        grad = logistic_gradient(theta, X, y, lambda_)
        H = logistic_hessian(theta, X, y, lambda_)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ grad

        theta -= delta

        if np.linalg.norm(delta, ord=2) < tol:
            break

    return NewtonResult(theta=theta, costs=costs, iterations=iteration)


def plot_logistic_boundary(
    uv: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float, degree: int
) -> Path:
    """Plot decision boundary for logistic regression."""
    u_vals = np.linspace(-1.0, 1.5, 200)
    v_vals = np.linspace(-1.0, 1.5, 200)
    z = np.zeros((v_vals.size, u_vals.size))

    for i, u in enumerate(u_vals):
        features = map_feature(np.full_like(v_vals, u), v_vals, degree)
        z[:, i] = features @ theta

    plt.figure(figsize=(6, 4))
    pos = y == 1
    neg = y == 0
    plt.scatter(uv[pos, 0], uv[pos, 1], c="tab:blue", marker="+", label="y = 1")
    plt.scatter(
        uv[neg, 0],
        uv[neg, 1],
        marker="o",
        label="y = 0",
        s=30,
        facecolors="none",
        edgecolors="tab:orange",
    )
    plt.contour(u_vals, v_vals, z, levels=[0], linewidths=2, colors="tab:red")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(f"Regularized Logistic Regression Boundary (λ={lambda_})")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"logistic_lambda_{lambda_}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    # Polynomial regression parameters
    linear_degree = 5
    lambdas = [0.0, 1.0, 10.0]

    x, y = load_linear_data()
    X = polynomial_features(x, linear_degree)

    print("Regularized Linear Regression (normal equation)")
    for lambda_ in lambdas:
        theta = regularized_normal_equation(X, y, lambda_)
        theta = theta.reshape(-1, 1)
        l2_norm = float(np.linalg.norm(theta))
        plot_path = plot_linear_fit(x, y, theta, lambda_, linear_degree)
        print(f"λ={lambda_:>4}: θ={theta.ravel()}, ||θ||₂={l2_norm:.4f}, plot='{plot_path}'")

    # Logistic regression parameters
    uv, y_log = load_logistic_data()
    X_log = map_feature(uv[:, 0], uv[:, 1])

    print("\nRegularized Logistic Regression (Newton's method)")
    for lambda_ in lambdas:
        result = newton_method(X_log, y_log, lambda_)
        l2_norm = float(np.linalg.norm(result.theta))
        plot_path = plot_logistic_boundary(uv, y_log, result.theta, lambda_, degree=6)
        print(
            f"λ={lambda_:>4}: iterations={result.iterations:2}, "
            f"||θ||₂={l2_norm:.4f}, final cost={result.costs[-1]:.6f}, plot='{plot_path}'"
        )


if __name__ == "__main__":
    main()
