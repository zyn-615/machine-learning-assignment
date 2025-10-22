from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data")
X_PATH = DATA_DIR / "ex4x.dat"
Y_PATH = DATA_DIR / "ex4y.dat"


@dataclass
class NewtonResult:
    theta: np.ndarray
    iterations: int
    cost_history: np.ndarray
    gradient_norms: np.ndarray


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset and add intercept column."""
    X = np.loadtxt(X_PATH, dtype=float)
    y = np.loadtxt(Y_PATH, dtype=float)
    if y.ndim > 1:
        y = y.ravel()
    m = X.shape[0]
    intercept = np.ones((m, 1))
    X_aug = np.hstack((intercept, X))
    return X_aug, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp and log later on
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    m = X.shape[0]
    h = sigmoid(X @ theta)
    eps = 1e-12
    h = np.clip(h, eps, 1 - eps)
    cost = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    return float(cost)


def gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    h = sigmoid(X @ theta)
    return (X.T @ (h - y)) / m


def hessian(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    h = sigmoid(X @ theta)
    diag = h * (1 - h)
    return (X.T * diag) @ X / m


def newton_method(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 15,
    tol: float = 1e-6,
    damping: float = 1e-6,
) -> NewtonResult:
    theta = np.zeros(X.shape[1])
    cost_history = []
    grad_norms = []

    for iteration in range(1, max_iter + 1):
        grad = gradient(theta, X, y)
        H = hessian(theta, X, y)
        # Add small damping term to improve numerical stability
        H += damping * np.eye(H.shape[0])

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(H, grad, rcond=None)

        theta = theta - delta
        cost = compute_cost(theta, X, y)
        grad_norm = np.linalg.norm(grad, ord=2)

        cost_history.append(cost)
        grad_norms.append(grad_norm)

        if grad_norm < tol:
            break

    return NewtonResult(
        theta=theta,
        iterations=len(cost_history),
        cost_history=np.array(cost_history),
        gradient_norms=np.array(grad_norms),
    )


def plot_decision_boundary(result: NewtonResult, X: np.ndarray, y: np.ndarray) -> None:
    theta = result.theta
    # Original (without intercept) exam scores
    exam1 = X[:, 1]
    exam2 = X[:, 2]

    admitted = y == 1
    not_admitted = y == 0

    plt.figure(figsize=(6, 5))
    plt.scatter(exam1[admitted], exam2[admitted], marker="+", label="Admitted")
    plt.scatter(exam1[not_admitted], exam2[not_admitted], marker="o", label="Not admitted")

    x_values = np.linspace(exam1.min() - 5, exam1.max() + 5, 100)
    if abs(theta[2]) < 1e-12:
        # Vertical boundary
        x0 = -theta[0] / theta[1]
        plt.axvline(x=x0, color="red", linestyle="--", label="Decision boundary")
    else:
        y_values = -(theta[0] + theta[1] * x_values) / theta[2]
        plt.plot(x_values, y_values, "r--", label="Decision boundary")

    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.close()


def plot_cost_history(result: NewtonResult) -> None:
    plt.figure(figsize=(6, 4))
    iterations = np.arange(1, result.cost_history.size + 1)
    plt.plot(iterations, result.cost_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title("Newton's Method Convergence")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("cost_history.png", dpi=150)
    plt.close()


def main() -> None:
    X, y = load_dataset()
    result = newton_method(X, y, max_iter=20, tol=1e-7)
    plot_decision_boundary(result, X, y)
    plot_cost_history(result)

    threshold_point = np.array([1.0, 20.0, 80.0])
    prob_admit = float(sigmoid(result.theta @ threshold_point))
    prob_not_admit = 1.0 - prob_admit

    print("=== Newton's Method Logistic Regression ===")
    print(f"Iterations: {result.iterations}")
    print(f"Theta: {result.theta}")
    print(f"Final cost: {result.cost_history[-1]:.6f}")
    print(f"Final gradient norm: {result.gradient_norms[-1]:.4e}")
    print(f"Probability admitted (20, 80): {prob_admit:.6f}")
    print(f"Probability NOT admitted (20, 80): {prob_not_admit:.6f}")
    print("Saved plots: decision_boundary.png, cost_history.png")


if __name__ == "__main__":
    main()
