import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set matplotlib backend to Agg for non-interactive use
import matplotlib
matplotlib.use('Agg')

def load_data():
    """Load training data"""
    x = np.loadtxt('data/ex1Data/ex1x.dat')
    y = np.loadtxt('data/ex1Data/ex1y.dat')
    return x, y

def compute_cost(X, y, theta):
    """Compute cost function J(theta)"""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient_descent(X, y, learning_rate=0.07, iterations=1500):
    """Implement gradient descent"""
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def generate_plots():
    """Generate all plots for the experiment report"""

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Load data
    x, y = load_data()
    m = len(x)
    X = np.column_stack([np.ones(m), x])

    # Run gradient descent
    theta, cost_history = gradient_descent(X, y)

    # 1. Plot training data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='red', marker='o', s=50, alpha=0.7, label='Training data')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Height (meters)', fontsize=12)
    plt.title('Training Data: Height vs Age', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_data.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot regression line with training data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='red', marker='o', s=50, alpha=0.7, label='Training data')

    # Generate regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    X_line = np.column_stack([np.ones(len(x_line)), x_line])
    y_line = X_line.dot(theta)
    plt.plot(x_line, y_line, 'b-', linewidth=2, label=f'Linear regression\nh(x) = {theta[0]:.3f} + {theta[1]:.3f}x')

    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Height (meters)', fontsize=12)
    plt.title('Linear Regression: Height vs Age', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/regression_line.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Cost function convergence
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost J(θ)', fontsize=12)
    plt.title('Cost Function Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/cost_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 3D Cost function surface
    theta0_vals = np.linspace(-3, 3, 100)
    theta1_vals = np.linspace(-1, 1, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            t = np.array([theta0, theta1])
            J_vals[i, j] = compute_cost(X, y, t)

    Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

    # 3D Surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Theta0, Theta1, J_vals.T, cmap='viridis', alpha=0.8)

    # Mark the optimal theta
    optimal_cost = compute_cost(X, y, theta)
    ax.scatter([theta[0]], [theta[1]], [optimal_cost], color='red', s=100, label='Optimal θ')

    ax.set_xlabel('θ₀', fontsize=12)
    ax.set_ylabel('θ₁', fontsize=12)
    ax.set_zlabel('J(θ)', fontsize=12)
    ax.set_title('Cost Function J(θ) - 3D Surface', fontsize=14, fontweight='bold')

    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig('plots/cost_surface_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contour(Theta0, Theta1, J_vals.T, levels=20, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
    plt.scatter(theta[0], theta[1], color='red', s=100, marker='x',
               linewidth=3, label=f'Optimal θ\n({theta[0]:.3f}, {theta[1]:.3f})')
    plt.xlabel('θ₀', fontsize=12)
    plt.ylabel('θ₁', fontsize=12)
    plt.title('Cost Function J(θ) - Contour Plot', fontsize=14, fontweight='bold')
    plt.legend()
    plt.colorbar(contour)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/cost_contour.png', dpi=300, bbox_inches='tight')
    plt.close()

    return theta, cost_history

if __name__ == "__main__":
    print("Generating plots for experiment report...")
    theta, cost_history = generate_plots()

    print("Generated plots:")
    print("- plots/training_data.png")
    print("- plots/regression_line.png")
    print("- plots/cost_convergence.png")
    print("- plots/cost_surface_3d.png")
    print("- plots/cost_contour.png")

    print(f"\nFinal parameters: θ₀ = {theta[0]:.6f}, θ₁ = {theta[1]:.6f}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print("All plots saved successfully!")