import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    def __init__(self, learning_rate=0.07):
        self.learning_rate = learning_rate
        self.theta = None
        self.cost_history = []

    def load_data(self):
        """Load training data from files"""
        x = np.loadtxt('data/ex1Data/ex1x.dat')
        y = np.loadtxt('data/ex1Data/ex1y.dat')
        return x, y

    def add_intercept(self, x):
        """Add intercept term (bias) to feature matrix"""
        m = len(x)
        return np.column_stack([np.ones(m), x])

    def compute_cost(self, X, y, theta):
        """Compute cost function J(theta)"""
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost

    def gradient_descent(self, X, y, iterations=1500):
        """Implement gradient descent algorithm"""
        m = len(y)
        self.theta = np.zeros(X.shape[1])

        print(f"Initial theta: {self.theta}")

        for i in range(iterations):
            predictions = X.dot(self.theta)
            errors = predictions - y

            # Gradient descent update rule
            gradient = (1/m) * X.T.dot(errors)
            self.theta = self.theta - self.learning_rate * gradient

            # Compute and store cost
            cost = self.compute_cost(X, y, self.theta)
            self.cost_history.append(cost)

            # Record first iteration results
            if i == 0:
                print(f"After 1st iteration: theta0 = {self.theta[0]:.6f}, theta1 = {self.theta[1]:.6f}")

        print(f"Final theta after convergence: theta0 = {self.theta[0]:.6f}, theta1 = {self.theta[1]:.6f}")
        return self.theta

    def predict(self, x):
        """Make predictions using learned parameters"""
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        X = self.add_intercept(x)
        return X.dot(self.theta)

    def plot_data_and_line(self, x, y):
        """Plot training data and regression line"""
        plt.figure(figsize=(10, 6))

        # Plot training data
        plt.scatter(x, y, c='red', marker='o', label='Training data')

        # Plot regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = self.predict(x_line)
        plt.plot(x_line, y_line, 'b-', label='Linear regression')

        plt.xlabel('Age in years')
        plt.ylabel('Height in meters')
        plt.title('Linear Regression: Height vs Age')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def visualize_cost_function(self, X, y):
        """Visualize cost function J(theta) as 3D surface"""
        # Create grid of theta values
        theta0_vals = np.linspace(-3, 3, 100)
        theta1_vals = np.linspace(-1, 1, 100)

        # Initialize cost matrix
        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

        # Compute cost for each theta combination
        for i, theta0 in enumerate(theta0_vals):
            for j, theta1 in enumerate(theta1_vals):
                theta = np.array([theta0, theta1])
                J_vals[i, j] = self.compute_cost(X, y, theta)

        # Create meshgrid for plotting
        Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

        # 3D Surface plot
        fig = plt.figure(figsize=(15, 5))

        # Surface plot
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(Theta0, Theta1, J_vals.T, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('θ₀')
        ax1.set_ylabel('θ₁')
        ax1.set_zlabel('J(θ)')
        ax1.set_title('Cost Function J(θ) - 3D Surface')

        # Mark the optimal theta
        optimal_cost = self.compute_cost(X, y, self.theta)
        ax1.scatter([self.theta[0]], [self.theta[1]], [optimal_cost],
                   color='red', s=100, label='Optimal θ')

        # Contour plot
        ax2 = fig.add_subplot(132)
        contour = ax2.contour(Theta0, Theta1, J_vals.T, levels=20)
        ax2.scatter(self.theta[0], self.theta[1], color='red', s=100,
                   marker='x', linewidth=3, label='Optimal θ')
        ax2.set_xlabel('θ₀')
        ax2.set_ylabel('θ₁')
        ax2.set_title('Cost Function J(θ) - Contour Plot')
        ax2.legend()

        # Cost history
        ax3 = fig.add_subplot(133)
        ax3.plot(self.cost_history)
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Cost J(θ)')
        ax3.set_title('Cost Function During Training')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    # Initialize linear regression model
    lr = LinearRegression(learning_rate=0.07)

    # Load data
    print("Loading training data...")
    x, y = lr.load_data()
    print(f"Loaded {len(x)} training examples")

    # Add intercept term
    X = lr.add_intercept(x)

    # Plot initial data
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='blue', marker='o')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.title('Training Data: Height vs Age')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Run gradient descent
    print("\nRunning gradient descent...")
    theta = lr.gradient_descent(X, y, iterations=1500)

    # Plot data with regression line
    print("\nPlotting regression line...")
    lr.plot_data_and_line(x, y)

    # Make predictions
    print("\nMaking predictions:")
    age_35 = np.array([3.5])
    age_7 = np.array([7.0])

    height_35 = lr.predict(age_35)[0]
    height_7 = lr.predict(age_7)[0]

    print(f"Predicted height for 3.5-year-old: {height_35:.3f} meters")
    print(f"Predicted height for 7-year-old: {height_7:.3f} meters")

    # Visualize cost function
    print("\nVisualizing cost function...")
    lr.visualize_cost_function(X, y)

    print(f"\nFinal model: h(x) = {theta[0]:.3f} + {theta[1]:.3f}x")

if __name__ == "__main__":
    main()