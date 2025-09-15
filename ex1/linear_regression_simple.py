import numpy as np
import matplotlib.pyplot as plt

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

    print(f"Initial theta: {theta}")

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y

        # Gradient descent update
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient

        # Store cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Print first iteration result
        if i == 0:
            print(f"After 1st iteration: theta0 = {theta[0]:.6f}, theta1 = {theta[1]:.6f}")

    print(f"Final theta: theta0 = {theta[0]:.6f}, theta1 = {theta[1]:.6f}")
    return theta, cost_history

def main():
    print("=== Linear Regression Experiment ===\n")

    # 1. Load data
    print("1. Loading training data...")
    x, y = load_data()
    print(f"   Loaded {len(x)} training examples")
    print(f"   Age range: {x.min():.1f} - {x.max():.1f} years")
    print(f"   Height range: {y.min():.3f} - {y.max():.3f} meters\n")

    # 2. Add intercept term
    m = len(x)
    X = np.column_stack([np.ones(m), x])
    print("2. Added intercept term (bias)")
    print(f"   Feature matrix shape: {X.shape}\n")

    # 3. Run gradient descent
    print("3. Running gradient descent...")
    theta, cost_history = gradient_descent(X, y)

    print(f"   Final cost: {cost_history[-1]:.6f}")
    print(f"   Model: h(x) = {theta[0]:.3f} + {theta[1]:.3f}x\n")

    # 4. Make predictions
    print("4. Making predictions:")
    ages_to_predict = [3.5, 7.0]

    for age in ages_to_predict:
        X_pred = np.array([1, age])
        height_pred = X_pred.dot(theta)
        print(f"   Age {age} years -> Height {height_pred:.3f} meters")

    print("\n5. Cost function analysis:")
    print(f"   Initial cost: {cost_history[0]:.6f}")
    print(f"   Final cost: {cost_history[-1]:.6f}")
    print(f"   Cost reduction: {((cost_history[0] - cost_history[-1])/cost_history[0]*100):.1f}%")

    # Save results to file
    with open('results.txt', 'w') as f:
        f.write("Linear Regression Results\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Training examples: {len(x)}\n")
        f.write(f"Learning rate: 0.07\n")
        f.write(f"Iterations: 1500\n\n")
        f.write("Parameters after convergence:\n")
        f.write(f"theta0 (intercept): {theta[0]:.6f}\n")
        f.write(f"theta1 (slope): {theta[1]:.6f}\n\n")
        f.write(f"Model equation: h(x) = {theta[0]:.3f} + {theta[1]:.3f}x\n\n")
        f.write("Predictions:\n")
        for age in ages_to_predict:
            X_pred = np.array([1, age])
            height_pred = X_pred.dot(theta)
            f.write(f"Age {age} years -> Height {height_pred:.3f} meters\n")
        f.write(f"\nFinal cost: {cost_history[-1]:.6f}\n")

    print(f"\n6. Results saved to 'results.txt'")

if __name__ == "__main__":
    main()