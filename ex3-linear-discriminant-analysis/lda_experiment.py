"""
Linear Discriminant Analysis (LDA) Implementation
Experiment 3: Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def load_data(filename):
    """Load data from file"""
    return np.loadtxt(filename)


class LDA:
    """Linear Discriminant Analysis implementation"""

    def __init__(self):
        self.w = None  # projection direction
        self.means = None  # class means
        self.overall_mean = None

    def fit_two_class(self, X1, X2):
        """
        Fit LDA for 2 classes

        Parameters:
        -----------
        X1, X2: numpy arrays of shape (n_samples, n_features)
            Data for class 1 and class 2

        Returns:
        --------
        w: numpy array
            Projection direction (normalized)
        """
        # Calculate means
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)

        # Calculate within-class scatter matrix Sw
        S1 = np.zeros((X1.shape[1], X1.shape[1]))
        for x in X1:
            x_minus_mean = (x - mean1).reshape(-1, 1)
            S1 += x_minus_mean @ x_minus_mean.T

        S2 = np.zeros((X2.shape[1], X2.shape[1]))
        for x in X2:
            x_minus_mean = (x - mean2).reshape(-1, 1)
            S2 += x_minus_mean @ x_minus_mean.T

        Sw = S1 + S2

        # Calculate projection direction
        # w = Sw^(-1) * (mean1 - mean2)
        mean_diff = mean1 - mean2
        self.w = np.linalg.inv(Sw) @ mean_diff

        # Normalize w
        self.w = self.w / np.linalg.norm(self.w)

        self.means = [mean1, mean2]

        return self.w

    def fit_multi_class(self, X_list, labels=None):
        """
        Fit LDA for multiple classes

        Parameters:
        -----------
        X_list: list of numpy arrays
            Each array contains data for one class
        labels: list of strings (optional)
            Class labels

        Returns:
        --------
        W: numpy array of shape (n_features, n_components)
            Projection directions
        """
        n_classes = len(X_list)
        n_features = X_list[0].shape[1]

        # Calculate overall mean
        all_data = np.vstack(X_list)
        overall_mean = np.mean(all_data, axis=0)
        self.overall_mean = overall_mean

        # Calculate class means
        class_means = [np.mean(X, axis=0) for X in X_list]
        self.means = class_means

        # Calculate within-class scatter matrix Sw
        Sw = np.zeros((n_features, n_features))
        for X in X_list:
            class_mean = np.mean(X, axis=0)
            for x in X:
                x_minus_mean = (x - class_mean).reshape(-1, 1)
                Sw += x_minus_mean @ x_minus_mean.T

        # Calculate between-class scatter matrix Sb
        Sb = np.zeros((n_features, n_features))
        for i, X in enumerate(X_list):
            n_samples = X.shape[0]
            mean_diff = (class_means[i] - overall_mean).reshape(-1, 1)
            Sb += n_samples * (mean_diff @ mean_diff.T)

        # Solve generalized eigenvalue problem: Sb * w = lambda * Sw * w
        # This is equivalent to solving: Sw^(-1) * Sb * w = lambda * w
        try:
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

            # Sort eigenvectors by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select top k eigenvectors (k = n_classes - 1 for LDA)
            n_components = min(n_classes - 1, n_features)
            W = eigenvectors[:, :n_components]

            # Normalize
            for i in range(n_components):
                W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

            self.w = W
            return W
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Using pseudo-inverse.")
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]
            n_components = min(n_classes - 1, n_features)
            W = eigenvectors[:, :n_components]
            for i in range(n_components):
                W[:, i] = W[:, i] / np.linalg.norm(W[:, i])
            self.w = W
            return W

    def project(self, X):
        """Project data onto the LDA direction(s)"""
        if self.w.ndim == 1:
            return X @ self.w
        else:
            return X @ self.w

    def get_line_points(self, X1, X2, margin=1.0):
        """
        Get points for plotting the decision boundary line

        Parameters:
        -----------
        X1, X2: numpy arrays
            Data points
        margin: float
            Extension margin for the line

        Returns:
        --------
        line_points: numpy array of shape (2, 2)
            Start and end points of the line
        """
        if self.w.ndim > 1:
            w = self.w[:, 0]
        else:
            w = self.w

        # Get data range
        all_data = np.vstack([X1, X2])
        x_min, x_max = all_data[:, 0].min() - margin, all_data[:, 0].max() + margin
        y_min, y_max = all_data[:, 1].min() - margin, all_data[:, 1].max() + margin

        # The line passes through the midpoint of the two means
        # and is perpendicular to w
        midpoint = (self.means[0] + self.means[1]) / 2

        # Perpendicular direction to w
        w_perp = np.array([-w[1], w[0]])

        # Find line endpoints within the plot range
        t_values = []

        # Check intersections with boundaries
        # x = x_min
        if abs(w_perp[0]) > 1e-10:
            t = (x_min - midpoint[0]) / w_perp[0]
            y = midpoint[1] + t * w_perp[1]
            if y_min <= y <= y_max:
                t_values.append(t)

        # x = x_max
        if abs(w_perp[0]) > 1e-10:
            t = (x_max - midpoint[0]) / w_perp[0]
            y = midpoint[1] + t * w_perp[1]
            if y_min <= y <= y_max:
                t_values.append(t)

        # y = y_min
        if abs(w_perp[1]) > 1e-10:
            t = (y_min - midpoint[1]) / w_perp[1]
            x = midpoint[0] + t * w_perp[0]
            if x_min <= x <= x_max:
                t_values.append(t)

        # y = y_max
        if abs(w_perp[1]) > 1e-10:
            t = (y_max - midpoint[1]) / w_perp[1]
            x = midpoint[0] + t * w_perp[0]
            if x_min <= x <= x_max:
                t_values.append(t)

        if len(t_values) >= 2:
            t_values.sort()
            t1, t2 = t_values[0], t_values[-1]
        else:
            t1, t2 = -10, 10

        point1 = midpoint + t1 * w_perp
        point2 = midpoint + t2 * w_perp

        return np.array([point1, point2])


def plot_two_class_lda(X1, X2, lda, title="LDA for 2 Classes", show_projections=False):
    """
    Plot LDA results for 2 classes

    Parameters:
    -----------
    X1, X2: numpy arrays
        Data for class 1 and class 2
    lda: LDA object
        Fitted LDA model
    title: str
        Plot title
    show_projections: bool
        Whether to show projection points
    """
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', s=50, label='Class 1 (Red)', edgecolors='k')
    plt.scatter(X2[:, 0], X2[:, 1], c='blue', marker='o', s=50, label='Class 2 (Blue)', edgecolors='k')

    # Plot decision boundary (perpendicular to projection direction)
    line_points = lda.get_line_points(X1, X2)
    plt.plot(line_points[:, 0], line_points[:, 1], 'k-', linewidth=2, label='Decision Boundary')

    if show_projections:
        # Project points onto the line
        all_points = np.vstack([X1, X2])

        # Midpoint between means
        midpoint = (lda.means[0] + lda.means[1]) / 2

        # For each point, find its projection on the decision boundary
        w_perp = np.array([-lda.w[1], lda.w[0]])

        for i, x in enumerate(X1):
            # Projection formula: proj = midpoint + ((x - midpoint) · w_perp) * w_perp
            t = np.dot(x - midpoint, w_perp)
            proj = midpoint + t * w_perp
            plt.plot([x[0], proj[0]], [x[1], proj[1]], 'r--', alpha=0.3, linewidth=1)
            plt.scatter(proj[0], proj[1], c='darkred', marker='s', s=30, zorder=5)

        for i, x in enumerate(X2):
            t = np.dot(x - midpoint, w_perp)
            proj = midpoint + t * w_perp
            plt.plot([x[0], proj[0]], [x[1], proj[1]], 'b--', alpha=0.3, linewidth=1)
            plt.scatter(proj[0], proj[1], c='darkblue', marker='s', s=30, zorder=5)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    return plt


def plot_multi_class_lda(X_list, labels, lda, title="LDA for Multiple Classes"):
    """
    Plot LDA results for multiple classes

    Parameters:
    -----------
    X_list: list of numpy arrays
        Data for each class
    labels: list of str
        Class labels
    lda: LDA object
        Fitted LDA model
    title: str
        Plot title
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    plt.figure(figsize=(8, 6))

    # Plot original data
    for i, (X, label, color) in enumerate(zip(X_list, labels, colors[:len(X_list)])):
        plt.scatter(X[:, 0], X[:, 1], c=color, marker='o', s=50,
                   label=f'{label}', edgecolors='k')

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    return plt


def plot_lda_projection_2d(X_list, labels, lda, title="LDA Projection to 2D"):
    """
    Plot LDA projection for multiple classes onto 2D space

    Parameters:
    -----------
    X_list: list of numpy arrays
        Data for each class
    labels: list of str
        Class labels
    lda: LDA object
        Fitted LDA model
    title: str
        Plot title
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    plt.figure(figsize=(8, 6))

    # Project data
    for i, (X, label, color) in enumerate(zip(X_list, labels, colors[:len(X_list)])):
        X_proj = lda.project(X)
        if X_proj.ndim == 1:
            # 1D projection, add zero for y-axis
            X_proj = np.column_stack([X_proj, np.zeros(len(X_proj))])

        plt.scatter(X_proj[:, 0], X_proj[:, 1] if X_proj.shape[1] > 1 else np.zeros(len(X_proj)),
                   c=color, marker='o', s=50, label=f'{label}', edgecolors='k')

    plt.xlabel('LD1', fontsize=12)
    plt.ylabel('LD2' if lda.w.shape[1] > 1 else '', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt


def main():
    """Main function to run all experiments"""

    # Load data
    print("Loading data...")
    red_data = load_data('data/ex3red.dat')
    blue_data = load_data('data/ex3blue.dat')
    green_data = load_data('data/ex3green.dat')

    print(f"Red data shape: {red_data.shape}")
    print(f"Blue data shape: {blue_data.shape}")
    print(f"Green data shape: {green_data.shape}")

    # ========================================
    # Part 1: LDA for 2 Classes
    # ========================================
    print("\n" + "="*50)
    print("Part 1: LDA for 2 Classes (Red vs Blue)")
    print("="*50)

    lda_2class = LDA()
    w = lda_2class.fit_two_class(red_data, blue_data)

    print(f"\nProjection direction w: {w}")
    print(f"Class 1 (Red) mean: {lda_2class.means[0]}")
    print(f"Class 2 (Blue) mean: {lda_2class.means[1]}")

    # Plot 1: Original data with decision boundary
    plot_two_class_lda(red_data, blue_data, lda_2class,
                       title="LDA for 2 Classes: Red vs Blue")
    plt.savefig('lda_2class_boundary.png', dpi=300, bbox_inches='tight')
    print("\nSaved: lda_2class_boundary.png")
    plt.close()

    # Plot 2: With projections
    plot_two_class_lda(red_data, blue_data, lda_2class,
                       title="LDA for 2 Classes with Projections",
                       show_projections=True)
    plt.savefig('lda_2class_projections.png', dpi=300, bbox_inches='tight')
    print("Saved: lda_2class_projections.png")
    plt.close()

    # Calculate separation metric
    red_proj = lda_2class.project(red_data)
    blue_proj = lda_2class.project(blue_data)

    print(f"\nProjected Red data mean: {np.mean(red_proj):.4f}")
    print(f"Projected Blue data mean: {np.mean(blue_proj):.4f}")
    print(f"Separation distance: {abs(np.mean(red_proj) - np.mean(blue_proj)):.4f}")

    # ========================================
    # Part 2: LDA for 3 Classes
    # ========================================
    print("\n" + "="*50)
    print("Part 2: LDA for 3 Classes (Red, Blue, Green)")
    print("="*50)

    lda_3class = LDA()
    X_list = [red_data, blue_data, green_data]
    labels = ['Red', 'Blue', 'Green']

    W = lda_3class.fit_multi_class(X_list, labels)

    print(f"\nProjection matrix W shape: {W.shape}")
    print(f"First discriminant direction: {W[:, 0]}")
    if W.shape[1] > 1:
        print(f"Second discriminant direction: {W[:, 1]}")

    # Plot 1: Original data
    plot_multi_class_lda(X_list, labels, lda_3class,
                        title="LDA for 3 Classes: Original Data")
    plt.savefig('lda_3class_original.png', dpi=300, bbox_inches='tight')
    print("\nSaved: lda_3class_original.png")
    plt.close()

    # Plot 2: Projected data
    plot_lda_projection_2d(X_list, labels, lda_3class,
                          title="LDA Projection for 3 Classes")
    plt.savefig('lda_3class_projection.png', dpi=300, bbox_inches='tight')
    print("Saved: lda_3class_projection.png")
    plt.close()

    # Calculate class separations
    print("\nClass separation analysis:")
    for i, (X, label) in enumerate(zip(X_list, labels)):
        X_proj = lda_3class.project(X)
        print(f"{label} - Projected mean: {np.mean(X_proj, axis=0)}")

    print("\n" + "="*50)
    print("All experiments completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()