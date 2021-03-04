import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    """
    Logistic Regression implementation
    Usage:
    """

    def __init__(self, kernel, lambda_=1.):
        """
        lambda_: float, regularization parameter
        """
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        """ Solve KLR using
        X: array (n_samples, n_features) \\
        y: array of -1 or 1 (n_samples,1)
        """

        self.X_train = X
        n_samples = X.shape[0]
        num_iter = 10
        eps = 1e-6
        K = self.kernel.gram(X)

        # Initialization
        alpha = np.zeros((n_samples, 1))

        for i in range(num_iter):
            alpha_old = alpha

            m = K @ alpha
            W = sigmoid(m) * sigmoid(-m)
            z = m + y / sigmoid(-y * m)

            # Solve WKRR
            sqrt_W = np.sqrt(W)
            alpha = sqrt_W * np.linalg.solve(sqrt_W * K * sqrt_W.T + n_samples * self.lambda_ * np.eye(n_samples),
                                             sqrt_W * y)

            if np.sum((alpha - alpha_old) ** 2) < eps:
                break

        self.alphas = alpha
        return self.alphas

    def predict(self, X):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = self.kernel.gram(X, self.X_train)
        y = np.dot(K, self.alpha)
        return y

    def predict_classes(self, X, threshold=0):
        """
        X: array (n_samples, n_features)\\
        Return: 0 and 1 array (n_samples,)
        """
        K = self.kernel.gram(X, self.X_train)
        y = np.dot(K, self.alpha)
        return np.where(y > threshold, 1, 0)
