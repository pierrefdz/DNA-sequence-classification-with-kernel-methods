import numpy as np
import cvxopt

class RidgeRegression():
    """
    Ridge Regression implementation

    Usage:
    """

    def __init__(self, kernel, alpha):
        """
        alpha: float > 0, default=1.0
            Regularization strength
        """
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, X, y):
        """ Solve KRR using alpha = (K + lambda n I)^(-1) y \\
        X: array (n_samples, n_features) \\
        y: array of 0 or 1 (n_samples,)   
        """
        self.X_train = X
        n_samples = X.shape[0]
        K = self.kernel.gram(X)
        self.alphas = np.linalg.solve(K+self.alpha*np.eye(n_samples), y)
        return self.alphas

    def predict(self, X):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = self.kernel.gram(X, self.X_train)
        y = np.dot(K, self.alphas)
        return y

    def predict_classes(self, X, threshold=0.5):
        """ 
        X: array (n_samples, n_features)\\
        Return: 0 and 1 array (n_samples,)
        """
        K = self.kernel.gram(X, self.X_train)
        y = np.dot(K, self.alphas)
        return np.where(y>threshold, 1, 0)
