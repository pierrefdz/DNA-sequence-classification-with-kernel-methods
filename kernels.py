import numpy as np

class Kernel():
    """ Abstract Kernel class"""

    def __init__(self):
        pass

    def similarity(self, x, y):
        """ Similarity between 2 feature vectors (depends on the type of kernel)"""
        return -1

    def gram(self, X1, X2=None):
        """ Compute the gram matrix of a data vector X where the (i,j) entry is defined as <Xi,Xj>\\
        X1: data vector (n_samples_1 x n_features)
        X2: data vector (n_samples_2 x n_features), if None compute the gram matrix for (X1,X1)
        """
        if X2 is None: 
            X2=X1
        n_samples_1 = X1.shape[0]
        n_samples_2 = X2.shape[0]
        G = np.zeros((n_samples_1, n_samples_2))
        for ii in range(n_samples_1):
            for jj in range(n_samples_2):
                G[ii,jj] = self.similarity(X1[ii], X2[jj])
        return G


class LinearKernel(Kernel):

    def __init__(self):
        super().__init__()

    def similarity(self, x, y):
        """ linear kernel : k(x,y) = <x,y> \\
        x, y: array (n_features,)
        """
        return np.dot(x,y)


class GaussianKernel(Kernel):

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def similarity(self, x, y):
        """ gaussian kernel : k(x,y) = 1/ sqrt(2 pi sigma2)^n * exp( - ||x-y||^2 / 2 sigma^2 )\\
        x, y: array (n_features,)
        """
        norm_fact = (np.sqrt(2 * np.pi) * self.sigma) ** len(x)
        return np.exp(-np.linalg.norm(x-y)**2 / (2 * self.sigma**2)) / norm_fact


class PolynomialKernel(Kernel):

    def __init__(self, gamma=1, coef0=1, degree=3):
        super().__init__()
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def similarity(self, x, y):
        """ polynomial kernel : k(x,y) = (gamma <x,y> + r)^d \\
        x, y: array (n_features,)
        """
        return (self.gamma * np.dot(x,y) + self.coef0)**self.degree


class SpectrumKernel(Kernel):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def similarity(self, x, y):
        """ Spectrum kernel \\
        x, y: string
        """
        substr_x, counts_x = np.unique([x[i:i+self.k] for i in range(len(x)-self.k+1)], return_counts=True)
        return np.sum(np.char.count(y, substr_x)*counts_x)