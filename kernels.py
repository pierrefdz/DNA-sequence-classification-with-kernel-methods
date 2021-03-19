import numpy as np
import scipy.sparse as sparse

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

    def __init__(self, sigma,normalize=True):
        super().__init__()
        self.sigma = sigma
        self.normalize = normalize

    def similarity(self, x, y):
        """ gaussian kernel : k(x,y) = 1/ sqrt(2 pi sigma2)^n * exp( - ||x-y||^2 / 2 sigma^2 )\\
        x, y: array (n_features,)
        """

        if self.normalize:
            norm_fact = (np.sqrt(2 * np.pi) * self.sigma) ** len(x)
            return np.exp(-np.linalg.norm(x-y)**2 / (2 * self.sigma**2)) / norm_fact
        else:
            return np.exp(-np.linalg.norm(x-y)**2 / (2 * self.sigma**2))


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


class MismatchKernel(Kernel):

    def __init__(self, k, m, neighbours, kmer_set):
        super().__init__()
        self.k = k
        self.m = m
        self.kmer_set = kmer_set
        self.neighbours = neighbours

#     def similarity(self, x, y):
#         """ Mismatch kernel \\
#         x, y: string
#         """
#         substr_x = [c for c in x]
#         substr_x = np.array([substr_x[i:i + self.k] for i in range(len(x) - self.k + 1)])
#
#         substr_y = [c for c in y]
#         substr_y = np.array([substr_y[i:i + self.k] for i in range(len(y) - self.k + 1)])
#
#         sp = 0
#         for i in range(len(substr_x)):
#             sp += np.sum(np.sum(substr_x[i] != substr_y, axis=1) <= self.m)
#         return sp

    def neighbour_embed_data(self, X):
        """
        Embed data with neighbours.
        X: array of string
        """
        X_emb = [{} for i in range(len(X))]
        for i in range(len(X)):
            x = X[i]
            kmer_x = [x[j:j + self.k] for j in range(len(X[0]) - self.k + 1)]
            for kmer in kmer_x:
                neigh_kmer = self.neighbours[kmer]
                for neigh in neigh_kmer:
                    idx_neigh = self.kmer_set[neigh]
                    if idx_neigh in X_emb[i]:
                        X_emb[i][idx_neigh] += 1
                    else:
                        X_emb[i][idx_neigh] = 1
        return X_emb
    
    def one_hot_embed_data(self, X):
        """
        Embed data with one hot encoding.
        X: array of string
        """
        X_emb = [{} for i in range(len(X))]
        for i in range(len(X)):
            x = X[i]
            kmer_x = [x[j:j + self.k] for j in range(len(X[0]) - self.k + 1)]
            for kmer in kmer_x:
                X_emb[i][self.kmer_set[kmer]] = 1
        return X_emb
    
    def to_sparse(self, X_emb):
        """
        Embed data to sparse matrix.
        X_emb: list of dict.
        """
        data, row, col = [], [], []
        for i in range(len(X_emb)):
            x = X_emb[i]
            data += list(x.values())
            row += list(x.keys())
            col += [i for j in range(len(x))]
        X_sm = sparse.coo_matrix((data, (row, col)))
        return X_sm

    def similarity(self, x, y):
        """ Mismatch kernel \\
        x, y: dict
        """
        sp = 0
        for idx_neigh in x:
            if idx_neigh in y:
                sp += x[idx_neigh] * y[idx_neigh]
        return sp

    def gram(self, X1, X2=None):
        """ Compute the gram matrix of a data vector X where the (i,j) entry is defined as <Xi,Xj>\\
        X1: array of string (n_samples_1,)
        X2: array of string (n_samples_2,), if None compute the gram matrix for (X1,X1)
        """
        
        X1_emb = self.neighbour_embed_data(X1)
        X1_sm = self.to_sparse(X1)
        
        if X2 is None:
            X2 = X1
        X2_emb = self.one_hot_embed_data(X2)
        X2_sm = self.to_sparse(X2)

        # Reshape matrices if the sizes are different
        nadd_row = abs(X1_sm.shape[0] - X2_sm.shape[0])
        if X1_sm.shape[0] > X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row-1], [X2_sm.shape[1]-1])))
            X2_sm = sparse.vstack((X2_sm, add_row))
        elif X1_sm.shape[0] < X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row - 1], [X1_sm.shape[1] - 1])))
            X1_sm = sparse.vstack((X1_sm, add_row))

        G = (X1_sm.T * X2_sm).todense().astype('float')
        return G