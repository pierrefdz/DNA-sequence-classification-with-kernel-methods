import numpy as np
import cvxopt
from cvxopt import matrix
import cvxpy as cp

class SVM():
    """
    SVM implementation
    
    Usage:
        svm = SVM(kernel='linear', C=1)
        svm.fit(X_train, y_train)
        svm.predict(X_test)
    """

    def __init__(self, kernel, C=1.0):
        """
        C: float > 0, default=1.0, regularization parameter
        """
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):

        self.X_train = X
        n_samples = X.shape[0]
        self.X_train_gram = self.kernel.gram(X)

        #Define the optimization problem to solve
        P = (y @ y.T) * self.X_train_gram
        q = -np.ones(n_samples)
        G = np.block([[np.eye(n_samples)],[-np.eye(n_samples)]])
        h = np.concatenate((self.C*np.ones(n_samples),np.zeros(n_samples)))

        #Solve the problem
        
        #With cvxopt
        P=matrix(P)
        q=matrix(q)
        G=matrix(G)
        h=matrix(h)
        solver = cvxopt.solvers.qp(P=P,q=q,G=G,h=h)
        x = solver['x']
        self.alphas = np.squeeze(y)*np.squeeze(np.array(x))
        
        """
        #With cvxpy
        x = cp.Variable(n_samples)
        objective = cp.Minimize((cp.quad_form(x, P) + 2 * q.T @ x))
        constraints = [G @ x <= h]
        prob = cp.Problem(objective,constraints)
        prob.solve()
        print(x.value)
        self.alphas = np.squeeze(y)*np.array(x.value)
        """

        #Retrieve the support vectors
        self.support_vectors_indices = np.squeeze(np.array(x)) > 1e-4
        self.alphas = self.alphas[self.support_vectors_indices]
        self.support_vectors = self.X_train[self.support_vectors_indices]

        print(len(self.support_vectors), "support vectors out of",len(self.X_train), "training samples")

        return self.alphas


    def predict(self, X):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = self.kernel.gram(X, self.support_vectors)
        y = np.dot(K, self.alphas)
        return y

    def predict_classes(self, X, threshold=0):
        """
        X: array (n_samples, n_features)\\
        Return: 0 and 1 array (n_samples,)
        """
        K = self.kernel.gram(X, self.support_vectors)
        y = np.dot(K, self.alphas)
        return np.where(y > threshold, 1, -1)

