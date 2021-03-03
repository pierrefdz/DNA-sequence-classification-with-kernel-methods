import numpy as np
import cvxopt

class SVM():
    """
    SVM implementation
    
    Usage:
        svm = SVM(kernel='linear', C=1)
        svm.fit(X_train, y_train)
        svm.predict(X_test)
    """

    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
