"""
Sandbox file to try things messily 
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC

from kernels import LinearKernel, GaussianKernel, PolynomialKernel

from classifiers.logistic_regression import LogisticRegression
from classifiers.ridge_regression import RidgeRegression
from classifiers.svm import SVM

# %% Read csv files
# shape (2000,1): string
X0_train = pd.read_csv("data/Xtr0.csv", sep=",", index_col=0).values
X1_train = pd.read_csv("data/Xtr1.csv", sep=",", index_col=0).values
X2_train = pd.read_csv("data/Xtr2.csv", sep=",", index_col=0).values

# shape (2000,100): float
X0_mat100_train = pd.read_csv("data/Xtr0_mat100.csv", sep=" ", header=None).values
X1_mat100_train = pd.read_csv("data/Xtr1_mat100.csv", sep=" ", header=None).values
X2_mat100_train = pd.read_csv("data/Xtr2_mat100.csv", sep=" ", header=None).values

# shape (2000,1): string
X0_test = pd.read_csv("data/Xte0.csv", sep=",", index_col=0).values
X1_test = pd.read_csv("data/Xte1.csv", sep=",", index_col=0).values
X2_test = pd.read_csv("data/Xte2.csv", sep=",", index_col=0).values

# shape (2000,100): float
X0_mat100_test = pd.read_csv("data/Xte0_mat100.csv", sep=" ", header=None).values
X1_mat100_test = pd.read_csv("data/Xte1_mat100.csv", sep=" ", header=None).values
X2_mat100_test = pd.read_csv("data/Xte2_mat100.csv", sep=" ", header=None).values

# shape (2000,1): 0 or 1
Y0_train = pd.read_csv("data/Ytr0.csv", sep=",", index_col=0).values
Y1_train = pd.read_csv("data/Ytr1.csv", sep=",", index_col=0).values
Y2_train = pd.read_csv("data/Ytr2.csv", sep=",", index_col=0).values


"""
# %%
kernel = LinearKernel()
ridge = RidgeRegression(kernel=kernel, alpha=0.01)
ridge.fit(X0_mat100_train, Y0_train)
# %%
ridge.predict(X0_mat100_train)

# KRR with sklearn
# ridge = KernelRidge(alpha=0.001, kernel='linear')
# ridge.fit(X0_mat100_train, Y0_train2)
# ridge.predict(X0_mat100_train)

# Kernel logistic regression
# logreg = LogisticRegression(kernel=kernel, lambda_=0.01)
# Y0_train_scaled = np.where(Y0_train == 0, -1, 1)
# logreg.fit(X0_mat100_train, Y0_train)
# logreg.predict(X0_mat100_train)

"""

Y0_train_scaled = np.where(Y0_train == 0, -1, 1)

print("Real classes distribution:")

print(len(Y0_train_scaled), "samples")
print(np.sum(Y0_train_scaled== 1), "positive")
print(np.sum(Y0_train_scaled==-1), "negative")

compare_svms = True

if compare_svms:

    #Sklearn SVM
    svm_scikit = SVC(C=100.0,kernel='linear')
    svm_scikit.fit(X0_mat100_train, Y0_train_scaled)
    svm_scikit_classes = svm_scikit.predict(X0_mat100_train)

    #My SVM
    my_svm = SVM(kernel=LinearKernel(),C=100.0)
    my_svm.fit(X0_mat100_train, Y0_train_scaled)

    y_pred = my_svm.predict(X0_mat100_train)
    y_classes_pred = my_svm.predict_classes(X0_mat100_train)


    print("Accuracy on train (sklearn SVM):", np.sum(svm_scikit_classes==np.squeeze(Y0_train_scaled))/len(Y0_train_scaled))
    print("Accuracy on train (my SVM):", np.sum(np.squeeze(y_classes_pred)==np.squeeze(Y0_train_scaled))/len(Y0_train_scaled))
    print("Similarity between sklearn SVM and my SVM:",np.sum(np.squeeze(y_classes_pred)==np.squeeze(svm_scikit_classes))/len(Y0_train_scaled))


