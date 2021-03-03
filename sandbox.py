"""
Sandbox file to try things messily 
"""

# %% Imports
import numpy as np
import pandas as pd

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

# %%
kernel = LinearKernel()
ridge = RidgeRegression(kernel=kernel, alpha=1)
ridge.fit(X0_mat100_train, Y0_train)
# %%
ridge.predict(X0_mat100_train)

