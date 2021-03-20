## IMPORTS ##

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from kernels import LinearKernel, GaussianKernel, PolynomialKernel, SpectrumKernel, MismatchKernel
from classifiers.svm import SVM

## PARAMETERS ##


kernel = 'rbf' # 'linear' 'rbf' or 'poly' #TODO: Add support for spectrum and mismatch
C = 10.0 #Parameter C for SVM
gamma = 10.0 #Parameter gamma for SVM (only for 'rbf' or 'poly')
coef0 = 1.0 #Parameter coef0 for SVM (only for 'poly')
degree = 3 #Parameter degree for SVM (only for 'poly')

shuffle = True #Shuffle the data
rescale_y = True #Rescale labels to -1 and 1

## LOAD DATA ##

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


## PREPROCESS DATA ##

#Rescaling labels
Y0_train = np.where(Y0_train == 0, -1, 1)
Y1_train = np.where(Y1_train == 0, -1, 1)
Y2_train = np.where(Y2_train == 0, -1, 1)

#Shuffling
if shuffle:

    shuffling_0 = np.random.permutation(len(X0_mat100_train))
    X0_train = X0_train[shuffling_0]
    X0_mat100_train = X0_mat100_train[shuffling_0]
    Y0_train = Y0_train[shuffling_0]

    shuffling_1 = np.random.permutation(len(X1_mat100_train))
    X1_train = X1_train[shuffling_1]
    X1_mat100_train = X1_mat100_train[shuffling_1]
    Y1_train = Y1_train[shuffling_1]

    shuffling_2 = np.random.permutation(len(X2_mat100_train))
    X2_train = X2_train[shuffling_2]
    X2_mat100_train = X2_mat100_train[shuffling_2]
    Y2_train = Y2_train[shuffling_2]


print("Kernel:", kernel)
print("C:", C)
if kernel == 'rbf' or kernel == 'poly':
    print("Gamma:", gamma)
if kernel == 'poly':
    print("Coef0:", coef0)
    print("Degree:", degree)
print()


## APPLY SVM ON DATASET 0 ##

print("Applying SVM on dataset 0...")

if kernel=='linear':
    svm = SVM(kernel=LinearKernel(),C=C)
elif kernel=='rbf':
    svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
elif kernel=='poly':
    svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)

svm.fit(X0_mat100_train, Y0_train)
pred_0 = svm.predict_classes(X0_mat100_test)

## APPLY SVM ON DATASET 1 ##

print("Applying SVM on dataset 1...")

if kernel=='linear':
    svm = SVM(kernel=LinearKernel(),C=C)
elif kernel=='rbf':
    svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
elif kernel=='poly':
    svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)

svm.fit(X1_mat100_train, Y1_train)
pred_1 = svm.predict_classes(X1_mat100_test)

## APPLY SVM ON DATASET 2 ##

print("Applying SVM on dataset 2...")

if kernel=='linear':
    svm = SVM(kernel=LinearKernel(),C=C)
elif kernel=='rbf':
    svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
elif kernel=='poly':
    svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)

svm.fit(X2_mat100_train, Y2_train)
pred_2 = svm.predict_classes(X2_mat100_test)

## CREATE SUBMISSION FILE ##

pred = np.concatenate([pred_0,pred_1,pred_2])
pred = np.where(pred == -1, 0, 1)
pred_df = pd.DataFrame()
pred_df['Bound'] = pred
pred_df.index.name = 'Id'
pred_df.to_csv('pred.csv', sep=',', header=True)