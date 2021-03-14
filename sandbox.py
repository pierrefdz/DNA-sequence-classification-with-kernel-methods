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

## Preprocessing

split_ratio = 0.8
shuffle = True
rescale_y = True

#Stack the data
X_mat_train_full = np.vstack((X0_mat100_train,X1_mat100_train,X2_mat100_train))
Y_train_full = np.vstack((Y0_train,Y1_train,Y2_train))

if rescale_y:
    Y_train_full = np.where(Y_train_full == 0, -1, 1)

#Shuffle the data
if shuffle:
    shuffling = np.random.permutation(len(X_mat_train_full))
    X_mat_train_full = X_mat_train_full[shuffling]
    Y_train_full = Y_train_full[shuffling]

#Split the data into 
nb_in_train = int(split_ratio*len(X_mat_train_full))
X_mat_train,X_mat_val = X_mat_train_full[:nb_in_train],X_mat_train_full[nb_in_train:]
Y_train,Y_val = Y_train_full[:nb_in_train],Y_train_full[nb_in_train:]


print("Real classes distribution:")
print(len(Y_train_full), "samples")
print(np.sum(Y_train_full== 1), "positive")
print(np.sum(Y_train_full==-1), "negative")

print("Training classes distribution:")
print(len(Y_train), "samples")
print(np.sum(Y_train== 1), "positive")
print(np.sum(Y_train==-1), "negative")

print("Validation classes distribution:")
print(len(Y_val), "samples")
print(np.sum(Y_val== 1), "positive")
print(np.sum(Y_val==-1), "negative")

compare_svms = True

if compare_svms:

    #Sklearn SVM
    print("Applying Sklearn SVM...")
    svm_scikit = SVC(C=100.0,kernel='linear')
    svm_scikit.fit(X_mat_train, Y_train)
    svm_scikit_classes_train = svm_scikit.predict(X_mat_train)
    svm_scikit_classes_val = svm_scikit.predict(X_mat_val)

    print("Accuracy on train (sklearn SVM):", np.sum(svm_scikit_classes_train==np.squeeze(Y_train))/len(Y_train))
    print("Accuracy on val (sklearn SVM):", np.sum(svm_scikit_classes_val==np.squeeze(Y_val))/len(Y_val))

    #My SVM
    print("Applying my SVM...")
    my_svm = SVM(kernel=LinearKernel(),C=100.0)
    my_svm.fit(X_mat_train, Y_train)
    my_svm_classes_train = my_svm.predict_classes(X_mat_train)
    my_svm_classes_val = my_svm.predict_classes(X_mat_val)

    print("Accuracy on train (my SVM):", np.sum(np.squeeze(my_svm_classes_train)==np.squeeze(Y_train))/len(Y_train))    
    print("Accuracy on val (my SVM):", np.sum(np.squeeze(my_svm_classes_val)==np.squeeze(Y_val))/len(Y_val))

    #Comparison
    print("Similarity on train between sklearn SVM and my SVM:",np.sum(np.squeeze(my_svm_classes_train)==np.squeeze(svm_scikit_classes_train))/len(Y_train))
    print("Similarity on val between sklearn SVM and my SVM:",np.sum(np.squeeze(my_svm_classes_val)==np.squeeze(svm_scikit_classes_val))/len(Y_val))


