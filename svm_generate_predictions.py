##### IMPORTS #####

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from kernels import LinearKernel, GaussianKernel, PolynomialKernel, SpectrumKernel, MismatchKernel, SumKernel
from utils import create_kmer_set,m_neighbours,get_neighbours,load_neighbors,load_or_compute_neighbors
from classifiers.svm import SVM

##### PARAMETERS #####

kernel = 'sum' #'linear' 'rbf', 'poly', 'spectrum', 'mismatch' or 'sum'
C = 5.0 #Parameter C for SVM
gamma = 10.0 #Parameter gamma for SVM (only for 'rbf' or 'poly')
coef0 = 1.0 #Parameter coef0 for SVM (only for 'poly')
degree = 3 #Parameter degree for SVM (only for 'poly')
k = 15 #Parameter k for SVM (only for 'spectrum' and 'mismatch')
m = 3 #Parameter m for SVM (only for 'mismatch')
list_k = [5,8,10,12,13,15] #List of parameters k for sum of mismatch kernels (only for 'sum')
list_m = [1,1,1,2,2,3] #List of parameters m for sum of mismatch kernels (only for 'sum')
weights = [1.0,1.0,1.0,1.0,1.0,1.0] #List of weights for sum of mismatch kernels (only for 'sum')

shuffle = True #Shuffle the data

##### LOAD DATA #####

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


##### PREPROCESS DATA #####

#Rescaling labels
Y0_train = np.where(Y0_train == 0, -1, 1)
Y1_train = np.where(Y1_train == 0, -1, 1)
Y2_train = np.where(Y2_train == 0, -1, 1)

#If the kernel is 'mismatch', compute kmers neighbors
if kernel=='mismatch':

    neighbours_0, kmer_set_0 = load_or_compute_neighbors(0,k,m)
    neighbours_1, kmer_set_1 = load_or_compute_neighbors(1,k,m)
    neighbours_2, kmer_set_2 = load_or_compute_neighbors(2,k,m)

#Some verifications for sum kernel
if kernel == 'sum':
    assert(len(list_k)==len(list_m))
    assert(len(weights)==len(list_m))

#Shuffling
if shuffle:

    shuffling_0 = np.random.permutation(len(X0_mat100_train))
    X0_train = X0_train[shuffling_0][:,0]
    X0_mat100_train = X0_mat100_train[shuffling_0]
    Y0_train = Y0_train[shuffling_0]

    shuffling_1 = np.random.permutation(len(X1_mat100_train))
    X1_train = X1_train[shuffling_1][:,0]
    X1_mat100_train = X1_mat100_train[shuffling_1]
    Y1_train = Y1_train[shuffling_1]

    shuffling_2 = np.random.permutation(len(X2_mat100_train))
    X2_train = X2_train[shuffling_2][:,0]
    X2_mat100_train = X2_mat100_train[shuffling_2]
    Y2_train = Y2_train[shuffling_2]

else:
    X0_train = X0_train[:,0]
    X1_train = X1_train[:,0]
    X2_train = X2_train[:,0]

#Check if the kernel applies on matrices or strings
kernel_on_matrices = (kernel=='linear' or kernel=='rbf' or kernel=='poly')

#Put test matrices into the right format
X0_test = X0_test[:,0]
X1_test = X1_test[:,0]
X2_test = X2_test[:,0]

## PRINT CONFIGURATION ##

print("Kernel:", kernel)
print("C:", C)
if kernel == 'rbf' or kernel == 'poly':
    print("Gamma:", gamma)
if kernel == 'poly':
    print("Coef0:", coef0)
    print("Degree:", degree)
if kernel== 'spectrum':
    print("K:",k)
if kernel == 'sum':
    print("List of Ks:",list_k)
    print("List of Ms:",list_m)
    print("Weights:", weights)
print()

##### APPLY SVM ON DATASET 0 #####

print("Applying SVM on dataset 0...")

if kernel=='linear':
    svm = SVM(kernel=LinearKernel(),C=C)
elif kernel=='rbf':
    svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
elif kernel=='poly':
    svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)
elif kernel=='spectrum':
    svm = SVM(kernel=SpectrumKernel(k=k),C=C)
elif kernel=='mismatch':
    svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_0, kmer_set=kmer_set_0,normalize=True), C=C)
elif kernel=='sum':
    dataset_nbr = 0 
    kernels = []
    for k,m in zip(list_k,list_m):
        neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
        kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
    svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)

if kernel_on_matrices:
    svm.fit(X0_mat100_train, Y0_train)
    pred_0 = svm.predict_classes(X0_mat100_test)

else:
    svm.fit(X0_train, Y0_train)
    pred_0 = svm.predict_classes(X0_test)

##### APPLY SVM ON DATASET 1 #####

print("Applying SVM on dataset 1...")

if kernel=='linear':
    svm = SVM(kernel=LinearKernel(),C=C)
elif kernel=='rbf':
    svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
elif kernel=='poly':
    svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)
elif kernel=='spectrum':
    svm = SVM(kernel=SpectrumKernel(k=k),C=C)
elif kernel=='mismatch':
    svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_1, kmer_set=kmer_set_1,normalize=True), C=C)
elif kernel=='sum':
    dataset_nbr = 1
    kernels = []
    for k,m in zip(list_k,list_m):
        neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
        kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
    svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)

if kernel_on_matrices:
    svm.fit(X1_mat100_train, Y1_train)
    pred_1 = svm.predict_classes(X1_mat100_test)

else:
    svm.fit(X1_train, Y1_train)
    pred_1 = svm.predict_classes(X1_test)


##### APPLY SVM ON DATASET 2 #####

print("Applying SVM on dataset 2...")

if kernel=='linear':
    svm = SVM(kernel=LinearKernel(),C=C)
elif kernel=='rbf':
    svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
elif kernel=='poly':
    svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)
elif kernel=='spectrum':
    svm = SVM(kernel=SpectrumKernel(k=k),C=C)
elif kernel=='mismatch':
    svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_2, kmer_set=kmer_set_2,normalize=True), C=C)
elif kernel=='sum':
    dataset_nbr = 2
    kernels = []
    for k,m in zip(list_k,list_m):
        neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
        kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
    svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)

if kernel_on_matrices:
    svm.fit(X2_mat100_train, Y2_train)
    pred_2 = svm.predict_classes(X2_mat100_test)

else:
    svm.fit(X2_train, Y2_train)
    pred_2 = svm.predict_classes(X2_test)

##### CREATE SUBMISSION FILE #####

pred = np.concatenate([pred_0.squeeze(),pred_1.squeeze(),pred_2.squeeze()])
pred = np.where(pred == -1, 0, 1)
pred_df = pd.DataFrame()
print(pred.shape)
print(pred)
pred_df['Bound'] = pred
pred_df.index.name = 'Id'
pred_df.to_csv('pred.csv', sep=',', header=True)