## IMPORTS ##

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from kernels import LinearKernel, GaussianKernel, PolynomialKernel, SpectrumKernel, MismatchKernel, SumKernel
from utils import create_kmer_set,m_neighbours,get_neighbours
from classifiers.svm import SVM

## PARAMETERS ##


kernel = 'sum' #'linear' 'rbf', 'poly', 'spectrum' ot 'mismatch' (unsure if 'spectrum' and 'mismatch' work perfectly)
C = 1.0 #Parameter C for SVM
gamma = 10.0 #Parameter gamma for SVM (only for 'rbf' or 'poly')
coef0 = 1.0 #Parameter coef0 for SVM (only for 'poly')
degree = 3 #Parameter degree for SVM (only for 'poly')
k = 15 #Parameter k for SVM (only for 'spectrum' and 'mismatch')
m = 3 #Parameter m for SVM (only for 'mismatch')

shuffle = True #Shuffle the data

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

#If the kernel is 'mismatch', compute kmers neighbors
if kernel=='mismatch':

    #Dataset0
    try:
        # Load
        neighbours_0, kmer_set_0 = pickle.load(open('neighbours_0'+str(k)+'_'+str(m)+'.p', 'rb'))
        print('Neighbors correctly loaded')
    except:
        print('No file found, creating kmers neighbors')
        kmer_set_0 = create_kmer_set(X0_train[:,0], k, kmer_set={})
        kmer_set_0 = create_kmer_set(X0_test[:,0], k, kmer_set_0)
        neighbours_0 = get_neighbours(kmer_set_0, m)
        
        # Save neighbours and kmer set
        pickle.dump([neighbours_0, kmer_set_0], open('neighbours_0'+str(k)+'_'+str(m)+'.p', 'wb'))

    #Dataset1
    try:
        # Load
        neighbours_1, kmer_set_1 = pickle.load(open('neighbours_1'+str(k)+'_'+str(m)+'.p', 'rb'))
        print('Neighbors correctly loaded')
    except:
        print('No file found, creating kmers neighbors')
        kmer_set_1 = create_kmer_set(X1_train[:,0], k, kmer_set={})
        kmer_set_1 = create_kmer_set(X1_test[:,0], k, kmer_set_1)
        neighbours_1 = get_neighbours(kmer_set_1, m)
        
        # Save neighbours and kmer set
        pickle.dump([neighbours_1, kmer_set_1], open('neighbours_1'+str(k)+'_'+str(m)+'.p', 'wb'))

    #Dataset2
    try:
        # Load
        neighbours_2, kmer_set_2 = pickle.load(open('neighbours_2'+str(k)+'_'+str(m)+'.p', 'rb'))
        print('Neighbors correctly loaded')
    except:
        print('No file found, creating kmers neighbors')
        kmer_set_2 = create_kmer_set(X2_train[:,0], k, kmer_set={})
        kmer_set_2 = create_kmer_set(X2_test[:,0], k, kmer_set_2)
        neighbours_2 = get_neighbours(kmer_set_2, m)
        
        # Save neighbours and kmer set
        pickle.dump([neighbours_2, kmer_set_2], open('neighbours_2'+str(k)+'_'+str(m)+'.p', 'wb'))

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
print()

## APPLY SVM ON DATASET 0 ##

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
    #Dataset0
    k=15
    m=3
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_0_'+str(k)+'_'+str(m)+'.p', 'rb'))
    print('Neighbors correctly loaded')
    kernel_15 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)
    k = 12
    m = 2
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_0_'+str(k)+'_'+str(m)+'.p', 'rb'))
    kernel_12 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)
    k = 8
    m = 1
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_0_'+str(k)+'_'+str(m)+'.p', 'rb'))
    kernel_8 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)

    svm = SVM(kernel=SumKernel(kernels=[kernel_15, kernel_12,kernel_8], weights=[1.0, 1.0, 1.0]), C=C)

if kernel_on_matrices:
    svm.fit(X0_mat100_train, Y0_train)
    pred_0 = svm.predict_classes(X0_mat100_test)

else:
    svm.fit(X0_train, Y0_train)
    pred_0 = svm.predict_classes(X0_test)

## APPLY SVM ON DATASET 1 ##

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
    #Dataset1
    k=15
    m=3
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_1_'+str(k)+'_'+str(m)+'.p', 'rb'))
    print('Neighbors correctly loaded')
    kernel_15 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)
    k = 12
    m = 2
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_1_'+str(k)+'_'+str(m)+'.p', 'rb'))
    kernel_12 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)
    k = 8
    m = 1
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_1_'+str(k)+'_'+str(m)+'.p', 'rb'))
    kernel_8 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)

    svm = SVM(kernel=SumKernel(kernels=[kernel_15, kernel_12,kernel_8], weights=[1.0, 1.0, 1.0]), C=C)

if kernel_on_matrices:
    svm.fit(X1_mat100_train, Y1_train)
    pred_1 = svm.predict_classes(X1_mat100_test)

else:
    svm.fit(X1_train, Y1_train)
    pred_1 = svm.predict_classes(X1_test)


## APPLY SVM ON DATASET 2 ##

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
    #Dataset2
    k=15
    m=3
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_2_'+str(k)+'_'+str(m)+'.p', 'rb'))
    print('Neighbors correctly loaded')
    kernel_15 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)
    k = 12
    m = 2
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_2_'+str(k)+'_'+str(m)+'.p', 'rb'))
    kernel_12 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)
    k = 8
    m = 1
    neighbours, kmer_set = pickle.load(open('saved_neighbors/neighbours_2_'+str(k)+'_'+str(m)+'.p', 'rb'))
    kernel_8 = MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True)

    svm = SVM(kernel=SumKernel(kernels=[kernel_15, kernel_12,kernel_8], weights=[1.0, 1.0, 1.0]), C=C)

if kernel_on_matrices:
    svm.fit(X2_mat100_train, Y2_train)
    pred_2 = svm.predict_classes(X2_mat100_test)

else:
    svm.fit(X2_train, Y2_train)
    pred_2 = svm.predict_classes(X2_test)

## CREATE SUBMISSION FILE ##

pred = np.concatenate([pred_0.squeeze(),pred_1.squeeze(),pred_2.squeeze()])
pred = np.where(pred == -1, 0, 1)
pred_df = pd.DataFrame()
print(pred.shape)
print(pred)
pred_df['Bound'] = pred
pred_df.index.name = 'Id'
pred_df.to_csv('pred.csv', sep=',', header=True)