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
coef0 = 10.0 #Parameter coef0 for SVM (only for 'poly')
degree = 3 #Parameter degree for SVM (only for 'poly')
k = 10 #Parameter k for SVM (only for 'spectrum' and 'mismatch')
m = 1 #Parameter m for SVM (only for 'mismatch')
list_k = [8,12] #List of parameters k for sum of mismatch kernels (only for 'sum')
list_m = [1,2] #List of parameters m for sum of mismatch kernels (only for 'sum')
weights = [1.0,1.0] #List of weights for sum of mismatch kernels (only for 'sum')

shuffle = False #Shuffle the data
k_fold = 5 #Number of folds for cross_validation

cross_validate_0 = True #Choose to cross_validate on dataset 0 or not
cross_validate_1 = True #Choose to cross_validate on dataset 1 or not
cross_validate_2 = True #Choose to cross_validate on dataset 2 or not

test_on_little_data = False #Test on little data. /!\ Before running real tests, make sure this is set to "False"

##### LOAD DATA #####

# shape (2000,1): string
X0_train = pd.read_csv("data/Xtr0.csv", sep=",", index_col=0).values
X1_train = pd.read_csv("data/Xtr1.csv", sep=",", index_col=0).values
X2_train = pd.read_csv("data/Xtr2.csv", sep=",", index_col=0).values

# shape (2000,100): float
X0_mat100_train = pd.read_csv("data/Xtr0_mat100.csv", sep=" ", header=None).values
X1_mat100_train = pd.read_csv("data/Xtr1_mat100.csv", sep=" ", header=None).values
X2_mat100_train = pd.read_csv("data/Xtr2_mat100.csv", sep=" ", header=None).values

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


#Take a small fraction of the data for tests 
if test_on_little_data:

    frac = 0.2

    X0_train = X0_train[:int(frac*len(X0_train))]
    X0_mat100_train = X0_mat100_train[:int(frac*len(X0_mat100_train))]
    Y0_train = Y0_train[:int(frac*len(Y0_train))]

    X1_train = X1_train[:int(frac*len(X1_train))]
    X1_mat100_train = X1_mat100_train[:int(frac*len(X1_mat100_train))]
    Y1_train = Y1_train[:int(frac*len(Y1_train))]

    X2_train = X2_train[:int(frac*len(X2_train))]
    X2_mat100_train = X2_mat100_train[:int(frac*len(X2_mat100_train))]
    Y2_train = Y2_train[:int(frac*len(Y2_train))]

#Check if the kernel applies on matrices or strings
kernel_on_matrices = (kernel=='linear' or kernel=='rbf' or kernel=='poly')

##### PRINT CONFIGURATION #####

print("Kernel:", kernel)
print("C:", C)
if kernel == 'rbf' or kernel == 'poly':
    print("Gamma:", gamma)
if kernel == 'poly':
    print("Coef0:", coef0)
    print("Degree:", degree)
if kernel== 'spectrum' or kernel == 'mismatch':
    print("K:",k)
if kernel == 'mismatch':
    print("M:",m)
if kernel == 'sum':
    print("List of Ks:",list_k)
    print("List of Ms:",list_m)
    print("Weights:", weights)
print()

##### CROSS-VALIDATE ON DATASET 0 #####

if cross_validate_0:

    print("Cross-validating on dataset 0...")

    if kernel=='linear':
        svm = SVM(kernel = LinearKernel(), C=C)
    elif kernel=='rbf':
        svm = SVM(kernel = GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False), C=C)
    elif kernel=='poly':
        svm = SVM(kernel = PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree), C=C)
    elif kernel=='spectrum':
        svm = SVM(kernel = SpectrumKernel(k=k), C=C)
    elif kernel=='mismatch':
        svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_0, kmer_set=kmer_set_0,normalize=True), C=C)
    elif kernel=='sum':
        dataset_nbr = 0 
        kernels = []
        for k,m in zip(list_k,list_m):
            neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
            kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
        svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)

    val_accs_0 = []

    split = np.linspace(0,len(X0_mat100_train),num=k_fold+1).astype(int)
    #print(split)

    for i in range(k_fold):

        print("Doing fold", i+1,"...")
        print()

        frac_val = 1.0/k_fold

        indices_val = np.arange(len(X0_mat100_train))[split[i]:split[i+1]]
        indices_train = np.concatenate([np.arange(len(X0_mat100_train))[0:split[i]],np.arange(len(X0_mat100_train))[split[i+1]:]]) 

        X0_mat100_train_,X0_mat100_val_ = X0_mat100_train[indices_train],X0_mat100_train[indices_val]
        X0_train_,X0_val_ = X0_train[indices_train],X0_train[indices_val]
        Y0_train_,Y0_val_ = Y0_train[indices_train],Y0_train[indices_val]

        print('Doing SVM')
        
        if kernel_on_matrices:
            svm.fit(X0_mat100_train_, Y0_train_)
            pred_train = svm.predict_classes(X0_mat100_train_)
            pred_val = svm.predict_classes(X0_mat100_val_)

        else:
            svm.fit(X0_train_, Y0_train_)
            pred_train = svm.predict_classes(X0_train_)
            pred_val = svm.predict_classes(X0_val_)

        train_acc = np.sum(np.squeeze(pred_train)==np.squeeze(Y0_train_)) / len(Y0_train_)
        val_acc = np.sum(np.squeeze(pred_val)==np.squeeze(Y0_val_)) / len(Y0_val_)

        print("Accuracy on train:", train_acc)
        print("Accuracy on val:", val_acc)
        val_accs_0.append(val_acc.copy())

    print(val_accs_0)
    print("Mean accuracy on val over the k folds (dataset 0):", np.mean(val_accs_0))


##### CROSS-VALIDATE ON DATASET 1 #####

if cross_validate_1:

    print("Cross-validating on dataset 1...")

    if kernel=='linear':
        svm = SVM(kernel = LinearKernel(), C=C)
    elif kernel=='rbf':
        svm = SVM(kernel = GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False), C=C)
    elif kernel=='poly':
        svm = SVM(kernel = PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree), C=C)
    elif kernel=='spectrum':
        svm = SVM(kernel = SpectrumKernel(k=k), C=C)
    elif kernel=='mismatch':
        svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_1, kmer_set=kmer_set_1,normalize=True), C=C)
    elif kernel=='sum':
        dataset_nbr = 1
        kernels = []
        for k,m in zip(list_k,list_m):
            neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
            kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
        svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)

    val_accs_1 = []

    split = np.linspace(0,len(X1_mat100_train),num=k_fold+1).astype(int)
    #print(split)

    for i in range(k_fold):

        print("Doing fold", i+1,"...")
        print()

        frac_val = 1.0/k_fold

        indices_val = np.arange(len(X1_mat100_train))[split[i]:split[i+1]]
        indices_train = np.concatenate([np.arange(len(X1_mat100_train))[0:split[i]],np.arange(len(X1_mat100_train))[split[i+1]:]]) 

        X1_mat100_train_,X1_mat100_val_ = X1_mat100_train[indices_train], X1_mat100_train[indices_val]
        X1_train_,X1_val_ = X1_train[indices_train], X1_train[indices_val]
        Y1_train_,Y1_val_ = Y1_train[indices_train], Y1_train[indices_val]

        print('Doing SVM')
        
        if kernel_on_matrices:
            svm.fit(X1_mat100_train_, Y1_train_)
            pred_train = svm.predict_classes(X1_mat100_train_)
            pred_val = svm.predict_classes(X1_mat100_val_)

        else:
            svm.fit(X1_train_, Y1_train_)
            pred_train = svm.predict_classes(X1_train_)
            pred_val = svm.predict_classes(X1_val_)

        train_acc = np.sum(np.squeeze(pred_train)==np.squeeze(Y1_train_)) / len(Y1_train_)
        val_acc = np.sum(np.squeeze(pred_val)==np.squeeze(Y1_val_)) / len(Y1_val_)

        print("Accuracy on train:", train_acc)
        print("Accuracy on val:", val_acc)
        val_accs_1.append(val_acc.copy())

    print(val_accs_1)
    print("Mean accuracy on val over the k folds (dataset 1):", np.mean(val_accs_1))


##### CROSS-VALIDATE ON DATASET 2 #####

if cross_validate_2:

    print("Cross-validating on dataset 2...")

    if kernel=='linear':
        svm = SVM(kernel = LinearKernel(), C=C)
    elif kernel=='rbf':
        svm = SVM(kernel = GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False), C=C)
    elif kernel=='poly':
        svm = SVM(kernel = PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree), C=C)
    elif kernel=='spectrum':
        svm = SVM(kernel = SpectrumKernel(k=k), C=C)
    elif kernel=='mismatch':
        svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_2, kmer_set=kmer_set_2,normalize=True), C=C)
    elif kernel=='sum':
        dataset_nbr = 2
        kernels = []
        for k,m in zip(list_k,list_m):
            neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
            kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
        svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)

    val_accs_2 = []

    split = np.linspace(0,len(X2_mat100_train),num=k_fold+1).astype(int)
    #print(split)

    for i in range(k_fold):

        print("Doing fold", i+1,"...")
        print()

        frac_val = 1.0/k_fold

        indices_val = np.arange(len(X2_mat100_train))[split[i]:split[i+1]]
        indices_train = np.concatenate([np.arange(len(X2_mat100_train))[0:split[i]],np.arange(len(X2_mat100_train))[split[i+1]:]]) 

        X2_mat100_train_,X2_mat100_val_ = X2_mat100_train[indices_train],X2_mat100_train[indices_val]
        X2_train_,X2_val_ = X2_train[indices_train],X2_train[indices_val]
        Y2_train_,Y2_val_ = Y2_train[indices_train],Y2_train[indices_val]

        print('Doing SVM')
        
        if kernel_on_matrices:
            svm.fit(X2_mat100_train_, Y2_train_)
            pred_train = svm.predict_classes(X2_mat100_train_)
            pred_val = svm.predict_classes(X2_mat100_val_)

        else:
            svm.fit(X2_train_, Y2_train_)
            pred_train = svm.predict_classes(X2_train_)
            pred_val = svm.predict_classes(X2_val_)

        train_acc = np.sum(np.squeeze(pred_train)==np.squeeze(Y2_train_)) / len(Y2_train_)
        val_acc = np.sum(np.squeeze(pred_val)==np.squeeze(Y2_val_)) / len(Y2_val_)

        print("Accuracy on train:", train_acc)
        print("Accuracy on val:", val_acc)
        val_accs_2.append(val_acc.copy())

    print(val_accs_2)
    print("Mean accuracy on val over the k folds (dataset 2):", np.mean(val_accs_2))




print("Summary:")

if cross_validate_0:
    print("Mean accuracy on val over the k folds (dataset 0):", np.mean(val_accs_0))
if cross_validate_1:
    print("Mean accuracy on val over the k folds (dataset 1):", np.mean(val_accs_1))
if cross_validate_2:
    print("Mean accuracy on val over the k folds (dataset 2):", np.mean(val_accs_2))