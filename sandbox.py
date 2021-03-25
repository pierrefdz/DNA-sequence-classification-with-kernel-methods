"""
Sandbox file to try things messily 
"""

# Imports
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from kernels import LinearKernel, GaussianKernel, PolynomialKernel, SpectrumKernel, MismatchKernel
from classifiers.logistic_regression import LogisticRegression
from classifiers.ridge_regression import RidgeRegression
from classifiers.svm import SVM

# Read csv files

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

# Kernel logistic regression
# logreg = LogisticRegression(kernel=kernel, lambda_=0.01)
# Y0_train_scaled = np.where(Y0_train == 0, -1, 1)
# logreg.fit(X0_mat100_train, Y0_train)
# logreg.predict(X0_mat100_train)

"""

## Preprocessing

fraction_of_data = 0.2 #Put a small value for faster tests
split_ratio = 0.8 #Ratio of data in train set
shuffle = True #Shuffle the data
rescale_y = True #Rescale labels to -1 and 1

#Stack the data
X_mat_train_full = np.vstack((X0_mat100_train,X1_mat100_train,X2_mat100_train))
X_train_full = np.vstack((X0_train,X1_train,X2_train))
Y_train_full = np.vstack((Y0_train,Y1_train,Y2_train))

if rescale_y:
    Y_train_full = np.where(Y_train_full == 0, -1, 1)

#Shuffle the data
if shuffle:
    shuffling = np.random.permutation(len(X_mat_train_full))
    X_mat_train_full = X_mat_train_full[shuffling]
    X_train_full = X_train_full[shuffling]
    Y_train_full = Y_train_full[shuffling]

#Take a fraction of the data
nb_samples = int(fraction_of_data*len(X_mat_train_full))
X_mat_train_full = X_mat_train_full[:nb_samples]
X_train_full = X_train_full[:nb_samples]
Y_train_full = Y_train_full[:nb_samples]

#Split the data into train and val
nb_in_train = int(split_ratio*len(X_mat_train_full))
X_mat_train,X_mat_val = X_mat_train_full[:nb_in_train],X_mat_train_full[nb_in_train:]
X_train,X_val = X_train_full[:nb_in_train],X_train_full[nb_in_train:]
Y_train,Y_val = Y_train_full[:nb_in_train],Y_train_full[nb_in_train:]


print("Real classes distribution:")
print(len(Y_train_full), "samples")
print(np.sum(Y_train_full== 1), "positive")
print(np.sum(Y_train_full==-1), "negative")
print()

print("Training classes distribution:")
print(len(Y_train), "samples")
print(np.sum(Y_train== 1), "positive")
print(np.sum(Y_train==-1), "negative")
print()

print("Validation classes distribution:")
print(len(Y_val), "samples")
print(np.sum(Y_val== 1), "positive")
print(np.sum(Y_val==-1), "negative")
print()


test_our_svm = False
test_spectrum = False
test_mismatch = True

if test_our_svm:

    ## Test our SVM implementation

    #Parameters
    kernel = 'poly' # 'linear' 'rbf' or 'poly'
    C = 1.0
    gamma = 1/(X_mat_train.shape[1] * X_mat_train.var())
    coef0 = 1.0
    degree = 3

    print("Kernel:", kernel)
    print("C:", C)
    if kernel != 'linear':
        print("Gamma:", gamma)
    if kernel == 'poly':
        print("Coef0:", coef0)
        print("Degree:", degree)
    print()

    #Our SVM
    print("Applying our SVM...")
    if kernel=='linear':
        our_svm = SVM(kernel=LinearKernel(),C=C)
    elif kernel=='rbf':
        our_svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5/gamma),normalize=False),C=C)
    elif kernel=='poly':
        our_svm = SVM(kernel=PolynomialKernel(gamma=gamma,coef0=coef0,degree=degree),C=C)
    our_svm.fit(X_mat_train, Y_train)
    our_svm_classes_train = our_svm.predict_classes(X_mat_train)
    our_svm_classes_val = our_svm.predict_classes(X_mat_val)

    print("Accuracy on train (our SVM):", np.sum(np.squeeze(our_svm_classes_train)==np.squeeze(Y_train))/len(Y_train))    
    print("Accuracy on val (our SVM):", np.sum(np.squeeze(our_svm_classes_val)==np.squeeze(Y_val))/len(Y_val))


if test_spectrum:

    ## First tests using the spectrum kernel (does not work for the moment)

    #Parameters
    C = 1.0
    k = 12

    print("Kernel: Spectrum")
    print("C:", C)
    print("K:", k)
    print()

    #Our SVM
    print("Applying our SVM...")

    our_svm = SVM(kernel=SpectrumKernel(k=k),C=C)

    our_svm.fit(X_train[:,0], Y_train)
    our_svm_classes_train = our_svm.predict_classes(X_train[:,0])
    our_svm_classes_val = our_svm.predict_classes(X_val[:,0])

    print("Accuracy on train (our SVM):", np.sum(np.squeeze(our_svm_classes_train)==np.squeeze(Y_train))/len(Y_train))    
    print("Accuracy on val (our SVM):", np.sum(np.squeeze(our_svm_classes_val)==np.squeeze(Y_val))/len(Y_val))


if test_mismatch:

    def create_kmer_set(X, k, kmer_set={}):
        """
        Return a set of all kmers appearing in the dataset.
        """
        len_seq = len(X[0])
        idx = len(kmer_set)
        for i in range(len(X)):
            x = X[i]
            kmer_x = [x[i:i + k] for i in range(len_seq - k + 1)]
            for kmer in kmer_x:
                if kmer not in kmer_set:
                    kmer_set[kmer] = idx
                    idx += 1
        return kmer_set


    def m_neighbours(kmer, m, recurs=0):
        """
        Return a list of neighbours kmers (up to m mismatches).
        """
        if m == 0:
            return [kmer]

        letters = ['G', 'T', 'A', 'C']
        k = len(kmer)
        neighbours = m_neighbours(kmer, m - 1, recurs + 1)

        for j in range(len(neighbours)):
            neighbour = neighbours[j]
            for i in range(recurs, k - m + 1):
                for l in letters:
                    neighbours.append(neighbour[:i] + l + neighbour[i + 1:])
        return list(set(neighbours))


    def get_neighbours(kmer_set, m):
        """
        Find the neighbours given a set of kmers.
        """
        kmers_list = list(kmer_set.keys())
        kmers = np.array(list(map(list, kmers_list)))
        num_kmers, kmax = kmers.shape
        neighbours = {}
        for i in range(num_kmers):
            neighbours[kmers_list[i]] = []

        for i in tqdm(range(num_kmers)):
            kmer = kmers_list[i]
            kmer_neighbours = m_neighbours(kmer, m)
            for neighbour in kmer_neighbours:
                if neighbour in kmer_set:
                    neighbours[kmer].append(neighbour)
        return neighbours

    k = 12
    m = 2

    try:
        # Load
        neighbours, kmer_set = pickle.load(open('neighbours_0'+str(k)+'_'+str(m)+'.p', 'rb'))
        print('Neighbors correctly loaded')
    except:
        print('No file found, creating kmers neighbors')
        kmer_set = create_kmer_set(X0_train[:,0], k)
        kmer_set = create_kmer_set(X0_test[:,0], k, kmer_set)
        neighbours = get_neighbours(kmer_set, m)
        
        # Save neighbours and kmer set
        pickle.dump([neighbours, kmer_set], open('neighbours_0'+str(k)+'_'+str(m)+'.p', 'wb'))

    print('Doing SVM')
    C = 1
    svm = SVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set), C=C)
    
    X0_train, X0_val = X0_train[:1600], X0_train[1600:]

    Y0_train = np.where(Y0_train == 0, -1, 1)
    Y0_train, Y0_val = Y0_train[:1600], Y0_train[1600:]

    svm.fit(X0_train[:,0], Y0_train)

    pred_train = svm.predict_classes(X0_train[:,0])
    # pred = np.where(pred == -1, 0, 1) 
    print( np.sum(np.squeeze(pred_train)==np.squeeze(Y0_train)) / len(Y0_train) )
    
    pred_val = svm.predict_classes(X0_val[:,0])
    # pred = np.where(pred == -1, 0, 1)
    print( np.sum(np.squeeze(pred_val)==np.squeeze(Y0_val)) / len(Y0_val) )
    
    
    