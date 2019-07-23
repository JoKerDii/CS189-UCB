# lda mnist code

import numpy as np

# LDA
def lda_mean(X,y):
    Mean = np.zeros([len(set(y)), X.shape[1]])
    for i, label in enumerate(set(y)):
        X_class = X[y==label,:]
        Nc = X_class.shape[0]
        Mean[i, :] = X_class.mean(axis=0)
    return Mean

def lda_covariance(X,y):
    Cov = 0
    N = len(y)
    for i, label in enumerate(set(y)):
        X_class = X[y==label,:]
        Nc = X_class.shape[0]
        Cov += np.cov(X_class.T) * Nc
    return Cov / N

def lda_P(X,y):
    N = len(y)
    P = np.zeros([len(set(y))])
    for i, label in enumerate(set(y)):
        X_class = X[y==label,:]
        Nc = X_class.shape[0]
        P[i] = Nc / N
    return P

def predict(X_test,Mean,Cov,P): # X_val or X_test
    PreMat = np.linalg.pinv(Cov)
    L = Mean.dot(PreMat).dot(X_test.T).T \
            - 1/2*np.diag(Mean.dot(PreMat).dot(Mean.T)) \
            + np.log(P)
    yp = np.argmax(L.T, axis=0)
    return yp

def accuracy(X_test,y_test,Mean,Cov,P):
    yp = predict(X_test,Mean,Cov,P)
    N = len(y_test)
    error = (yp != y_test).sum()
    return 1 - error/N
    
def normalize(X):
    X_n = np.zeros(X.shape)
    for i in range(X.shape[0]):
        x = X[i, :]
        X_n[i, :] = (x/(np.sqrt(x.dot(x))+1e-20))
    return X_n


# load data
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
import random

mnist = io.loadmat("mnist-data/%s_data.mat" % "mnist")
print("\nloaded %s data!" % mnist)
fields = "test_data", "training_data", "training_labels"
for field in fields:
    print(field, mnist[field].shape)
    
mTraining = mnist['training_data']
mLabel = mnist['training_labels']
mTest = mnist['test_data']
train_label = np.concatenate((mTraining, mLabel), axis=1)
Mcolname = np.array([str(i) for i in np.arange(mTraining.shape[1] + 1)])
train_label_df = pd.DataFrame(data = train_label, columns = Mcolname)
num_of_rows = int(train_label_df.shape[0] * 1/6)
train_label_df.sample(frac = 1)

train_label_df = np.array(train_label_df)
X_train, y_train = normalize(train_label_df[10000:,:-1]),train_label_df[10000:,-1]
X_val,y_val = normalize(train_label_df[:10000,:-1]),train_label_df[:10000,-1]

sample_size = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000])
accuracy_matrix = np.zeros(len(sample_size))

for i, N in enumerate(sample_size):
    idx = random.sample(range(50000), N)
    Mean = lda_mean(X_train[idx, :], y_train[idx])
    Cov = lda_covariance(X_train[idx, :], y_train[idx])
    P = lda_P(X_train[idx, :], y_train[idx])
    accuracy_matrix[i] = accuracy(X_val, y_val,Mean,Cov,P)

plt.plot(sample_size, accuracy_matrix * 100)
plt.xscale('log')
plt.xlabel('Sample size')
plt.ylabel('Accuracy(%)')
plt.show()

# best sample size = 250
N = 250
idx = random.sample(range(50000), N)
Mean = lda_mean(X_train[idx, :], y_train[idx])
Cov = lda_covariance(X_train[idx, :], y_train[idx])
P = lda_P(X_train[idx, :], y_train[idx])
yp_lda = predict(X_val,Mean,Cov,P)
accuracy(X_val,y_val,Mean,Cov,P)

yp_lda = predict(mTest,Mean,Cov,P)