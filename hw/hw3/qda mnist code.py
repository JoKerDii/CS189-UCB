import numpy as np

def qda_mean(X,y):
    Mean = np.zeros([len(set(y)), X.shape[1]])
    for i, label in enumerate(set(y)):
        X_class = X[y==label,:]
        Nc = X_class.shape[0]
        Mean[i, :] = X_class.mean(axis=0)
    return Mean

def qda_covariance(X,y,a):
    Cov = np.zeros([X.shape[1], X.shape[1], len(set(y))])
    N = len(y)
    for i, label in enumerate(set(y)):
        X_class = X[y==label,:]
        Cov[:, :, i] = np.cov(X_class.T) + a*np.eye(X.shape[1])
    return Cov

def qda_P(X,y):
    N = len(y)
    P = np.zeros([len(set(y))])
    for i, label in enumerate(set(y)):
        X_class = X[y==label,:]
        Nc = X_class.shape[0]
        P[i] = Nc / N
    return P

def qda_Q(X_test,Mean,Cov,P):
    i = np.eye(Cov.shape[0], Cov.shape[0])
    PreMat = np.linalg.lstsq(Cov,i)[0]
    Const = - 1/2*np.log(np.linalg.det(Cov)+1e-20) + np.log(P)
    Qc = np.array([(-1/2*(x-Mean).dot(PreMat).dot(x-Mean)) + Const for x in X_test])
    return Qc

def predict(X_test,Mean,Cov,P): # X_val or X_test
    nC = len(P)
    L = np.zeros([nC, len(X_test)])
    for i in range(nC):
        L[i, :] = qda_Q(X_test, Mean[i], Cov[:, :, i], P[i])
    yp = np.argmax(L, axis=0)
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
	
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    Mean = qda_mean(X_train[idx, :], y_train[idx])
    Cov = qda_covariance(X_train[idx, :], y_train[idx],1e-8)
    P = qda_P(X_train[idx, :], y_train[idx])
    accuracy_matrix[i] = accuracy(X_val, y_val,Mean,Cov,P)

plt.plot(sample_size, accuracy_matrix * 100)
plt.xscale('log')
plt.xlabel('Sample size')
plt.ylabel('Accuracy(%)')
plt.show()

# best sample size = 200
N = 200
idx = random.sample(range(50000), N)
Mean = qda_mean(X_train[idx, :], y_train[idx])
Cov = qda_covariance(X_train[idx, :], y_train[idx],1e-8)
P = qda_P(X_train[idx, :], y_train[idx])
yp_lda = predict(X_val,Mean,Cov,P)
accuracy(X_val,y_val,Mean,Cov,P)

yp_lda_m = predict(mTest,Mean,Cov,P)

################### QDA performs better than LDA