# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 09:48:51 2019

@author: dizhen
"""

"""
MNIST - CODE

1. data loading
2. data partitioning (sets 10,000 validation set) without sklearn
3. SVM, training samples are 100,200,500,1000,2000,5000,10000
4. hyperparameter tuning, using 10000 training samples to find best C
5. build model with 10000 samples and predict test data
"""

## data loading 
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
for data_name in ["mnist", "spam", "cifar10"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)
    
mnist = io.loadmat("data/%s_data.mat" % "mnist")
mTraining = np.array(mnist['training_data'])
mLabel = np.array(mnist['training_labels'])
mTest = np.array(mnist['test_data'])

## data partitioning (sets 10,000 validation set) without sklearn
train_label = np.concatenate((mTraining, mLabel), axis=1)
Mcolname = np.array([str(i) for i in np.arange(mTraining.shape[1] + 1)])
train_label_df = pd.DataFrame(data = train_label, columns = Mcolname)
num_of_rows = int(train_label_df.shape[0] * 1/6)
train_label_df.sample(frac = 1)

X_train,y_train = train_label_df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(mTraining.shape[1])]],
                                     train_label_df.loc[num_of_rows+1:,'784']
X_val,y_val = train_label_df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(mTraining.shape[1])]],
                                 train_label_df.loc[:num_of_rows,'784']

## SVM, training samples are 100,200,500,1000,2000,5000,10000
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

acclist=[]
def trainingAll(arr,C):
    """
    A function to train data iterately with a sample or 
    different number of samples and a C value or different 
    C values.
    """
    for i in arr:
        for j in C:
            svm_clf=Pipeline((
            ('scaler',StandardScaler()),
            ('linear_svc',LinearSVC(C=j,loss='hinge'))
            ))
            svm_clf.fit(X_train[:i],y_train[:i])
            pred_svm=svm_clf.predict(X_val)
            acclist.append(accuracy_score(y_val,pred_svm))
            print("svm classification result,C="+ str(j) + 
                  " and training samples = "+ str(i) + ':')
            print("accuracy_score:",accuracy_score(y_val,pred_svm))
            
arr = np.array([100,200,500, 1000, 2000, 5000, 10000])
C = np.array([1])
trainingAll(arr,C)
"""For 100,200,500,1000,2000,5000,10000 samples, the accuracy is 
0.60622,0.63976,0.73736,0.76498,0.78926,0.81452,0.85722
"""

# save
nplist = np.array(acclist)
np.savetxt("m-acc.csv", nplist, delimiter=",")

# plot
errorM = 1-nplist
fig = plt.figure()
plt.plot(arr, errorM, label="Error rate", color="black")
plt.title("Error Rate Curve With SVM")
plt.xlabel("Number Of Sample")
plt.ylabel("Error rate")
plt.show()
fig.savefig('error rate curve of M.png')

# hyperparameter tuning, using 10000 training samples to find best C
arrC = np.array([10000])
Cs = np.array([0.001, 0.01, 0.1, 1, 10])
acclist=[]
trainingAll(arrC,Cs)
"""
For C = 0.001,0.01,0.1,1,10, the accuracy is 0.83168,0.88092,0.8794,
0.85796,0.83754, so that the best C is 0.01
"""
## build model with 10000 samples and predict test data
train_label_df.sample(frac = 1)
X_train,y_train = train_label_df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(mTraining.shape[1])]], 
                                     train_label_df.loc[num_of_rows+1:,'784']
X_val,y_val = train_label_df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(mTraining.shape[1])]],
                                 train_label_df.loc[:num_of_rows,'784']

svm_clf=Pipeline((
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=0.01,loss='hinge'))
        ))

svm_clf.fit(X_train[:10000],y_train[:10000])
y_pred = svm_clf.predict(X_val)
print('svm classification result:')
print("accuracy_score:",accuracy_score(y_val,y_pred))
predOfTest = svm_clf.predict(mTest)

## save to csv
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv('submission_m.csv', index_label='Id')
results_to_csv(predOfTest)


### improve the model

# grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'kernel':('linear', 'rbf'),
              'C':(0.1,0.01,0.001),
              'gamma': (1,2,3,'auto'),
              'decision_function_shape':('ovo','ovr'),
              'shrinking':(True,False)}
svm = SVC()
clf = GridSearchCV(svm, parameters)
clf.fit(X_train[:10000],y_train[:10000])

from sklearn.model_selection import cross_val_score
print("accuracy:"+
      str(np.average(cross_val_score(clf,
                                     X_train[:10000],
                                     y_train[:10000], 
                                     scoring='accuracy'))))
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
clf.score(X_train[:10000], y_train[:10000])

# prediction
y_pred_2 = clf.predict(X_val)
print('svm classification result:')
print("accuracy_score:",accuracy_score(y_val,y_pred_2))
predOfTest_2 = clf.predict(mTest)

## save to csv
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv('submission_m_2.csv', index_label='Id')
results_to_csv(predOfTest_2)












