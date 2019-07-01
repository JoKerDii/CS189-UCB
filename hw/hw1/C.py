# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 09:25:17 2019

@author: dizhen
"""
"""
CIFAR-10 - CODE
1. data loading
2. data partitioning (set 5000 validation set) without sklearn
3. SVM, training samples are 100,200,500,1000,2000,5000
4. build model by 5000 samples and predict test data 
"""

# data loading
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
for data_name in ["mnist", "spam", "cifar10"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)  

cifar10 = io.loadmat("data/%s_data.mat" % "cifar10")
cTraining = np.array(cifar10['training_data'])
cLabel = np.array(cifar10['training_labels'])
cTest = np.array(cifar10['test_data'])

# data partitioning (set 5000 validation set) without sklearn
train_label = np.concatenate((cTraining, cLabel), axis=1)
Ccolname = np.array([str(i) for i in np.arange(cTraining.shape[1] + 1)])
train_label_df = pd.DataFrame(data = train_label, columns = Ccolname)
num_of_rows = int(train_label_df.shape[0] * 1/10)
train_label_df.sample(frac = 1)

X_train,y_train = train_label_df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(cTraining.shape[1])]],
                                     train_label_df.loc[num_of_rows+1:,'3072']
X_val,y_val = train_label_df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(cTraining.shape[1])]],
                                 train_label_df.loc[:num_of_rows,'3072']

# SVM,training samples are 100,200,500,1000,2000,5000
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

arr = np.array([100, 200, 500, 1000, 2000, 5000])
arrC = np.array([5000])
C = np.array([1])

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
            
trainingAll(arr,C)
"""For 100,200,500,1000,2000,5000 samples, C = 1, the accuracies are 
0.1888,0.1816, 0.2012,0.2118,0.2314,0.2422
"""
# save
nplist = np.array(acclist)
np.savetxt("c-acc.csv", nplist, delimiter=",")

errorM = 1-nplist
# Plot 
fig = plt.figure()
plt.plot(arr, nplist, label="Error rate", color="black")
plt.title("Error Rate Curve With SVM")
plt.xlabel("Number Of Sample")
plt.ylabel("Error rate")
plt.show()
fig.savefig('error rate curve of C.png')

# build model by 5000 samples and predict test data 
num_of_rows = int(train_label_df.shape[0] * 1/5)
train_label_df.sample(frac = 1)
X_train,y_train = train_label_df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(cTraining.shape[1])]],
                                     train_label_df.loc[num_of_rows+1:,'32']
X_val,y_val = train_label_df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(cTraining.shape[1])]],
                                 train_label_df.loc[:num_of_rows,'32']

svm_clf=Pipeline((
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=1,loss='hinge'))
        ))
svm_clf.fit(X_train[:5000],y_train[:5000])
y_pred = svm_clf.predict(X_val)
print('svm classification result:')
print("accuracy_score:",accuracy_score(y_val,y_pred))
predOfTest = svm_clf.predict(cTest)

# save to csv
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv('submission_c.csv', index_label='Id')
results_to_csv(predOfTest)




### improve model
# grid search
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 
              'C':(0.001,0.01,0.1),
              'gamma': (1,2,3,'auto'),
              'decision_function_shape':('ovo','ovr'),
              'shrinking':(True,False)}
svm = SVC()
clf = GridSearchCV(svm, parameters)
clf.fit(X_train[:5000],y_train[:5000])

from sklearn.model_selection import cross_val_score
print("accuracy:"+str(np.average(cross_val_score(clf, 
                                                 X_train[:5000],
                                                 y_train[:5000],
                                                 scoring='accuracy'))))
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
clf.score(X_train[:5000], y_train[:5000]) 

# prediction
y_pred_2 = clf.predict(X_val)
print('svm classification result:')
print("accuracy_score:",accuracy_score(y_val,y_pred_2))
predOfTest_2 = clf.predict(cTest)
    
# save to csv
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv('submission_s_2.csv', index_label='Id')
results_to_csv(predOfTest_2)