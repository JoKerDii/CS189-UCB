# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 09:54:58 2019

@author: dizhen
"""

"""
SPAM - CODE
1. data loading
2. data partitioning (set 20% validation set) without sklearn
3. SVM, training samples are 100,200,500,1000,2000,all
4. 5-fold cross validation and tune hyperparameter C
5. build model with all the data and predict test data

"""
## data loading
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
        
spam = io.loadmat("data/%s_data.mat" % "spam")
sTraining = np.array(spam['training_data'])
sLabel = np.array(spam['training_labels'])
sTest = np.array(spam['test_data'])

## data partitioning (set 20% validation set) without sklearn
train_label = np.concatenate((sTraining, sLabel), axis=1)
Scolname = np.array([str(i) for i in np.arange(sTraining.shape[1] + 1)])
train_label_df = pd.DataFrame(data = train_label, columns = Scolname)
num_of_rows = int(train_label_df.shape[0] * 1/5)
train_label_df.sample(frac = 1)

X_train,y_train = train_label_df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(sTraining.shape[1])]],
                                     train_label_df.loc[num_of_rows+1:,'32']
X_val,y_val = train_label_df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(sTraining.shape[1])]],
                                 train_label_df.loc[:num_of_rows,'32']

## SVM training samples are 100,200,500,1000,2000,all
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

arr = np.array([100,200,500, 1000, 2000, 4137])
C = np.array([1])
trainingAll(arr,C)
"""
For 100,200,500,1000,2000,4137 samples and C = 1, the accuracies are
0.7671497584541063,0.7855072463768116,0.8009661835748793,0.7874396135265701,
0.7990338164251207,0.8048309178743961
"""

# save
nplist = np.array(acclist)
np.savetxt("s-acc.csv", nplist, delimiter=",")
errorM = 1-nplist
fig = plt.figure()
plt.plot(arr, errorM, label="Error rate", color="black")
plt.title("Error Rate Curve With SVM")
plt.xlabel("Number Of Sample")
plt.ylabel("Error rate")
plt.show()
fig.savefig('error rate curve of S.png')

# 5-fold cross validation and tuning hyperparameter
def my_k_fold(df, k_fold, C):
    """
    A function for k fold cross validation and computing the mean
    of accuracy of each C value.
    """
    acc = []
    accmean = []
    test_size = 1/k_fold
    num_of_rows = int(df.shape[0] * test_size)
    for j in C:
        for i in np.arange(k_fold):
            # split data
            df.sample(frac = 1)
            X_train,y_train = df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(sTraining.shape[1])]],
                                     df.loc[num_of_rows+1:,'32']
            X_val,y_val = df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(sTraining.shape[1])]],
                                 df.loc[:num_of_rows,'32']

            # build model
            svm_clf = Pipeline((
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=j,loss='hinge'))
        ))
            svm_clf.fit(X_train, y_train)
            
            # use model to predict test data
            y_pred = svm_clf.predict(X_val)
            
            # accuracy
            acc.append(accuracy_score(y_val, y_pred))
        
        # store mean of accuracy    
        accmean.append(sum(acc)/len(acc))
        print("svm cross validation result, when C="+ str(j))
        print("mean of accuracy score:",sum(acc)/len(acc))
    return accmean

Cs = np.array([0.001, 0.01, 0.1, 1, 10,100])
my_k_fold(train_label_df,k_fold = 5,C = Cs)
"""
with C = 0.001,0.01,0.1,1,10,100, means of accuracy are 0.7843478260869565,
 0.7935265700483092, 0.7952979066022545, 0.7985024154589373, 0.7988019323671499,
 0.7969726247987119. So the best C is 0.01
"""

# build model with all the data and predict test data
num_of_rows = int(train_label_df.shape[0] * 1/5)
train_label_df.sample(frac = 1)
X_train,y_train = train_label_df.loc[num_of_rows+1:,
                                     [str(i) for i in np.arange(sTraining.shape[1])]],
                                     train_label_df.loc[num_of_rows+1:,'32']
X_val,y_val = train_label_df.loc[:num_of_rows,
                                 [str(i) for i in np.arange(sTraining.shape[1])]],
                                 train_label_df.loc[:num_of_rows,'32']

svm_clf=Pipeline((
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=0.01,loss='hinge'))
        ))

svm_clf.fit(X_train,y_train)
y_pred = svm_clf.predict(X_val)
print('svm classification result:')
print("accuracy_score:",accuracy_score(y_val,y_pred))
predOfTest=svm_clf.predict(sTest)

# save to csv
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv('submission_s.csv', index_label='Id')
results_to_csv(predOfTest)    


### improve model
# grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'kernel':('linear', 'rbf'), 
              'C':(0.001,0.01,0.1,1,10,100,1000),
              'gamma': (1,2,3,'auto'),
              'decision_function_shape':('ovo','ovr'),
              'shrinking':(True,False)}
svm = SVC()
clf = GridSearchCV(svm, parameters)
clf.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
print("accuracy:"+str(np.average(cross_val_score(clf,
                                                 X_train,
                                                 y_train,
                                                 scoring='accuracy'))))
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
clf.score(X_train, y_train) 

# prediction
y_pred_2 = clf.predict(X_val)
print('svm classification result:')
print("accuracy_score:",accuracy_score(y_val,y_pred_2))
predOfTest_2 = clf.predict(sTest)

## save to csv
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv('submission_s_2.csv', index_label='Id')
results_to_csv(predOfTest_2)






