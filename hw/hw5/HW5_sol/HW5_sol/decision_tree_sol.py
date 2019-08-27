
# coding: utf-8

# In[4]:


"""
To prepare the starter code, copy this file over to decision_tree_starter.py
and go through and handle all the inline TODO(cathywu)s.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number

class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None, max_features=None, is_random_forest=False):
        self.max_depth = max_depth
        self.features = feature_labels
        self.is_random_forest = is_random_forest # If True, the decision tree is a part of a random forest
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.max_features = max_features
        
    @staticmethod
    def entropy(y):
        if y.size == 0:
            return 0
        p0 = np.where(y < 0.5)[0].size / y.size
        if np.abs(p0) < 1e-10 or np.abs(1 - p0) < 1e-10:
            return 0
        return -p0 * np.log(p0) - (1 - p0) * np.log(1 - p0)

    @staticmethod
    def information_gain(X, y, thresh):
        base = DecisionTree.entropy(y)
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        entropy = p0 * DecisionTree.entropy(y0) + p1 * DecisionTree.entropy(y1)
        return base - entropy

    @staticmethod
    def gini_impurity(y):
        if y.size == 0:
            return 0
        p0 = np.where(y < 0.5)[0].size / y.size
        if np.abs(p0) < 1e-10 or np.abs(1 - p0) < 1e-10:
            return 0
        return 1.0 - p0**2 - (1 - p0)**2

    @staticmethod
    def gini_purification(X, y, thresh):
        base = DecisionTree.gini_impurity(y)
        y0 = y[np.where(X < thresh)[0]]
        p0 = y0.size / y.size
        y1 = y[np.where(X >= thresh)[0]]
        p1 = y1.size / y.size
        gini_impurity = p0 * DecisionTree.gini_impurity(
            y0) + p1 * DecisionTree.gini_impurity(y1)
        return base - gini_impurity

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1
    
    def segmenter(self, X, y):
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresholds = np.array([
                np.linspace(
                    np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([
                    self.information_gain(X[:, i], y, t) for t in thresholds[i, :]
                ])
            gains = np.nan_to_num(np.array(gains))
            split_idx, thresh_idx = np.unravel_index(
                np.argmax(gains), gains.shape)
            return split_idx, thresholds[split_idx, thresh_idx]
    
    def rf_segmenter(self, X, y):
        feature_idx = np.random.choice(X.shape[1], self.max_features, replace=False)
        feature_idx.sort()
        split_idx, threshold = self.segmenter(X[:, feature_idx], y)
        return feature_idx[split_idx], threshold
    
    def fit(self, X, y):
        if self.max_depth > 0:
            if self.is_random_forest:
                self.split_idx, self.thresh = self.rf_segmenter(X, y)
            else:
                self.split_idx, self.thresh = self.segmenter(X, y)
            X0, y0, X1, y1 = self.split(
                X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(
                X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self, level=0):
        if self.pred is None:
            ret = "\t"*level+"%s < %s" % (self.features[self.split_idx],self.thresh)+"\n"
            ret += self.right.__repr__(level+1)
            ret += self.left.__repr__(level+1)
        else:
            ret = "\t"*level+'leave node:'+str(self.pred)+"\n"
        return ret
    
class RandomForest():
    
    def __init__(self, params=None, feature_labels=None, n=200, m=1):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        max_depth = params['max_depth']
        self.decision_trees = [
            DecisionTree(feature_labels=feature_labels, max_depth=max_depth, max_features=m, is_random_forest=True)
            for i in range(self.n)
        ]
        
    def __repr__(self):
        ret = ""
        for dt in self.decision_trees:
            ret += dt.__repr__() + "\n"
        return ret
        
    def fit(self, X, y):
        for i in range(self.n):
            idx = np.random.choice(X.shape[0], X.shape[0])
            newX, newy = X[idx, :], y[idx]
            self.decision_trees[i].fit(newX, newy)
        return self

    def predict(self, X):
        yhat = [self.decision_trees[i].predict(X) for i in range(self.n)]
        return np.array(np.round(np.mean(yhat, axis=0)), dtype=np.bool)

def preprocess_titanic(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
    data = np.delete(data, onehot_cols, axis=1)
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=np.float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) *
                 (data[:, i] < -1 + eps)][:, i] = mode
    return data, onehot_features

if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    # dataset = 'census'
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        
        # TODO: preprocess titanic dataset
        # Notes: 
        # 1. Some data points are missing their labels
        # 2. Some features are not numerical but categorical
        # 3. Some values are missing for some features
        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n preprocessing the titanic dataset")
        X, onehot_features = preprocess_titanic(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess_titanic(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(np.delete(data[0, 1:], [1, 5, 7, 8])) + onehot_features
    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam-dataset/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]
    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    # Basic decision tree
    print("\n simplified decision tree")
    dt = DecisionTree(max_depth=5, feature_labels=features)
    dt.fit(X_train, y_train)
    dt_predictions = dt.predict(X_train)
    dt_predictions_val = dt.predict(X_val)
    print('The training accuracy is', np.sum(dt_predictions==y_train)/len(y_train))
    print('The validation accuracy is', np.sum(dt_predictions_val==y_val)/len(y_val))
    # Question 2.5.2
    # Visualize the tree and find a sample path for prediction on a sample from each class
    print("Tree structure\n", dt.__repr__())

    # Random forest
    print("\n random forest")
    # rf = RandomForest(params, n=N, m=2)
    rf = RandomForest(params, feature_labels=features, n=N, m=np.int(np.sqrt(X.shape[1])))
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_train)
    rf_predictions_val = rf.predict(X_val)
    print('The training accuracy is', np.sum(rf_predictions==y_train)/len(y_train))
    print('The validation accuracy is', np.sum(rf_predictions_val==y_val)/len(y_val)) 


# In[5]:


# Question 2.5.3  
# Find the most common splits at roots for random forest
roots = []
for dt in rf.decision_trees:
    roots.append("%s < %s" % (dt.features[dt.split_idx],dt.thresh))
counter = Counter(roots)
print(counter.most_common())


# In[6]:


# Question 2.5.4
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

val_accuracies = []
for depth in range(1, 41):
    dt = DecisionTree(max_depth=depth, feature_labels=features)
    dt.fit(X_train, y_train)
    dt_predictions_val = dt.predict(X_val)
    val_acc = np.sum(dt_predictions_val==y_val)/len(y_val)
    val_accuracies.append(val_acc)
plt.figure()
plt.plot(list(range(1, 41)), val_accuracies)
plt.show()
print('The depth with the best accuracy is ', np.argmax(val_accuracies)+1)


# In[ ]:


# Question 2.7
# Visualize a depth-3 decision tree for titanic dataset
d_tree = DecisionTree(max_depth=3, feature_labels=features)
d_tree.fit(X_train, y_train)
print("Tree structure\n", d_tree.__repr__())


# In[ ]:


# Titanic testing
testing_label_path = 'datasets/titanic/titanic_testing_labels.csv'
testing_label = genfromtxt(testing_label_path, delimiter=',', dtype=None)
testing_label = np.array(testing_label[1:,0], dtype=np.int)
testing_acc_dt = np.sum(dt.predict(Z)==testing_label)/len(testing_label)
testing_acc_rf = np.sum(rf.predict(Z)==testing_label)/len(testing_label)
print('testing accuracy for decision tree:', testing_acc_dt)
print('testing accuracy for random forest:', testing_acc_rf)


# In[ ]:


# Spam testing
testing_label_path = 'datasets/spam-dataset/spam_test_labels.txt'
testing_label = genfromtxt(testing_label_path, delimiter=',', dtype=None)
testing_label = np.array(testing_label[1:,1], dtype=np.int)
testing_acc_dt = np.sum(dt.predict(Z)==testing_label)/len(testing_label)
testing_acc_rf = np.sum(rf.predict(Z)==testing_label)/len(testing_label)
print('testing accuracy for decision tree:', testing_acc_dt)
print('testing accuracy for random forest:', testing_acc_rf)

