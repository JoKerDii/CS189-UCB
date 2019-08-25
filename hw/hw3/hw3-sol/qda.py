import numpy as np
import scipy.cluster
import scipy.io
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt

import learners

def train_val_split(data, labels, val_size):
    num_items = len(data)
    assert num_items == len(labels)
    assert val_size >= 0
    if val_size < 1.0:
        val_size = int(num_items * val_size)
    train_size = num_items - val_size
    idx = np.random.permutation(num_items)
    data_train = data[idx][:train_size]
    label_train = labels[idx][:train_size]
    data_val = data[idx][train_size:]
    label_val = labels[idx][train_size:]
    return data_train, data_val, label_train, label_val

mnist = scipy.io.loadmat("./mnist-data/mnist_data.mat")

train_data, train_labels = mnist['training_data'], mnist['training_labels']
labels = np.unique(train_labels)

# Part a
mnist_fitted = {}
for label in labels:
    class_indices = (train_labels == label).flatten()
    data = train_data[class_indices]
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    mnist_fitted[label] = (mean, cov)

# Part b
class_indices = (train_labels == labels[0]).flatten()
data = train_data[class_indices]
ncov = np.corrcoef(data, rowvar=False)
ncov[np.isnan(ncov)] = 0
ncov = np.abs(ncov)
plt.imshow(ncov, cmap=matplotlib.cm.Greys_r)
plt.colorbar()
plt.show()


# Part c (Gaussian/linear discriminant analysis)
train_data, val_data, train_labels, val_labels = train_val_split(train_data, train_labels, val_size=10000)
val_labels.flatten()
num_training = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
lda_errors = []
qda_errors = []

for num in num_training:
    gda = learners.GDA()
    gda.fit(train_data[:num], train_labels[:num])
    err = 1 - gda.evaluate(val_data, val_labels)
    lda_errors.append(err)
    err = 1 - gda.evaluate(val_data, val_labels, mode="qda")
    qda_errors.append(err)

# Part c.1
plt.figure(figsize=(8, 8))
plt.plot(num_training, lda_errors)
plt.xlabel("Number of Training Examples")
plt.ylabel("Error Rate")
plt.title("LDA Classification (Digits)")
plt.ylim((0, 1))
plt.show()
print(lda_errors)

#[0.34, 0.346, 0.62, 0.36, 0.21, 0.16, 0.14, 0.13, 0.13]

# Part c.2

plt.figure(figsize=(8, 8))
plt.plot(num_training, qda_errors)
plt.xlabel("Number of Training Examples")
plt.ylabel("Error Rate")
plt.title("QDA Classification (Digits)")
plt.ylim((0, 1))
plt.show()
print(qda_errors)

#[0.90, 0.90, 0.78, 0.51, 0.40, 0.26, 0.17, 0.15, 0.15]
