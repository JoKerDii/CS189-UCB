import os

from scipy import io
from sklearn import svm, metrics
import numpy as np
import matplotlib.pyplot as plt


def predict_kaggle(name, training_data, training_labels, test_data, C=1.0):
    model = train(training_data, training_labels, C=C)
    test_labels = model.predict(test_data)

    filename = os.path.join(os.path.split(__file__)[0], '%s_solution.csv' % name)
    f = open(filename, 'w')
    f.write("Id,Category\n")
    for i, y in enumerate(test_labels):
        f.write(str(i + 1) + ',' + str(y) + '\n')
    f.close()


class funtionMayNeed(object):

    def __init__(self, dataset_name, val_size, seed=1):
        self.dataset_name = dataset_name
        self.val_size = val_size
        self.data = None
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.load_data()

    def split(self, data, labels, val_size):
        num_items = len(data)
        assert num_items == len(labels)
        assert val_size >= 0
        if val_size < 1.0:
            val_size = int(num_items * val_size)
        train_size = num_items - val_size
        idx = self.random.permutation(num_items)
        data_train = data[idx][:train_size]
        label_train = labels[idx][:train_size]
        data_val = data[idx][train_size:]
        label_val = labels[idx][train_size:]
        return data_train, data_val, label_train, label_val

    def load_data(self):
        this_dir = os.path.split(__file__)[0]
        data_dir = os.path.join(this_dir,
                                self.dataset_name,
                                "data",
                                "%s_data.mat" % self.dataset_name)
        self.data = io.loadmat(data_dir)
        print("loaded", self.dataset_name)
        fields = 'test_data', 'training_data', 'training_labels'
        for field in fields:
            print(field, self.data[field].shape)

    def train(self, X_train, Y_train, C=1.0):
        model = svm.SVC(kernel='linear', C=C)
        model.fit(X_train, Y_train)
        return model

    def train_all_sizes(self, input_train_sizes):
        training_data, training_labels = self.data['training_data'], self.data['training_labels']
        X_train, X_val, Y_train, Y_val = self.split(training_data,
                                                    training_labels,
                                                    self.val_size)
        print(X_train.shape, X_val.shape, self.val_size)
        train_sizes, train_accuracies, val_accuracies = [], [], []
        end_training = False
        for train_size in input_train_sizes:
            if train_size > X_train.shape[0]:
                train_size = X_train.shape[0]
                end_training = True
            model = self.train(X_train[:train_size], Y_train[:train_size])
            Y_train_prediction = model.predict(X_train[:train_size])
            train_accuracy = metrics.accuracy_score(Y_train[:train_size], Y_train_prediction)
            train_sizes.append(train_size)
            train_accuracies.append(train_accuracy)
            Y_val_prediction = model.predict(X_val)
            val_accuracy = metrics.accuracy_score(Y_val, Y_val_prediction)
            val_accuracies.append(val_accuracy)
            print("train_size", train_size,
                  "train accuracy", train_accuracy,
                  "val accuracy", val_accuracy)
            if end_training:
                break
        return train_sizes, train_accuracies, val_accuracies

    def predict_kaggle(self, train_size, C=1.0):
        num_examples = self.data['training_data'].shape[0]
        val_size = num_examples - train_size
        training_data, _, training_labels, _ = self.split(self.data['training_data'],
                                                          self.data['training_labels'],
                                                          val_size)
        test_data = self.data['test_data']
        model = self.train(training_data, training_labels, C=C)
        test_labels = model.predict(test_data)

        filename = os.path.join(os.path.split(__file__)[0], '%s_solution.csv' % self.dataset_name)
        f = open(filename, 'w')
        f.write("Id,Category\n")
        for i, y in enumerate(test_labels):
            f.write(str(i + 1) + ',' + str(y) + '\n')
        f.close()

    def hyperopt_experiment(self, train_size, C_range):
        # It doesn't matter if the splits match between this experiment and any other experiment.
        training_data, training_labels = self.data['training_data'], self.data['training_labels']
        X_train, X_val, Y_train, Y_val = self.split(training_data,
                                                    training_labels,
                                                    self.val_size)
        results = []
        for C in C_range:
            model = self.train(X_train[:train_size], Y_train[:train_size], C)
            Y_prediction = model.predict(X_val)
            val_accuracy = metrics.accuracy_score(Y_val, Y_prediction)
            results.append((C, val_accuracy))
            print("C", C, "accuracy", val_accuracy)
        return sorted(results, key=lambda x: x[1])[-1][0]

    def cv_experiment(self, k, C_range):
        training_data, training_labels = self.data['training_data'], self.data['training_labels']
        num_examples = len(training_data)
        idx = self.random.permutation(num_examples)
        x_parts = [None]*k
        y_parts = [None]*k
        part_size = num_examples // k
        for i in range(k):
            si = i * part_size
            if i == k-1:
                ei = num_examples
            else:
                ei = (i+1) * part_size
            x_parts[i] = training_data[idx][si:ei]
            y_parts[i] = training_labels[idx][si:ei]
        assert np.sum(list(map(lambda x: x.shape[0], x_parts))) == num_examples
        assert np.sum(list(map(lambda x: x.shape[0], y_parts))) == num_examples
        results = []
        for C in C_range:
            val_scores = []
            for i in range(0, k):
                X_val, Y_val = x_parts[i], y_parts[i]
                X_train = np.concatenate(x_parts[:i] + x_parts[i+1:], axis=0)
                assert X_train.shape[0] + X_val.shape[0] == num_examples
                Y_train = np.concatenate(y_parts[:i] + y_parts[i+1:], axis=0)
                assert Y_train.shape[0] + Y_val.shape[0] == num_examples
                model = self.train(X_train, Y_train, C)
                val_scores.append(metrics.accuracy_score(Y_val, model.predict(X_val)))
            results.append((C, np.mean(val_scores)))
            print("C", results[-1][0], "mean val accuracy: ", results[-1][1])
        return sorted(results, key=lambda x: x[1])[-1][0]

    def plot_result(self, train_sizes, train_accuracies, val_accuracies, title, filename):
        plt.figure()
        plt.title(title)
        plt.xlabel("Number of Examples")
        plt.ylabel("Accuracy")
        plt.plot(train_sizes, train_accuracies, label="Training Set")
        plt.plot(train_sizes, val_accuracies, label="Validation Set")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(__file__)[0], filename))

    def svm_experiment(self, input_train_sizes):
        train_sizes, train_accuracies, val_accuracies = self.train_all_sizes(input_train_sizes)
        self.plot_result(train_sizes,
                         train_accuracies,
                         val_accuracies,
                         "%s results" % self.dataset_name,
                         "%s_plot.png" % self.dataset_name)


def execute():
    # See HomeworkOneSolution.split for solution to problem 2.
    hw1_spam = funtionMayNeed("spam", val_size=0.2)
    hw1_cifar10 = funtionMayNeed("cifar10", val_size=5000)
    hw1_mnist = funtionMayNeed("mnist", val_size=10000)

    # Output solution to problem 3.
    hw1_spam.svm_experiment([100, 200, 500, 1000, 5000])
    hw1_cifar10.svm_experiment([100, 200, 500, 1000, 5000])
    hw1_mnist.svm_experiment([100, 200, 500, 1000, 5000, 10000])

    # Output solution to problem 4.
    mnist_C = hw1_mnist.hyperopt_experiment(train_size=10000,
                                            C_range=np.power(10.0, np.arange(-8, 0)))
    print("mnist_C", mnist_C)

    # Output solution to problem 5.
    spam_C = hw1_spam.cv_experiment(k=5,
                                    C_range=np.power(10.0, np.arange(-6, 2)))
    print("spam_C", spam_C)
    # Output solution to problem 6.
    hw1_spam.predict_kaggle(5000, C=spam_C)
    hw1_mnist.predict_kaggle(10000, C=mnist_C)
    hw1_cifar10.predict_kaggle(5000)

if __name__ == "__main__":
    execute()
