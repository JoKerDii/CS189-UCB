### learners.py


import math
import numpy as np
import random
import scipy.stats


class GDA:
    """Perform Gaussian discriminant analysis (both LDA and QDA)."""

    def evaluate(self, X, y, mode="lda"):
        """Predict and evaluate the accuracy using zero-one loss.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The true labels.

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            float: The zero-one loss of the learner.

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
        """
        pred = self.predict(X, mode=mode)
        return np.sum(pred == y) / y.shape[0]

    def fit(self, X, classes):
        """Train the GDA model (both LDA and QDA).

        Args:
            X (np.ndarray): The design matrix.
            classes (list): A list of the different classes.
        """
        self.classes = np.array(classes)
        num_examples = X.shape[0]
        num_features = X.shape[1] - 1
        self.fitted_params = []
        self.pooled_cov = np.zeros((num_features, num_features))
        for c in classes:
            data = X[X[:, -1] == c][:, :-1]
            mean = np.mean(data, axis=0)
            unscaled_cov = np.dot((data - mean).T, data - mean)
            self.pooled_cov = np.add(self.pooled_cov, unscaled_cov)
            scaled_cov = unscaled_cov / data.shape[0]
            prior = data.shape[0] / num_examples
            self.fitted_params.append((c, mean, scaled_cov, prior))
        self.pooled_cov = self.pooled_cov / num_examples

    def predict(self, X, mode="lda"):
        """Use the fitted model to make predictions.

        Args:
            X (np.ndarray): The feature matrix.

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            np.ndarray: The array of predictions.

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
        """
        if mode == "lda":
            pred = np.array([
                (scipy.stats.multivariate_normal.logpdf(
                    X, allow_singular=True, cov=self.pooled_cov, mean=mean)
                    + math.log(prior))
                for (c, mean, cov, prior) in self.fitted_params
            ])
            return self.classes[np.argmax(pred, axis=0)]
        elif mode == "qda":
            pred = np.array([
                (scipy.stats.multivariate_normal.logpdf(
                    X, allow_singular=True, cov=cov, mean=mean)
                    + math.log(prior))
                for (c, mean, cov, prior) in self.fitted_params
            ])
            return self.classes[np.argmax(pred, axis=0)]
        else:
            raise RuntimeError("Unknown mode!")


class LogisticRegression:
    """Perform logistic regression."""

    def __init__(self, eps=0.1, lambd=0.1):
        """Initializes the hyperparameters.

        Optional:
            eps (float): The gradient descent step size.
            lambd (float): The regularization hyperparameter.
        """
        self.cost = []
        self.eps = eps
        self.lambd = lambd

    def evaluate(self, X, y):
        """Evaluate the performance of the logistic regression model.

        Args:
            X (np.ndarray): The features.
            y (np.ndarray): The labels.

        Returns:
            float: The accuracy of the model.
        """
        pred = self.predict(X)
        return np.sum(pred == y) / y.shape[0]

    def fit(
            self, X, y, cost_interval=1, decr_learning_rate=False,
            iterations=100, learning_rate_fn=None, stochastic=False,
            verbose=False):
        """Train the logistic regression model.

        Args:
            X (np.ndarray): The feature matrix.
            classes (list): A list of the different classes.

        Optional:
            cost_interval (int): How frequently we should compute the cost.
            decr_learning_rate (boolean): Whether or not to decrease the
                learning rate over time. Defaults to false.
            iterations (int): The number of iterations in the training.
            learning_rate_fn (function): A function to apply to the learning
                rate.
            stochastic (boolean): Whether to use SGD or not. Defaults to batch
                GD.
            verbose (boolean): Whether to print out the cost or not.
        """
        self.w = np.zeros(X.shape[1])
        if learning_rate_fn is None and decr_learning_rate:
            learning_rate_fn = lambda eps, t: eps / (t + 1)
        elif not decr_learning_rate:
            learning_rate_fn = lambda eps, t: eps
        for t in range(iterations):
            index = random.randint(0, X.shape[0] - 1)
            X_t = X[index] if stochastic else X
            s_t = scipy.special.expit(np.dot(X_t, self.w))
            y_t = y[index] if stochastic else y
            if t % cost_interval == 0:
                s = scipy.special.expit(np.dot(X, self.w))
                train_cost = np.dot(y, np.log(s)) + np.dot(1 - y, np.log(1 - s))
                reg_cost = self.lambd * np.linalg.norm(self.w) ** 2
                cost = -train_cost + reg_cost
                self.cost.append(cost)
                if verbose:
                    print("Iteration {}, Cost {}".format(t, cost))
            train_grad = np.dot(X_t.T, s_t - y_t)
            reg_grad = 2 * self.lambd * self.w
            eps = learning_rate_fn(self.eps, t)
            self.w = self.w - eps * (train_grad + reg_grad)

    def predict(self, X):
        """Classifies the data.

        Args:
            X (np.ndarray): The design matrix.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return scipy.special.expit(np.dot(X, self.w)) > 0.5


class Whiten:
    """Standardizes data."""

    def fit(self, X):
        """Compute the means and standard deviations.

        Args:
            X (np.ndarray): The data to whiten.
        """
        self.means = np.mean(X, axis=0)
        self.norms = (np.linalg.norm(X - self.means, axis=0)
            / math.sqrt(X.shape[0]))

    def transform(self, X):
        """Whiten further data based on computed means and norms.

        Args:
            X (np.ndarray): The data to whiten.

        Returns:
            np.ndarray: The whitened data.
        """
        return (X - self.means) / self.norms

