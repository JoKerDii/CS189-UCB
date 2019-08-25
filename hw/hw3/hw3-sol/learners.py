## learners.py
## Author: Sinho Chewi

import math
import numpy as np
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

    def fit(self, X, y):
        """Train the GDA model (both LDA and QDA).

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The true labels.
        """
        self.classes = np.unique(y)
        num_examples = X.shape[0]
        num_features = X.shape[1] 
        self.fitted_params = []
        self.pooled_cov = np.zeros((num_features, num_features))
        for c in self.classes:
            class_indices = (y == c).flatten()
            data = X[class_indices]
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
            return self.classes[np.argmax(pred, axis=0)].reshape((-1, 1))
        elif mode == "qda":
            pred = np.array([
                (scipy.stats.multivariate_normal.logpdf(
                    X, allow_singular=True, cov=cov, mean=mean)
                    + math.log(prior))
                for (c, mean, cov, prior) in self.fitted_params
            ])
            return self.classes[np.argmax(pred, axis=0)].reshape((-1, 1))
        else:
            raise RuntimeError("Unknown mode!")
