### utils.py


import csv
import itertools
import numpy as np


### TRAINING/VALIDATION SPLIT


def train_val_split(data, num_val=None, val_split=0.8):
    """Split the data into a training set and a validation set. This is a
    destructive operation.

    Args:
        data (np.ndarray): An iterable collection of data.
    
    Optional:
        num_val (int): The number of examples to place in the validation set.
        val_split (float): The fraction of the data which should be training
            data.
    
    Returns:
        tuple: A pair containing (validation set, training set).
    """
    if num_val is None:
        num_val = int(len(data) * val_split)
    np.random.shuffle(data)
    return data[:num_val], data[num_val:]


### HYPERPARAMETER TUNING


def hyperparameter_grid(hyperparameters):
    """Returns all of the points in a grid of hyperparameters.

    Args:
        hyperparameters (dict): The keys are the names of the hyperparameters,
            and the values are lists of possible values for the hyperparameter.

    Returns:
        list: A list of dictionaries, where each dictionary contains options for
            the learning algorithm.
    """
    hyp_list = hyperparameters.keys()
    grid = itertools.product(*[hyperparameters[hyp] for hyp in hyp_list])
    hyperparameter_list = []
    for tup in grid:
        options = {}
        for (hyperparameter, value) in zip(hyp_list, tup):
            options[hyperparameter] = value
        hyperparameter_list.append(options)
    return hyperparameter_list


def random_hyperparameter_search(learner, X_train, y_train, X_val, y_val, 
        hyperparameter_list):
    """Given a list of hyperparameters to try, returns the validation accuracies
    for each choice of hyperparameters.

    Args:
        learner: The learning algorithm.
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        X_val (np.ndarray): The validation data.
        y_val (np.ndarray): The validation labels.
        hyperparameter_list (list): A list of dictionaries, where each
            dictionary contains options for the learning algorithm.

    Returns:
        list: A list of (options, accuracy) tuples.
    """
    results = []
    for options in hyperparameter_list:
        model = learner(**options)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        results.append((options, acc))
    return results


### CSV WRITER


def write_kaggle_csv(file_name, predictions):
    """Writes the results in a CSV file.

    Args:
        file_name (str): The name of the file to store the results.
        predictions (np.ndarray): An array of predictions.
    """
    example_id = 0
    with open(file_name, "w") as csv_file:
        kaggle_writer = csv.writer(csv_file, delimiter=",")
        kaggle_writer.writerow(["Id", "Category"])
        for prediction in predictions:
            kaggle_writer.writerow([example_id, int(prediction)])
            example_id += 1

