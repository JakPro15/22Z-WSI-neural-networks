import warnings
from math import isnan
from random import shuffle
from typing import Callable, Sequence

import numpy as np

from multilayer_perceptron import MultilayerPerceptron
from training_helpers import mse

# ignore overflow warnings - NaNs resulting from them are handled
warnings.filterwarnings("ignore")


def _do_batch(
    X_train: Sequence[np.ndarray], Y_train: Sequence[np.ndarray],
    X_validation: Sequence[np.ndarray], Y_validation: Sequence[np.ndarray],
    perceptron: MultilayerPerceptron, learning_rate: float, batch_size: int,
    data_point_indexes: Sequence[int], batch_index: int
) -> float:
    """
    Executes a single batch for stochastic gradient descent, updating the
    perceptron. Returns the MSE on the validation set after the batch.
    Raises OverflowError if the error turns out to be NaN.
    """
    changes = []
    for element_index in range(batch_size):
        changes.append(perceptron.train(
            X_train[data_point_indexes[batch_index + element_index]],
            Y_train[data_point_indexes[batch_index + element_index]]
        ))
    perceptron.apply_changes(changes, learning_rate)
    error = mse(perceptron.predict_all(X_validation), Y_validation)
    return error


def _do_epoch(
    X_train: Sequence[np.ndarray], Y_train: Sequence[np.ndarray],
    X_validation: Sequence[np.ndarray], Y_validation: Sequence[np.ndarray],
    perceptron: MultilayerPerceptron,
    learning_rate: float, batch_size: int, mean_square_errors: list[float],
    best_error: float, best_perceptron: MultilayerPerceptron
) -> tuple[float, MultilayerPerceptron, bool]:
    """
    Executes a single epoch for stochastic gradient descent.
    Raises OverflowError if values get too large to handle.
    Returns the new best error and perceptron.
    Third return indicates if the SGD function should end (NaN encountered).
    """
    data_point_indexes = list(range(len(X_train)))
    shuffle(data_point_indexes)
    batch_index = 0
    while batch_index + batch_size <= len(X_train):
        error = _do_batch(
            X_train, Y_train, X_validation, Y_validation, perceptron,
            learning_rate, batch_size, data_point_indexes, batch_index
        )
        if isnan(error):
            return best_error, best_perceptron, True
        mean_square_errors.append(error)
        if error <= best_error:
            best_error = error
            best_perceptron = perceptron.copy()
        batch_index += batch_size
    return best_error, best_perceptron, False


def stochastic_gradient_descent(
    X_train: Sequence[np.ndarray], Y_train: Sequence[np.ndarray],
    X_validation: Sequence[np.ndarray], Y_validation: Sequence[np.ndarray],
    activation: tuple[Callable[[float], float], Callable[[float], float]],
    layer_widths: Sequence[int], learning_rate: float, epochs: int,
    batch_size: int
) -> tuple[MultilayerPerceptron, float, list[float]]:
    """
    Trains a multilayer perceptron using the given training and validation
    datasets and stochastic gradient descent.
    The given training and validation sets should be normalized.
    activation should be a tuple of 2 functions: the activation function and
    its derivative.
    layer_widths specifies the widths of the hidden layers; input and output
    layers' widths are set based on the given X and Y widths.
    Returns the perceptron and mean square errors on validation set list over
    the trainings.
    Batch size equal to the length of the training set makes the algorithm
    equivalent to simple gradient descent.
    """
    assert len(X_train) == len(Y_train)
    assert len(X_train) > 0
    assert len(X_validation) == len(Y_validation)
    assert len(X_validation) > 0
    assert learning_rate > 0
    assert 0 <= batch_size <= len(X_train)

    all_widths = [len(X_train[0])] + layer_widths + [len(Y_train[0])]
    perceptron = MultilayerPerceptron.initialize(all_widths, activation)
    best_error = mse(perceptron.predict_all(X_validation), Y_validation)
    best_perceptron = perceptron.copy()
    mean_square_errors = [best_error]
    for _ in range(epochs):
        best_error, best_perceptron, end = _do_epoch(
            X_train, Y_train, X_validation, Y_validation, perceptron,
            learning_rate, batch_size, mean_square_errors, best_error,
            best_perceptron
        )
        if end:
            break
    return best_perceptron, best_error, mean_square_errors
