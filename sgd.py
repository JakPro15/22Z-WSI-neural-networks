from typing import Sequence, Callable
from multilayer_perceptron import MultilayerPerceptron
import numpy as np
from random import shuffle
from math import inf


def mse(predicted: Sequence[np.ndarray], real: Sequence[np.ndarray]) -> float:
    """
    Calculates the mean square error of the given values predicted by the
    perceptron.
    """
    return np.mean(np.square(np.array(real) - predicted))


def stochastic_gradient_descent(
    X_train: Sequence[np.ndarray], Y_train: Sequence[np.ndarray],
    X_validation: Sequence[np.ndarray], Y_validation: Sequence[np.ndarray],
    activation: tuple[Callable[[float], float], Callable[[float], float]],
    layer_widths: Sequence[int], learning_rate: float, epochs: int,
    batch_size: int
) -> tuple[MultilayerPerceptron, list[float]]:
    """
    Trains a multilayer perceptron using the given training and validation
    datasets and stochastic gradient descent.
    activation should be a tuple of 2 functions: the activation function and
    its derivative.
    layer_widths specifies the widths of the hidden layers; input and output
    layers' widths are set based on the given X and Y widths.
    Returns the perceptron and mean square errors on validation set list over
    the trainings.
    """
    assert len(X_train) == len(Y_train)
    assert len(X_train) > 0
    assert len(X_validation) == len(Y_validation)
    assert len(X_validation) > 0
    assert learning_rate > 0

    all_widths = [len(X_train[0])] + layer_widths + [len(Y_train[0])]
    perceptron = MultilayerPerceptron.initialize(all_widths, activation)
    data_point_indexes = list(range(len(X_train)))
    mean_square_errors = []
    best_error = inf
    best_perceptron = perceptron
    for _ in range(epochs):
        data_point_indexes = list(range(len(X_train)))
        shuffle(data_point_indexes)
        i = 0
        while i + batch_size <= len(X_train):
            changes = []
            for j in range(batch_size):
                changes.append(perceptron.train(
                    X_train[data_point_indexes[i + j]],
                    Y_train[data_point_indexes[i + j]]
                ))
            perceptron.apply_changes(changes, learning_rate)
            error = mse(perceptron.predict_all(X_validation), Y_validation)
            mean_square_errors.append(error)
            if error <= best_error:
                best_error = error
                best_perceptron = perceptron.copy()
            i += batch_size
        print(f"epoch done {_}")
    return best_perceptron, best_error, mean_square_errors
