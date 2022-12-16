from typing import Sequence, Callable
from multilayer_perceptron import MultilayerPerceptron
import numpy as np
from random import choice
from math import inf


def _mse(predicted: Sequence[np.ndarray], real: Sequence[np.ndarray]) -> float:
    """
    Calculates the mean square error of the given values predicted by the
    perceptron.
    """
    return np.mean(np.square(np.array(real) - predicted))


def stochastic_gradient_descent(
    X_train: Sequence[np.ndarray], y_train: Sequence[np.ndarray],
    X_validation: Sequence[np.ndarray], y_validation: Sequence[np.ndarray],
    activation: tuple[Callable[[float], float], Callable[[float], float]],
    layer_widths: Sequence[int], learning_rate: float, max_attempts: int
) -> tuple[MultilayerPerceptron, list[float]]:
    """
    Trains a multilayer perceptron using the given training and validation
    datasets and stochastic gradient descent.
    activation should be a tuple of 2 functions: the activation function and
    its derivative.
    layer_widths specifies the widths of the hidden layers; input and output
    layers' widths are set based on the given X and y widths.
    Returns the perceptron and mean square errors on validation set list over
    the trainings.
    """
    assert len(X_train) == len(y_train)
    assert len(X_train) > 0
    assert len(X_validation) == len(y_validation)
    assert len(X_validation) > 0
    assert learning_rate > 0

    all_widths = [len(X_train[0])] + layer_widths + [len(y_train[0])]
    perceptron = MultilayerPerceptron.initialize(all_widths, activation)
    data_point_indexes = list(range(len(X_train)))
    mean_square_errors = [inf]
    for _ in range(max_attempts):
        i = choice(data_point_indexes)
        perceptron.train(X_train[i], y_train[i], learning_rate)
        error = _mse(perceptron.predict_all(X_validation), y_validation)
        mean_square_errors.append(error)
        # if error >= mean_square_errors[-2]:
        #     break
    return perceptron, mean_square_errors[1:]
