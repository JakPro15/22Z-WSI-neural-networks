from typing import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt

from sgd import mse, stochastic_gradient_descent
from training_helpers import (ACTIVATIONS, get_normalizations,
                              normalize_sequence, triple_split)


def generate_dataset(
    function: Callable[[np.ndarray], np.ndarray | float], points_amount: int,
    input_size: int, parameters: tuple[float, float]
) -> tuple[list[np.ndarray], list[np.ndarray | float]]:
    """
    Generates a dataset approximately based on the given function with
    points_amount elements.
    Returns a tuple of the attributes and targets of the generated dataset.
    """
    attributes = []
    for _ in range(points_amount):
        value = np.random.normal(*parameters, input_size)
        attributes.append(value)
    targets = [function(X)  # * np.random.normal(1, 0.025)
               for X in attributes]
    return attributes, targets


def generate_line_plot(
    args: Sequence[float], values: Sequence[float], title: str,
    xlabel: str, ylabel: str,
) -> None:
    """
    Generates a 2-dimensional matplotlib graph for the given data.
    args and values lists should have the same length.
    """
    plt.figure()
    plt.plot(args, values, '-', markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def squares_sum_test():
    EPOCHS = 50
    BATCH_SIZE = 20
    X_all_original, Y_all_original = generate_dataset(
        lambda X: np.array([(X ** 2).sum()]), 1000, 3,
        (0, 10)
    )
    normalize_X, normalize_Y, denormalize_Y = get_normalizations(
        X_all_original, Y_all_original, True
    )
    X_all = normalize_sequence(X_all_original, normalize_X)
    Y_all = normalize_sequence(Y_all_original, normalize_Y)

    print(np.mean(X_all, 0))
    print(np.std(X_all, 0))
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = \
        triple_split(X_all, Y_all, 0.8, 0.1)

    best_perceptron, best_error, mses = stochastic_gradient_descent(
        X_train, Y_train, X_validation, Y_validation,
        ACTIVATIONS["relu"], [3, 3], 0.1, EPOCHS, BATCH_SIZE
    )
    print(f"Best achieved MSE: {best_error}")
    generate_line_plot(
        range(EPOCHS * (len(X_train) // BATCH_SIZE)), mses,
        "Mean square errors during training",
        "Batch number", "MSE"
    )

    print(f"{mse(best_perceptron.predict_all(X_all), Y_all)=}")
    print(f"{mse(best_perceptron.predict_all(X_test), Y_test)=}")
    best_perceptron.normalize = normalize_X
    best_perceptron.denormalize = denormalize_Y
    print(
        f"{mse(best_perceptron.predict_all(X_all_original), Y_all_original)=}"
    )


if __name__ == "__main__":
    squares_sum_test()
