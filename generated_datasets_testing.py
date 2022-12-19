from typing import Callable

import numpy as np

from perceptron_training import stochastic_gradient_descent
from plots import generate_line_plot
from training_helpers import (ACTIVATIONS, get_normalizations, mse,
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


def squares_sum_test() -> None:
    EPOCHS = 50
    BATCH_SIZE = 20
    X_all, Y_all = generate_dataset(
        lambda X: np.array([(X ** 2).sum()]), 1000, 3,
        (100, 10)
    )

    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = \
        triple_split(X_all, Y_all, 0.8, 0.1)

    normalize_X, normalize_Y, denormalize_Y = get_normalizations(
        X_train, Y_train
    )
    X_train_normal = normalize_sequence(X_train, normalize_X)
    Y_train_normal = normalize_sequence(Y_train, normalize_Y)
    X_validation_normal = normalize_sequence(X_validation, normalize_X)
    Y_validation_normal = normalize_sequence(Y_validation, normalize_Y)
    X_test_normal = normalize_sequence(X_test, normalize_X)
    Y_test_normal = normalize_sequence(Y_test, normalize_Y)

    best_perceptron, best_error, mses = stochastic_gradient_descent(
        X_train_normal, Y_train_normal, X_validation_normal,
        Y_validation_normal, ACTIVATIONS["relu"], [3, 3], 0.1, EPOCHS,
        BATCH_SIZE
    )
    print(f"Best achieved MSE: {best_error}")
    generate_line_plot(
        range(len(mses)), mses,
        "Mean square errors during training",
        "Batch number", "MSE"
    )

    train_normal_mse = mse(
        best_perceptron.predict_all(X_train_normal), Y_train_normal
    )
    valid_normal_mse = mse(
        best_perceptron.predict_all(X_validation_normal), Y_validation_normal
    )
    test_normal_mse = mse(
        best_perceptron.predict_all(X_test_normal), Y_test_normal
    )
    best_perceptron.normalize = normalize_X
    best_perceptron.denormalize = denormalize_Y
    test_mse = mse(
        best_perceptron.predict_all(X_test), Y_test
    )

    print(f"{train_normal_mse=}")
    print(f"{valid_normal_mse=}")
    print(f"{test_normal_mse=}")
    print(f"{test_mse=}")


def xor_test() -> None:
    def double_xor(attributes: np.ndarray) -> np.ndarray:
        integer_attributes = [int(attribute * 1e6) for attribute in attributes]
        return np.array([float(integer_attributes[0] ^ integer_attributes[1]),
                         float(integer_attributes[2] ^ integer_attributes[3])])

    EPOCHS = 100
    BATCH_SIZE = 20
    X_all, Y_all = generate_dataset(
        double_xor, 1000, 4, (100, 10)
    )

    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = \
        triple_split(X_all, Y_all, 0.8, 0.1)

    normalize_X, normalize_Y, denormalize_Y = get_normalizations(
        X_train, Y_train
    )
    X_train_normal = normalize_sequence(X_train, normalize_X)
    Y_train_normal = normalize_sequence(Y_train, normalize_Y)
    X_validation_normal = normalize_sequence(X_validation, normalize_X)
    Y_validation_normal = normalize_sequence(Y_validation, normalize_Y)
    X_test_normal = normalize_sequence(X_test, normalize_X)
    Y_test_normal = normalize_sequence(Y_test, normalize_Y)

    best_perceptron, best_error, mses = stochastic_gradient_descent(
        X_train_normal, Y_train_normal, X_validation_normal,
        Y_validation_normal, ACTIVATIONS["relu"], [3, 3, 3], 0.1, EPOCHS,
        BATCH_SIZE
    )
    print(f"Best achieved MSE: {best_error}")
    generate_line_plot(
        range(len(mses)), mses,
        "Mean square errors during training",
        "Batch number", "MSE"
    )

    train_normal_mse = mse(
        best_perceptron.predict_all(X_train_normal), Y_train_normal
    )
    valid_normal_mse = mse(
        best_perceptron.predict_all(X_validation_normal), Y_validation_normal
    )
    test_normal_mse = mse(
        best_perceptron.predict_all(X_test_normal), Y_test_normal
    )
    best_perceptron.normalize = normalize_X
    best_perceptron.denormalize = denormalize_Y
    test_mse = mse(
        best_perceptron.predict_all(X_test), Y_test
    )

    print(f"{train_normal_mse=}")
    print(f"{valid_normal_mse=}")
    print(f"{test_normal_mse=}")
    print(f"{test_mse=}")


if __name__ == "__main__":
    squares_sum_test()
    # xor_test()
