from typing import Callable, Sequence

import numpy as np

from multilayer_perceptron import MultilayerPerceptron
from perceptron_training import stochastic_gradient_descent
from plots import generate_line_plot
from training_helpers import (ACTIVATIONS, get_normalizations, mse,
                              normalize_sequence, triple_split)


def _generate_dataset(
    function: Callable[[np.ndarray], np.ndarray | float], points_amount: int,
    input_size: int, parameters: tuple[float, float]
) -> tuple[list[np.ndarray], list[np.ndarray | float]]:
    """
    Generates a dataset approximately based on the given function with
    points_amount elements.
    parameters are parameters of the normal distribution from which points are
    randomized. Noise is added to the targets.
    Returns a tuple of the attributes and targets of the generated dataset.
    """
    attributes = []
    for _ in range(points_amount):
        value = np.random.normal(*parameters, input_size)
        attributes.append(value)
    targets = [function(X) * np.random.normal(1, 0.025)
               for X in attributes]
    return attributes, targets


def _get_ready_sets(
    dataset_function: Callable[[np.ndarray], np.ndarray], dataset_size: int,
    X_length: int, dataset_parameters: tuple[int, int]
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray],
           Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray],
           Sequence[np.ndarray], Sequence[np.ndarray],
           Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], np.ndarray]]:
    """
    Returns the normalized training, validation and testing datasets and the
    original (not normalized) testing dataset generated from the given
    arguments. Also returns the normalize_X and denormalize_Y functions from
    the normalization used.
    """
    X_all, Y_all = _generate_dataset(
        dataset_function, dataset_size, X_length, dataset_parameters
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
    return (
        X_train_normal, Y_train_normal, X_validation_normal,
        Y_validation_normal, X_test_normal, Y_test_normal,
        X_test, Y_test, normalize_X, denormalize_Y
    )


def _do_testing(
    best_perceptron: MultilayerPerceptron,
    X_train: Sequence[np.ndarray], Y_train: Sequence[np.ndarray],
    X_validation: Sequence[np.ndarray], Y_validation: Sequence[np.ndarray],
    X_test: Sequence[np.ndarray], Y_test: Sequence[np.ndarray],
    X_test_original: Sequence[np.ndarray],
    Y_test_original: Sequence[np.ndarray],
    normalize_X: Callable[[np.ndarray], np.ndarray],
    denormalize_Y: Callable[[np.ndarray], np.ndarray]
):
    """
    Executes the testing of a trained perceptron, prints results to stdout.
    """
    train_mse = mse(
        best_perceptron.predict_all(X_train), Y_train
    )
    valid_mse = mse(
        best_perceptron.predict_all(X_validation), Y_validation
    )
    test_mse = mse(
        best_perceptron.predict_all(X_test), Y_test
    )
    best_perceptron.normalize = normalize_X
    best_perceptron.denormalize = denormalize_Y
    test_original_mse = mse(
        best_perceptron.predict_all(X_test_original), Y_test_original
    )

    print(f"{train_mse=}")
    print(f"{valid_mse=}")
    print(f"{test_mse=}")
    print(f"{test_original_mse=}")


def conduct_test(
    dataset_function: Callable[[np.ndarray], np.ndarray], dataset_size: int,
    X_length: int, dataset_parameters: tuple[int, int], epochs: int,
    batch_size: int, shape: list[int], activation: str, learning_rate: float
) -> None:
    """
    Generates a dataset based on the given function using generate_dataset.
    Trains a perceptron with the given parameters on it and prints the testing
    results and a graph of MSEs during the learning.
    """
    (
        X_train, Y_train, X_validation, Y_validation, X_test, Y_test,
        X_test_original, Y_test_original, normalize_X, denormalize_Y
    ) = _get_ready_sets(
        dataset_function, dataset_size, X_length, dataset_parameters
    )

    best_perceptron, best_error, mses = stochastic_gradient_descent(
        X_train, Y_train, X_validation,
        Y_validation, ACTIVATIONS[activation], shape, learning_rate,
        epochs, batch_size
    )
    print(f"Best achieved MSE: {best_error}")
    generate_line_plot(
        range(len(mses)), mses, "Mean square errors during training",
        "Batch number", "MSE"
    )

    _do_testing(
        best_perceptron, X_train, Y_train, X_validation, Y_validation, X_test,
        Y_test, X_test_original, Y_test_original, normalize_X, denormalize_Y
    )


def squares_sum_test() -> None:
    def squares_sum(attributes: np.ndarray) -> np.ndarray:
        return np.array([(attributes ** 2).sum()])

    conduct_test(squares_sum, 1000, 3, (0, 10), 50, 20, [3, 3], "relu", 0.1)


def xor_test() -> None:
    def double_xor(attributes: np.ndarray) -> np.ndarray:
        integer_attributes = [int(attribute * 1e6) for attribute in attributes]
        return np.array([float(integer_attributes[0] ^ integer_attributes[1]),
                         float(integer_attributes[2] ^ integer_attributes[3])])

    conduct_test(
        double_xor, 1000, 4, (100, 10), 100, 20, [3, 3, 3], "relu", 0.1
    )


if __name__ == "__main__":
    # squares_sum_test()
    xor_test()
