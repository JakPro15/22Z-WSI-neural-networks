from typing import Callable, Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from sgd import stochastic_gradient_descent
from matplotlib import pyplot as plt


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


def triple_split(
    X_all: Sequence[np.ndarray], y_all: Sequence[np.ndarray],
    train_size: float, valid_size: float
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray],
           Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Splits the given dataset into three parts (training, validation and
    testing), with the given proportion.
    """
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X_all, y_all, train_size=train_size
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining,
        train_size=round(valid_size / (1 - train_size), 3)
    )
    return X_train, y_train, X_validation, y_validation, X_test, y_test


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


def atest_xor():
    SIZE = 100
    X_all, y_all = generate_dataset(
        lambda X: np.array([(X ** 2).sum()]), 2000, 3,
        (0, 1)
    )
    X_train, y_train, X_validation, y_validation, X_test, y_test = \
        triple_split(X_all, y_all, 0.8, 0.1)

    perceptron, mses = stochastic_gradient_descent(
        X_train, y_train, X_validation, y_validation,
        (lambda x: max(x, 0.), lambda x: float(x >= 0)), [3, 2, 2], 0.01, SIZE
    )
    generate_line_plot(range(SIZE), [np.sqrt(mse) for mse in mses],
                       "a", "b", "mses")
    print(mses)


if __name__ == "__main__":
    atest_xor()
