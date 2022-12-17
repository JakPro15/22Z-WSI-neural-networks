from typing import Callable, Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from sgd import stochastic_gradient_descent, mse
from matplotlib import pyplot as plt


ACTIVATIONS = {
    "relu": (lambda x: max(x, 0.),
             lambda x: float(x >= 0)),
    "tanh": (lambda x: np.tanh(x),
             lambda x: (1 / np.cosh(x)) ** 2),
    "logistic": (lambda x: np.exp(x) / (1 + np.exp(x)),
                 lambda x: np.exp(x) / ((1 + np.exp(x)) ** 2))
}


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


def normalize_data(
    attributes: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray],
           Callable[[np.ndarray], np.ndarray]]:
    """
    Makes the given data have the average of 0 and the standard deviation of 1
    in each column.
    Returns the altered dataset and a function to denormalize the results.
    """
    attributes_means = np.mean(attributes, 0)
    attributes_deviations = np.std(attributes, 0)
    attributes -= attributes_means
    attributes /= attributes_deviations
    target_means = np.mean(targets, 0)
    target_deviations = np.std(targets, 0)
    targets -= target_means
    targets /= target_deviations

    def normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalizes the given attributes so they can be run through the
        perceptron.
        """
        return (data - attributes_means) / attributes_deviations

    def denormalize(results: np.ndarray) -> np.ndarray:
        """
        Denormalizes the given results back to the original scale.
        """
        return results * target_deviations + target_means

    return attributes, targets, normalize, denormalize


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


def squares_sum_test():
    EPOCHS = 100
    BATCH_SIZE = 20
    X_all_original, y_all_original = generate_dataset(
        lambda X: np.array([(X ** 2).sum()]), 1000, 3,
        (0, 10)
    )
    X_all, y_all, normalize, denormalize = normalize_data(X_all_original, y_all_original)
    print(np.mean(X_all, 0))
    print(np.std(X_all, 0))
    X_train, y_train, X_validation, y_validation, X_test, y_test = \
        triple_split(X_all, y_all, 0.8, 0.1)

    best_perceptron, best_error, mses = stochastic_gradient_descent(
        X_train, y_train, X_validation, y_validation,
        ACTIVATIONS["logistic"], [3, 3], 1, EPOCHS, BATCH_SIZE
    )
    print(f"Best achieved MSE: {best_error}")
    generate_line_plot(
        range(EPOCHS * (len(X_train) // BATCH_SIZE)), mses,
        "Mean square errors during training",
        "Batch number", "MSE"
    )

    assert np.allclose(X_all, [normalize(X) for X in X_all_original])
    assert np.allclose(y_all_original, [denormalize(y) for y in y_all])
    print(f"{mse(best_perceptron.predict_all(X_all), y_all)=}")
    print(f"{mse(best_perceptron.predict_all(X_test), y_test)=}")
    best_perceptron.normalize = normalize
    best_perceptron.denormalize = denormalize
    print(f"{mse(best_perceptron.predict_all(X_all_original), y_all_original)=}")


if __name__ == "__main__":
    squares_sum_test()
