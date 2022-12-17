from typing import Sequence, Callable, TypeVar, Hashable
import numpy as np
from sklearn.model_selection import train_test_split


ACTIVATIONS = {
    "relu": (lambda x: max(x, 0.),
             lambda x: float(x >= 0)),
    "tanh": (lambda x: np.tanh(x),
             lambda x: (1 / np.cosh(x)) ** 2),
    "logistic": (lambda x: np.exp(x) / (1 + np.exp(x)),
                 lambda x: np.exp(x) / ((1 + np.exp(x)) ** 2))
}


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
    X_all: Sequence[np.ndarray], Y_all: Sequence[np.ndarray],
    train_size: float, valid_size: float
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray],
           Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Splits the given dataset into three parts (training, validation and
    testing), with the given proportion.
    """
    X_train, X_remaining, Y_train, Y_remaining = train_test_split(
        X_all, Y_all, train_size=train_size
    )
    X_validation, X_test, Y_validation, Y_test = train_test_split(
        X_remaining, Y_remaining,
        train_size=round(valid_size / (1 - train_size), 3)
    )
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


ClassType = TypeVar("ClassType", bound=Hashable)


def prepare_targets(
    Y: Sequence[ClassType]
) -> tuple[Sequence[np.ndarray], list[ClassType]]:
    """
    Converts a list of classification targets (classes) into a list of arrays
    ready to be used for training of a perceptron.
    Also returns a list of original classes.
    """
    classes = list(set(Y))
    classes_dict = {
        element: np.array([
            0. if j != i else 1.
            for j in range(len(classes))
        ])
        for i, element in enumerate(classes)
    }
    return [classes_dict[element] for element in Y], classes
