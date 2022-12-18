from typing import Callable, Hashable, Sequence, TypeVar

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


def get_normalizations(
    attributes: Sequence[np.ndarray], targets: Sequence[np.ndarray],
    denormalize: bool = False
) -> tuple[Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], np.ndarray]] | \
     tuple[Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], np.ndarray]]:
    """
    Returns functions that normalize the attributes and targets from the given
    dataset.
    If denormalize is given and True, also returns the target denormalization
    function.
    """
    attributes_means = np.mean(attributes, 0)
    attributes_deviations = np.std(attributes, 0)
    target_means = np.mean(targets, 0)
    target_deviations = np.std(targets, 0)
    normalizations = (
        lambda X: (X - attributes_means) / attributes_deviations,
        lambda Y: (Y - target_means) / target_deviations
    )
    if denormalize:
        return (
            *normalizations,
            lambda Y: Y * target_deviations + target_means
        )
    else:
        return normalizations


def normalize_sequence(
    sequence: Sequence[np.ndarray],
    normalize: Callable[[np.ndarray], np.ndarray]
) -> Sequence[np.ndarray]:
    """
    Returns the normalized version of the given sequence using the given
    normalization function.
    """
    return [normalize(element) for element in sequence]


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


def mse(predicted: Sequence[np.ndarray], real: Sequence[np.ndarray]) -> float:
    """
    Calculates the mean square error of the given values predicted by the
    perceptron.
    """
    return np.mean(np.square(np.array(real) - predicted))
