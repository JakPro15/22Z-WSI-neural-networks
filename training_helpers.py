from collections import Counter
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
    attributes: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> tuple[Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], np.ndarray]]:
    """
    Returns three functions:
        one normalizing the attributes of a point of data
        one normalizing the targets of a point of data
        one denormalizing the targets of a point of data
    calculated for the given dataset.
    """
    attributes_means = np.mean(attributes, 0)
    attributes_deviations = np.std(attributes, 0)
    target_means = np.mean(targets, 0)
    target_deviations = np.std(targets, 0)
    return (
        lambda X: (X - attributes_means) / attributes_deviations,
        lambda Y: (Y - target_means) / target_deviations,
        lambda Y: Y * target_deviations + target_means
    )


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
    Y: Sequence[ClassType], classes: list[ClassType]
) -> Sequence[np.ndarray]:
    """
    Converts a list of classification targets (classes) into a list of arrays
    ready to be used for training of a perceptron.
    classes is the list of possible classes.
    """
    classes_dict = {
        element: np.array([
            0. if j != i else 1.
            for j in range(len(classes))
        ])
        for i, element in enumerate(classes)
    }
    return [classes_dict[element] for element in Y]


def mse(predicted: Sequence[np.ndarray], real: Sequence[np.ndarray]) -> float:
    """
    Calculates the mean square error of the given values predicted by the
    perceptron.
    """
    return np.mean(np.square(np.array(real) - predicted))


def get_confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int]
) -> list[list[int]]:
    """
    Returns a 3x3 confusion matrix for the given real and predicted targets,
    where possible target values (classes) are 0, 1 or 2.
    """
    frequencies = Counter(zip(targets, predictions))
    return [[frequencies[(0, 0)], frequencies[(0, 1)], frequencies[(0, 2)]],
            [frequencies[(1, 0)], frequencies[(1, 1)], frequencies[(1, 2)]],
            [frequencies[(2, 0)], frequencies[(2, 1)], frequencies[(2, 2)]]]


def get_accuracy(confusion_matrix: Sequence[Sequence[int]]) -> float:
    """
    Returns accuracy metric calculated from the given 3x3 confusion matrix.
    """
    return (confusion_matrix[0][0] + confusion_matrix[1][1] +
            confusion_matrix[2][2]) / np.sum(confusion_matrix)
