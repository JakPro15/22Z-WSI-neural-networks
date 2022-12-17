import numpy as np
from pytest import approx

from training_helpers import (get_normalizations, mse, normalize_sequence,
                              prepare_targets)


def test_get_normalizations():
    data = [np.array([1, 3, 3]),
            np.array([3, 2, -3]),
            np.array([2, -4, -3])]
    targets = [np.array([3, 2, 1]),
               np.array([-3, -2, -1]),
               np.array([0, 2, 0])]
    normalize_X, normalize_Y, denormalize_Y = \
        get_normalizations(data, targets, True)
    assert np.allclose(
        normalize_X(np.array([0, 0, 0])),
        [-2 / np.sqrt(2 / 3), -(1 / 3) / np.sqrt(86 / 9), 1 / np.sqrt(8)]
    )
    assert np.allclose(
        normalize_X(np.array([2, -10, 500])),
        [0 / np.sqrt(2 / 3), -(31 / 3) / np.sqrt(86 / 9), 501 / np.sqrt(8)]
    )

    assert np.allclose(
        normalize_Y(np.array([0, 0, 0])),
        [0, -(2 / 3) / np.sqrt(32 / 9), 0]
    )
    assert np.allclose(
        normalize_Y(np.array([2, -10, 500])),
        [2 / np.sqrt(6), -(32 / 3) / np.sqrt(32 / 9), 500 / np.sqrt(2 / 3)]
    )

    assert np.allclose(
        denormalize_Y(normalize_Y(np.array([0, 0, 0]))), np.array([0, 0, 0])
    )
    assert np.allclose(
        denormalize_Y(normalize_Y(np.array([2, -10, 500]))),
        np.array([2, -10, 500])
    )


def test_normalize_sequence():
    assert np.allclose(
        normalize_sequence(np.array([2, 3, -4, 5]), lambda X: X * 2),
        [4, 6, -8, 10]
    )
    assert np.allclose(
        normalize_sequence(np.array([2, 3, -4, 5]), lambda X: (X - 4) / 2),
        [-1, -0.5, -4, 0.5]
    )
    assert np.allclose(
        normalize_sequence(np.array([2, 3, -4, 5]), lambda X: X + 100),
        [102, 103, 96, 105]
    )


def test_prepare_targets():
    targets = ['a', 'a', 'b', 'c', 'b']
    prepared, classes = prepare_targets(targets)
    assert set(classes) == {'a', 'b', 'c'}
    assert np.array_equal(
        [
            np.array([
                1. if class_ == val else 0.
                for class_ in classes
            ])
            for val in targets
        ], prepared
    )


def test_mse():
    predicted = [
        [0.5, 0.8, 0.2, 0.9],
        [0.1, 0.2, 0, 1],
        [0.5, 0.85, 0.25, 0.1]
    ]
    real = [
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ]
    assert mse(predicted[0], real[0]) == approx(0.285)
    assert mse(predicted[1], real[1]) == approx(0.0125)
    assert mse(predicted[2], real[2]) == approx(0.08625)
    assert mse(predicted, real) == approx(0.1279, abs=0.0001)
