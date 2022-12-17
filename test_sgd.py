from sgd import mse
from pytest import approx


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
