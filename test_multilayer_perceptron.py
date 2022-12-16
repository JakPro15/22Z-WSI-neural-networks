from multilayer_perceptron import MultilayerPerceptron
import numpy as np


def test_prediction_1_layer():
    weights = [
        np.array([[1, 0, 2, 1],
                  [0, -1, 2, 0],
                  [1, 2, 3, 4]])
    ]
    biases = [np.array([-1, 0, 1])]
    perceptron = MultilayerPerceptron(weights, biases,
                                      (lambda x: x,
                                       lambda x: 1))
    assert np.array_equal(
        perceptron.predict(np.array([4, 3, 2, 1])),
        np.array([8, 1, 21])
    )
    assert np.allclose(
        perceptron.predict(np.array([0.5, 1.5, -0.5, 0.25])),
        np.array([-1.25, -2.5, 4])
    )
    assert np.array_equal(
        perceptron.predict(np.array([0, 0, 0, 0])),
        np.array([-1, 0, 1])
    )
    assert np.array_equal(
        perceptron.predict_all([
            np.array([1, 0, 0, 0]),
            np.array([0, -1, 0, 1]),
            np.array([1, 0, -1, 1]),
        ]),
        [
            np.array([0, 0, 2]),
            np.array([0, 1, 3]),
            np.array([-1, -2, 3]),
        ]
    )


def test_prediction_more_layers():
    weights = [
        np.array([[1, 2, 3],
                  [4, 5, 6]]),
        np.array([[-1, 1],
                  [1, -1]]),
        np.array([[2, 0.5]])
    ]
    biases = [np.array([-2, 2]),
              np.array([-0.5, 4]),
              np.array([1])]
    perceptron = MultilayerPerceptron(weights, biases,
                                      (lambda x: max(x, 0),
                                       lambda x: int(x > 0)))
    assert np.allclose(
        perceptron.predict(np.array([1, 0, 1])),
        np.array([20])
    )
    assert np.allclose(
        perceptron.predict(np.array([1, -2, 11])),
        np.array([68])
    )
    assert np.array_equal(
        perceptron.predict_all([
            np.array([1, 0, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
        ]),
        [
            np.array([12]),
            np.array([3]),
            np.array([14]),
        ]
    )


def test_weight_update_final_layer():
    weights = [
        np.array([[1, 0, 2, 1],
                  [0, -1, 2, 0],
                  [1, 2, 3, 4]])
    ]
    biases = [np.array([-1, 0, 1])]
    perceptron = MultilayerPerceptron(weights, biases,
                                      (lambda x: x,
                                       lambda x: 1))
    perceptron.propagate(np.array([4, 3, 2, 1]), 1,
                         np.array([1, 1, 1]))

    assert np.array_equal(perceptron.weights[0],
           np.array([[-27, -21, -12, -6],
                     [0, -1, 2, 0],
                     [-79, -58, -37, -16]]))

    assert np.array_equal((perceptron.biases[0]),
           np.array([-8, 0, -19]))

    weights = [
        np.array([[1., 0., 2., 1.],
                  [0., -1., 2., 0.],
                  [1., 2., 3., 4.]])
    ]
    biases = [np.array([-1., 0., 1.])]
    perceptron = MultilayerPerceptron(weights, biases,
                                      (lambda x: x,
                                       lambda x: 1.))

    perceptron.propagate(np.array([0.5, 1.5, -0.5, 0.25]), 1.,
                         np.array([1., 1., 1.]))

    assert np.allclose(perceptron.weights[0],
           np.array([[2.125, 3.375, 0.875, 1.5625],
                     [1.75, 4.25, 0.25, 0.875],
                     [-0.5, -2.5, 4.5, 3.25]]))

    assert np.allclose(perceptron.biases[0],
           np.array([1.25, 3.5, -2.]))


def test_weight_update_more_layers():
    weights = [
        np.array([[1, 2, 3],
                  [4, 5, 6]]),
        np.array([[-1, 1],
                  [1, -1]]),
        np.array([[2, 0.5]])
    ]
    biases = [np.array([-2, 2]),
              np.array([-0.5, 4]),
              np.array([1])]
    perceptron = MultilayerPerceptron(weights, biases,
                                      (lambda x: max(x, 0),
                                       lambda x: int(x > 0)))

    perceptron.propagate(np.array([1, 0, 1]), 1, np.array([1]))
