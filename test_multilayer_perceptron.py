import numpy as np

from multilayer_perceptron import MultilayerPerceptron


def test_prediction_1_layer():
    weights = [
        np.array([[1., 0., 2., 1.],
                  [0., -1., 2., 0.],
                  [1., 2., 3., 4.]])
    ]
    biases = [np.array([-1., 0., 1.])]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: x, lambda x: 1.)
    )
    assert np.array_equal(
        perceptron.predict(np.array([4., 3., 2., 1.])),
        np.array([8., 1., 21.])
    )
    assert np.allclose(
        perceptron.predict(np.array([0.5, 1.5, -0.5, 0.25])),
        np.array([-1.25, -2.5, 4.])
    )
    assert np.array_equal(
        perceptron.predict(np.array([0., 0., 0., 0.])),
        np.array([-1., 0., 1.])
    )
    assert np.array_equal(
        perceptron.predict_all([
            np.array([1., 0., 0., 0.]),
            np.array([0., -1., 0., 1.]),
            np.array([1., 0., -1., 1.]),
        ]),
        [
            np.array([0., 0., 2.]),
            np.array([0., 1., 3.]),
            np.array([-1., -2., 3.]),
        ]
    )


def test_prediction_more_layers():
    weights = [
        np.array([[1., 2., 3.],
                  [4., 5., 6.]]),
        np.array([[-1., 1.],
                  [1., -1.]]),
        np.array([[2., 0.5]])
    ]
    biases = [np.array([-2., 2.]),
              np.array([-0.5, 4.]),
              np.array([1.])]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: max(x, 0.), lambda x: int(x > 0.))
    )
    assert np.allclose(
        perceptron.predict(np.array([1., 0., 1.])),
        np.array([20.])
    )
    assert np.allclose(
        perceptron.predict(np.array([1., -2., 11.])),
        np.array([68.])
    )
    assert np.array_equal(
        perceptron.predict_all([
            np.array([1., 0., 0.]),
            np.array([0., -1., 0.]),
            np.array([0., 0., 1.]),
        ]),
        [
            np.array([12.]),
            np.array([3.]),
            np.array([14.]),
        ]
    )


def test_weight_update_final_layer():
    weights = [
        np.array([[1., 0., 2., 1.],
                  [0., -1., 2., 0.],
                  [1., 2., 3., 4.]])
    ]
    biases = [np.array([-1., 0., 1.])]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: x, lambda x: 1.)
    )
    changes = perceptron.train(
        np.array([4., 3., 2., 1.]), np.array([1., 1., 1.])
    )
    perceptron.apply_changes([changes], 1.)

    assert np.array_equal(
        perceptron.weights[0],
        np.array([[-27., -21., -12., -6.],
                  [0., -1., 2., 0.],
                  [-79., -58., -37., -16.]])
    )
    assert np.array_equal(
        perceptron.biases[0], np.array([-8., 0., -19.])
    )

    weights = [
        np.array([[1., 0., 2., 1.],
                  [0., -1., 2., 0.],
                  [1., 2., 3., 4.]])
    ]
    biases = [np.array([-1., 0., 1.])]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: x, lambda x: 1.)
    )

    changes = perceptron.train(
        np.array([0.5, 1.5, -0.5, 0.25]), np.array([1., 1., 1.])
    )
    perceptron.apply_changes([changes], 1.)

    assert np.allclose(
        perceptron.weights[0],
        np.array([[2.125, 3.375, 0.875, 1.5625],
                  [1.75, 4.25, 0.25, 0.875],
                  [-0.5, -2.5, 4.5, 3.25]])
    )

    assert np.allclose(
        perceptron.biases[0], np.array([1.25, 3.5, -2.])
    )


def test_weight_update_more_layers():
    weights = [
        np.array([[1., 2., 3.],
                  [4., 5., 6.]]),
        np.array([[-1., 1.],
                  [1., -1.]]),
        np.array([[2., 0.5]])
    ]
    biases = [
        np.array([-2., 2.]),
        np.array([-0.5, 4.]),
        np.array([1.])
    ]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: max(x, 0.), lambda x: float(x >= 0.))
    )
    changes = perceptron.train(np.array([1., 0., 1.]), np.array([1.]))
    perceptron.apply_changes([changes], 1.)

    assert np.allclose(
        perceptron.weights[2],
        np.array([-178.5, 0.5])
    )
    assert np.allclose(
        perceptron.biases[2],
        np.array([-18])
    )
    assert np.allclose(
        perceptron.weights[1],
        np.array([
            [-77., -455.],
            [1., -1.]
        ])
    )
    assert np.allclose(
        perceptron.biases[1],
        np.array([-38.5, 4.])
    )
    assert np.allclose(
        perceptron.weights[0],
        np.array([
            [39., 2., 41.],
            [-34., 5., -32.]
        ])
    )
    assert np.allclose(
        perceptron.biases[0],
        np.array([36., -36.])
    )


def assert_lists_equal(
    list1: list[np.ndarray], list2: list[np.ndarray]
) -> bool:
    assert len(list1) == len(list2)
    for element1, element2 in zip(list1, list2):
        assert np.allclose(element1, element2)


def test_get_empty_changes():
    weights = [
        np.array([[1., 2., 3.],
                  [4., 5., 6.]]),
        np.array([[-1., 1.],
                  [1., -1.]]),
        np.array([[2., 0.5]])
    ]
    biases = [
        np.array([-2., 2.]),
        np.array([-0.5, 4.]),
        np.array([1.])
    ]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: x, lambda x: 1)
    )
    changes = perceptron.get_empty_changes()

    assert_lists_equal(changes[0], [
        np.array([[0., 0., 0.],
                  [0., 0., 0.]]),
        np.array([[0., 0.],
                  [0., 0.]]),
        np.array([[0., 0.]])
    ])
    assert_lists_equal(changes[1], [
        np.array([0., 0.]),
        np.array([0., 0.]),
        np.array([0.])
    ])


def test_apply_changes():
    weights = [
        np.array([[1., 2., 3.],
                  [4., 5., 6.]]),
        np.array([[-1., 1.],
                  [1., -1.]]),
        np.array([[2., 0.5]])
    ]
    biases = [
        np.array([-2., 2.]),
        np.array([-0.5, 4.]),
        np.array([1.])
    ]
    perceptron = MultilayerPerceptron(
        weights, biases, (lambda x: x, lambda x: 1)
    )
    changes = [
        ([np.array([[0.1, 0., 0.],
                    [0., 1., 0.]]),
          np.array([[0., 0.11],
                    [0., 0.]]),
          np.array([[0., 0.]])],
         [np.array([0., 0.]),
          np.array([0., 0.]),
          np.array([-0.1])]),
        ([np.array([[0.2, 0., 0.],
                    [0., 0., 0.]]),
          np.array([[0., 0.11],
                    [0., 0.]]),
          np.array([[0., 0.]])],
         [np.array([0., 0.]),
          np.array([0., 0.]),
          np.array([0.3])]),
        ([np.array([[0.3, 0., 0.],
                    [0., -1., 0.]]),
          np.array([[21., 0.11],
                    [0., 0.]]),
          np.array([[0., 0.]])],
         [np.array([0., 0.]),
          np.array([0., 0.]),
          np.array([0.1])]),
    ]
    perceptron.apply_changes(changes, 1)
    assert_lists_equal(perceptron.weights, [
        np.array([[0.8, 2., 3.],
                  [4., 5., 6.]]),
        np.array([[-8., 0.89],
                  [1., -1.]]),
        np.array([[2., 0.5]])
    ])
    assert_lists_equal(perceptron.biases, [
        np.array([-2., 2.]),
        np.array([-0.5, 4.]),
        np.array([0.9])
    ])

    changes = [
        ([np.array([[10., 0., 0.],
                    [0., 0., 0.]]),
          np.array([[0., 11.],
                    [0., 0.]]),
          np.array([[0., 0.]])],
         [np.array([0., 0.]),
          np.array([0., -20.]),
          np.array([0.])])
    ]
    perceptron.apply_changes(changes, 0.1)
    assert_lists_equal(perceptron.weights, [
        np.array([[-0.2, 2., 3.],
                  [4., 5., 6.]]),
        np.array([[-8., -0.21],
                  [1., -1.]]),
        np.array([[2., 0.5]])
    ])
    assert_lists_equal(perceptron.biases, [
        np.array([-2., 2.]),
        np.array([-0.5, 6.]),
        np.array([0.9])
    ])
