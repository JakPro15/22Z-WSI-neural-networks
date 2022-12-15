from typing import Any, Callable, Sequence
import numpy as np


class MultilayerPerceptron:
    def __init__(
        self, weights: Sequence[np.ndarray],
        activation: Callable[[float], float]
    ) -> None:
        """
        Creates a multilayer perceprtion with the given weights and the
        activation function.
        Each element of weights should be a matrix - a list of lists of
        weights for each neuron in the layer, given as a numpy ndarray.
        activation is the activation function applied on all neuron outputs
        except the last layer.
        """
        self.weights = weights
        self.activation = np.vectorize(activation)

    def predict(self, attributes: np.ndarray) -> np.ndarray:
        """
        Predicts the target vector for the given attributes vector.
        """
        data = attributes
        for i, layer in enumerate(self.weights):
            data = np.matmul(data, layer)
            if i < len(self.weights) - 1:
                data = self.activation(data)
        return data

    def predict_all(self, all_attributes: list[tuple[float]]) -> list[Any]:
        """
        Returns a list of vectors predicted for each element of the input list.
        """
        results = []
        for attributes in all_attributes:
            results.append(self.predict(attributes))
        return results
