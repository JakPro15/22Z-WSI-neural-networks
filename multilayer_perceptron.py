from typing import Callable, Sequence
import numpy as np


class MultilayerPerceptron:
    def __init__(
        self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
        activation: tuple[Callable[[float], float]]
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
        self.biases = biases
        self.activation = np.vectorize(activation[0])
        self.activation_derivative = np.vectorize(activation[1])

    def predict(self, attributes: np.ndarray) -> np.ndarray:
        """
        Predicts the target vector for the given attributes vector.
        """
        data = attributes
        for layer, (weights, biases) in \
                enumerate(zip(self.weights, self.biases)):
            data = np.matmul(weights, data) + biases
            if layer < len(self.weights) - 1:
                data = self.activation(data)
        return data

    def propagate(self, attributes: np.ndarray,
                  learning_rate: float, target: np.ndarray) -> np.ndarray:
        """
        Predicts the target vector for the given attributes vector.
        """
        data = attributes
        sums = [data]
        inputs = [data]
        for layer, (weights, biases) in \
                enumerate(zip(self.weights, self.biases)):
            data = np.matmul(weights, data) + biases
            sums.append(data)
            if layer < len(self.weights) - 1:
                data = self.activation(data)
            inputs.append(data)

        for i in reversed(range(len(self.weights))):
            if np.array_equal(self.weights[i], self.weights[-1]):
                deltas = []
                for j in range(len(self.weights[i])):
                    delta = (inputs[-1][j] - target[j]) * \
                             self.activation_derivative(sums[-1][j])
                    deltas.append(delta)
                    self.biases[i][j] -= learning_rate * delta
                    for k in range(len(self.weights[i][j])):
                        change = -learning_rate * delta * inputs[-2][k]
                        self.weights[i][j][k] += change
            else:
                prev_deltas = deltas
                deltas = []
                for j in range(len(self.weights[i])):
                    delta_weight_sum = 0
                    for ll in range(len(self.weights[i + 1])):
                        delta_weight_sum += prev_deltas[ll] * \
                            self.weights[i + 1][ll][j]
                    delta = delta_weight_sum * \
                        self.activation_derivative(sums[i][j])
                    deltas.append(delta)
                    self.biases[i][j] -= learning_rate * delta
                    for k in range(len(self.weights[i][j])):
                        change = -learning_rate * delta * inputs[i][k]
                        self.weights[i][j][k] += change

        return data

    def predict_all(
        self, all_attributes: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Returns a list of vectors predicted for each element of the input list.
        """
        results = []
        for attributes in all_attributes:
            results.append(self.predict(attributes))
        return results
