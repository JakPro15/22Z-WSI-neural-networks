from typing import Callable, Sequence
import numpy as np
from typing_extensions import Self


class MultilayerPerceptron:
    def __init__(
        self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
        activation: tuple[Callable[[float], float], Callable[[float], float]]
    ) -> None:
        """
        Creates a multilayer perceprtion with the given weights and the
        activation function.
        Each element of weights should be a matrix - a list of lists of
        weights for each neuron in the layer, given as a numpy ndarray.
        activation is the activation function applied on all neuron outputs
        except the last layer, given as a tuple of the function itself and its
        derivative.
        """
        self.weights = weights
        self.biases = biases
        self.activation = np.vectorize(activation[0])
        self.activation_derivative = np.vectorize(activation[1])

    @classmethod
    def initialize(
        cls, layer_widths: Sequence[int],
        activation: tuple[Callable[[float], float], Callable[[float], float]]
    ) -> Self:
        """
        Creates a multilayer perceptron with the given dimensions and
        activation function and initializes its weights and biases.
        layer_widths are all layer widths including the input and output.
        activation is the activation function applied on all neuron outputs
        except the last layer, given as a tuple of the function itself and its
        derivative.
        """
        weights = []
        biases = []
        weights_max = 1 / np.sqrt(layer_widths[0])
        for i, (input_width, output_width) in \
                enumerate(zip(layer_widths[:-1], layer_widths[1:])):
            weights.append(np.array(
                [[
                    np.random.uniform(-weights_max, weights_max)
                    # last layer gets weights of 0
                    if i != len(layer_widths) - 2 else 0.
                    for _ in range(input_width)
                ] for _ in range(output_width)]
            ))
            biases.append(np.array([
                np.random.uniform(-weights_max, weights_max)
                if i != len(layer_widths) - 2 else 0.
                for _ in range(output_width)
            ]))
        return cls(weights, biases, activation)

    def forward_propagate(
        self, attributes: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Calculates the result based on the given attributes (first layer
        inputs) and returns the sums and inputs at all layers, and the final
        outputs.
        """
        data = attributes
        sums = []
        inputs = []
        for layer, (weights, biases) in \
                enumerate(zip(self.weights, self.biases)):
            inputs.append(data)
            data = np.matmul(weights, data) + biases
            sums.append(data)
            if layer < len(self.weights) - 1:
                data = self.activation(data)
        return sums, inputs, data

    def back_propagate(
        self, sums: list[np.ndarray], inputs: list[np.ndarray],
        errors: np.ndarray, learning_rate: float
    ) -> None:
        """
        Based on values calculated in the forward propagation, executes
        backpropagation, updating weights and biases on the way.
        """
        deltas = []
        for i in reversed(range(len(self.weights))):
            prev_deltas = deltas
            deltas = []
            for j in range(len(self.weights[i])):
                if i == len(self.weights) - 1:
                    delta = errors[j] * self.activation_derivative(sums[i][j])
                else:
                    delta = (
                        prev_deltas * self.weights[i + 1][:, j]
                    ).sum() * self.activation_derivative(sums[i][j])
                deltas.append(delta)
                self.biases[i][j] -= learning_rate * delta
                self.weights[i][j] -= inputs[i] * learning_rate * delta

    def train(
        self, attributes: np.ndarray, targets: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        """
        Trains the perceptron based on the given piece of data.
        Updates weights and biases via backpropagation.
        """
        sums, inputs, outputs = self.forward_propagate(attributes)
        self.back_propagate(sums, inputs, outputs - targets, learning_rate)

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
