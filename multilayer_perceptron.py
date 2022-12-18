from copy import deepcopy
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
        self.normalize = None
        self.denormalize = None

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
            weights.append(np.array([
                [
                    np.random.uniform(-weights_max, weights_max)
                    # output layer gets weights of 0
                    if i != len(layer_widths) - 2 else 0.
                    for _ in range(input_width)
                ] for _ in range(output_width)
            ]))
            biases.append(np.array([
                np.random.uniform(-weights_max, weights_max)
                # output layer gets biases of 0
                if i != len(layer_widths) - 2 else 0.
                for _ in range(output_width)
            ]))
        return cls(weights, biases, activation)

    def copy(self) -> Self:
        """
        Returns a copy of the perceptron.
        """
        return type(self)(deepcopy(self.weights), deepcopy(self.biases),
                          (self.activation, self.activation_derivative))

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

    def get_empty_changes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns empty copies of the weights and biases of the perceptron,
        to which changes can be saved by basck_propagate.
        """
        weights_changes = []
        biases_changes = []
        for neuron_weights in self.weights:
            weights_changes.append(np.zeros(neuron_weights.shape, dtype=float))
            biases_changes.append(
                np.zeros(neuron_weights.shape[0], dtype=float)
            )
        return weights_changes, biases_changes

    def back_propagate(
        self, sums: list[np.ndarray], inputs: list[np.ndarray],
        errors: np.ndarray
    ) -> None:
        """
        Based on values calculated in the forward propagation, executes
        backpropagation. Returns the changes in weights and biases calculated.
        """
        weights_changes, biases_changes = self.get_empty_changes()
        deltas = []
        for layer_index in reversed(range(len(self.weights))):
            prev_deltas = deltas
            deltas = []
            for neuron_index in range(len(self.weights[layer_index])):
                if layer_index == len(self.weights) - 1:
                    delta = errors[neuron_index]
                else:
                    delta = (
                        prev_deltas *
                        self.weights[layer_index + 1][:, neuron_index]
                    ).sum() * self.activation_derivative(
                        sums[layer_index][neuron_index]
                    )
                deltas.append(delta)
                biases_changes[layer_index][neuron_index] = delta
                weights_changes[layer_index][neuron_index] = \
                    inputs[layer_index] * delta
        return weights_changes, biases_changes

    def train(
        self, attributes: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Trains the perceptron based on the given piece of data.
        Calculates changes in weights and biases via backpropagation and
        returns them.
        """
        sums, inputs, outputs = self.forward_propagate(attributes)
        weights_changes, biases_changes = \
            self.back_propagate(sums, inputs, outputs - targets)
        return weights_changes, biases_changes

    def apply_changes(
        self, changes: list[tuple[np.ndarray, np.ndarray]],
        learning_rate: float
    ) -> None:
        """
        Applies the average of the given changes to the weights and biases of
        the perceptron.
        """
        if len(changes) > 1:
            total_weights_changes, total_biases_changes = \
                self.get_empty_changes()
            all_weights_changes, all_biases_changes = zip(*changes)
            for weights_changes, biases_changes in \
                    zip(all_weights_changes, all_biases_changes):
                for i in range(len(weights_changes)):
                    total_weights_changes[i] += weights_changes[i]
                    total_biases_changes[i] += biases_changes[i]
        else:
            total_weights_changes = changes[0][0]
            total_biases_changes = changes[0][1]
        for i in range(len(self.weights)):
            self.weights[i] -= \
                learning_rate * total_weights_changes[i] / len(changes)
            self.biases[i] -= \
                learning_rate * total_biases_changes[i] / len(changes)

    def predict(self, attributes: np.ndarray) -> np.ndarray:
        """
        Predicts the target vector for the given attributes vector.
        """
        if self.normalize is not None:
            data = self.normalize(attributes)
        else:
            data = attributes
        for layer, (weights, biases) in \
                enumerate(zip(self.weights, self.biases)):
            data = np.matmul(weights, data) + biases
            if layer < len(self.weights) - 1:
                data = self.activation(data)
        if self.denormalize is not None:
            return self.denormalize(data)
        else:
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
