from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np
from typing_extensions import Self


class MultilayerPerceptron:
    def __init__(
        self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
        activation: tuple[Callable[[float], float], Callable[[float], float]],
        normalize: Callable[[np.ndarray], np.ndarray] = None,
        denormalize: Callable[[np.ndarray], Any] = None
    ) -> None:
        """
        Creates a multilayer perceptron with the given weights and the given
        activation function.
        Each element of weights should be a matrix - a list of lists of
        weights for each neuron in the layer, given as a numpy ndarray.
        activation is the activation function applied on all neuron outputs
        except the last layer, given as a tuple of the function itself and its
        derivative.
        normalize and denormalize can be set after the training in order to
        make the perceptron work on original (not normalized) data.
        """
        self.weights = weights
        self.biases = biases
        self.scalar_activation = activation
        self.activation = np.vectorize(activation[0])
        self.activation_derivative = np.vectorize(activation[1])
        self.normalize = normalize
        self.denormalize = denormalize

    @classmethod
    def initialize(
        cls, layer_widths: Sequence[int],
        activation: tuple[Callable[[float], float], Callable[[float], float]],
        normalize: Callable[[np.ndarray], np.ndarray] = None,
        denormalize: Callable[[np.ndarray], np.ndarray] = None
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
                    # output layer gets weights and biases of 0
                    if i != len(layer_widths) - 2 else 0.
                    for _ in range(input_width)
                ] for _ in range(output_width)
            ]))
            biases.append(np.array([
                np.random.uniform(-weights_max, weights_max)
                if i != len(layer_widths) - 2 else 0.
                for _ in range(output_width)
            ]))
        return cls(weights, biases, activation, normalize, denormalize)

    def copy(self) -> Self:
        """
        Returns a copy of the perceptron.
        """
        return type(self)(deepcopy(self.weights), deepcopy(self.biases),
                          self.scalar_activation, self.normalize,
                          self.denormalize)

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

    def get_empty_changes(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Returns empty copies of the weights and biases of the perceptron,
        to which changes can be saved by back_propagate.
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Based on values calculated in the forward propagation, executes
        backpropagation. Returns the gradients of weights and biases
        calculated.
        """
        weights_gradient, biases_gradient = self.get_empty_changes()
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
                biases_gradient[layer_index][neuron_index] = delta
                weights_gradient[layer_index][neuron_index] = \
                    inputs[layer_index] * delta
        return weights_gradient, biases_gradient

    def train(
        self, attributes: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Trains the perceptron based on the given piece of data.
        Calculates changes in weights and biases via backpropagation and
        returns them.
        The given piece of data should already be normalized.
        """
        sums, inputs, outputs = self.forward_propagate(attributes)
        return self.back_propagate(sums, inputs, outputs - targets)

    def apply_changes(
        self, gradients: list[tuple[np.ndarray, np.ndarray]],
        learning_rate: float
    ) -> None:
        """
        Applies the changes based on the average of the given gradients to the
        weights and biases of the perceptron.
        """
        total_weights_gradients, total_biases_gradients = \
            self.get_empty_changes()
        for weights_changes, biases_changes in gradients:
            for i in range(len(weights_changes)):
                total_weights_gradients[i] += weights_changes[i]
                total_biases_gradients[i] += biases_changes[i]
        for i in range(len(self.weights)):
            self.weights[i] -= \
                learning_rate * total_weights_gradients[i] / len(gradients)
            self.biases[i] -= \
                learning_rate * total_biases_gradients[i] / len(gradients)

    def predict(self, attributes: np.ndarray) -> Any:
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
    ) -> list[Any]:
        """
        Returns a list of vectors predicted for each element of the input list.
        """
        results = []
        for attributes in all_attributes:
            results.append(self.predict(attributes))
        return results
