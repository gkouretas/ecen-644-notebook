"""
George Kouretas -- ECEN 644
Back-propogation algorithm developed from scratch
"""

import numpy as np
import copy

from activation_funcs import ActivationFunction

from dataclasses import dataclass

@dataclass
class NeuralNetInfo:
    """Container for holding iterative neural net metrics for analysis"""
    error: list[float]
    weights: list[list]
    biases: list[list]

    @property
    def epochs(self): return len(self.error)

class NeuralNet:
    def __init__(self, input_size: int, hidden_size: tuple[int, ...], output_size: int, activation_func: ActivationFunction = None) -> None:
        """Neural network class with forward/backward propogation

        Args:
            input_size (int): Size of input layer
            hidden_size (tuple[int, ...]): Hidden layer shape. 
            Length of tuple = number of hidden layer, with each element equating to the number of neurons per hidden layer
            output_size (int): Size of output layer
            activation_func (ActivationFunction, optional): Activation function. Defaults to None, in which case an activation function must be set via `set_activation_function`
        """
        # Size for input, hidden layers, and output
        self._input_size = (input_size, 1)
        self._hidden_size = hidden_size
        self._output_size = (output_size, 1)

        # Compute number of total layers (# of hidden layers + input layer + output layer)
        self._number_of_layers = len(self._hidden_size) + 2

        # Weights
        # The weights are represented as a list of length = (# of layers - 1)
        # with each element being an m x n matrix, 
        # where m = # of neurons in the next layer
        # and n = # of neurons in current layer
        self._weights: list[np.ndarray] = []

        # Biases
        # The biases are represented as a list of length = (# of layers - 1)
        # with each element being an m x 1 matrix,
        # where m = # of neurons in the next layer
        self._biases:  list[np.ndarray] = []

        # Neural network outputs
        # Length will be # of layers
        # no need to pre-initialize since this will be dynamically populated
        self._network: list[np.ndarray] = []

        for idx in range(len(hidden_size)):
            # For the first weight/bias element, reference the input size for the m dimension
            if idx == 0:
                self._weights.append(np.random.randn(hidden_size[0], self._input_size[0]))
                self._biases.append(np.random.randn(hidden_size[0], 1))

            # m = current layer size, n = previous layer size
            else:
                self._weights.append(np.random.randn(hidden_size[idx], hidden_size[idx - 1]))
                self._biases.append(np.random.randn(hidden_size[idx], 1))

        # m = output layer size, n = previous layer size
        self._weights.append(np.random.randn(self._output_size[0], hidden_size[idx]))
        self._biases.append(np.random.randn(self._output_size[0], 1))

        # copy initial weights/biases to allow for resetting network to initial state
        self._initial_weights   = copy.deepcopy(self._weights)
        self._initial_biases    = copy.deepcopy(self._biases)

        # initialize iteration counter, iteration container
        self._iter      : int                  = -1
        self._iterations: NeuralNetInfo        = NeuralNetInfo([], [], [])

        # activation function
        self._activation_func: ActivationFunction = activation_func

    def train(self, inputs: np.ndarray, outputs: np.ndarray, epoch: int, rate: float, log_decimation: int = 100):
        """Train model

        Args:
            inputs (np.ndarray): Array of size (m x n)
                                 m: number of items in training data
                                 n: input vector of the size specified with `input_size`
            outputs (np.ndarray): Array of size (m x n)
                                 m: number of items in training data
                                 n: output vector of the size specified with `output_size`
            epoch (int): Number of training iterations
            rate (float): Training rate multiplier
            log_decimation (int, optional): Decimation counter for epoch / error printing. Defaults to 100.
        """
        # Confirm an activation function has been initialized
        if self._activation_func is None:
            print("Activation function is uninitialized")
            return
        
        # Perform training
        for self._iter in range(epoch):
            # Empty container for container error for training data
            _errors = []

            # Iterate across training inputs/outputs
            for i, o in zip(inputs, outputs):
                
                # Resize vector to proper size, convert to numpy array
                i = self._resize(i, self._input_size)
                o = self._resize(o, self._output_size)
                
                # Reset neural-network values
                self._network = [i]

                # Perform forward propogation
                prediction = self._forward_prop()

                # Compute error
                error = 0.5 * np.sum(np.power(prediction - o, 2))
                _errors.append(error)

                # Perform backward propogation w/ using computed error/learning rate
                self._backward_prop(prediction - o, rate)

            # Populate iteration info
            self._iterations.error.append(_errors)
            self._iterations.weights.append(np.concatenate([w.flatten() for w in self._weights]))
            self._iterations.biases.append(np.concatenate([b.flatten() for b in self._biases]))

            # Print update
            if ((self._iter + 1) % log_decimation) == 0:
                print(f"Epoch ({self._iter+1}/{epoch}): average error = {np.average(self._iterations.error[-1]):.2f}")

    def set_activation_function(self, activation_func: ActivationFunction):
        """Set activation functions ot be used across hidden/output layers

        Args:
            activation_func (ActivationFunction): Activation function
        """
        self._activation_func = activation_func

    def predict(self, x) -> np.ndarray:
        """Predict the output given the input

        Args:
            x: Input vector, of size specified with `input_size`

        Returns:
            Prediction, of the size specified with `output_size`
        """
        x = self._resize(x, self._input_size)
        self._network = [x]
        return self._forward_prop()

    def reset(self):
        """Reset neural network to initial weights/biases"""
        self._weights = copy.deepcopy(self._initial_weights)
        self._biases = copy.deepcopy(self._initial_biases)
        self._network = []
        self._iterations = NeuralNetInfo([], [], [])

    def _forward_prop(self) -> np.ndarray:
        """Forward propogation"""
        # Iterate through layers' weights and biases
        for w, b in zip(self._weights, self._biases):
            # Get previous layer (input layer for first iteration)
            i = self._network[-1]

            # Compute neurons in network by applying the activation
            # function to the equation w*i + b. To achieve the shape (m, 1)
            # where m: # of neurons in the layer
            self._network.append(
                self._activation_func.compute(
                    np.dot(w, i) + b
                )
            )
        
        # Return output layer
        return copy.copy(self._network[-1])

    def _backward_prop(self, err: float, rate: float) -> np.ndarray:
        """Backward propogation"""
        # Initialize delta to None
        delta = None

        for idx in reversed(range(len(self._weights))):
            # First iteration
            if delta is None:
                # Define current layer's weights/biases
                w = self._weights[idx]
                b = self._biases[idx]

                # Define current layer and previous layer.
                # Since the size of the network is 1 greater than the weights/biases
                # due to it including the input layer, the index must be added by 1
                layer = self._network[idx + 1]
                previous_layer = self._network[idx]

                # Compute delta weight vector, which is done by multiplying the error by the partial derivative of the layer
                delta = err * self._activation_func.derivative(layer)

                # Compute the change in weights and biases
                w -= rate * np.dot(delta, previous_layer.T)
                b -= rate * delta
            else:
                # Define current and output layer's weights
                w = self._weights[idx]
                w_ahead = self._weights[idx + 1]

                # Define current layer's biases
                b = self._biases[idx]

                # Define current and previous layer outputs
                layer = self._network[idx + 1]
                previous_layer = self._network[idx]

                # Compute delta vector
                delta = np.dot(w_ahead.T, delta) * self._activation_func.derivative(layer)
                w -= rate * np.dot(delta, previous_layer.T)
                b -= rate * delta

    def _resize(self, x, size):
        if np.shape(x) != size:
            try:
                return np.array(x).reshape(size)
            except Exception:
                print(f"Unable to reshape input vector to size {self._input_size} ({x})")
        
        return np.asarray(x)