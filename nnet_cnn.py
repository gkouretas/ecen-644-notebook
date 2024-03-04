from activation_funcs import ActivationFunction
from loss_funcs import LossFunction

import copy
import numpy as np
from typing import Iterable
from scipy.signal import convolve2d, correlate2d

import abc
from dataclasses import dataclass

@dataclass(frozen = True)
class InputShape:
    depth: int
    height: int
    width: int

    @property
    def shape(self): return (self.depth, self.height, self.width)

class _Layer(abc.ABC):
    def __init__(self): pass

    @abc.abstractmethod
    def _forward_prop(self, input_: np.ndarray) -> np.ndarray: 
        pass

    @abc.abstractmethod
    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, rate: int) -> np.ndarray: 
        pass

class Conv2D(_Layer):
    def __init__(self, input_depth: int, input_height: int, input_width: int, depth: int, kernel_size: int) -> None:
        super().__init__()
        self._input_shape = \
            InputShape(input_depth, input_height, input_width)
        
        self._depth = depth
        self._kernel_size = kernel_size

        self._kernels = \
            np.random.randn(
                self._depth, 
                self._input_shape.depth, 
                self._kernel_size, 
                self._kernel_size
            )

        self._biases = \
            np.random.randn(
                self._depth, 
                self._input_shape.height - self._kernel_size + 1, 
                self._input_shape.width - self._kernel_size + 1
            )

        self._output = \
            np.zeros(self._biases.shape)

    def _forward_prop(self, input_: np.ndarray):
        for d in range(self._depth):
            for k in self._kernels[d]:
                self._output[d] = self._biases[d] + correlate2d(input_[d], k, "valid")

        return self._output

    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, rate: float): 
        _input_grad = np.zeros(self._input_shape.shape)
        _kernel_grad = np.zeros((self._kernel_size, self._kernel_size))

        self._biases -= rate * gradient_
        
        for d in range(self._depth):
            for d_in in range(self._input_shape.depth):
                _kernel_grad = correlate2d(input_[d_in], gradient_[d], "valid")
                _input_grad[d_in] += convolve2d(gradient_[d], self._kernels[d, d_in], "full")
                self._kernels[d, d_in] -= rate * _kernel_grad

        return _input_grad

class MaxPool2D(_Layer):
    def __init__(self, input_depth: int, input_height: int, input_width: int):
        super().__init__()
        self._input_shape = InputShape(input_depth, input_height, input_width)
        self._output = np.zeros(
            (self._input_shape.depth, 
             self._input_shape.height//2, 
             self._input_shape.width//2)
        )

    def _forward_prop(self, input_: np.ndarray):
        for d in range(self._input_shape.depth):
            for h in range(self._input_shape.height//2):
                for w in range(self._input_shape.width//2):
                    self._output[d, h, w] = \
                        np.max(input_[d, (2*h):(2*(h+1)), (2*w):(2*(w+1))], axis = (0,1))
                    
        return self._output

    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, _):
        print(gradient_.shape)
        _input_grad = np.zeros(input_.shape)
        
        for h_ in range(self._input_shape.height//2):
            for w_ in range(self._input_shape.width//2):
                for d in range(self._input_shape.depth):
                    for h in range(self._input_shape.height):
                        for w in range(self._input_shape.width):
                            h_inc, w_inc = np.where(input_[d, (2*h):(2*(h+1)), (2*w):(2*(w+1))] == self._output[d, h_, w_])
                            if np.size(h_inc) > 0 and np.size(w_inc) > 0:
                                _input_grad[d, 2*h + h_inc[-1], 2*w + w_inc[-1]] = gradient_[d, h_, w_]
                                
        return _input_grad

class Flatten(_Layer):
    def __init__(self):
        super().__init__()

    def _forward_prop(self, input_: np.ndarray) -> np.ndarray:
        return np.reshape(input_, (np.prod(input_.shape), 1))
    
    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, _) -> np.ndarray:
        return np.reshape(gradient_, input_.shape)

class Dense(_Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__()
        self._weight: np.ndarray = np.random.randn(output_shape, input_shape)
        self._bias: np.ndarray = np.random.randn(output_shape, 1)

    def _forward_prop(self, input_: np.ndarray) -> np.ndarray:
        return np.dot(self._weight, input_) + self._bias
    
    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, rate: int) -> np.ndarray:
        new_gradient = np.dot(self._weight.T, gradient_)
        self._weight -= rate * np.dot(gradient_, input_.T) 
        self._bias -= rate * gradient_
        return new_gradient

class CNN:
    def __init__(self, layers: Iterable[_Layer | ActivationFunction], loss: LossFunction) -> None:
        self._layers = layers   # NN layers
        self._loss = loss       # Loss function

    def train(self, input_, output_, epochs: int, rate: float):
        for epoch in range(epochs):
            _error = []
            for i, o in zip(input_, output_):
                # Store inputs to later use for back prop.
                _inputs = []

                for layer in self._layers:
                    print(f"In: {i.shape}")
                    _inputs.append(i)
                    if isinstance(layer, _Layer):
                        i = layer._forward_prop(i)
                    elif isinstance(layer, ActivationFunction): 
                        i = layer.compute(i)
                    print(f"Out: {i.shape}")

                _error.append(self._loss.compute(actual = o, predicted = i))
                grad_error = self._loss.derivative(actual = o, predicted = i)

                for layer in reversed(self._layers):
                    if isinstance(layer, _Layer):
                        grad_error = layer._backward_prop(_inputs.pop(), grad_error, rate)
                    elif isinstance(layer, ActivationFunction): 
                        grad_error *= layer.derivative(_inputs.pop())

            print(f"Epoch: {epoch+1}/{epochs}. Error = {np.average(_error)}")

if __name__ == "__main__":
    from activation_funcs import Sigmoid
    from loss_funcs import BinaryXEntropyLoss
    c2 = CNN(
        [
            Conv2D(50, 25, 25, 3, 3),       # Conv2D layer
            Sigmoid(),                      # Activation layer
            MaxPool2D(3, 23, 23),           # MaxPool2D layer
            Sigmoid(),                      # Activation layer
            Flatten(),                      # Flatten
            Dense(3 * (23//2) * (23//2), 10),      # Dense layer
            Sigmoid()                       # Activation layer
        ],
        loss = BinaryXEntropyLoss()
    )
    # # c2 = Conv2D(50, 25, 25, 3, 3)
    # # c2._forward_prop(np.array(np.zeros((50, 25, 25))))
    # c2 = MaxPool2D(50, 25, 25)
    # c2._forward_prop(np.array(np.random.randn(50, 25, 25)))
    # print(c2._output.shape)

    c2.train(np.random.randn(100, 50, 25, 25), np.random.randn(10,1), epochs = 20, rate = 0.01)