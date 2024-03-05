"""
George Kouretas -- ECEN 644
CNN from scratch. Includes CNN architecture w/ configurable conv2D, maxPool2D, flatten, and dense layers
"""
from activation_funcs import ActivationFunction
from loss_funcs import LossFunction

import copy
import numpy as np
from typing import Iterable
from scipy.signal import convolve2d as conv2
from scipy.signal import correlate2d as xcorr2

import abc
from dataclasses import dataclass

"""Dataclass to contain information concerning the input shape, used for readability"""
@dataclass(frozen = True)
class InputShape:
    depth: int
    height: int
    width: int

    @property
    def shape(self): return (self.depth, self.height, self.width)

"""Abstract class for a layer, with required methods for forward/backward propogation"""
class _Layer(abc.ABC):
    def __init__(self): pass

    @abc.abstractmethod
    def _forward_prop(self, input_: np.ndarray) -> np.ndarray: 
        pass

    @abc.abstractmethod
    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, rate: int) -> np.ndarray: 
        pass

"""Conv2D layer"""
class Conv2D(_Layer):
    def __init__(self, input_depth: int, input_height: int, input_width: int, output_depth: int, kernel_size: int) -> None:
        super().__init__()
        self._input_shape = \
            InputShape(input_depth, input_height, input_width)
        
        self._output_depth = output_depth
        self._kernel_size = kernel_size

        self._kernels = \
            np.random.randn(
                self._output_depth, 
                self._input_shape.depth, 
                self._kernel_size, 
                self._kernel_size
            )

        self._biases = \
            np.random.randn(
                self._output_depth, 
                self._input_shape.height - self._kernel_size + 1, 
                self._input_shape.width - self._kernel_size + 1
            )

        self._output = \
            np.zeros(self._biases.shape)

    def _forward_prop(self, input_: np.ndarray):
        # Iterate across output depth (kernels)
        for d in range(self._output_depth):
            for d_in in range(self._input_shape.depth):
                # output = sum(b + xcorr2d(input, kernel)). 
                # Computes "valid" shape, which corresponds to the configured shape (H-K+1, W-K+1).
                self._output[d] += self._biases[d] + xcorr2(input_[d_in], self._kernels[d, d_in], "valid")

        return self._output

    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, rate: float): 
        _input_grad = np.zeros(self._input_shape.shape)
        _kernel_grad = np.zeros((self._kernel_size, self._kernel_size))

        self._biases -= rate * gradient_
        
        for d in range(self._output_depth):
            for d_in in range(self._input_shape.depth):
                _input_grad[d_in] += conv2(gradient_[d], self._kernels[d, d_in], "full")
                _kernel_grad = xcorr2(input_[d_in], gradient_[d], "valid")
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
                    
        return copy.deepcopy(self._output)

    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, _):
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
        self._weight: np.ndarray    = np.random.randn(output_shape, input_shape)
        self._bias: np.ndarray      = np.random.randn(output_shape, 1)

    def _forward_prop(self, input_: np.ndarray) -> np.ndarray:
        return np.dot(self._weight, input_) + self._bias
    
    def _backward_prop(self, input_: np.ndarray, gradient_: np.ndarray, rate: int) -> np.ndarray:
        new_gradient = np.dot(self._weight.T, gradient_)
        self._weight -= rate * np.dot(gradient_, input_.T) 
        self._bias -= rate * gradient_
        return new_gradient

CNNLayer = _Layer | ActivationFunction # CNN layer may be a layer or activation function

class CNN:
    def __init__(self, layers: Iterable[CNNLayer], loss: LossFunction) -> None:
        self._layers = layers   # NN layers
        self._loss = loss       # Loss function
        self._errors = []

    def train(self, input_, output_, epochs: int, rate: float):
        self._errors.clear()
        for epoch in range(epochs):
            _error = []
            for i, o in zip(input_, output_):
                # Store inputs to later use for back prop.
                _inputs = []

                # print("######## Forward prop ########")
                for layer in self._layers:
                    _inputs.append(copy.deepcopy(i))
                    if isinstance(layer, _Layer):
                        i = layer._forward_prop(i)
                    elif isinstance(layer, ActivationFunction): 
                        i = layer.compute(i)

                o = np.reshape(o, (np.size(o), 1))

                # Compute loss for forward iteration
                _error.append(self._loss.compute(o, i))

                # Compute gradient of loss
                grad_error = self._loss.derivative(o, i)

                # Nackwards propogation
                for layer in reversed(self._layers):
                    # Pop input from input list, which will yield the elements in reverse
                    prev_input = _inputs.pop()

                    # Perform gradient descent for layer/activation function
                    if isinstance(layer, _Layer):
                        grad_error = layer._backward_prop(prev_input, grad_error, rate)
                    elif isinstance(layer, ActivationFunction): 
                        grad_error = np.multiply(grad_error, layer.derivative(prev_input))

            self._errors.append(np.average(_error))
            print(f"Epoch: {epoch+1}/{epochs}. Error = {self._errors[-1]}")