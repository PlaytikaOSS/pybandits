# MIT License
#
# Copyright (c) 2023 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Literal

import jax
import numpy as np
from scipy.special import erf

_Array = np.ndarray | jax.Array

VIMethods = Literal["advi", "fullrank_advi"]
ActivationFunctions = Literal["tanh", "relu", "sigmoid", "gelu"]
OptaxKind = Literal["optimizer", "lr_scheduler"]

_LOGIT_CLIPPING_THRESHOLD = 15


def _numpy_relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function for NumPy."""
    return np.maximum(0, x)


def _numpy_gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function for NumPy."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2.0)))


def _numpy_sigmoid(x):
    """Stable sigmoid activation function for NumPy."""
    x = np.clip(x, -_LOGIT_CLIPPING_THRESHOLD, _LOGIT_CLIPPING_THRESHOLD)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
