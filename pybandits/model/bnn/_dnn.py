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
"""Shared base for deep-network components (the Bayesian heads and the deterministic backbone).

Both the per-arm ``BaseBayesianNeuralNetwork`` heads and the shared ``MLPBackbone`` encoder run
feed-forward layers with the same set of element-wise activations. ``DNNMixin`` holds the single
source of truth for those activation maps (NumPy + JAX) and the ``activation``-name → callable
resolution, so the two component families stay in lockstep.

``DNNMixin`` inherits :class:`PyBanditsBaseModel` so that ``activation`` is a first-class Pydantic
field; concrete subclasses supply the default value for their domain.
"""

from typing import Any, ClassVar, Dict

import jax
import numpy as np

from pybandits.base import PyBanditsBaseModel
from pybandits.model.bnn._typing import ActivationFunctions, _numpy_gelu, _numpy_relu, _numpy_sigmoid


class DNNMixin(PyBanditsBaseModel):
    """Pydantic base providing the shared activation field, maps and resolvers for DNN components."""

    activation: ActivationFunctions

    # Default embedding width when a categorical/backbone embedding_dim is omitted: n_features // divisor
    # (min 1). Shared by BaseBayesianNeuralNetwork's categorical embeddings and MLPBackbone's default
    # output width, since both are "pick a reasonable embedding size" defaults for a DNN component.
    embedding_dim_divisor: ClassVar[int] = 4

    _numpy_activations: ClassVar[Dict[str, Any]] = {
        "tanh": np.tanh,
        "relu": _numpy_relu,
        "sigmoid": _numpy_sigmoid,
        "gelu": _numpy_gelu,
    }
    _jax_activations: ClassVar[Dict[str, Any]] = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
        "gelu": lambda x: jax.nn.gelu(x, approximate=False),
    }

    @property
    def numpy_activation(self) -> Any:
        """NumPy activation callable for this component's ``activation``."""
        return self._numpy_activations[self.activation]

    @property
    def jax_activation(self) -> Any:
        """JAX activation callable for this component's ``activation``."""
        return self._jax_activations[self.activation]
