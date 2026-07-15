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
"""Shared deterministic DNN backbone (preprocessor) for the contextual-bandit meta-model.

The backbone maps a context matrix ``(n, n_features)`` to an embedding ``(n, embedding_dim)`` with
point-estimate (deterministic) weights. It is shared across arms and trained jointly with the per-arm
Bayesian heads inside one SVI pass (its weights enter the joint NumPyro model as ``numpyro.param``).
Pydantic immutability is preserved by returning a new instance from :meth:`with_weights_and_biases`.

It shares its activation field, maps and resolvers with the Bayesian heads via :class:`DNNMixin`.
"""

from typing import ClassVar, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from pydantic import ConfigDict, NonNegativeInt, PositiveInt
from typing_extensions import Self

from pybandits.model.bnn._dnn import DNNMixin
from pybandits.model.bnn._svi import forward_layers
from pybandits.model.bnn._typing import ActivationFunctions


class MLPBackbone(DNNMixin):
    """A plain MLP backbone with deterministic (point-estimate) weights.

    Architecture is ``[n_features] + hidden_dims + [embedding_dim]``. All layers except the last are
    activated; the final layer is a linear projection to the embedding space (the per-arm Bayesian
    heads supply the decision non-linearity). Point-estimate weights/biases are stored as plain lists
    (``weights``/``biases``) so the model serialises via ``get_state``/``from_state``; numpy views
    (``weight_arrays``/``bias_arrays``) are rebuilt on init for the forward pass.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Uniform-init half-width numerator: limit = sqrt(numerator / fan), shared by He and Glorot/Xavier.
    _init_limit_numerator: ClassVar[float] = 6.0
    # Activations whose negative-half saturation calls for He init (variance ~ 2/fan_in).
    _he_activations: ClassVar[Tuple[str, ...]] = ("relu", "gelu")

    n_features: PositiveInt
    embedding_dim: PositiveInt
    hidden_dims: List[PositiveInt]
    activation: ActivationFunctions = "relu"
    random_seed: Optional[NonNegativeInt] = None
    weights: List[List[List[float]]]  # per layer: (input_dim, output_dim)
    biases: List[List[float]]  # per layer: (output_dim,)

    # Site-name prefixes for the backbone's deterministic params inside the joint NumPyro model.
    weight_var_name: ClassVar[str] = "backbone_weight"
    bias_var_name: ClassVar[str] = "backbone_bias"

    @property
    def weight_arrays(self) -> List[np.ndarray]:
        """Per-layer weight matrices as numpy arrays (consumed by the forward pass / joint SVI engine)."""
        return [np.asarray(w, dtype=float) for w in self.weights]

    @property
    def bias_arrays(self) -> List[np.ndarray]:
        """Per-layer bias vectors as numpy arrays (consumed by the forward pass / joint SVI engine)."""
        return [np.asarray(b, dtype=float) for b in self.biases]

    @classmethod
    def get_layer_params_name(cls, layer_ind: int) -> Tuple[str, str]:
        """NumPyro ``param`` names for a backbone layer's weight/bias (namespaced away from heads).

        Parameters
        ----------
        layer_ind : int
            Zero-based layer index.

        Returns
        -------
        Tuple[str, str]
            The ``(weight_name, bias_name)`` site names for the layer.
        """
        return f"{cls.weight_var_name}_{layer_ind}", f"{cls.bias_var_name}_{layer_ind}"

    @classmethod
    def _init_limit(cls, input_dim: PositiveInt, output_dim: PositiveInt, activation: ActivationFunctions) -> float:
        """Uniform-init half-width for a layer, chosen by activation.

        * **He** uniform — ``sqrt(6 / fan_in)`` — for ``relu``/``gelu``.
        * **Xavier/Glorot** uniform — ``sqrt(6 / (fan_in + fan_out))`` — for ``tanh``/``sigmoid``.

        Parameters
        ----------
        input_dim : PositiveInt
            Fan-in of the layer.
        output_dim : PositiveInt
            Fan-out of the layer.
        activation : ActivationFunctions
            Activation applied after the layer; selects He vs Xavier/Glorot.

        Returns
        -------
        float
            The uniform-init half-width ``limit`` (weights drawn from ``U(-limit, limit)``).

        References
        ----------
        He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers.
        https://arxiv.org/abs/1502.01852

        Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward networks.
        https://proceedings.mlr.press/v9/glorot10a.html
        """
        if activation in cls._he_activations:
            return float(np.sqrt(cls._init_limit_numerator / input_dim))
        return float(np.sqrt(cls._init_limit_numerator / (input_dim + output_dim)))

    @classmethod
    def _init_params(
        cls,
        n_features: PositiveInt,
        hidden_dims: List[PositiveInt],
        embedding_dim: PositiveInt,
        activation: ActivationFunctions,
        random_seed: Optional[int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Activation-aware (He/Xavier) uniform weights + zero biases; deterministic given ``random_seed``.

        Parameters
        ----------
        n_features : int
            Input (context) dimensionality.
        hidden_dims : List[int]
            Hidden-layer widths.
        embedding_dim : int
            Output (embedding) dimensionality.
        activation : ActivationFunctions
            Activation used for the init scheme (He vs Xavier/Glorot).
        random_seed : Optional[int]
            Seed for reproducible initialisation.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Per-layer ``(weights, biases)`` arrays.
        """
        rng = np.random.default_rng(random_seed)
        dims = [n_features, *hidden_dims, embedding_dim]
        weights, biases = [], []
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            limit = cls._init_limit(input_dim, output_dim, activation)
            weights.append(rng.uniform(-limit, limit, size=(input_dim, output_dim)))
            biases.append(np.zeros(output_dim))
        return weights, biases

    @classmethod
    def cold_start(
        cls,
        n_features: PositiveInt,
        hidden_dims: List[PositiveInt],
        embedding_dim: PositiveInt,
        activation: ActivationFunctions = "relu",
        random_seed: Optional[NonNegativeInt] = None,
    ) -> Self:
        """Initialise an MLP backbone with activation-dependent (He/Xavier) weights and zero biases.

        Parameters
        ----------
        n_features : PositiveInt
            Input (context) dimensionality.
        hidden_dims : List[PositiveInt]
            Hidden-layer widths.
        embedding_dim : PositiveInt
            Output (embedding) dimensionality.
        activation : ActivationFunctions, default="relu"
            Element-wise activation applied after every layer except the last.
        random_seed : Optional[NonNegativeInt], default=None
            Seed for reproducible initialisation.

        Returns
        -------
        MLPBackbone
            A freshly initialised backbone.
        """
        weights, biases = cls._init_params(n_features, list(hidden_dims), embedding_dim, activation, random_seed)
        return cls(
            n_features=n_features,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            random_seed=random_seed,
            weights=[w.tolist() for w in weights],
            biases=[b.tolist() for b in biases],
        )

    def reset(self) -> Self:
        """Return a fresh backbone with the same architecture / seed (weights re-initialised).

        Returns
        -------
        MLPBackbone
            A cold-started backbone with this instance's configuration.
        """
        return self.cold_start(
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            activation=self.activation,
            random_seed=self.random_seed,
        )

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Deterministic forward pass producing the shared embedding.

        Parameters
        ----------
        x : np.ndarray of shape (n, n_features)
            Context matrix.

        Returns
        -------
        np.ndarray of shape (n, embedding_dim)
            The shared embedding.
        """
        weights_biases = list(zip(self.weight_arrays, self.bias_arrays))
        return np.asarray(
            forward_layers(
                next_layer_input=x,
                weights_biases=weights_biases,
                activation_fn=self.numpy_activation,
                linear_fn=lambda a, w, b: a @ w + b,
                backend=np,
                use_residual_connections=False,
            )
        )

    def forward_jax(self, x: jnp.ndarray, weights_biases: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> jnp.ndarray:
        """JAX forward pass on ``x`` using the supplied (e.g. ``numpyro.param``) weight/bias pairs.

        Used inside the joint NumPyro model where the backbone weights are optimised as params.

        Parameters
        ----------
        x : jnp.ndarray of shape (n, n_features)
            Context matrix.
        weights_biases : List[Tuple[jnp.ndarray, jnp.ndarray]]
            Per-layer ``(weight, bias)`` arrays (the ``numpyro.param`` point estimates).

        Returns
        -------
        jnp.ndarray of shape (n, embedding_dim)
            The shared embedding.
        """
        return forward_layers(
            next_layer_input=x,
            weights_biases=weights_biases,
            activation_fn=self.jax_activation,
            linear_fn=lambda a, w, b: jnp.dot(a, w) + b,
            backend=jnp,
            use_residual_connections=False,
        )

    def with_weights_and_biases(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> Self:
        """Return a new instance with the given weight/bias arrays (Pydantic immutability).

        Parameters
        ----------
        weights : List[np.ndarray]
            Per-layer weight matrices.
        biases : List[np.ndarray]
            Per-layer bias vectors.

        Returns
        -------
        Self
            A new backbone carrying the supplied parameters.
        """
        return self.model_copy(
            update={
                "weights": [np.asarray(w).tolist() for w in weights],
                "biases": [np.asarray(b).tolist() for b in biases],
            }
        )
