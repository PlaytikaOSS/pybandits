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

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from pydantic import ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from pybandits.model.bnn._dnn import DNNMixin
from pybandits.model.bnn._svi import forward_layers
from pybandits.model.bnn._typing import ActivationFunctions
from pybandits.model.bnn.config import CategoricalFeatureConfig, FeaturesConfig


class MLPBackbone(DNNMixin):
    """A plain MLP backbone with deterministic (point-estimate) weights.

    Architecture is ``[input_dim] + hidden_dims + [embedding_dim]`` (``input_dim`` is ``n_features``
    with each categorical column widened to its one-hot block). All layers except the last are
    activated; the final layer is a linear projection to the embedding space (the per-arm Bayesian
    heads supply the decision non-linearity). Point-estimate weights/biases are stored as plain lists
    (``weights``/``biases``) so the model serialises via ``get_state``/``from_state``; numpy views
    (``weight_arrays``/``bias_arrays``) are rebuilt on init for the forward pass.

    Two optional knobs control how the shared backbone behaves under repeated joint training (both are
    no-ops at their defaults, and are preserved by :meth:`reset` — they are configuration, not state):

    * ``l2_anchoring`` — weight of a quadratic penalty pulling each ``update()`` call's weights *and*
      biases back toward their values at the start of that call, the point-estimate analogue of the
      heads' KL-to-previous-posterior anchor. Bounds per-call drift, which otherwise compounds under
      continual small-batch updates and degrades online performance. It trades off against the
      likelihood term (which scales with batch and network size), so tune it per deployment — start
      around ``1e3``-``1e5`` and increase until drift stabilises without stalling the backbone.
    * ``lr`` — a backbone-only learning rate for the joint SVI pass (heads keep the one from their
      ``update_kwargs``); ``0.0`` freezes the backbone entirely. Prefer ``l2_anchoring`` as the primary
      drift control: empirically it beats any ``lr`` setting, freeze included.

    When cold-starting a :class:`~pybandits.meta_model.cmab_meta_model.CmabMetaModel`, pass these as
    ``backbone_l2_anchoring`` / ``backbone_lr``.

    ``categorical_features`` (``{column_index: cardinality}``, on raw context columns) declares columns
    holding integer category codes; each is **one-hot expanded** before the first layer instead of being
    consumed as a continuous value. One-hot into a dense layer *is* a full-rank learned embedding of the
    category — ``onehot_K @ W`` selects row ``k`` of ``W``, and that row is category ``k``'s learned
    vector. The Bayesian heads instead use a low-rank embedding table (see
    :class:`~pybandits.model.bnn.config.CategoricalFeatureConfig`), which buys a *posterior* per
    category; the backbone's weights are deterministic point estimates, so a table here could only lose
    rank without buying anything. Cost is ``cardinality x hidden_dims[0]`` extra deterministic weights
    per feature, which is why very large cardinalities warn (see ``_one_hot_warn_cardinality``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Uniform-init half-width numerator: limit = sqrt(numerator / fan), shared by He and Glorot/Xavier.
    _init_limit_numerator: ClassVar[float] = 6.0
    # Activations whose negative-half saturation calls for He init (variance ~ 2/fan_in).
    _he_activations: ClassVar[Tuple[str, ...]] = ("relu", "gelu")
    # Above this cardinality a one-hot block's first-layer weights get expensive (K x hidden_dims[0]
    # deterministic weights, all serialised as nested lists) and cold_start warns.
    _one_hot_warn_cardinality: ClassVar[int] = 500

    n_features: PositiveInt
    embedding_dim: PositiveInt
    hidden_dims: List[PositiveInt]
    activation: ActivationFunctions = "relu"
    random_seed: Optional[NonNegativeInt] = None
    weights: List[List[List[float]]]  # per layer: (input_dim, output_dim)
    biases: List[List[float]]  # per layer: (output_dim,)
    l2_anchoring: NonNegativeFloat = 0.0
    lr: Optional[NonNegativeFloat] = None
    categorical_features: Dict[NonNegativeInt, PositiveInt] = Field(default_factory=dict)

    # Site-name prefixes for the backbone's deterministic params inside the joint NumPyro model.
    weight_var_name: ClassVar[str] = "backbone_weight"
    bias_var_name: ClassVar[str] = "backbone_bias"

    @staticmethod
    def _as_float32_arrays(nested: List[List[float]]) -> List[np.ndarray]:
        """Convert per-layer nested lists to ``float32`` numpy arrays (JAX's default working precision)."""
        return [np.asarray(arr, dtype=np.float32) for arr in nested]

    @staticmethod
    def _build_feature_config(n_features: int, categorical_features: Dict[int, int]) -> FeaturesConfig:
        """Column layout of the raw context, each categorical widened to its one-hot block.

        ``embedding_dim = cardinality`` because a one-hot expansion *is* an identity embedding, so
        ``FeaturesConfig.total_output_dim`` is exactly the width the first layer sees. Configs are
        emitted in ``column_index`` order so the expanded column layout does not depend on the order
        the ``categorical_features`` dict happened to be written (or deserialised) in.

        Parameters
        ----------
        n_features : int
            Raw context width (before expansion).
        categorical_features : Dict[int, int]
            ``{column_index: cardinality}`` on raw context columns.

        Returns
        -------
        FeaturesConfig
            The layout, validated (out-of-range ``column_index`` raises).
        """
        return FeaturesConfig(
            n_features=n_features,
            categorical_features_configs=[
                CategoricalFeatureConfig(column_index=column_index, cardinality=cardinality, embedding_dim=cardinality)
                for column_index, cardinality in sorted(categorical_features.items())
            ],
        )

    @property
    def feature_config(self) -> FeaturesConfig:
        """This backbone's raw-context column layout (see :meth:`_build_feature_config`)."""
        return self._build_feature_config(self.n_features, self.categorical_features)

    @property
    def input_dim(self) -> PositiveInt:
        """Width the first layer sees: numerical columns plus one column per category code."""
        return self.feature_config.total_output_dim

    @model_validator(mode="after")
    def _check_input_layer(self) -> Self:
        """Validate the categorical layout and that layer 0 was built for the expanded width.

        Catches a state whose ``categorical_features`` and stored ``weights`` disagree — e.g. a
        serialized backbone loaded against a changed feature layout — at construction rather than
        with a shape error inside the SVI pass.

        Returns
        -------
        Self
            The validated instance.
        """
        expected = self.input_dim  # also raises on an out-of-range column_index
        if self.weights and len(self.weights[0]) != expected:
            raise ValueError(
                f"First layer expects {len(self.weights[0])} input columns but the feature layout gives "
                f"{expected} (n_features={self.n_features}, categorical_features={self.categorical_features})."
            )
        return self

    @property
    def weight_arrays(self) -> List[np.ndarray]:
        """Per-layer weight matrices as numpy arrays (consumed by the forward pass / joint SVI engine)."""
        return self._as_float32_arrays(self.weights)

    @property
    def bias_arrays(self) -> List[np.ndarray]:
        """Per-layer bias vectors as numpy arrays (consumed by the forward pass / joint SVI engine)."""
        return self._as_float32_arrays(self.biases)

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
        input_dim: PositiveInt,
        hidden_dims: List[PositiveInt],
        embedding_dim: PositiveInt,
        activation: ActivationFunctions,
        random_seed: Optional[int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Activation-aware (He/Xavier) uniform weights + zero biases; deterministic given ``random_seed``.

        Parameters
        ----------
        input_dim : int
            Width the first layer consumes — the raw context width with each categorical column
            replaced by its one-hot block (``FeaturesConfig.total_output_dim``).
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
        dims = [input_dim, *hidden_dims, embedding_dim]
        weights, biases = [], []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = cls._init_limit(fan_in, fan_out, activation)
            weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            biases.append(np.zeros(fan_out))
        return weights, biases

    @classmethod
    def cold_start(
        cls,
        n_features: PositiveInt,
        hidden_dims: List[PositiveInt],
        embedding_dim: PositiveInt,
        activation: ActivationFunctions = "relu",
        random_seed: Optional[NonNegativeInt] = None,
        l2_anchoring: NonNegativeFloat = 0.0,
        lr: Optional[NonNegativeFloat] = None,
        categorical_features: Optional[Dict[NonNegativeInt, PositiveInt]] = None,
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
        l2_anchoring : NonNegativeFloat, default=0.0
            Weight of the per-update anchoring penalty on the backbone's params (``0.0`` disables it).
        lr : Optional[NonNegativeFloat], default=None
            Backbone-only learning rate for joint SVI (``None`` shares the heads'; ``0.0`` freezes).
        categorical_features : Optional[Dict[NonNegativeInt, PositiveInt]], default=None
            Raw-context columns holding integer category codes, as ``{column_index: cardinality}``.
            Each is one-hot expanded before the first layer instead of being consumed as a continuous
            value; columns absent from this dict are treated as numerical. When cold-starting a
            ``CmabMetaModel``, pass this as ``categorical_features`` (or ``backbone_categorical_features``).

        Returns
        -------
        MLPBackbone
            A freshly initialised backbone.
        """
        categorical_features = dict(categorical_features or {})
        first_layer_width = hidden_dims[0] if hidden_dims else embedding_dim  # hidden_dims=[] is one linear layer
        for column_index, cardinality in sorted(categorical_features.items()):
            if cardinality > cls._one_hot_warn_cardinality:
                warnings.warn(
                    f"Categorical column {column_index} has cardinality {cardinality}: one-hot expansion adds "
                    f"{cardinality} x {first_layer_width} deterministic first-layer weights to the backbone state. "
                    "Consider hashing or grouping rare levels upstream.",
                    stacklevel=2,
                )
        feature_config = cls._build_feature_config(n_features, categorical_features)
        weights, biases = cls._init_params(
            feature_config.total_output_dim, list(hidden_dims), embedding_dim, activation, random_seed
        )
        return cls(
            n_features=n_features,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            random_seed=random_seed,
            weights=[w.tolist() for w in weights],
            biases=[b.tolist() for b in biases],
            l2_anchoring=l2_anchoring,
            lr=lr,
            categorical_features=categorical_features,
        )

    def reset(self) -> Self:
        """Return a fresh backbone with the same architecture / seed / knobs (weights re-initialised).

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
            l2_anchoring=self.l2_anchoring,
            lr=self.lr,
            categorical_features=self.categorical_features,
        )

    def _expand(self, x: Any, backend: Any) -> Any:
        """Replace each declared categorical column with its one-hot block (numerical columns first).

        Backend-agnostic (NumPy for :meth:`embed`, ``jax.numpy`` under the joint SVI trace) and a pure
        function of ``x``, so it needs no parameters of its own and leaves every caller's signature
        alone. A no-op when no categorical column is declared.

        Parameters
        ----------
        x : np.ndarray or jnp.ndarray of shape (n, n_features)
            Raw context matrix.
        backend : module
            ``numpy`` or ``jax.numpy``.

        Returns
        -------
        np.ndarray or jnp.ndarray of shape (n, input_dim)
            The expanded matrix fed to the first layer.
        """
        feature_config = self.feature_config
        if not feature_config.has_categorical:
            return x
        blocks = [x[:, feature_config.numerical_indices]]
        for config in feature_config.categorical_features_configs:
            codes = x[:, config.column_index].astype(backend.int32)
            # ponytail: (n, K) broadcast comparison rather than eye(K)[codes], which would materialise a
            # K x K identity; switch to a gather on layer 0's weights if a cardinality in the thousands
            # ever makes the dense (n, K) block itself the problem.
            blocks.append((codes[:, None] == backend.arange(config.cardinality)).astype(x.dtype))
        return backend.concatenate(blocks, axis=1)

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
        self.check_context_matrix(context=x)
        weights_biases = list(zip(self.weight_arrays, self.bias_arrays))
        return np.asarray(
            forward_layers(
                next_layer_input=self._expand(x, np),
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
            next_layer_input=self._expand(x, jnp),
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
