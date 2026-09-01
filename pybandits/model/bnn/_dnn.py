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

from math import ceil
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
    # Bounds clamping the divisor rule for a *categorical* embedding (see default_categorical_embedding_dim).
    # embedding_dim_min needs no binary special case: the cardinality - 1 cap dominates it below 5, so a
    # 2-level feature still gets a single axis and a 3-level one gets two.
    embedding_dim_min: ClassVar[int] = 4
    embedding_dim_max: ClassVar[int] = 64

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

    def check_context_matrix(self, context: np.ndarray) -> None:
        """
        Validate the context input.

        Context must be an array-like with numeric values and the correct number of columns.
        Categorical columns are validated to contain valid integer indices within their vocab range.

        Shared by the Bayesian heads and the deterministic backbone, which differ only in what their
        ``feature_config`` describes: the head's own input for the former, the raw context for the
        latter. Both checks guard a *silent* failure on the backbone side — its ``_expand`` reads named
        columns, so a too-wide context would be truncated rather than rejected by the first matmul, and
        an out-of-range code one-hots to an all-zero block rather than raising.

        Subclasses must expose a ``feature_config``.

        Parameters
        ----------
        context : np.ndarray
            Matrix of contextual features of shape ``(n_samples, n_cols)``.

        Raises
        ------
        AttributeError
            If the context is not array-like or does not have ``feature_config.n_features`` columns.
        ValueError
            If the context is non-numeric, or a categorical column holds a non-integer-valued or
            out-of-range code.
        """
        expected_cols = self.feature_config.n_features

        try:
            n_cols_context = context.shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {expected_cols} columns: {e}.")

        if not np.issubdtype(context.dtype, np.number):
            raise ValueError("Context array must contain only numeric values.")

        if n_cols_context != expected_cols:
            raise AttributeError(f"Shape mismatch: context must have {expected_cols} columns.")

        configs = self.feature_config.categorical_features_configs
        if configs:
            col_indices = [c.column_index for c in configs]
            cardinalities = np.array([c.cardinality for c in configs])
            raw_cat_cols = context[:, col_indices]
            if not np.all(np.isfinite(raw_cat_cols) & (raw_cat_cols == np.floor(raw_cat_cols))):
                raise ValueError("Categorical feature columns must contain finite integer-valued indices.")
            cat_cols = raw_cat_cols.astype(int)
            in_range = (cat_cols >= 0) & (cat_cols < cardinalities[np.newaxis, :])
            if not np.all(in_range):
                bad_idx = int(np.argmax(~np.all(in_range, axis=0)))
                cfg = configs[bad_idx]
                raise ValueError(
                    f"Categorical feature at column {cfg.column_index} (index {bad_idx}) has indices out of range "
                    f"[0, {cfg.cardinality})."
                )

    @classmethod
    def default_categorical_embedding_dim(cls, cardinality: int) -> int:
        """Embedding width for a categorical feature whose ``embedding_dim`` was not given explicitly.

        ``ceil(cardinality / embedding_dim_divisor)``, clamped into
        ``[embedding_dim_min, embedding_dim_max]`` and then capped at ``cardinality - 1``.

        The ``cardinality - 1`` cap is the point of the rule: an embedding table ``E (K, d)`` followed by
        a layer ``V (d, h)`` is the map ``onehot_K -> R^h`` constrained to rank ``d``, and since the
        layer's bias absorbs the mean of the ``K`` rows, ``d = K - 1`` already spans every such map. So
        ``d >= K - 1`` buys nothing and ``d < K - 1`` costs expressiveness — the divisor rule alone would
        compress a 4-level feature onto a single learned axis. ``embedding_dim_max`` bounds the other end,
        where the divisor rule would otherwise grow the table (and its variational parameters) linearly
        in cardinality.

        The cap is also what handles small cardinalities without a special case: a binary feature gets
        ``2 - 1 = 1`` and a 3-level one gets 2, regardless of ``embedding_dim_min``.

        Parameters
        ----------
        cardinality : int
            Number of distinct category codes.

        Returns
        -------
        int
            The embedding width to use (always ``>= 1``).
        """
        compressed = max(cls.embedding_dim_min, ceil(cardinality / cls.embedding_dim_divisor))
        # The outer max is load-bearing only for cardinality 1 (a single-level feature, which
        # _add_cat_scenario in tests/test_transfer.py does draw): cardinality - 1 is then 0, which
        # CategoricalFeatureConfig.embedding_dim rejects as non-positive.
        return max(1, min(cardinality - 1, cls.embedding_dim_max, compressed))

    @property
    def numpy_activation(self) -> Any:
        """NumPy activation callable for this component's ``activation``."""
        return self._numpy_activations[self.activation]

    @property
    def jax_activation(self) -> Any:
        """JAX activation callable for this component's ``activation``."""
        return self._jax_activations[self.activation]
