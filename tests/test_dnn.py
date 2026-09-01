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
"""Unit tests for ``DNNMixin``, shared by the Bayesian heads and the deterministic backbone.

``check_context_matrix`` is exercised through *both* subclasses: they differ only in what their
``feature_config`` describes (the head's own input vs the raw context), so one implementation has to
serve both, and a test against either alone would not show that.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pybandits.model.bnn._dnn import DNNMixin
from pybandits.model.bnn.backbone import MLPBackbone
from pybandits.model.bnn.network import BayesianNeuralNetwork

# The two DNNMixin subclasses, as names a factory can dispatch on.
_COMPONENT_KINDS = ["head", "backbone"]


@pytest.fixture
def make_component() -> Callable[..., DNNMixin]:
    """Factory for either ``DNNMixin`` subclass over the same declared feature layout.

    Both are built without hidden layers: nothing here trains, and the shallowest architecture keeps
    the cold-start cost negligible across the parametrization.
    """

    def _make(kind: str, n_features: int, categorical_features: Optional[Dict[int, int]] = None) -> DNNMixin:
        if kind == "head":
            return BayesianNeuralNetwork.cold_start(
                n_features=n_features, hidden_dim_list=[], categorical_features=categorical_features
            )
        return MLPBackbone.cold_start(
            n_features=n_features, hidden_dims=[], embedding_dim=2, categorical_features=categorical_features
        )

    return _make


def _context(
    rng: np.random.Generator, n_rows: int, n_features: int, categorical_features: Dict[int, int]
) -> np.ndarray:
    """Uniform context with valid integer codes written into every declared categorical column."""
    context = rng.uniform(low=-100.0, high=100.0, size=(n_rows, n_features))
    for column_index, cardinality in categorical_features.items():
        context[:, column_index] = rng.integers(0, cardinality, size=n_rows)
    return context


class TestCheckContextMatrix:
    """Context validation shared by the Bayesian heads and the deterministic backbone."""

    n_features = 4
    n_rows = 5
    # One categorical column, low enough in cardinality to be legal for either subclass.
    categorical_features = {2: 3}

    @pytest.mark.parametrize("kind", _COMPONENT_KINDS)
    @pytest.mark.parametrize("with_categorical", [False, True])
    def test_accepts_a_valid_context(
        self,
        kind: str,
        with_categorical: bool,
        make_component: Callable[..., DNNMixin],
        rng: np.random.Generator,
    ) -> None:
        """A numeric context of the declared width, with in-range codes, passes."""
        categorical_features = self.categorical_features if with_categorical else {}
        component = make_component(kind, self.n_features, categorical_features or None)
        component.check_context_matrix(context=_context(rng, self.n_rows, self.n_features, categorical_features))

    @pytest.mark.parametrize("kind", _COMPONENT_KINDS)
    @pytest.mark.parametrize("column_delta", [-3, -1, 1, 3])
    def test_rejects_the_wrong_width(
        self,
        kind: str,
        column_delta: int,
        make_component: Callable[..., DNNMixin],
        rng: np.random.Generator,
    ) -> None:
        """Any width but the declared one raises, whether the context is too narrow or too wide.

        Too *wide* matters for the backbone specifically: its one-hot expansion reads named columns, so
        surplus columns would be silently truncated rather than rejected by the first matmul.
        """
        component = make_component(kind, self.n_features, self.categorical_features)
        with pytest.raises(AttributeError, match="Shape mismatch"):
            component.check_context_matrix(context=rng.random((self.n_rows, self.n_features + column_delta)))

    @pytest.mark.parametrize("kind", _COMPONENT_KINDS)
    def test_rejects_a_non_numeric_context(
        self, kind: str, make_component: Callable[..., DNNMixin], rng: np.random.Generator
    ) -> None:
        """A context holding strings raises rather than reaching the forward pass."""
        component = make_component(kind, self.n_features)
        context = rng.uniform(size=(self.n_rows, self.n_features)).astype(object)
        context[:, 0] = context[:, 0].astype(str)
        with pytest.raises(ValueError, match="numeric"):
            component.check_context_matrix(context=context)

    @pytest.mark.parametrize("kind", _COMPONENT_KINDS)
    @pytest.mark.parametrize(
        "invalid_context",
        ["not_an_array", None, 42, [["not_numeric", 1], [2, "also_not_numeric"]], {1: 2, 3: 4}, True, [1, 2, 3], [[1]]],
    )
    def test_rejects_a_non_array_context(
        self, kind: str, invalid_context: Any, make_component: Callable[..., DNNMixin]
    ) -> None:
        """Anything that is not a 2-D numeric array raises instead of failing later with a shape error."""
        component = make_component(kind, 2)
        with pytest.raises((AttributeError, ValueError)):
            component.check_context_matrix(context=invalid_context)

    @pytest.mark.parametrize("kind", _COMPONENT_KINDS)
    @pytest.mark.parametrize(
        "bad_code, match",
        [
            (3, "out of range"),
            (-1, "out of range"),
            (1.5, "integer-valued"),
            (np.nan, "integer-valued"),
            (np.inf, "integer-valued"),
        ],
    )
    def test_rejects_bad_category_codes(
        self,
        kind: str,
        bad_code: float,
        match: str,
        make_component: Callable[..., DNNMixin],
        rng: np.random.Generator,
    ) -> None:
        """Codes outside ``[0, cardinality)``, or non-integers, raise.

        On the backbone side this is what stops an invalid code from one-hotting to an all-zero block;
        on the head side it stops it from indexing the wrong embedding row.
        """
        column_index = next(iter(self.categorical_features))
        component = make_component(kind, self.n_features, self.categorical_features)
        context = _context(rng, self.n_rows, self.n_features, self.categorical_features)
        context[0, column_index] = bad_code
        with pytest.raises(ValueError, match=match):
            component.check_context_matrix(context=context)


@pytest.mark.parametrize(
    "cardinality, expected_embedding_dim",
    # Full rank (cardinality - 1, since the next layer's bias absorbs the mean of the rows) while the
    # feature is small; then the divisor rule; then capped so a large table cannot grow without bound.
    [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (9, 4), (32, 8), (100, 25), (256, 64), (10_000, 64)],
)
def test_default_categorical_embedding_dim(cardinality: int, expected_embedding_dim: int) -> None:
    """The automatic embedding width neither collapses small cardinalities nor grows unboundedly."""
    assert DNNMixin.default_categorical_embedding_dim(cardinality) == expected_embedding_dim


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=2, max_value=5),
    cardinality=st.integers(min_value=1, max_value=12),
    hidden_dim_list=st.lists(st.integers(min_value=2, max_value=4), min_size=0, max_size=1),
)
def test_default_categorical_embedding_dim_is_what_cold_start_assigns(
    n_features: int, cardinality: int, hidden_dim_list: List[int]
) -> None:
    """``cold_start`` widths come from the rule, so the table above pins real model shapes."""
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        categorical_features={n_features - 1: cardinality},
    )
    assert bnn.feature_config.categorical_features_configs[0].embedding_dim == (
        DNNMixin.default_categorical_embedding_dim(cardinality)
    )
