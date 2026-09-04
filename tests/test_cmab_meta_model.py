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
"""Unit tests for the unified joint-VI cMAB engine: MLPBackbone + CmabMetaModel."""

import itertools
from typing import List, Optional, Set

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pybandits.base import ActionId
from pybandits.meta_model import CmabMetaModelMO, CmabMetaModelSO
from pybandits.model import BayesianNeuralNetwork
from pybandits.model.bnn.backbone import MLPBackbone

# Shared strategies for the backbone's architecture and its configuration knobs (the ones that must
# survive reset / serialization and be reachable as ``backbone_``-prefixed cold-start kwargs). Used by
# the tests that only build/serialise a backbone; the SVI-training tests below keep fixed architectures
# instead, so their runtime stays bounded.
_HIDDEN_DIMS = st.lists(st.integers(min_value=1, max_value=8), min_size=0, max_size=2)
_BACKBONE_KNOBS = [
    ("l2_anchoring", st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)),
    ("lr", st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))),
    ("categorical_features", st.just({0: 3})),
]


class TestMLPBackbone:
    """The shared deterministic encoder: shape, determinism, serialization."""

    n_features = 6
    hidden = [4]
    embedding_dim = 4
    n_rows = 9
    backbone_seed = 1

    # A raw-context column of integer category codes, one-hot expanded by the backbone.
    categorical_features = {0: 3}

    def _context(
        self,
        rng: np.random.Generator,
        n_rows: int = None,
        n_features: int = None,
        categorical_features: Optional[dict] = None,
    ) -> np.ndarray:
        """Gaussian context, with valid integer codes written into any declared categorical column."""
        context = rng.normal(size=(n_rows or self.n_rows, n_features or self.n_features))
        for column_index, cardinality in (categorical_features or {}).items():
            context[:, column_index] = rng.integers(0, cardinality, size=len(context))
        return context

    @pytest.mark.parametrize("with_categorical", [False, True])
    @given(
        n_rows=st.integers(min_value=1, max_value=12),
        n_features=st.integers(min_value=1, max_value=8),
        embedding_dim=st.integers(min_value=1, max_value=6),
        hidden_dims=_HIDDEN_DIMS,
    )
    def test_embed_output_shape(
        self, with_categorical: bool, n_rows: int, n_features: int, embedding_dim: int, hidden_dims: List[int]
    ) -> None:
        """``embed`` maps ``(n_rows, n_features)`` to ``(n_rows, embedding_dim)``.

        The output width is unchanged by a one-hot expanded categorical column: expansion widens the
        *input* to the first layer only.
        """
        rng = np.random.default_rng(0)
        categorical_features = self.categorical_features if with_categorical else None
        bb = MLPBackbone.cold_start(
            n_features=n_features,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            random_seed=1,
            categorical_features=categorical_features,
        )
        x = self._context(rng, n_rows=n_rows, n_features=n_features, categorical_features=categorical_features)
        assert bb.embed(x).shape == (n_rows, embedding_dim)

    def test_embed_is_deterministic(self, rng: np.random.Generator) -> None:
        """``embed`` is a pure function: identical input yields identical output."""
        bb = MLPBackbone.cold_start(
            n_features=self.n_features, hidden_dims=self.hidden, embedding_dim=self.embedding_dim, random_seed=1
        )
        x = self._context(rng)
        np.testing.assert_array_equal(bb.embed(x), bb.embed(x))

    @pytest.mark.parametrize("with_categorical", [False, True])
    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "sigmoid"])
    @given(hidden_dims=_HIDDEN_DIMS)
    def test_serialization_round_trip(
        self, activation: str, with_categorical: bool, hidden_dims: List[int], rng: np.random.Generator
    ) -> None:
        """A backbone survives ``model_dump_json`` / ``model_validate`` with identical embeddings."""
        categorical_features = self.categorical_features if with_categorical else None
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=hidden_dims,
            embedding_dim=self.embedding_dim,
            activation=activation,
            random_seed=1,
            categorical_features=categorical_features,
        )
        restored = MLPBackbone.model_validate_json(bb.model_dump_json())
        x = self._context(rng, categorical_features=categorical_features)
        np.testing.assert_allclose(bb.embed(x), restored.embed(x))

    @pytest.mark.parametrize("knob_name, value_strategy", _BACKBONE_KNOBS)
    @given(data=st.data(), hidden_dims=_HIDDEN_DIMS)
    def test_training_knobs_survive_reset_and_serialization(
        self, knob_name: str, value_strategy: st.SearchStrategy, data: st.DataObject, hidden_dims: List[int]
    ) -> None:
        """A knob set at ``cold_start`` persists through ``reset`` and a JSON round-trip, at any value."""
        value = data.draw(value_strategy)
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=hidden_dims,
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
            **{knob_name: value},
        )
        assert getattr(bb, knob_name) == value
        assert getattr(bb.reset(), knob_name) == value
        assert getattr(MLPBackbone.model_validate_json(bb.model_dump_json()), knob_name) == value

    # ----------------------------------------------------------------- one-hot categorical expansion
    @given(cardinalities=st.lists(st.integers(min_value=2, max_value=5), min_size=1, max_size=3, unique=False))
    def test_one_hot_expansion_widens_first_layer_only(self, cardinalities: List[int]) -> None:
        """Each categorical column costs its ``cardinality`` in first-layer inputs, not one column.

        ``n_features`` keeps reporting the *raw* context width — the expansion is internal — while
        ``input_dim`` (and hence layer 0's fan-in) grows to ``n_numerical + sum(cardinality)``.
        """
        categorical_features = {index: cardinality for index, cardinality in enumerate(cardinalities)}
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=self.hidden,
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
            categorical_features=categorical_features,
        )
        expected = self.n_features - len(categorical_features) + sum(cardinalities)
        assert bb.n_features == self.n_features
        assert bb.input_dim == expected
        assert bb.weight_arrays[0].shape[0] == expected

    def test_one_hot_expansion_matches_between_numpy_and_jax(self, rng: np.random.Generator) -> None:
        """``embed`` (NumPy, predict path) and ``forward_jax`` (traced, training path) agree.

        Both call the same ``_expand``, so this cannot catch a bad column layout — one implementation
        cannot diverge from itself. What it does guard is that *both* call sites expand at all: the
        model must not train on an encoding it does not predict with. The residual it measures is the
        NumPy path's float64 context meeting float32 weights.
        """
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=self.hidden,
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
            categorical_features=self.categorical_features,
        )
        x = self._context(rng, categorical_features=self.categorical_features)
        weights_biases = [(jnp.asarray(w), jnp.asarray(b)) for w, b in zip(bb.weight_arrays, bb.bias_arrays)]
        from_jax = np.asarray(bb.forward_jax(jnp.asarray(x, dtype=jnp.float32), weights_biases))
        np.testing.assert_allclose(bb.embed(x), from_jax, atol=1e-5)

    def test_one_hot_expansion_is_not_ordinal(self, rng: np.random.Generator) -> None:
        """A categorical column is a lookup, not a magnitude: doubling the code is not doubling the input.

        Guards the whole point of the expansion — with the column consumed as a continuous value, the
        first layer's response to code ``2`` would be exactly twice its response to code ``1``.
        """
        column_index, cardinality = next(iter(self.categorical_features.items()))
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=[],  # no hidden layer: embed is the raw first-layer projection
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
            categorical_features=self.categorical_features,
        )
        codes = np.zeros((cardinality, self.n_features))
        codes[:, column_index] = np.arange(cardinality)
        embedded = bb.embed(codes)
        steps = np.diff(embedded, axis=0)
        # Consumed as a magnitude, every unit step in the code would move the projection identically.
        assert not np.allclose(steps, steps[0])

    def test_high_cardinality_warns_without_hidden_layers(self) -> None:
        """The high-cardinality warning reports layer 0's fan-out, which is the embedding when
        ``hidden_dims`` is empty (a legal single-layer backbone)."""
        cardinality = MLPBackbone._one_hot_warn_cardinality + 1
        with pytest.warns(UserWarning, match=f"{cardinality} x {self.embedding_dim}"):
            bb = MLPBackbone.cold_start(
                n_features=self.n_features,
                hidden_dims=[],
                embedding_dim=self.embedding_dim,
                random_seed=self.backbone_seed,
                categorical_features={0: cardinality},
            )
        assert bb.input_dim == self.n_features - 1 + cardinality

    def test_first_layer_width_mismatch_raises(self) -> None:
        """A state whose stored weights disagree with its categorical layout is rejected at construction.

        This is the deserialization guard: a backbone saved under one feature layout must not silently
        load against another and fail later inside the SVI pass.
        """
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=self.hidden,
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
            categorical_features=self.categorical_features,
        )
        state = bb.model_dump()
        state["categorical_features"] = {0: 5}  # widens the expected layer-0 fan-in by 2
        with pytest.raises(ValueError, match="First layer expects"):
            MLPBackbone.model_validate(state)


class TestCmabMetaModel:
    """The unified per-arm-heads (+ optional shared backbone) joint-VI meta-model."""

    n_features = 6
    hidden = [4]
    # Fixed (not hypothesis-drawn) architectures: the tests here run real SVI, so their cost has to stay
    # bounded — _HIDDEN_DIMS explores the architecture space in TestMLPBackbone, where nothing trains.
    backbone_hidden_dims = [8]
    embedding_dim = 4
    n_rows = 9
    action_ids: Set[ActionId] = {"a", "b", "c"}
    valid_subset: Set[ActionId] = {"a", "b"}
    absent_arm: ActionId = "c"
    n_objectives = 2
    quantity_dim = 1
    num_steps = 15
    backbone_seed = 7
    meta_seed = 13
    # joint-minibatch config: batch_size < minibatch_n engages the single-plate minibatch path.
    minibatch_size = 24
    minibatch_n = 160
    # MLPBackbone's l2_anchoring / lr disabled-state sentinels (the field defaults); other values are
    # hypothesis-generated per test.
    l2_anchoring_disabled = 0.0
    lr_freeze = 0.0
    # One raw-context column carrying integer category codes, for the backbone's one-hot expansion.
    categorical_features = {0: 3}
    # Smallest cardinality whose derived embedding_dim isn't 1 — derived from the rule itself (as
    # test_transfer.py's _cardinality_bands does) rather than a hardcoded literal, so a widened-head
    # test stays valid regardless of default_categorical_embedding_dim's exact formula.
    _wide_cat_cardinality = next(
        c for c in itertools.count(1) if BayesianNeuralNetwork.default_categorical_embedding_dim(c) != 1
    )

    # ----------------------------------------------------------------- helpers / fixtures
    def _head_cold_start_kwargs(self, n_features: int = None) -> dict:
        """Fresh per-arm-head cold-start kwargs (the cold-start machinery pops consumed keys)."""
        return {
            "n_features": n_features or self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "random_seed": self.meta_seed,
        }

    def _build_meta(
        self,
        with_backbone: bool,
        batch_size: Optional[int] = None,
        l2_anchoring: float = 0.0,
        lr: Optional[float] = None,
        categorical_features: Optional[dict] = None,
    ) -> CmabMetaModelSO:
        """Build a meta-model, optionally with a shared backbone and/or a minibatch ``batch_size``.

        The backbone's own knobs live on the ``MLPBackbone``, so they go to its ``cold_start`` here
        rather than into the meta-model's own ``kwargs``.
        """
        update_kwargs = {"num_steps": self.num_steps}
        if batch_size is not None:
            update_kwargs["batch_size"] = batch_size
        kwargs = {
            "n_features": self.embedding_dim if with_backbone else self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": update_kwargs,
            "random_seed": self.meta_seed,
        }
        if with_backbone:
            kwargs["backbone"] = MLPBackbone.cold_start(
                n_features=self.n_features,
                hidden_dims=self.backbone_hidden_dims,
                embedding_dim=self.embedding_dim,
                random_seed=self.backbone_seed,
                l2_anchoring=l2_anchoring,
                lr=lr,
                categorical_features=categorical_features,
            )
        return CmabMetaModelSO(action_ids=self.action_ids, kwargs=kwargs)

    def _n_rows(self, batch_size: Optional[int]) -> int:
        """Enough rows to trigger the minibatch path (batch_size < N) when set, else the small default."""
        return self.minibatch_n if batch_size is not None else self.n_rows

    def _context(self, rng: np.random.Generator, n_rows: int = None) -> np.ndarray:
        return rng.normal(size=(n_rows or self.n_rows, self.n_features))

    def _balanced_batch(self, rng: np.random.Generator, n_rows: int = None) -> tuple:
        n_rows = n_rows or self.n_rows
        arms = sorted(self.action_ids)
        actions = [arms[i % len(arms)] for i in range(n_rows)]
        rewards = rng.integers(0, 2, size=n_rows).tolist()
        return actions, rewards, self._context(rng, n_rows)

    def _head_mu(self, meta: CmabMetaModelSO, arm: ActionId) -> np.ndarray:
        return np.array(meta.actions[arm].model_params.bnn_layer_params[0].weight.mu)

    @pytest.fixture
    def meta_no_backbone(self) -> CmabMetaModelSO:
        """No-backbone unified meta-model over ``action_ids`` (BNN heads on raw context)."""
        return CmabMetaModelSO(action_ids=self.action_ids, kwargs=self._head_cold_start_kwargs())

    @pytest.fixture
    def meta_backbone(self) -> CmabMetaModelSO:
        """Shared-backbone unified meta-model (heads on the embedding)."""
        backbone = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=self.backbone_hidden_dims,
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
        )
        head_kwargs = self._head_cold_start_kwargs(n_features=self.embedding_dim)
        head_kwargs["hidden_dim_list"] = [3]
        head_kwargs["backbone"] = backbone
        return CmabMetaModelSO(action_ids=self.action_ids, kwargs=head_kwargs)

    # ----------------------------------------------------------------- construction / sampling
    def test_action_ids(self, meta_no_backbone: CmabMetaModelSO) -> None:
        """``action_ids`` exposes exactly the configured action set."""
        assert set(meta_no_backbone.action_ids) == self.action_ids

    def test_input_dim_reports_raw_features(
        self, meta_no_backbone: CmabMetaModelSO, meta_backbone: CmabMetaModelSO
    ) -> None:
        """``input_dim`` is the raw feature count on both paths (backbone hides the embedding width)."""
        assert meta_no_backbone.input_dim == self.n_features
        assert meta_backbone.input_dim == self.n_features

    def test_sample_proba_shape(self, meta_no_backbone: CmabMetaModelSO, rng: np.random.Generator) -> None:
        """``sample_proba`` returns one ``(prob, logit)`` list per (valid) arm, length == context rows."""
        context = self._context(rng)
        result = meta_no_backbone.sample_proba(rng=rng, context=context)
        assert set(result) == self.action_ids
        assert all(len(per_arm) == self.n_rows for per_arm in result.values())
        assert all(0.0 <= prob <= 1.0 for prob, _ in result["a"])
        restricted = meta_no_backbone.sample_proba(rng=rng, valid_action_ids=self.valid_subset, context=context)
        assert set(restricted) == self.valid_subset

    def test_sample_proba_requires_context(self, meta_no_backbone: CmabMetaModelSO, rng: np.random.Generator) -> None:
        """``sample_proba`` raises when no context is supplied (context is a required keyword-only arg)."""
        with pytest.raises(TypeError):
            meta_no_backbone.sample_proba(rng=rng)

    def test_update_wrong_feature_count_raises(
        self, meta_no_backbone: CmabMetaModelSO, rng: np.random.Generator
    ) -> None:
        """A context with the wrong number of feature columns raises against ``input_dim``."""
        bad = rng.normal(size=(2, self.n_features + 1))
        with pytest.raises(AttributeError, match="feature columns"):
            meta_no_backbone.update(actions=["a", "b"], rewards=[1, 0], context=bad)

    # ----------------------------------------------------------------- training
    @pytest.mark.parametrize("batch_size", [None, minibatch_size])
    def test_update_changes_heads(self, batch_size: Optional[int], rng: np.random.Generator) -> None:
        """A joint update moves at least one head's posterior (no backbone), full-batch and minibatched.

        ``batch_size`` set (< N) exercises the single global ``data`` plate + arm-indexing minibatch path.
        """
        meta = self._build_meta(with_backbone=False, batch_size=batch_size)
        before = {a: self._head_mu(meta, a).copy() for a in meta.action_ids}
        actions, rewards, context = self._balanced_batch(rng, n_rows=self._n_rows(batch_size))
        meta.update(actions=actions, rewards=rewards, context=context)
        assert any(not np.allclose(before[a], self._head_mu(meta, a)) for a in meta.action_ids)

    @pytest.mark.parametrize("batch_size", [None, minibatch_size])
    def test_backbone_update_changes_backbone_and_heads(
        self, batch_size: Optional[int], rng: np.random.Generator
    ) -> None:
        """A joint update trains both the shared backbone and the per-arm heads, full-batch and minibatched.

        With ``batch_size`` set the backbone runs on the minibatch (not all N) via the single ``data`` plate.
        """
        meta = self._build_meta(with_backbone=True, batch_size=batch_size)
        w_before = [w.copy() for w in meta.backbone.weight_arrays]
        mu_before = self._head_mu(meta, "a").copy()
        actions, rewards, context = self._balanced_batch(rng, n_rows=self._n_rows(batch_size))
        meta.update(actions=actions, rewards=rewards, context=context)
        assert any(not np.allclose(b, a) for b, a in zip(w_before, meta.backbone.weight_arrays))
        assert not np.allclose(mu_before, self._head_mu(meta, "a"))

    @settings(deadline=None, max_examples=5)
    @given(l2_anchoring=st.floats(min_value=1e4, max_value=1e8, allow_nan=False, allow_infinity=False))
    def test_backbone_l2_anchoring_reduces_drift(self, l2_anchoring: float) -> None:
        """A positive ``l2_anchoring`` keeps the backbone's weights *and* biases closer to their
        pre-update values (it anchors every parameter, biases included, unlike L2 weight decay).

        Locally-seeded ``rng``, not the session-scoped fixture: that fixture keeps advancing across
        examples, so a hypothesis replay would see different data and flag a spurious flaky failure.
        """
        rng = np.random.default_rng(0)
        actions, rewards, context = self._balanced_batch(rng)
        unanchored = self._build_meta(with_backbone=True, l2_anchoring=self.l2_anchoring_disabled)
        anchored = self._build_meta(with_backbone=True, l2_anchoring=l2_anchoring)
        w_initial = [w.copy() for w in unanchored.backbone.weight_arrays]
        b_initial = [b.copy() for b in unanchored.backbone.bias_arrays]
        unanchored.update(actions=actions, rewards=rewards, context=context)
        anchored.update(actions=actions, rewards=rewards, context=context)
        drift_unanchored_w = sum(np.sum((a - b) ** 2) for a, b in zip(w_initial, unanchored.backbone.weight_arrays))
        drift_anchored_w = sum(np.sum((a - b) ** 2) for a, b in zip(w_initial, anchored.backbone.weight_arrays))
        drift_unanchored_b = sum(np.sum((a - b) ** 2) for a, b in zip(b_initial, unanchored.backbone.bias_arrays))
        drift_anchored_b = sum(np.sum((a - b) ** 2) for a, b in zip(b_initial, anchored.backbone.bias_arrays))
        assert drift_anchored_w < drift_unanchored_w
        assert drift_anchored_b < drift_unanchored_b

    def test_backbone_lr_zero_freezes_backbone_exactly(self, rng: np.random.Generator) -> None:
        """``lr=0.0`` produces exactly zero backbone movement, while heads still train normally."""
        actions, rewards, context = self._balanced_batch(rng)
        meta = self._build_meta(with_backbone=True, lr=self.lr_freeze)
        w_before = [w.copy() for w in meta.backbone.weight_arrays]
        mu_before = self._head_mu(meta, "a").copy()
        meta.update(actions=actions, rewards=rewards, context=context)
        for before, after in zip(w_before, meta.backbone.weight_arrays):
            np.testing.assert_array_equal(before, after)
        assert not np.allclose(mu_before, self._head_mu(meta, "a"))

    @settings(deadline=None, max_examples=5)
    @given(
        slow_lr=st.floats(min_value=1e-4, max_value=1e-1, allow_nan=False, allow_infinity=False),
        speedup=st.floats(min_value=2.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_backbone_lr_scales_backbone_drift(self, slow_lr: float, speedup: float) -> None:
        """A smaller ``lr`` produces proportionally less backbone movement, same seed/data.

        Locally-seeded ``rng`` — see :meth:`test_backbone_l2_anchoring_reduces_drift`.
        """
        rng = np.random.default_rng(0)
        actions, rewards, context = self._balanced_batch(rng)
        fast = self._build_meta(with_backbone=True, lr=slow_lr * speedup)
        slow = self._build_meta(with_backbone=True, lr=slow_lr)
        w_initial = [w.copy() for w in fast.backbone.weight_arrays]
        fast.update(actions=actions, rewards=rewards, context=context)
        slow.update(actions=actions, rewards=rewards, context=context)
        drift_fast = sum(np.sum((a - b) ** 2) for a, b in zip(w_initial, fast.backbone.weight_arrays))
        drift_slow = sum(np.sum((a - b) ** 2) for a, b in zip(w_initial, slow.backbone.weight_arrays))
        assert drift_slow < drift_fast

    @settings(deadline=None, max_examples=5)
    @given(
        l2_anchoring=st.floats(min_value=1e4, max_value=1e8, allow_nan=False, allow_infinity=False),
        lr=st.floats(min_value=1e-4, max_value=1e-1, allow_nan=False, allow_infinity=False),
    )
    def test_backbone_l2_anchoring_and_lr_compose(self, l2_anchoring: float, lr: float) -> None:
        """Setting both backbone training knobs together doesn't crash and still trains everything.

        Locally-seeded ``rng`` — see :meth:`test_backbone_l2_anchoring_reduces_drift`.
        """
        rng = np.random.default_rng(0)
        actions, rewards, context = self._balanced_batch(rng)
        meta = self._build_meta(with_backbone=True, l2_anchoring=l2_anchoring, lr=lr)
        w_before = [w.copy() for w in meta.backbone.weight_arrays]
        mu_before = self._head_mu(meta, "a").copy()
        meta.update(actions=actions, rewards=rewards, context=context)
        assert any(not np.allclose(b, a) for b, a in zip(w_before, meta.backbone.weight_arrays))
        assert not np.allclose(mu_before, self._head_mu(meta, "a"))

    @pytest.mark.parametrize("knob_name, value_strategy", _BACKBONE_KNOBS)
    @given(data=st.data())
    def test_backbone_training_knobs_serialization_round_trip(
        self, knob_name: str, value_strategy: st.SearchStrategy, data: st.DataObject
    ) -> None:
        """Backbone training knobs (on the nested ``backbone``) survive a full ``CmabMetaModel`` JSON round
        trip, for any valid value."""
        value = data.draw(value_strategy)
        meta = self._build_meta(with_backbone=True, **{knob_name: value})
        restored = CmabMetaModelSO.model_validate_json(meta.model_dump_json())
        assert getattr(restored.backbone, knob_name) == value

    @pytest.mark.parametrize("knob_name, value_strategy", _BACKBONE_KNOBS)
    @given(data=st.data())
    def test_backbone_training_knobs_reach_backbone_via_cold_start_kwargs(
        self, knob_name: str, value_strategy: st.SearchStrategy, data: st.DataObject
    ) -> None:
        """``backbone_``-prefixed cold-start kwargs thread through onto the built ``MLPBackbone``, prefix
        stripped, for any valid value."""
        value = data.draw(value_strategy)
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "backbone_hidden_dims": self.backbone_hidden_dims,
            "backbone_embedding_dim": self.embedding_dim,
            f"backbone_{knob_name}": value,
        }
        meta = CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)
        assert getattr(meta.backbone, knob_name) == value

    def test_unprefixed_categorical_features_route_to_the_backbone(self) -> None:
        """A plain ``categorical_features=`` reaches the backbone when one is built, not the heads.

        ``column_index`` addresses raw context columns, and with a backbone only the backbone sees
        those — the heads sit on the embedding — so the same top-level argument has to mean "one-hot
        these raw columns" rather than landing on heads that no longer receive them.
        """
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "backbone_hidden_dims": self.backbone_hidden_dims,
            "backbone_embedding_dim": self.embedding_dim,
            "categorical_features": self.categorical_features,
        }
        meta = CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)
        assert meta.backbone.categorical_features == self.categorical_features
        assert meta.representative_bnn.feature_config.categorical_features_configs == []
        assert meta.representative_bnn.feature_config.n_features == self.embedding_dim
        assert meta.input_dim == self.n_features  # raw context width, expansion is internal

    def test_categorical_features_stay_on_the_heads_without_a_backbone(self) -> None:
        """The other half of the routing contract: with no backbone the heads keep their own embeddings.

        Nothing is re-routed, because the heads *are* what sees the raw context in that configuration.
        """
        kwargs = self._head_cold_start_kwargs()
        kwargs["categorical_features"] = self.categorical_features
        meta = CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)
        assert meta.backbone is None
        head_configs = meta.representative_bnn.feature_config.categorical_features_configs
        assert [(cfg.column_index, cfg.cardinality) for cfg in head_configs] == list(self.categorical_features.items())

    def test_categorical_features_given_both_ways_raises(self) -> None:
        """The prefixed and unprefixed spellings mean the same thing here, so asking for both is an error."""
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "backbone_hidden_dims": self.backbone_hidden_dims,
            "backbone_embedding_dim": self.embedding_dim,
            "categorical_features": self.categorical_features,
            "backbone_categorical_features": self.categorical_features,
        }
        with pytest.raises(TypeError, match="not both"):
            CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)

    def test_backbone_categorical_update_trains_and_validates_codes(self, rng: np.random.Generator) -> None:
        """A joint update over one-hot expanded context trains everything and rejects invalid codes.

        The update path runs the backbone under a JAX trace, where an out-of-range code would silently
        one-hot to an all-zero block instead of raising, so the check has to happen before the trace.
        """
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "backbone_hidden_dims": self.backbone_hidden_dims,
            "backbone_embedding_dim": self.embedding_dim,
            "backbone_random_seed": self.backbone_seed,
            "categorical_features": self.categorical_features,
        }
        meta = CmabMetaModelSO(action_ids=self.action_ids, kwargs=kwargs)
        actions, rewards, context = self._balanced_batch(rng)
        for column_index, cardinality in self.categorical_features.items():
            context[:, column_index] = rng.integers(0, cardinality, size=len(context))
        w_before = [w.copy() for w in meta.backbone.weight_arrays]
        mu_before = self._head_mu(meta, "a").copy()
        meta.update(actions=actions, rewards=rewards, context=context)
        assert any(not np.allclose(b, a) for b, a in zip(w_before, meta.backbone.weight_arrays))
        assert not np.allclose(mu_before, self._head_mu(meta, "a"))

        context[0, 0] = max(self.categorical_features.values())  # one past the largest valid code
        with pytest.raises(ValueError, match="out of range"):
            meta.update(actions=actions, rewards=rewards, context=context)

    @given(random_seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_backbone_inherits_meta_model_random_seed(self, random_seed: int) -> None:
        """The backbone falls back to the meta-model's ``random_seed``, so one top-level seed makes the
        whole model (backbone init included) reproducible; ``backbone_random_seed`` overrides it."""
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "backbone_hidden_dims": self.backbone_hidden_dims,
            "backbone_embedding_dim": self.embedding_dim,
            "random_seed": random_seed,
        }
        meta = CmabMetaModelSO(action_ids=self.valid_subset, kwargs=dict(kwargs))
        assert meta.backbone.random_seed == random_seed

        overridden = CmabMetaModelSO(
            action_ids=self.valid_subset, kwargs={**kwargs, "backbone_random_seed": self.backbone_seed}
        )
        assert overridden.backbone.random_seed == self.backbone_seed

    @pytest.mark.parametrize("knob_name, value_strategy", _BACKBONE_KNOBS)
    @given(data=st.data())
    def test_backbone_training_knobs_require_backbone_hidden_dims(
        self, knob_name: str, value_strategy: st.SearchStrategy, data: st.DataObject
    ) -> None:
        """A ``backbone_``-prefixed knob passed via cold-start kwargs without ``backbone_hidden_dims``
        raises, at any value — the same ``TypeError`` as ``backbone_embedding_dim``/``backbone_activation``.
        """
        value = data.draw(value_strategy)
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            f"backbone_{knob_name}": value,
        }
        with pytest.raises(TypeError, match="only apply with a backbone"):
            CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)

    def test_reset_restores_backbone(self, meta_backbone: CmabMetaModelSO, rng: np.random.Generator) -> None:
        """``reset`` restores the backbone (same seed) after an update."""
        w_initial = [w.copy() for w in meta_backbone.backbone.weight_arrays]
        actions, rewards, context = self._balanced_batch(rng)
        meta_backbone.update(actions=actions, rewards=rewards, context=context)
        meta_backbone.reset()
        for initial, restored in zip(w_initial, meta_backbone.backbone.weight_arrays):
            np.testing.assert_allclose(initial, restored)

    def test_update_leaves_absent_arm_untouched(
        self, meta_no_backbone: CmabMetaModelSO, rng: np.random.Generator
    ) -> None:
        """An arm absent from the update batch keeps its head posterior unchanged."""
        mu_before = self._head_mu(meta_no_backbone, self.absent_arm).copy()
        actions: List[ActionId] = sorted(self.valid_subset) * (self.n_rows // 2)
        rewards = rng.integers(0, 2, size=len(actions)).tolist()
        meta_no_backbone.update(actions=actions, rewards=rewards, context=self._context(rng, n_rows=len(actions)))
        np.testing.assert_array_equal(self._head_mu(meta_no_backbone, self.absent_arm), mu_before)

    @pytest.mark.parametrize("batch_size", [None, minibatch_size])
    @pytest.mark.parametrize("with_backbone", [False, True])
    def test_zero_lr_leaves_posteriors_and_backbone_unchanged(
        self, with_backbone: bool, batch_size: Optional[int], rng: np.random.Generator
    ) -> None:
        """With SGD step_size=0 the joint pass leaves every head's posterior and the backbone as-is.

        Mirrors ``test_advi_zero_lr_posterior_equals_prior``: mu_init > 0 forces ``init_to_value`` so the
        guide loc seeds exactly at the prior mu and stays there under zero-gradient steps; the shared
        backbone (``numpyro.param``) likewise receives zero updates. Also covers the minibatch path
        (``batch_size`` set), where the invariant must still hold despite noisy minibatch gradients.
        """
        update_kwargs = {
            "num_steps": self.num_steps,
            "method": "advi",
            "optimizer_type": "sgd",
            "optimizer_kwargs": {"step_size": 0.0},
        }
        if batch_size is not None:
            update_kwargs["batch_size"] = batch_size
        head_kwargs = {
            "n_features": self.embedding_dim if with_backbone else self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": update_kwargs,
            "dist_type": "normal",
            "dist_params_init": {"mu": 0.5, "sigma": 0.3},
            "random_seed": self.meta_seed,
        }
        if with_backbone:
            head_kwargs["backbone"] = MLPBackbone.cold_start(
                n_features=self.n_features,
                hidden_dims=self.backbone_hidden_dims,
                embedding_dim=self.embedding_dim,
                random_seed=self.backbone_seed,
            )
        meta = CmabMetaModelSO(action_ids=self.valid_subset, kwargs=head_kwargs)
        w_before = [w.copy() for w in meta.backbone.weight_arrays] if with_backbone else []

        arms = sorted(self.valid_subset)
        n = self._n_rows(batch_size)
        actions = [arms[i % len(arms)] for i in range(n)]
        rewards = rng.integers(0, 2, size=n).tolist()
        meta.update(actions=actions, rewards=rewards, context=self._context(rng, n_rows=n))

        atol = 0.5 * np.finfo(np.float32).eps * 100
        rtol = np.finfo(np.float32).eps * 100
        for arm in meta.action_ids:
            params = meta.actions[arm].model_params
            for prior_layer, post_layer in zip(params.bnn_layer_params_init, params.bnn_layer_params):
                np.testing.assert_allclose(post_layer.weight.params["mu"], prior_layer.weight.params["mu"], atol=atol)
                np.testing.assert_allclose(
                    post_layer.weight.params["sigma"], prior_layer.weight.params["sigma"], rtol=rtol
                )
        for before, after in zip(w_before, meta.backbone.weight_arrays if with_backbone else []):
            np.testing.assert_allclose(before, after)

    # ----------------------------------------------------------------- validation
    def test_requires_at_least_one_action(self) -> None:
        """An empty actions dict is rejected at construction."""
        with pytest.raises((ValueError, AttributeError), match="[Aa]t least one action"):
            CmabMetaModelSO(actions={})

    def test_rejects_non_advi_vi_method(self) -> None:
        """With a shared backbone, the joint engine is ADVI-only: a non-advi VI method is rejected.

        With no backbone there is nothing to train jointly (``update`` dispatches to each arm's own
        independent ``update``), so this restriction only applies when a backbone is configured.
        """
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"method": "fullrank_advi"},
            "backbone_hidden_dims": self.hidden,
            "backbone_embedding_dim": self.embedding_dim,
        }
        with pytest.raises(NotImplementedError, match="advi"):
            CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)

    def test_rejects_head_whose_first_layer_disagrees_with_the_embedding(self) -> None:
        """A head carrying categorical columns of its own is rejected at construction.

        Its first layer is wider than the ``embedding_dim`` the backbone feeds it (the categorical
        columns address raw context, which only the backbone sees), so the mismatch is a construction
        error rather than a shape blow-up inside the joint pass' forward pass.
        """
        head = BayesianNeuralNetwork.cold_start(
            n_features=self.embedding_dim,
            hidden_dim_list=self.hidden,
            categorical_features={0: self._wide_cat_cardinality},
            update_kwargs={"num_steps": self.num_steps},
        )
        assert head.feature_config.total_output_dim != self.embedding_dim
        backbone = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=self.backbone_hidden_dims,
            embedding_dim=self.embedding_dim,
            random_seed=self.backbone_seed,
        )
        with pytest.raises(AttributeError, match="first layer width"):
            CmabMetaModelSO(actions={arm: head for arm in self.valid_subset}, backbone=backbone)

    def test_no_backbone_allows_non_advi_vi_method(self) -> None:
        """With no shared backbone, a non-advi VI method is allowed (each arm trains independently)."""
        kwargs = {
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"method": "fullrank_advi"},
        }
        meta = CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)
        assert meta.backbone is None

    # ----------------------------------------------------------------- MO + quantitative heads
    @pytest.mark.parametrize("batch_size", [None, minibatch_size])
    def test_mo_update_trains_objectives(self, batch_size: Optional[int], rng: np.random.Generator) -> None:
        """MO heads train via the joint pass and sample MO tuples; ``batch_size`` is ignored (full-batch).

        Minibatching is single-objective only, so an MO head with ``batch_size`` set falls back to the
        full-batch path (``_heads_support_minibatching`` is False) and still trains.
        """
        update_kwargs = {"num_steps": self.num_steps}
        if batch_size is not None:
            update_kwargs["batch_size"] = batch_size
        kwargs = {
            "n_objectives": self.n_objectives,
            "n_features": self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": update_kwargs,
            "random_seed": self.meta_seed,
        }
        meta = CmabMetaModelMO(action_ids=self.valid_subset, kwargs=kwargs)
        n = self._n_rows(batch_size)
        assert meta._heads_support_minibatching is False  # MO always falls back to full-batch
        context = self._context(rng, n_rows=n)
        sp = meta.sample_proba(rng=rng, context=context)
        assert isinstance(sp["a"][0], list) and len(sp["a"][0]) == self.n_objectives
        mu0 = np.array(meta.actions["a"].models[0].model_params.bnn_layer_params[0].weight.mu).copy()
        arms = sorted(self.valid_subset)
        actions = [arms[i % len(arms)] for i in range(n)]
        rewards = rng.integers(0, 2, size=(n, self.n_objectives)).tolist()
        meta.update(actions=actions, rewards=rewards, context=context)
        assert not np.allclose(mu0, np.array(meta.actions["a"].models[0].model_params.bnn_layer_params[0].weight.mu))

    def test_mo_update_with_backbone_rejects_wrong_objective_count(self, rng: np.random.Generator) -> None:
        """With a shared backbone, the joint engine's own MO reward-shape check rejects a wrong shape.

        MO heads bypass ``BaseModelMO.update()``'s own check on this path (the joint engine never
        calls each head's ``update()``), so ``CmabMetaModel.update()`` must enforce it directly here —
        this covers that enforcement (``test_mo_update_trains_objectives`` above only exercises the
        no-backbone dispatch path, where each head's own check already applies).
        """
        kwargs = {
            "n_objectives": self.n_objectives,
            "n_features": self.embedding_dim,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "random_seed": self.meta_seed,
            "backbone": MLPBackbone.cold_start(
                n_features=self.n_features,
                hidden_dims=self.backbone_hidden_dims,
                embedding_dim=self.embedding_dim,
                random_seed=self.backbone_seed,
            ),
        }
        meta = CmabMetaModelMO(action_ids=self.valid_subset, kwargs=kwargs)
        arms = sorted(self.valid_subset)
        actions = [arms[i % len(arms)] for i in range(self.n_rows)]
        context = self._context(rng, n_rows=self.n_rows)
        wrong_rewards = rng.integers(0, 2, size=(self.n_rows, self.n_objectives + 1)).tolist()
        with pytest.raises(AttributeError, match="objectives"):
            meta.update(actions=actions, rewards=wrong_rewards, context=context)

    def test_quantitative_update_trains_inner_bnn(self, rng: np.random.Generator) -> None:
        """Quantitative heads ([quantity ‖ context] input) train via the joint pass; sample_proba is callable."""
        kwargs = {
            "dimension": self.quantity_dim,
            "n_features": self.n_features,
            "base_model_cold_start_kwargs": {
                "hidden_dim_list": self.hidden,
                "update_kwargs": {"num_steps": self.num_steps},
            },
            "random_seed": self.meta_seed,
        }
        meta = CmabMetaModelSO(quantitative_action_ids=self.valid_subset, kwargs=kwargs)
        context = self._context(rng, n_rows=8)
        sp = meta.sample_proba(rng=rng, context=context)
        assert callable(sp["a"][0][0])
        mu0 = np.array(meta.actions["a"].bnn.model_params.bnn_layer_params[0].weight.mu).copy()
        quantities = rng.random(8).tolist()
        meta.update(
            actions=["a", "b"] * 4, rewards=rng.integers(0, 2, size=8).tolist(), quantities=quantities, context=context
        )
        assert not np.allclose(mu0, np.array(meta.actions["a"].bnn.model_params.bnn_layer_params[0].weight.mu))
