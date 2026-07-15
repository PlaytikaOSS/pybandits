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

from typing import List, Optional, Set

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pybandits.base import ActionId
from pybandits.meta_model import CmabMetaModelMO, CmabMetaModelSO
from pybandits.model.bnn.backbone import MLPBackbone


class TestMLPBackbone:
    """The shared deterministic encoder: shape, determinism, serialization."""

    n_features = 6
    hidden = [4]
    embedding_dim = 4
    n_rows = 9
    backbone_seed = 1

    def _context(self, rng: np.random.Generator, n_rows: int = None, n_features: int = None) -> np.ndarray:
        return rng.normal(size=(n_rows or self.n_rows, n_features or self.n_features))

    @given(
        n_rows=st.integers(min_value=1, max_value=12),
        n_features=st.integers(min_value=1, max_value=8),
        embedding_dim=st.integers(min_value=1, max_value=6),
    )
    def test_embed_output_shape(self, n_rows: int, n_features: int, embedding_dim: int) -> None:
        """``embed`` maps ``(n_rows, n_features)`` to ``(n_rows, embedding_dim)``."""
        rng = np.random.default_rng(0)
        bb = MLPBackbone.cold_start(n_features=n_features, hidden_dims=[4], embedding_dim=embedding_dim, random_seed=1)
        assert bb.embed(rng.normal(size=(n_rows, n_features))).shape == (n_rows, embedding_dim)

    def test_embed_is_deterministic(self, rng: np.random.Generator) -> None:
        """``embed`` is a pure function: identical input yields identical output."""
        bb = MLPBackbone.cold_start(
            n_features=self.n_features, hidden_dims=self.hidden, embedding_dim=self.embedding_dim, random_seed=1
        )
        x = self._context(rng)
        np.testing.assert_array_equal(bb.embed(x), bb.embed(x))

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "sigmoid"])
    def test_serialization_round_trip(self, activation: str, rng: np.random.Generator) -> None:
        """A backbone survives ``model_dump_json`` / ``model_validate`` with identical embeddings."""
        bb = MLPBackbone.cold_start(
            n_features=self.n_features,
            hidden_dims=self.hidden,
            embedding_dim=self.embedding_dim,
            activation=activation,
            random_seed=1,
        )
        restored = MLPBackbone.model_validate_json(bb.model_dump_json())
        x = self._context(rng)
        np.testing.assert_allclose(bb.embed(x), restored.embed(x))


class TestCmabMetaModel:
    """The unified per-arm-heads (+ optional shared backbone) joint-VI meta-model."""

    n_features = 6
    hidden = [4]
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

    # ----------------------------------------------------------------- helpers / fixtures
    def _head_cold_start_kwargs(self, n_features: int = None) -> dict:
        """Fresh per-arm-head cold-start kwargs (the cold-start machinery pops consumed keys)."""
        return {
            "n_features": n_features or self.n_features,
            "hidden_dim_list": self.hidden,
            "update_kwargs": {"num_steps": self.num_steps},
            "random_seed": self.meta_seed,
        }

    def _build_meta(self, with_backbone: bool, batch_size: Optional[int] = None) -> CmabMetaModelSO:
        """Build a meta-model, optionally with a shared backbone and/or a minibatch ``batch_size``."""
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
                hidden_dims=[8],
                embedding_dim=self.embedding_dim,
                random_seed=self.backbone_seed,
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
            hidden_dims=[8],
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
                hidden_dims=[8],
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
            "embedding_dim": self.embedding_dim,
        }
        with pytest.raises(NotImplementedError, match="advi"):
            CmabMetaModelSO(action_ids=self.valid_subset, kwargs=kwargs)

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
                hidden_dims=[8],
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
