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
"""Unified contextual-bandit meta-model: per-arm BNN heads with an optional shared backbone.

One meta-model covers both cmab paths. With ``backbone=None`` each arm is an independent Bayesian
Neural Network; with a :class:`MLPBackbone` the arms share a deterministic encoder and own cheap
heads on top. Either way training is a **single joint SVI pass** (VI-only): the backbone weights
enter the NumPyro model as ``numpyro.param`` point estimates and each arm's head contributes its own
prefixed sample-sites (built via :meth:`BaseBayesianNeuralNetwork.emit_submodel`). With no backbone
the joint ELBO factorises across arms, so this is equivalent (full-batch) to independent per-arm VI.

Lives in its own module within the ``meta_model`` package so the heavy jax/numpyro import surface is
isolated to the cmab path.
"""

import functools
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, ClassVar, Generic, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from loguru import logger
from numpyro.distributions import Bernoulli as NumpyroBernoulli
from numpyro.infer import TraceMeanField_ELBO
from pydantic import ConfigDict, NonNegativeInt, PrivateAttr, model_validator

from pybandits.base import ActionId, BinaryReward
from pybandits.base_model import BaseModel
from pybandits.meta_model.base import BaseMetaModel, SampleProbaResult
from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BaseBayesianNeuralNetworkMO,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkDP,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
)
from pybandits.model.bnn._guide import ParameterizedScaleAutoNormal
from pybandits.model.bnn._svi import forward_layers, per_sample_linear, run_svi
from pybandits.model.bnn.backbone import MLPBackbone
from pybandits.quantitative_model import (
    BaseQuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetworkCC,
    QuantitativeBayesianNeuralNetworkDP,
)

CmabHeadType = TypeVar(
    "CmabHeadType",
    bound=BaseBayesianNeuralNetwork | BaseBayesianNeuralNetworkMO | BaseQuantitativeBayesianNeuralNetwork,
)


class CmabMetaModel(BaseMetaModel, Generic[CmabHeadType]):
    """Per-arm BNN heads + optional shared backbone, trained by one joint SVI pass (VI-only).

    Construction mirrors ``SmabMetaModel`` (``actions=`` or ``action_ids=`` + cold-start
    ``kwargs``) with an extra optional ``backbone``. The per-arm heads are full
    :class:`BayesianNeuralNetwork` instances (SO/CC/DP single-objective, or MO as a list of BNNs),
    so a head can itself be multi-layer. Generic over the head type so the cold-start machinery can
    resolve the concrete class from the ``actions`` field annotation (use the parameterized aliases
    at the bottom of this module).

    Head shapes
    -----------
    A "head" is one of three shapes, normalised by the ``_arm_units`` / ``_context_dim`` /
    ``representative_bnn`` helpers so the joint engine can treat them uniformly:

    * a single BNN (SO/CC/DP);
    * a multi-objective head — a list of per-objective BNNs (``BaseBayesianNeuralNetworkMO``);
    * a quantitative head — wraps a ``.bnn`` whose input is ``[quantity ‖ context]``.

    How the shared encoder is trained (anchoring penalty, backbone-only learning rate) is configured on
    the backbone itself — see :class:`~pybandits.model.bnn.backbone.MLPBackbone`.
    """

    actions: dict[ActionId, CmabHeadType]  # type: ignore[assignment]
    backbone: MLPBackbone | None = None
    random_seed: NonNegativeInt | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Divider between an arm's scope prefix and its per-head site names in the joint NumPyro model.
    # ``numpyro.handlers.scope`` prepends ``f"{prefix}{_scope_divider}"`` to every site it wraps.
    _scope_divider: ClassVar[str] = "/"
    _obj_prefix: ClassVar[str] = "obj"
    # tqdm progress-bar label for the joint SVI run.
    _svi_desc: ClassVar[str] = "cmab joint SVI"
    # Cold-start kwargs prefix marking a key as backbone-only: it is popped and forwarded, prefix
    # stripped, to ``MLPBackbone.cold_start`` (so a new backbone knob needs no change here).
    _backbone_kwargs_prefix: ClassVar[str] = "backbone_"

    _rng_key: Any = PrivateAttr(default=None)

    def __init__(
        self,
        actions: dict[ActionId, BaseModel] | None = None,
        action_ids: set[ActionId] | None = None,
        quantitative_action_ids: set[ActionId] | None = None,
        kwargs: dict[str, Any] | None = None,
        backbone: MLPBackbone | None = None,
        random_seed: NonNegativeInt | None = None,
    ):
        """Build from a pre-made ``actions`` dict or cold-start specs, plus an optional ``backbone``.

        The shared ``backbone`` / ``random_seed`` fields reach this constructor two ways: inside the
        ``kwargs`` bag on **cold start** (the backbone requested via ``backbone_hidden_dims`` plus any
        other ``backbone_``-prefixed knob — see ``_build_backbone_from_kwargs``; this is how
        ``cold_start`` reaches this class through the generic ``BaseMab.cold_start`` factory, so no
        bespoke cmab ``cold_start`` is needed), or as top-level keyword arguments on **pydantic
        (de)serialization**. Both are folded into the single bag here so ``BaseMetaModel.__init__``
        handles every field uniformly (no catch-all kwarg).
        """
        kwargs = kwargs or {}
        # Fold the explicitly-passed (de)serialization fields into the bag; ``setdefault`` yields to a
        # value already threaded through ``kwargs`` on the cold-start path.
        for field_name, value in (("backbone", backbone), ("random_seed", random_seed)):
            if value is not None:
                kwargs.setdefault(field_name, value)
        # Build the shared encoder (if requested) before delegating; this mutates ``kwargs`` in place —
        # popping the backbone knobs and rewriting ``n_features`` to the embedding width — so the per-arm
        # heads that ``BaseMetaModel.__init__`` cold-starts sit on the embedding. The built backbone is
        # stashed back into the bag as the ``backbone`` field for the base ``__init__`` to forward.
        if kwargs.get("backbone") is None:
            built = self._build_backbone_from_kwargs(kwargs)
            if built is not None:
                kwargs["backbone"] = built
        super().__init__(
            actions=actions,
            action_ids=action_ids,
            quantitative_action_ids=quantitative_action_ids,
            kwargs=kwargs,
        )

    @classmethod
    def _build_backbone_from_kwargs(cls, kwargs: dict[str, Any]) -> MLPBackbone | None:
        """Pop every ``backbone_``-prefixed key from ``kwargs`` and build the shared encoder (``None``
        if not requested).

        Mutates ``kwargs`` in place: each ``f"{_backbone_kwargs_prefix}{name}"`` key is popped and,
        prefix stripped, forwarded as ``name=`` to :meth:`MLPBackbone.cold_start` — so a new
        backbone-only knob needs a ``cold_start`` parameter there and nothing here. Keys not given keep
        ``MLPBackbone.cold_start``'s own defaults rather than duplicating them here, except
        ``random_seed``, which falls back to the meta-model's own so one top-level seed makes the whole
        model reproducible. When a backbone is built, ``n_features`` is rewritten to the embedding width
        so the per-arm heads sit on the embedding. Backbone-only knobs without ``hidden_dims`` (i.e. no
        backbone requested) raise, rather than being silently ignored.
        """
        prefix = cls._backbone_kwargs_prefix
        backbone_kwargs = {key[len(prefix) :]: kwargs.pop(key) for key in list(kwargs) if key.startswith(prefix)}
        if "hidden_dims" not in backbone_kwargs:
            if backbone_kwargs:
                set_params = sorted(f"{prefix}{name}" for name in backbone_kwargs)
                raise TypeError(f"{set_params} only apply with a backbone; pass backbone_hidden_dims=[...].")
            return None
        n_features = kwargs.get("n_features")
        if n_features is None:
            raise ValueError("n_features must be provided to build the shared backbone.")
        backbone_kwargs.setdefault("random_seed", kwargs.get("random_seed"))
        if backbone_kwargs.get("embedding_dim") is None:
            backbone_kwargs["embedding_dim"] = max(1, n_features // BaseBayesianNeuralNetwork.embedding_dim_divisor)
        backbone = MLPBackbone.cold_start(n_features=n_features, **backbone_kwargs)
        kwargs["n_features"] = backbone_kwargs["embedding_dim"]  # heads live on the embedding, not raw context
        return backbone

    def model_post_init(self, __context: Any) -> None:
        """Seed the JAX PRNG key used to drive the joint SVI pass."""
        self._rng_key = jax.random.PRNGKey(
            self.random_seed
            if self.random_seed is not None
            else int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        )

    @model_validator(mode="after")
    def _check_models(self) -> "CmabMetaModel":
        """Cross-arm consistency: every head must share input size and update kwargs.

        Adapted from the old cmab per-action meta-model's ``check_models``; the heads' input dim must
        also match the backbone's ``embedding_dim`` when a backbone is present, and the joint engine is
        ADVI-only, so that VI-method constraint is enforced here at construction time.

        Returns
        -------
        CmabMetaModel
            The validated instance.
        """
        heads = list(self.actions.values())
        first_dim = self._context_dim(heads[0])
        first_bnn = self._rep_bnn(heads[0])
        for head in heads[1:]:
            if first_dim != self._context_dim(head):
                raise AttributeError("All actions should have the same input size.")
            if first_bnn.update_kwargs != self._rep_bnn(head).update_kwargs:
                raise AttributeError("All actions should have the same update kwargs.")
        if self.backbone is not None and first_dim != self.backbone.embedding_dim:
            raise AttributeError(
                f"Head context dim ({first_dim}) must equal backbone.embedding_dim ({self.backbone.embedding_dim})."
            )
        # With no shared backbone there is nothing to train jointly (update() dispatches each arm's
        # head independently), so the joint engine's ADVI-only restriction does not apply.
        if self.backbone is not None and first_bnn.resolved_update_kwargs["method"] != "advi":
            raise NotImplementedError("Joint cmab training currently supports the 'advi' VI method only.")
        return self

    # --------------------------------------------------------------------- head-shape helpers
    @property
    def representative_bnn(self) -> BaseBayesianNeuralNetwork:
        """The first arm's underlying BNN, representative of the shared training config.

        Every head shares ``update_kwargs`` / optimizer / numerical eps (enforced by
        ``_check_models``), so any one BNN can drive the joint SVI setup. Returns the first arm's.

        Returns
        -------
        BaseBayesianNeuralNetwork
            The representative BNN.
        """
        return self._rep_bnn(next(iter(self.actions.values())))

    @property
    def input_dim(self) -> int:
        """Raw context columns the meta-model expects.

        The backbone's ``n_features`` when a shared encoder is present (heads then sit on the
        embedding), otherwise the per-arm head context dimension.

        Returns
        -------
        int
            Expected number of context feature columns.
        """
        if self.backbone is not None:
            return self.backbone.n_features
        return self._context_dim(next(iter(self.actions.values())))

    @classmethod
    def _rep_bnn(cls, head: BaseModel) -> BaseBayesianNeuralNetwork:
        """A representative underlying BNN (for update_kwargs / optimizer / eps).

        The first joint-model sub-unit's BNN; reuses ``_arm_units`` so the head-type dispatch
        lives in exactly one place.
        """
        return cls._arm_units(head)[0][0]

    @staticmethod
    def _context_dim(head: BaseModel) -> int:
        """Context columns the head consumes (must equal ``backbone.embedding_dim`` when present)."""
        if isinstance(head, BaseBayesianNeuralNetworkMO):
            return head.models[0].input_dim
        return cast(BaseBayesianNeuralNetwork, head).input_dim  # quantitative.input_dim excludes quantity cols

    @classmethod
    def _arm_units(cls, head: BaseModel) -> list[tuple[BaseBayesianNeuralNetwork, int | None, bool]]:
        """Decompose a head into joint-model sub-units.

        Returns ``(bnn, objective_index_or_None, is_quantitative)`` per trainable BNN: one quantitative
        unit, one per objective for MO, or a single unit otherwise.
        """
        if isinstance(head, BaseQuantitativeBayesianNeuralNetwork):
            return [(head.bnn, None, True)]
        if isinstance(head, BaseBayesianNeuralNetworkMO):
            return [(bnn, i, False) for i, bnn in enumerate(head.models)]
        return [(cast(BaseBayesianNeuralNetwork, head), None, False)]

    @classmethod
    def _unit_scope(cls, arm: ActionId, obj_index: int | None) -> str:
        """``numpyro.handlers.scope`` prefix isolating one trainable unit's sites in the joint trace.

        Per-arm for single-objective/quantitative heads; per-(arm, objective) for multi-objective.
        """
        return arm if obj_index is None else f"{arm}{cls._scope_divider}{cls._obj_prefix}{obj_index}"

    # --------------------------------------------------------------------- public meta-model API
    def sample_proba(
        self,
        rng: np.random.Generator,
        valid_action_ids: set[ActionId] | None = None,
        *,
        context: np.ndarray,
        **kwargs: Any,
    ) -> dict[ActionId, SampleProbaResult]:
        """Embed the context once (if a backbone is set), then sample each (valid) head on it."""
        # The actions manager already validated/cast the context; embed it once when a backbone is set,
        # otherwise pass it straight to the per-arm heads (which handle their own shape checks).
        z = self.backbone.embed(context) if self.backbone is not None else context
        return {
            action_id: model.sample_proba(context=z, rng=rng)
            for action_id, model in self.actions.items()
            if valid_action_ids is None or action_id in valid_action_ids
        }

    def update(
        self,
        actions: list[ActionId],
        rewards: list[BinaryReward] | list[list[BinaryReward]],
        context: np.ndarray,
        quantities: list[float | list[float] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run one joint SVI pass over the batch (VI-only), then drop it (no replay buffer).

        With no shared backbone there is nothing to train jointly: each arm's head is updated
        independently from only its own rows, exactly as it was before this meta-model existed —
        this is the path that must stay a faithful replica of the pre-PR per-arm-independent BNN,
        so that it remains a valid control arm for an NLB-vs-current-architecture live A/B test.
        """
        # Rows are aligned by the manager (_validate_update_params), which also cast context to float.
        if context.shape[1] != self.input_dim:
            raise AttributeError(f"context has {context.shape[1]} feature columns; expected {self.input_dim}.")

        if self.backbone is None:
            # Each arm's own update() runs (via _dispatch_per_action_update), so BaseModelMO.update()'s
            # own reward-shape check already applies — no need to duplicate it here.
            self._dispatch_per_action_update(actions=actions, rewards=rewards, quantities=quantities, context=context)
            return

        rewards_arr: np.ndarray = np.asarray(rewards)
        # The joint engine bypasses each head's own update() (and thus BaseModelMO's reward-shape
        # check), so multi-objective heads need that same 2-D-rewards validation enforced here instead.
        for action_id in set(actions):
            head = self.actions[action_id]
            if isinstance(head, BaseBayesianNeuralNetworkMO):
                n_obj = len(head.models)
                if rewards_arr.ndim != 2 or rewards_arr.shape[1] != n_obj:
                    raise AttributeError(
                        f"multi-objective action '{action_id}' expects rewards with {n_obj} objectives per row."
                    )

        arm_to_rows: dict[ActionId, list[int]] = defaultdict(list)
        for row_index, action_id in enumerate(actions):
            arm_to_rows[action_id].append(row_index)

        self._joint_svi_update(context, arm_to_rows, rewards_arr, quantities)
        self._increment_counters(arm_to_rows, rewards_arr)

    def reset(self) -> None:
        """Cold-start the backbone (same seed) and reset every head."""
        if self.backbone is not None:
            self.backbone = self.backbone.reset()
        for model in self.actions.values():
            model.reset()

    # --------------------------------------------------------------------- joint SVI engine
    def _joint_svi_update(
        self,
        context: np.ndarray,
        arm_to_rows: dict[ActionId, list[int]],
        rewards_arr: np.ndarray,
        quantities: list[float | list[float] | None] | None,
    ) -> None:
        """Train the (optional) backbone point estimates + all per-arm head posteriors in one SVI pass.

        All arms are optimised together against a single ELBO, via one of two NumPyro models depending
        on head shape:

        * single-BNN-per-arm heads (SO/CC/DP) use :meth:`_make_so_model` — one shared data plate spans
          every arm's rows pooled together, with each row's head params gathered by its arm index (the
          arm-indexing trick); minibatched or full-batch (``nullcontext``) makes no difference to this
          per-row computation.
        * multi-objective / quantitative heads (which the arm-indexing trick doesn't support — MO has
          multiple weight sets per arm, quantitative needs a per-row quantity concatenated) use
          :meth:`_full_batch_model` — a per-arm loop, one :meth:`BaseBayesianNeuralNetwork.emit_submodel`
          call per trainable unit, always full-batch.

        Either way, each head's sites are wrapped in ``numpyro.handlers.scope(prefix="{arm}/")``
        (``/objN`` for multi-objective) so the otherwise-identical per-head site names stay unique
        across arms in the shared trace. The joint guide (:meth:`_build_joint_guide`) seeds each arm's
        variational sites from that arm's current posterior, so with no backbone the ELBO factorises
        across arms and this reduces to independent per-arm VI; with a backbone the arms are coupled
        only through the shared embedding. After the SVI run the trained params are written back per
        arm (:meth:`_store_head`) and to the backbone (:meth:`_store_backbone`). Seeding from the
        current posterior each call is what makes the streaming (no replay buffer) update work: the
        prior for step *t+1* is the posterior of *t*.

        Parameters
        ----------
        context : np.ndarray of shape (n_samples, input_dim)
            Feature matrix for the batch (raw context, or backbone input).
        arm_to_rows : dict[ActionId, list[int]]
            Row indices of ``context`` belonging to each arm in the batch.
        rewards_arr : np.ndarray
            Rewards for the batch; 1-D for single-objective, 2-D ``(n_samples, n_objectives)`` for MO.
        quantities : list[float | list[float] | None] | None
            Per-sample quantities for quantitative heads; ``None`` for non-quantitative batches.
        """
        batch_arms = list(arm_to_rows)
        representative_bnn = self.representative_bnn

        x = jnp.asarray(context, dtype=jnp.float32)
        backbone_w, backbone_b, n_bb_layers = self._backbone_jax_params()
        n_samples = len(context)
        resolved_kwargs = representative_bnn.resolved_update_kwargs
        declare_backbone = functools.partial(self._declare_backbone, backbone_w, backbone_b, n_bb_layers)
        batch_size = resolved_kwargs.get("batch_size")

        if self._heads_support_minibatching:
            # One shared data plate for every arm's rows pooled together; _so_model subsamples it
            # (the arm-indexing trick) when batch_size < n_samples, else it's a full-batch nullcontext.
            action_index = jnp.asarray(self._build_action_index(arm_to_rows, batch_arms), dtype=jnp.int32)
            y_all = jnp.asarray(rewards_arr, dtype=jnp.int32)
            model = self._make_so_model(batch_arms, representative_bnn, batch_size, n_samples, declare_backbone)
            model_args: tuple[Any, ...] = (x, action_index, y_all)
        else:
            if batch_size is not None and batch_size < n_samples:
                logger.warning(
                    "Joint cMAB minibatching is supported for single-objective heads only; "
                    "multi-objective / quantitative heads train full-batch (batch_size ignored)."
                )
            arm_data = {
                arm: (jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(rewards_arr[rows], dtype=jnp.int32))
                for arm, rows in arm_to_rows.items()
            }
            # Per-arm quantity columns for quantitative heads (constants prepended to the head input);
            # follow the same row order as arm_data so they align with z[row_idx].
            arm_q = self._build_arm_quantities(arm_to_rows, quantities)
            model = functools.partial(self._full_batch_model, arm_q=arm_q, declare_backbone=declare_backbone)
            model_args = (x, arm_data)

        guide = self._build_joint_guide(model, batch_arms)
        epoch_factor_arrays = representative_bnn.build_kl_annealing_factors(
            representative_bnn.compute_epoch_steps(n_samples)
        )
        loss = TraceMeanField_ELBO(num_particles=resolved_kwargs["num_particles"])

        svi, params, _history, self._rng_key = run_svi(
            model=model,
            guide=guide,
            optimizer=self._get_joint_optimizer(representative_bnn, n_bb_layers),
            loss=loss,
            rng_key=self._rng_key,
            model_args=model_args,
            epoch_factor_arrays=epoch_factor_arrays,
            restore_best=resolved_kwargs.get("restore_best_svi_state", True),
            desc=self._svi_desc,
        )
        site_mu, site_sigma = representative_bnn.extract_advi_params(params)

        for arm in batch_arms:
            self._store_head(self.actions[arm], site_mu, site_sigma, arm)
        self._store_backbone(params, n_bb_layers)

    def _get_joint_optimizer(self, representative_bnn: BaseBayesianNeuralNetwork, n_bb_layers: int) -> Any:
        """The optimizer for the joint SVI pass.

        One optimizer for every parameter (backbone and heads alike) unless the shared backbone sets its
        own ``lr``, in which case an ``optax.multi_transform`` gives the backbone's params their own rate
        — everything else (optimizer type, remaining ``optimizer_kwargs``, lr schedule) stays shared with
        the heads. Gradient clipping is applied globally, *before* the per-group split, so the threshold
        keeps bounding the whole gradient's norm either way.

        ``backbone.lr=0.0`` uses ``optax.set_to_zero()`` rather than the adaptive optimizer at a zero
        rate, so the per-step update is architecturally zero whatever the optimizer's internals do.
        """
        if self.backbone is None or self.backbone.lr is None:
            return representative_bnn.obj_optimizer

        backbone_optimizer = (
            optax.set_to_zero()
            if not self.backbone.lr
            else representative_bnn.build_optax_optimizer(step_size=self.backbone.lr)
        )
        backbone_param_names: set[str] = {
            name for i in range(n_bb_layers) for name in self.backbone.get_layer_params_name(i)
        }

        def label_fn(params: dict[str, Any]) -> dict[str, str]:
            return {name: ("backbone" if name in backbone_param_names else "head") for name in params}

        return representative_bnn.clip_and_wrap_optimizer(
            optax.multi_transform(
                {"backbone": backbone_optimizer, "head": representative_bnn.build_optax_optimizer()}, label_fn
            )
        )

    def _declare_backbone(self, backbone_w: list, backbone_b: list, n_bb_layers: int) -> list:
        """Register the backbone's deterministic weights as ``numpyro.param`` (inside the trace).

        When ``backbone.l2_anchoring > 0``, also adds a ``numpyro.factor`` quadratic penalty pulling each
        layer's optimized weights *and* biases back toward ``backbone_w``/``backbone_b`` — the values at
        the start of this SVI run, i.e. the previous ``update()`` call's — to bound per-call drift (see
        :class:`~pybandits.model.bnn.backbone.MLPBackbone`). Declared outside the data plate, so the
        penalty is not scaled by the minibatch.
        """
        weights_biases = []
        for i in range(n_bb_layers):
            w_name, b_name = self.backbone.get_layer_params_name(i)
            w, b = numpyro.param(w_name, backbone_w[i]), numpyro.param(b_name, backbone_b[i])
            if self.backbone.l2_anchoring > 0:
                penalty = jnp.sum((w - backbone_w[i]) ** 2) + jnp.sum((b - backbone_b[i]) ** 2)
                numpyro.factor(f"backbone_anchor_{i}", -0.5 * self.backbone.l2_anchoring * penalty)
            weights_biases.append((w, b))
        return weights_biases

    def _full_batch_model(
        self,
        x_: jax.Array,
        arm_data_: dict[ActionId, Any],
        kl_annealing_factor: float = 1.0,
        *,
        arm_q: dict[ActionId, Any],
        declare_backbone: Callable[[], list],
    ) -> None:
        """The full-batch joint NumPyro model for MO/quantitative heads; bound by :meth:`_joint_svi_update`.

        Single-BNN-per-arm heads use :meth:`_so_model` instead (the arm-indexing trick doesn't extend
        to MO's multiple weight sets per arm or quantitative's per-row quantity concatenation).
        """
        z = self.backbone.forward_jax(x_, declare_backbone()) if self.backbone is not None else x_
        for arm, (row_idx, arm_rewards) in arm_data_.items():
            z_arm = z[row_idx]
            for bnn, obj_index, is_quantitative in self._arm_units(self.actions[arm]):
                x_unit = jnp.concatenate([arm_q[arm], z_arm], axis=1) if is_quantitative else z_arm
                y_unit = arm_rewards if obj_index is None else arm_rewards[:, obj_index]
                # scope() prepends "{arm}/" (and "/objN" for MO) to every site this unit emits, so
                # the otherwise-identical per-head site names stay unique across arms.
                with numpyro.handlers.scope(prefix=self._unit_scope(arm, obj_index), divider=self._scope_divider):
                    bnn.emit_submodel(x=x_unit, y=y_unit, kl_annealing_factor=cast(Any, kl_annealing_factor))

    @property
    def _heads_support_minibatching(self) -> bool:
        """Whether every head is a single-BNN head (the arm-indexing trick needs one BNN per arm).

        Multi-objective and quantitative heads are excluded (they keep the full-batch joint pass);
        whether minibatching is actually *requested* (``batch_size`` vs. the batch size) is the
        caller's concern, not this structural head-shape check.
        """
        return not any(
            isinstance(head, (BaseBayesianNeuralNetworkMO, BaseQuantitativeBayesianNeuralNetwork))
            for head in self.actions.values()
        )

    @staticmethod
    def _build_action_index(arm_to_rows: dict[ActionId, list[int]], batch_arms: list[ActionId]) -> np.ndarray:
        """Map each row to its arm's position in ``batch_arms`` (the arm-indexing trick's lookup).

        Parameters
        ----------
        arm_to_rows : dict[ActionId, list[int]]
            Row indices belonging to each arm.
        batch_arms : list[ActionId]
            Fixed arm order; the stacked per-arm head params follow this order.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            ``action_index[row]`` is the arm's index into the stacked head params.
        """
        n_samples = sum(len(rows) for rows in arm_to_rows.values())
        action_index = np.empty(n_samples, dtype=np.int32)
        for position, arm in enumerate(batch_arms):
            for row in arm_to_rows[arm]:
                action_index[row] = position
        return action_index

    def _make_so_model(
        self,
        batch_arms: list[ActionId],
        representative_bnn: BaseBayesianNeuralNetwork,
        batch_size: int | None,
        n_samples: int,
        declare_backbone: Callable[[], list],
    ) -> Callable:
        """Build the joint model for single-BNN-per-arm heads: arm-indexing + one shared data plate.

        Each arm's head sites are sampled globally (scoped) and stacked across arms into
        ``(num_arms, ...)`` tensors. A single data plate spans every arm's rows pooled together —
        a real ``numpyro.plate("data", size=N, subsample_size=B)`` when ``batch_size < N``, else a
        ``nullcontext`` (mirroring :meth:`BaseBayesianNeuralNetwork._observe`'s full-batch/minibatch
        split), so the exact same per-row computation serves both cases. Per row, the backbone embeds
        the row (or its raw context, no backbone), each row's head params are gathered by its arm
        index (``stacked[action]``), and one ``out`` Bernoulli is observed. Because the latent sites
        are the same scoped per-arm sites either way, the guide and posterior readback are unchanged.

        Returns a closure over ``self``/``batch_arms``/``representative_bnn``/``batch_size``/
        ``n_samples``/``declare_backbone`` (see :meth:`_so_model`); binding it here (rather than
        threading these as extra call args) keeps the NumPyro model's call signature to just the
        trace variables ``(x, action_index, y, kl_annealing_factor)`` that ``run_svi`` expects.

        Parameters
        ----------
        batch_arms : list[ActionId]
            Fixed arm order (matches ``_build_action_index``).
        representative_bnn : BaseBayesianNeuralNetwork
            Any head; supplies the shared activation / residual flag / feature config.
        batch_size : int | None
            Minibatch size (subsample from ``n_samples``); ``None`` (or ``>= n_samples``) is full-batch.
        n_samples : int
            Total rows (the data plate's ``size``).
        declare_backbone : Callable[[], list]
            Registers the backbone ``numpyro.param`` sites and returns the per-layer ``(w, b)``.

        Returns
        -------
        Callable
            The NumPyro model ``(x, action_index, y, kl_annealing_factor) -> None``.
        """
        return functools.partial(
            self._so_model,
            batch_arms=batch_arms,
            representative_bnn=representative_bnn,
            batch_size=batch_size,
            n_samples=n_samples,
            declare_backbone=declare_backbone,
        )

    def _so_model(
        self,
        x_: jax.Array,
        action_index_: jax.Array,
        y_: jax.Array,
        kl_annealing_factor: float = 1.0,
        *,
        batch_arms: list[ActionId],
        representative_bnn: BaseBayesianNeuralNetwork,
        batch_size: int | None,
        n_samples: int,
        declare_backbone: Callable[[], list],
    ) -> None:
        """The single-BNN-per-arm joint NumPyro model; built and bound by :meth:`_make_so_model`."""
        backbone_wb = declare_backbone() if self.backbone is not None else None
        per_arm = []
        for arm in batch_arms:
            ((bnn, obj_index, _q),) = self._arm_units(self.actions[arm])
            with numpyro.handlers.scope(prefix=self._unit_scope(arm, obj_index), divider=self._scope_divider):
                per_arm.append(bnn.sample_head_sites(cast(Any, kl_annealing_factor)))
        stacked_wb, stacked_emb = self._stack_head_sites(per_arm)

        use_minibatch = batch_size is not None and batch_size < n_samples
        plate_ctx = numpyro.plate("data", size=n_samples, subsample_size=batch_size) if use_minibatch else nullcontext()
        with plate_ctx as idx:
            ctx = x_[idx] if idx is not None else x_
            arm_of_row = action_index_[idx] if idx is not None else action_index_
            y_batch = y_[idx] if idx is not None else y_
            if self.backbone is not None:
                head_input = self.backbone.forward_jax(ctx, backbone_wb)
            else:
                head_input = self._per_sample_head_input(ctx, arm_of_row, stacked_emb, representative_bnn)
            per_sample_wb = [(w[arm_of_row], b[arm_of_row]) for w, b in stacked_wb]
            logit = forward_layers(
                next_layer_input=head_input,
                weights_biases=per_sample_wb,
                activation_fn=representative_bnn._jax_activation_fn,
                linear_fn=per_sample_linear,
                backend=jnp,
                use_residual_connections=representative_bnn.use_residual_connections,
            ).squeeze(-1)
            numpyro.sample("out", NumpyroBernoulli(logits=logit), obs=y_batch)

    @staticmethod
    def _stack_head_sites(per_arm: list[tuple[list, list]]) -> tuple[list, list]:
        """Stack per-arm sampled head sites across arms for the arm-indexing gather.

        Parameters
        ----------
        per_arm : list[tuple[list, list]]
            One ``(weights_biases, embedding_matrices)`` per arm (in ``batch_arms`` order).

        Returns
        -------
        tuple[list, list]
            ``(stacked_wb, stacked_emb)`` where each layer's ``(w, b)`` is stacked to
            ``(num_arms, in, out)`` / ``(num_arms, out)`` and each categorical embedding to
            ``(num_arms, cardinality, emb_dim)``.
        """
        n_layers = len(per_arm[0][0])
        stacked_wb = [
            (
                jnp.stack([per_arm[a][0][layer][0] for a in range(len(per_arm))]),
                jnp.stack([per_arm[a][0][layer][1] for a in range(len(per_arm))]),
            )
            for layer in range(n_layers)
        ]
        n_emb = len(per_arm[0][1])
        stacked_emb = [jnp.stack([per_arm[a][1][e] for a in range(len(per_arm))]) for e in range(n_emb)]
        return stacked_wb, stacked_emb

    @staticmethod
    def _per_sample_head_input(
        ctx: jax.Array, arm_of_row: jax.Array, stacked_emb: list, bnn: BaseBayesianNeuralNetwork
    ) -> jax.Array:
        """Build the no-backbone head input for a minibatch: numerical cols + per-sample categorical embeddings.

        For categorical features the per-row embedding is gathered from *that row's arm* head:
        ``stacked_emb[c][arm_of_row, category_value]``.

        Parameters
        ----------
        ctx : jax.Array of shape (B, n_features)
            Raw context minibatch.
        arm_of_row : jax.Array of shape (B,)
            Arm index per row (into the stacked embeddings' first axis).
        stacked_emb : list
            Per-categorical stacked embeddings ``(num_arms, cardinality, emb_dim)``; empty if none.
        bnn : BaseBayesianNeuralNetwork
            Representative head, for its ``feature_config``.

        Returns
        -------
        jax.Array
            The head input (``ctx`` itself when the head has no categorical embeddings).
        """
        if not stacked_emb:
            return ctx
        fc = bnn.feature_config
        parts = []
        if fc.numerical_indices:
            parts.append(ctx[:, fc.numerical_indices])
        for i, cfg in enumerate(fc.categorical_features_configs):
            cat_vals = ctx[:, cfg.column_index].astype(jnp.int32)
            parts.append(stacked_emb[i][arm_of_row, cat_vals])
        return jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _build_joint_guide(self, model: Callable, batch_arms: list[ActionId]) -> ParameterizedScaleAutoNormal:
        """Build the joint ADVI guide, seeding each unit's sites under its scope prefix.

        Each head contributes namespace-free guide-init arrays (means/sigmas from its current
        posterior); they are re-keyed under the same ``{arm}/`` (``/objN``) scope the model emits so the
        auto-guide initialises the matching sites — this is what carries last step's posterior into the
        next step's prior.

        Parameters
        ----------
        model : Callable
            The joint NumPyro model the guide wraps.
        batch_arms : list[ActionId]
            Arms present in this batch (each contributes its head's sites).

        Returns
        -------
        ParameterizedScaleAutoNormal
            The initialised joint guide.
        """
        values, site_sigmas, all_mus, all_sigmas = {}, {}, [], []
        for arm in batch_arms:
            for bnn, obj_index, _q in self._arm_units(self.actions[arm]):
                scope = f"{self._unit_scope(arm, obj_index)}{self._scope_divider}"
                v, s, mu, sg = bnn.collect_guide_init_arrays()
                values.update({f"{scope}{name}": val for name, val in v.items()})
                site_sigmas.update({f"{scope}{name}": val for name, val in s.items()})
                all_mus += mu
                all_sigmas += sg
        init_loc_fn, init_scale_fn = self.representative_bnn.build_guide_init_fns(
            values, site_sigmas, all_mus, all_sigmas
        )
        return ParameterizedScaleAutoNormal(model, init_loc_fn=init_loc_fn, init_scale_fn=init_scale_fn)

    def _backbone_jax_params(self) -> tuple[list, list, int]:
        """JAX copies of the backbone's deterministic weights/biases (empty when no backbone).

        Returns
        -------
        tuple[list, list, int]
            ``(weights, biases, n_layers)`` as jnp arrays; ``([], [], 0)`` with no backbone.
        """
        if self.backbone is None:
            return [], [], 0
        w = [jnp.asarray(arr, dtype=jnp.float32) for arr in self.backbone.weight_arrays]
        b = [jnp.asarray(arr, dtype=jnp.float32) for arr in self.backbone.bias_arrays]
        return w, b, len(w)

    def _build_arm_quantities(
        self, arm_to_rows: dict[ActionId, list[int]], quantities: list[float | list[float] | None] | None
    ) -> dict[ActionId, Any]:
        """Per-arm quantity matrices ``(n_arm_rows, dimension)`` for quantitative heads (jnp), else {}.

        Parameters
        ----------
        arm_to_rows : dict[ActionId, list[int]]
            Row indices of the batch belonging to each arm.
        quantities : list[float | list[float] | None] | None
            Per-sample quantities; must be present for every row of a quantitative arm.

        Returns
        -------
        dict[ActionId, Any]
            Quantity matrix per quantitative arm; non-quantitative arms are omitted.
        """
        arm_q: dict[ActionId, Any] = {}
        for arm, rows in arm_to_rows.items():
            head = self.actions[arm]
            if not isinstance(head, BaseQuantitativeBayesianNeuralNetwork):
                continue
            if quantities is None or any(quantities[i] is None for i in rows):
                raise ValueError(f"Quantitative action '{arm}' requires a quantity for every observation.")
            q = np.asarray([quantities[i] for i in rows], dtype=float).reshape(len(rows), head.dimension)
            arm_q[arm] = jnp.asarray(q, dtype=jnp.float32)
        return arm_q

    def _store_head(self, head: BaseModel, site_mu: dict, site_sigma: dict, arm: ActionId) -> None:
        """Write back a head's per-arm posterior (per objective for MO) from the joint SVI params.

        Parameters
        ----------
        head : BaseModel
            The arm's head to update in place.
        site_mu : dict
            Posterior means keyed by scoped site name (across all arms).
        site_sigma : dict
            Posterior stds keyed by scoped site name (across all arms).
        arm : ActionId
            The arm whose scoped sites to slice out of ``site_mu`` / ``site_sigma``.
        """
        for bnn, obj_index, _q in self._arm_units(head):
            # Slice this unit's scoped sites back to namespace-free names for the prefix-free readback.
            scope = f"{self._unit_scope(arm, obj_index)}{self._scope_divider}"
            unit_mu = {name[len(scope) :]: v for name, v in site_mu.items() if name.startswith(scope)}
            unit_sigma = {name[len(scope) :]: v for name, v in site_sigma.items() if name.startswith(scope)}
            bnn.model_params.bnn_layer_params = bnn.layer_params_from_posterior(unit_mu, unit_sigma)
            bnn.update_embedding_params_from_vi(unit_mu, unit_sigma)

    def _store_backbone(self, params: dict, n_bb_layers: int) -> None:
        """Write back the trained backbone point estimates (no-op when no backbone).

        Parameters
        ----------
        params : dict
            Fitted variational params from the SVI run (includes the backbone ``numpyro.param`` sites).
        n_bb_layers : int
            Number of backbone layers to read back.
        """
        if self.backbone is None:
            return
        layer_names = [self.backbone.get_layer_params_name(i) for i in range(n_bb_layers)]
        new_w = [np.asarray(params[wn]) for wn, _ in layer_names]
        new_b = [np.asarray(params[bn]) for _, bn in layer_names]
        self.backbone = self.backbone.with_weights_and_biases(new_w, new_b)

    def _increment_counters(self, arm_to_rows: dict[ActionId, list[int]], rewards_arr: np.ndarray) -> None:
        """Keep each head's success/failure counters in sync (the joint engine bypasses per-head ``update``).

        These counters feed the manager's adaptive-window stats. The joint SVI pass trains every arm at
        once instead of calling each head's ``update``, so we replay the rewards through the canonical
        :meth:`BaseModelSO.record_rewards` bookkeeping: the success/failure owner is the per-objective
        sub-model for MO heads and the head itself (a ``BaseModelSO``) for single-objective/quantitative.

        Parameters
        ----------
        arm_to_rows : dict[ActionId, list[int]]
            Row indices of the batch belonging to each arm.
        rewards_arr : np.ndarray
            Rewards for the batch (1-D single-objective, 2-D ``(n_samples, n_objectives)`` for MO).
        """
        for arm, rows in arm_to_rows.items():
            head = self.actions[arm]
            if isinstance(head, BaseBayesianNeuralNetworkMO):
                arm_rewards = rewards_arr[rows]
                for i, sub in enumerate(head.models):
                    sub.record_rewards(list(arm_rewards[:, i]))
            else:  # BNN or quantitative head — both BaseModelSO that own their counters
                head.record_rewards(list(np.asarray(rewards_arr[rows]).reshape(-1)))


# Module-level aliases for concrete parameterisations (required for pickling; see meta_model.py).
CmabMetaModelSO = CmabMetaModel[BayesianNeuralNetwork | QuantitativeBayesianNeuralNetwork]
CmabMetaModelCC = CmabMetaModel[BayesianNeuralNetworkCC | QuantitativeBayesianNeuralNetworkCC]
CmabMetaModelDP = CmabMetaModel[BayesianNeuralNetworkDP | QuantitativeBayesianNeuralNetworkDP]
CmabMetaModelMO = CmabMetaModel[BayesianNeuralNetworkMO]
CmabMetaModelMOCC = CmabMetaModel[BayesianNeuralNetworkMOCC]
