# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
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

"""Abstract meta-model base for action managers.

A meta-model sits between an actions manager (e.g. ``CmabActionsManager``)
and the per-action models. The manager delegates ``sample_proba``, ``update``,
and ``reset`` to the meta-model. The meta-model decides whether per-action
calls are independent (simple dispatch loop) or require coordinated state
(e.g. a shared learned representation).

This module holds only ``BaseMetaModel`` (the abstract base) plus the shared
construction/validation machinery. Concrete implementations live alongside it in
the ``meta_model`` package: ``SmabMetaModel`` (independent per-action
dispatch) in ``smab_meta_model.py`` and ``CmabMetaModel`` (per-arm heads +
optional shared backbone) in ``cmab_meta_model.py``.
"""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

import numpy as np
from pydantic import ConfigDict, field_validator

from pybandits.base import (
    ACTION_IDS_PREFIX,
    QUANTITATIVE_ACTION_IDS_PREFIX,
    ActionId,
    BinaryReward,
    MOProbability,
    MOProbabilityWeight,
    Probability,
    ProbabilityWeight,
    PyBanditsBaseModel,
    QuantitativeMOProbability,
    QuantitativeMOProbabilityWeight,
    QuantitativeProbability,
    QuantitativeProbabilityWeight,
)
from pybandits.base_model import BaseModel
from pybandits.model import Model, ModelMO
from pybandits.quantitative_model import QuantitativeModel
from pybandits.utils import classproperty, extract_argument_names_from_function

# Possible return shapes from a per-action sample_proba call.
SampleProbaResult = (
    list[Probability]
    | list[MOProbability]
    | list[ProbabilityWeight]
    | list[MOProbabilityWeight]
    | list[QuantitativeProbability]
    | list[QuantitativeMOProbability]
    | list[QuantitativeProbabilityWeight]
    | list[QuantitativeMOProbabilityWeight]
)

ActionModelType = TypeVar("ActionModelType", bound=BaseModel)


class BaseMetaModel(PyBanditsBaseModel, ABC):
    """Abstract base for objects that own per-action state and dispatch logic.

    The manager class (e.g. ``CmabActionsManager``) delegates ``sample_proba``,
    ``update``, and ``reset`` to a meta-model instance. The meta-model owns the
    per-action models (exposed via the ``actions`` field) and decides whether
    they are dispatched independently or with coordinated shared state.

    Subclasses may narrow the value type of ``actions`` (e.g.
    ``dict[ActionId, BayesianNeuralNetwork]``); Pydantic preserves identity of
    nested model instances, so mutations via ``update()`` remain visible to
    holders of the original references.
    """

    actions: dict[ActionId, BaseModel]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("actions", mode="after")
    @classmethod
    def _at_least_one_action_is_defined(cls, v: dict[ActionId, BaseModel]) -> dict[ActionId, BaseModel]:
        cls._check_action_count(v)
        return v

    def __init__(
        self,
        actions: dict[ActionId, BaseModel] | None = None,
        action_ids: set[ActionId] | None = None,
        quantitative_action_ids: set[ActionId] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        """Construct from a pre-built ``actions`` dict or cold-start specs.

        Everything travels in the single ``kwargs`` bag: the per-action model arguments *and* this
        meta-model's own extra fields (e.g. the cmab ``backbone`` / ``random_seed``). That bag is the
        same dict ``BaseMab`` threads through for unknown-argument detection, so it is mutated in place —
        ``_instantiate_actions`` pops the per-action model keys it consumes. The meta-model's own fields
        are pulled out of the bag with :func:`extract_argument_names_from_function` and forwarded to
        Pydantic; a field that doubles as a per-action cold-start key (``random_seed``) is snapshotted
        *before* the actions are built, and a meta-only field (``backbone``) is then popped so it
        does not linger as an apparent leftover for ``BaseMab``'s check.

        Subclasses whose fields also arrive as top-level kwargs on Pydantic (de)serialization simply
        fold those into the bag before delegating here (see :meth:`CmabMetaModel.__init__`), so this
        base needs no catch-all keyword argument.
        """
        kwargs = kwargs or {}
        field_names = [f for f in extract_argument_names_from_function(type(self)) if f != "actions"]
        field_kwargs = {k: kwargs[k] for k in field_names if k in kwargs}
        actions_dict = self._instantiate_actions(
            actions=actions,
            action_ids=action_ids,
            quantitative_action_ids=quantitative_action_ids,
            kwargs=kwargs,
        )
        for field in field_names:
            kwargs.pop(field, None)
        super().__init__(actions=actions_dict, **field_kwargs)

    @property
    def action_ids(self) -> list[ActionId]:
        """Action identifiers covered by this meta-model."""
        return list(self.actions.keys())

    @abstractmethod
    def sample_proba(
        self,
        rng: np.random.Generator,
        valid_action_ids: set[ActionId] | None = None,
        **kwargs: Any,
    ) -> dict[ActionId, SampleProbaResult]:
        """Sample per-action probabilities/scores under the current context.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central random generator from the bandit (for reproducibility).
        valid_action_ids : set[ActionId] | None
            If provided, restrict sampling to these action ids (the bandit's
            allowed actions after removing ``forbidden_actions``). When
            ``None`` all action ids are sampled. Meta-models with shared
            state may use this to skip unnecessary head/branch evaluations.
        **kwargs
            Additional sampling inputs. For contextual bandits this includes
            ``context: np.ndarray``.

        Returns
        -------
        dict[ActionId, SampleProbaResult]
            Per-action probability/score collection in whatever shape the
            underlying model returns from its ``sample_proba``.
        """

    @abstractmethod
    def update(
        self,
        actions: list[ActionId],
        rewards: list[BinaryReward] | list[list[BinaryReward]],
        quantities: list[float | list[float] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update per-action state from a batch of (action, reward, ...) tuples.

        Parameters
        ----------
        actions : list[ActionId]
            Selected action per sample.
        rewards : list[BinaryReward] | list[list[BinaryReward]]
            Reward per sample (scalar for single-objective; list per sample
            for multi-objective).
        quantities : list[float | list[float] | None] | None
            Per-sample quantity for quantitative actions; ``None`` for
            non-quantitative actions in the same batch.
        **kwargs
            Additional update inputs. For contextual bandits this includes
            ``context: np.ndarray``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset every per-action model to its cold-start state."""

    @staticmethod
    def _check_action_count(actions: dict[ActionId, BaseModel]) -> None:
        """Shared action-count guard: at least one action; a single action is deterministic."""
        if len(actions) == 0:
            raise AttributeError("At least one action should be defined.")
        if len(actions) == 1:
            warnings.warn("Only a single action was supplied. This MAB will be deterministic.")

    def _dispatch_per_action_update(
        self,
        actions: list[ActionId],
        rewards: list[BinaryReward] | list[list[BinaryReward]],
        quantities: list[float | list[float] | None] | None = None,
        **row_aligned_kwargs: Any,
    ) -> None:
        """Group rows by action and dispatch to each action's own ``update`` — independent per-action training.

        Shared by ``SmabMetaModel`` (always) and ``CmabMetaModel`` (when there is no shared backbone,
        so there is nothing to train jointly): each action's model is updated in isolation, from only
        that action's own rows, exactly as if it were the sole action in the batch.

        Parameters
        ----------
        actions : list[ActionId]
            The selected action per sample.
        rewards : list[BinaryReward] | list[list[BinaryReward]]
            The reward per sample (or per-objective list of rewards per sample).
        quantities : list[float | list[float] | None] | None
            Per-sample quantity for quantitative actions; ``None`` for non-quantitative actions.
        **row_aligned_kwargs : Any
            Additional per-sample data, row-aligned with ``actions``, forwarded (sliced to that
            action's own rows) to each action's ``update`` — e.g. ``context=...`` for
            ``CmabMetaModel``. ``smab`` passes none of these, so this base method carries no
            knowledge of any cmab-specific parameter.
        """
        rewards_dict: dict[ActionId, list[Any]] = defaultdict(list)
        extra_dicts = {name: defaultdict(list) for name in row_aligned_kwargs}
        quantities_dict = defaultdict(list) if quantities is not None else None

        for i, a in enumerate(actions):
            rewards_dict[a].append(rewards[i])
            for name, values in row_aligned_kwargs.items():
                extra_dicts[name][a].append(values[i])
            if quantities_dict is not None:
                quantities_dict[a].append(quantities[i])

        for a in set(actions):
            call_kwargs: dict[str, Any] = {"rewards": rewards_dict[a]}
            for name, values in row_aligned_kwargs.items():
                action_values = extra_dicts[name][a]
                call_kwargs[name] = np.array(action_values) if isinstance(values, np.ndarray) else action_values
            if quantities_dict is not None and any(quantities_dict[a]):
                call_kwargs["quantities"] = quantities_dict[a]
            self.actions[a].update(**call_kwargs)

    @classmethod
    def _instantiate_actions(
        cls,
        actions: dict[ActionId, BaseModel] | None,
        action_ids: set[ActionId] | None,
        quantitative_action_ids: set[ActionId] | None,
        kwargs: dict[str, Any],
    ) -> dict[ActionId, BaseModel]:
        """Construct per-action model instances from cold-start kwargs, or pass through a pre-built dict."""
        if actions is not None:
            return actions
        action_specific_kwargs, quantitative_action_specific_kwargs = cls._extract_action_specific_kwargs(kwargs)
        inner_action_ids = action_ids or set(action_specific_kwargs)
        inner_quantitative_action_ids = quantitative_action_ids or set(quantitative_action_specific_kwargs)
        if not inner_action_ids and not inner_quantitative_action_ids:
            raise AttributeError("At least one action should be defined.")
        overlap = set(inner_action_ids) & set(inner_quantitative_action_ids)
        if overlap:
            raise AttributeError(f"Actions cannot be both regular and quantitative: {overlap}.")
        (
            model_cold_start,
            quantitative_model_cold_start,
            action_general_kwargs,
            quantitative_action_general_kwargs,
        ) = cls._extract_action_model_class_and_attributes(kwargs)
        actions = {}
        for _action_ids, cold_start, general_kwargs, specific_kwargs in zip(
            [inner_action_ids, inner_quantitative_action_ids],
            [model_cold_start, quantitative_model_cold_start],
            [action_general_kwargs, quantitative_action_general_kwargs],
            [action_specific_kwargs, quantitative_action_specific_kwargs],
        ):
            for a in _action_ids:
                actions[a] = cold_start(**general_kwargs, **specific_kwargs.get(a, {}))
        return actions

    @staticmethod
    def _extract_action_specific_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
        """Split ``kwargs`` into per-action and per-quantitative-action sub-dicts, removing them from ``kwargs``."""
        action_specific_kwargs = defaultdict(dict)
        quantitative_action_specific_kwargs = defaultdict(dict)
        for keyword in list(kwargs):
            argument = kwargs[keyword]
            for prefix, target_kwargs in zip(
                [ACTION_IDS_PREFIX, QUANTITATIVE_ACTION_IDS_PREFIX],
                [action_specific_kwargs, quantitative_action_specific_kwargs],
            ):
                if keyword.startswith(prefix) and type(argument) is dict:
                    kwargs.pop(keyword)
                    inner_keyword = keyword.split(prefix)[1]
                    for action_id, value in argument.items():
                        target_kwargs[action_id][inner_keyword] = value
        return dict(action_specific_kwargs), dict(quantitative_action_specific_kwargs)

    @classmethod
    def _extract_action_model_class_and_attributes(
        cls, kwargs: dict[str, Any]
    ) -> tuple[Callable, Callable, dict[str, Any], dict[str, Any]]:
        """Extract cold-start callables and their kwargs from the manager's ``kwargs`` dict."""
        action_model_classes = cls._action_model_classes
        if len(action_model_classes) > 2:
            raise ValueError("Only up to two types of action models are supported.")
        quantitative_model_cold_start = model_cold_start = lambda **kw: None  # dummy callable
        action_general_kwargs = quantitative_action_general_kwargs = None
        for action_model_class in action_model_classes:
            if hasattr(action_model_class, "cold_start"):
                action_model_cold_start = action_model_class.cold_start
                action_model_attributes = extract_argument_names_from_function(action_model_cold_start)
                action_model_attributes = action_model_attributes + extract_argument_names_from_function(
                    action_model_class
                )
            else:
                action_model_cold_start = action_model_class
                action_model_attributes = extract_argument_names_from_function(action_model_cold_start)
            general_kwargs = {k: kwargs[k] for k in action_model_attributes if k in kwargs}

            if issubclass(action_model_class, (Model, ModelMO)):
                model_cold_start = action_model_cold_start
                action_general_kwargs = general_kwargs
            elif issubclass(action_model_class, QuantitativeModel):
                quantitative_model_cold_start = action_model_cold_start
                quantitative_action_general_kwargs = general_kwargs
            else:
                raise TypeError(f"Unsupported action model class: {action_model_class}")

        used_kwargs = (action_general_kwargs or {}) | (quantitative_action_general_kwargs or {})
        for k in used_kwargs:
            kwargs.pop(k)

        return (
            model_cold_start,
            quantitative_model_cold_start,
            action_general_kwargs,
            quantitative_action_general_kwargs,
        )

    @classproperty
    def _action_model_classes(cls) -> tuple[type[BaseModel], ...]:
        """Extract concrete action-model classes from the ``actions`` field annotation (``dict[ActionId, T]`` → ``T``).

        Raises ``TypeError`` when the annotation is unparameterised or still a TypeVar,
        since cold-starting from kwargs needs a concrete class to call ``cold_start`` on.
        """
        actions_annotation = cls.model_fields["actions"].annotation
        type_args = get_args(actions_annotation)  # (ActionId, T)
        if not type_args or len(type_args) < 2:
            raise TypeError(
                f"{cls.__name__}.actions has no concrete value type annotation; "
                "parameterise the meta-model with a concrete model class before cold-starting."
            )
        action_model_type = type_args[1]
        if isinstance(action_model_type, TypeVar):
            raise TypeError(
                f"{cls.__name__}.actions value type is still a TypeVar ({action_model_type}); "
                "parameterise the meta-model with a concrete model class before cold-starting."
            )
        if get_origin(action_model_type) in (Union, UnionType):
            return get_args(action_model_type)
        return (action_model_type,)
