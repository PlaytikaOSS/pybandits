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

"""Meta-model abstraction layer for action managers.

A meta-model sits between an actions manager (e.g. ``CmabActionsManager``)
and the per-action models. The manager delegates ``sample_proba``, ``update``,
and ``reset`` to the meta-model. The meta-model decides whether per-action
calls are independent (simple dispatch loop) or require coordinated state
(e.g. a shared learned representation).

One concrete class is provided:

- ``PerActionMetaModel[T]``: generic implementation wrapping a
  ``Dict[ActionId, T]`` with an independent per-action dispatch loop.
  Behaviour is identical to the pre-meta-model smab/cmab pattern.
  Domain validators (e.g. same number of objectives for smab; consistent
  input size / update method for cmab) live on the respective manager
  subclasses, not here.

The interface is intentionally minimal and will be extended as additional
meta-model patterns (e.g. shared-backbone neural-linear) are introduced.
"""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union, get_args, get_origin

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator

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
from pybandits.model import (
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
    Beta,
    BetaCC,
    BetaMO,
    BetaMOCC,
    Model,
    ModelMO,
)
from pybandits.quantitative_model import (
    QuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetworkCC,
    QuantitativeModel,
    Zooming,
    ZoomingCC,
)
from pybandits.utils import classproperty, extract_argument_names_from_function

# Possible return shapes from a per-action sample_proba call.
SampleProbaResult = Union[
    List[Probability],
    List[MOProbability],
    List[ProbabilityWeight],
    List[MOProbabilityWeight],
    List[QuantitativeProbability],
    List[QuantitativeMOProbability],
    List[QuantitativeProbabilityWeight],
    List[QuantitativeMOProbabilityWeight],
]

ActionModelType = TypeVar("ActionModelType", bound=BaseModel)


class BaseMetaModel(PyBanditsBaseModel, ABC):
    """Abstract base for objects that own per-action state and dispatch logic.

    The manager class (e.g. ``CmabActionsManager``) delegates ``sample_proba``,
    ``update``, and ``reset`` to a meta-model instance. The meta-model owns the
    per-action models (exposed via the ``actions`` field) and decides whether
    they are dispatched independently or with coordinated shared state.

    Subclasses may narrow the value type of ``actions`` (e.g.
    ``Dict[ActionId, BayesianNeuralNetwork]``); Pydantic preserves identity of
    nested model instances, so mutations via ``update()`` remain visible to
    holders of the original references.
    """

    actions: Dict[ActionId, BaseModel]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def action_ids(self) -> List[ActionId]:
        """Action identifiers covered by this meta-model."""
        return list(self.actions.keys())

    @abstractmethod
    def sample_proba(
        self,
        rng: np.random.Generator,
        valid_action_ids: Optional[Set[ActionId]] = None,
        **kwargs: Any,
    ) -> Dict[ActionId, SampleProbaResult]:
        """Sample per-action probabilities/scores under the current context.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central random generator from the bandit (for reproducibility).
        valid_action_ids : Optional[Set[ActionId]]
            If provided, restrict sampling to these action ids (the bandit's
            allowed actions after removing ``forbidden_actions``). When
            ``None`` all action ids are sampled. Meta-models with shared
            state may use this to skip unnecessary head/branch evaluations.
        **kwargs
            Additional sampling inputs. For contextual bandits this includes
            ``context: np.ndarray``.

        Returns
        -------
        Dict[ActionId, SampleProbaResult]
            Per-action probability/score collection in whatever shape the
            underlying model returns from its ``sample_proba``.
        """

    @abstractmethod
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        **kwargs: Any,
    ) -> None:
        """Update per-action state from a batch of (action, reward, ...) tuples.

        Parameters
        ----------
        actions : List[ActionId]
            Selected action per sample.
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]]
            Reward per sample (scalar for single-objective; list per sample
            for multi-objective).
        quantities : Optional[List[Union[float, List[float], None]]]
            Per-sample quantity for quantitative actions; ``None`` for
            non-quantitative actions in the same batch.
        **kwargs
            Additional update inputs. For contextual bandits this includes
            ``context: np.ndarray``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset every per-action model to its cold-start state."""

    @classmethod
    def _preprocess_actions(
        cls,
        actions: Optional[Dict[ActionId, BaseModel]],
        action_ids: Optional[Set[ActionId]],
        quantitative_action_ids: Optional[Set[ActionId]],
        kwargs: Dict[str, Any],
    ) -> Dict[ActionId, BaseModel]:
        """Return the canonical per-action dict the meta-model should store.

        Default behaviour (used by ``PerActionMetaModel`` and any subclass whose state is
        a 1:1 per-action mapping): if ``actions`` is None, cold-start the per-action models
        from ``action_ids`` / ``quantitative_action_ids`` and the remaining ``kwargs``.
        Shared-state meta-models (e.g. neural-linear) override this to materialise per-action
        regression heads on top of a jointly-trained backbone.
        """
        return cls._instantiate_actions(
            actions=actions,
            action_ids=action_ids,
            quantitative_action_ids=quantitative_action_ids,
            kwargs=kwargs,
        )

    @classmethod
    def _instantiate_actions(
        cls,
        actions: Optional[Dict[ActionId, BaseModel]],
        action_ids: Optional[Set[ActionId]],
        quantitative_action_ids: Optional[Set[ActionId]],
        kwargs: Dict[str, Any],
    ) -> Dict[ActionId, BaseModel]:
        """Construct per-action model instances from cold-start kwargs, or pass through a pre-built dict."""
        if actions is not None:
            return actions
        action_specific_kwargs, quantitative_action_specific_kwargs = cls._extract_action_specific_kwargs(kwargs)
        inner_action_ids = action_ids or set(action_specific_kwargs)
        inner_quantitative_action_ids = quantitative_action_ids or set(quantitative_action_specific_kwargs)
        if not inner_action_ids and not inner_quantitative_action_ids:
            raise AttributeError("At least one action should be defined.")
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
    def _extract_action_specific_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
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
        cls, kwargs: Dict[str, Any]
    ) -> Tuple[Callable, Callable, Dict[str, Any], Dict[str, Any]]:
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
    def _action_model_classes(cls) -> Tuple[Type[BaseModel], ...]:
        """Extract concrete action-model classes from the ``actions`` field annotation (``Dict[ActionId, T]`` → ``T``).

        Raises ``TypeError`` when the annotation is unparameterised or still a TypeVar,
        since cold-starting from kwargs needs a concrete class to call ``cold_start`` on.
        """
        actions_annotation = cls.model_fields["actions"].annotation
        type_args = get_args(actions_annotation)  # (ActionId, T)
        if not type_args or len(type_args) < 2:
            raise TypeError(
                f"{cls.__name__}.actions has no concrete value type annotation; "
                "parameterise the meta-model (e.g. PerActionMetaModel[Beta]) before cold-starting."
            )
        action_model_type = type_args[1]
        if isinstance(action_model_type, TypeVar):
            raise TypeError(
                f"{cls.__name__}.actions value type is still a TypeVar ({action_model_type}); "
                "parameterise the meta-model with a concrete model class before cold-starting."
            )
        if get_origin(action_model_type) is Union:
            return get_args(action_model_type)
        return (action_model_type,)


class PerActionMetaModel(BaseMetaModel, Generic[ActionModelType]):
    """Meta-model that dispatches independently to per-action models.

    Wraps a ``Dict[ActionId, ActionModelType]`` and routes each ``sample_proba`` /
    ``update`` / ``reset`` call to the relevant action's model. This is the
    historical cmab/smab pattern, lifted into the meta-model abstraction without
    any behavioural change.

    The ``actions`` dict is a regular Pydantic field. Pydantic v2 with its
    default ``revalidate_instances='never'`` does not deep-copy nested model
    instances during validation, so the original model references are preserved.
    This guarantees that mutations via ``update()`` are visible to callers that
    hold references to the original per-action model objects.

    Domain invariants (e.g. same number of objectives for smab; consistent
    input size / update method for cmab) are validated on the respective manager
    subclasses, not here.
    """

    actions: Dict[ActionId, ActionModelType]  # type: ignore[valid-type]

    @field_validator("actions", mode="after")
    @classmethod
    def at_least_one_action_is_defined(cls, v: Dict[ActionId, ActionModelType]) -> Dict[ActionId, ActionModelType]:
        if len(v) == 0:
            raise AttributeError("At least one action should be defined.")
        elif len(v) == 1:
            warnings.warn("Only a single action was supplied. This MAB will be deterministic.")
        return v

    def __init__(
        self,
        actions: Optional[Dict[ActionId, BaseModel]] = None,
        action_ids: Optional[Set[ActionId]] = None,
        quantitative_action_ids: Optional[Set[ActionId]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a meta-model from either a pre-built ``actions`` dict or cold-start specs.

        Mirrors the ``ActionsManager.__init__`` pattern: optional construction kwargs are
        resolved into a per-action dict, then handed to Pydantic's ``BaseModel.__init__``
        via ``super().__init__(actions=...)``.
        """
        actions_dict = type(self)._preprocess_actions(
            actions=actions,
            action_ids=action_ids,
            quantitative_action_ids=quantitative_action_ids,
            kwargs=kwargs or {},
        )
        super().__init__(actions=actions_dict)

    def sample_proba(
        self,
        rng: np.random.Generator,
        valid_action_ids: Optional[Set[ActionId]] = None,
        **kwargs: Any,
    ) -> Dict[ActionId, SampleProbaResult]:
        return {
            action_id: model.sample_proba(rng=rng, **kwargs)
            for action_id, model in self.actions.items()
            if valid_action_ids is None or action_id in valid_action_ids
        }

    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        context: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """Group rows by action and dispatch to each action's ``update``.

        ``context`` and ``quantities`` may be longer than ``actions`` when the
        adaptive-window manager prepends memory rows before calling here.  We
        slice to the trailing ``len(actions)`` rows so only the current batch
        is passed to each per-action model.
        """
        if context is not None:
            if len(context) < len(actions):
                raise ValueError(
                    f"context has {len(context)} rows but {len(actions)} actions were provided; "
                    "context must have at least as many rows as actions."
                )
            context = context[-len(actions) :]

        rewards_dict: Dict[ActionId, List[Any]] = defaultdict(list)
        context_dict: Dict[ActionId, List[Any]] = defaultdict(list)

        if quantities is None:
            for i, (a, r) in enumerate(zip(actions, rewards)):
                rewards_dict[a].append(r)
                if context is not None:
                    context_dict[a].append(context[i])
            for a in set(actions):
                call_kwargs: Dict[str, Any] = {"rewards": rewards_dict[a]}
                if context is not None:
                    call_kwargs["context"] = np.array(context_dict[a])
                self.actions[a].update(**call_kwargs)
        else:
            if len(quantities) < len(actions):
                raise ValueError(
                    f"quantities has {len(quantities)} elements but {len(actions)} actions were provided; "
                    "quantities must have at least as many elements as actions."
                )
            quantities = quantities[-len(actions) :]
            quantities_dict: Dict[ActionId, List[Any]] = defaultdict(list)
            for i, (a, r, q) in enumerate(zip(actions, rewards, quantities)):
                rewards_dict[a].append(r)
                if context is not None:
                    context_dict[a].append(context[i])
                quantities_dict[a].append(q)
            for a in set(actions):
                call_kwargs = {"rewards": rewards_dict[a]}
                if context is not None:
                    call_kwargs["context"] = np.array(context_dict[a])
                if any(quantities_dict[a]):
                    call_kwargs["quantities"] = quantities_dict[a]
                self.actions[a].update(**call_kwargs)

    def reset(self) -> None:
        for model in self.actions.values():
            model.reset()


class CmabPerActionMetaModel(PerActionMetaModel[ActionModelType], Generic[ActionModelType]):
    """Per-action meta-model for cmab variants.

    Adds a cross-action consistency validator (same input dim, same update method, same
    update kwargs) that the cmab branch needs at construction time. This lives on the
    meta-model rather than on ``CmabActionsManager`` because the constraint is about the
    per-action *models* (their BNN shape and training config), not about the manager.
    """

    @staticmethod
    def _maybe_crawl_model(model: Any) -> "BayesianNeuralNetwork":
        """Unwrap MO/quantitative wrappers to the base BNN."""
        from pybandits.model import BaseBayesianNeuralNetworkMO  # local import: avoids circular at module top
        from pybandits.quantitative_model import BaseQuantitativeBayesianNeuralNetwork

        if isinstance(model, BaseBayesianNeuralNetworkMO):
            return model.models[0]
        if isinstance(model, BaseQuantitativeBayesianNeuralNetwork):
            return model.bnn
        return model

    @staticmethod
    def _get_expected_context_size(model: Any) -> int:
        """Return the expected ``input_dim`` for the given cmab model."""
        from pybandits.model import BaseBayesianNeuralNetworkMO

        if isinstance(model, BaseBayesianNeuralNetworkMO):
            return model.models[0].input_dim
        return model.input_dim

    @model_validator(mode="after")
    def check_models(self) -> "CmabPerActionMetaModel":
        action_models = list(self.actions.values())
        first_action = action_models[0]
        test_first_action = self._maybe_crawl_model(first_action)
        for action in action_models[1:]:
            test_action = self._maybe_crawl_model(action)
            if not self._get_expected_context_size(first_action) == self._get_expected_context_size(action):
                raise AttributeError("All actions should have the same input size.")
            if not test_first_action.update_method == test_action.update_method:
                raise AttributeError("All actions should have the same update method.")
            if not test_first_action.update_kwargs == test_action.update_kwargs:
                raise AttributeError("All actions should have the same update kwargs.")
        return self


# Module-level aliases for concrete parameterisations.
# These are required for pickling: Python's pickle resolves a class by looking
# up ``cls.__qualname__`` on ``sys.modules[cls.__module__]``.  Pydantic sets the
# qualname of a parameterised generic to e.g.
# ``PerActionMetaModel[Union[Beta, Zooming]]``; defining the alias here makes
# that attribute accessible on this module.
PerActionMetaModelSmabSO = PerActionMetaModel[Union[Beta, Zooming]]
PerActionMetaModelSmabCC = PerActionMetaModel[Union[BetaCC, ZoomingCC]]
PerActionMetaModelSmabMO = PerActionMetaModel[BetaMO]
PerActionMetaModelSmabMOCC = PerActionMetaModel[BetaMOCC]

PerActionMetaModelCmabSO = CmabPerActionMetaModel[Union[BayesianNeuralNetwork, QuantitativeBayesianNeuralNetwork]]
PerActionMetaModelCmabCC = CmabPerActionMetaModel[Union[BayesianNeuralNetworkCC, QuantitativeBayesianNeuralNetworkCC]]
PerActionMetaModelCmabMO = CmabPerActionMetaModel[BayesianNeuralNetworkMO]
PerActionMetaModelCmabMOCC = CmabPerActionMetaModel[BayesianNeuralNetworkMOCC]
