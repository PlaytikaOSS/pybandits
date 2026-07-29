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

import json
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import (
    ValidationError,
)
from pydantic.dataclasses import dataclass
from pydantic.types import PositiveInt
from pytest import MonkeyPatch

import pybandits
from pybandits.actions_manager import CmabModelType
from pybandits.base import ActionId, Float01, PositiveProbability
from pybandits.cmab import (
    BaseCmabBernoulli,
    CmabBernoulli,
    CmabBernoulliBAI,
    CmabBernoulliCC,
    CmabBernoulliDP,
    CmabBernoulliMO,
    CmabBernoulliMOCC,
)
from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BaseBayesianNeuralNetworkMO,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkDP,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
    StudentTArray,
)
from pybandits.model.bnn._typing import ActivationFunctions
from pybandits.quantitative_model import (
    BaseQuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetworkCC,
    QuantitativeBayesianNeuralNetworkDP,
    QuantitativeModel,
    Segment,
)
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    DynamicPricingBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
)
from tests.utils import (
    apply_mock_update,
    mocked_cmab_training,
    sample_with_replacement,
    to_temporary_pickle,
)


@st.composite
def diff_strategy(draw):
    return draw(st.floats(min_value=0.001, max_value=0.5))


@st.composite
def value_strategy(draw, n_actions):
    return draw(st.lists(st.floats(min_value=0, max_value=2), min_size=n_actions, max_size=n_actions))


def mock_student_t_array(
    field_value: StudentTArray,
    diff: Any,
    monkeymodule: MonkeyPatch,
    label: Union[int, str],
) -> int:
    """
    Update the mu and sigma fields of a StudentTArray object.

    Parameters
    ----------
    field_value : StudentTArray
        The object to update.
    diff : Any
        The diff object for drawing random values.
    monkeymodule : MonkeyPatch
        The monkey module for patching.
    label : Union[int, str]
        The label for the diff draw.

    Returns
    -------
    label: int
        Updated label
    """
    for sub_field in ("mu", "sigma"):
        try:
            orig_value = getattr(field_value, sub_field)
            if isinstance(orig_value, list) and isinstance(orig_value[0], float):
                # StudentTArray with a list of floats
                new_value = [value + diff.draw(diff_strategy(), label=f"{label}") for value in orig_value]
                monkeymodule.setattr(field_value, sub_field, new_value)

            elif isinstance(orig_value, list) and isinstance(orig_value[0], list):
                # StudentTArray with a list of lists of floats
                new_value = [
                    [value + diff.draw(diff_strategy(), label=f"{label}") for value in sublist]
                    for sublist in orig_value
                ]
                monkeymodule.setattr(field_value, sub_field, new_value)

            label = int(label) + 1 if isinstance(label, (int, str)) else label + 1
        except AttributeError as e:
            raise ValueError(f"Invalid StudentTArray field: {sub_field}") from e
    return label


def _quantitative_callable(x, value):
    return min(sum(x) ** value, 1000)


@dataclass
class ModelTestConfig:
    """Configuration for parameterized CMAB (Contextual Multi-Armed Bandit) tests.

    Bundles the CMAB algorithm, bandit strategy, and model types used to generate
    test instances. Used with Hypothesis to create CMAB instances and actions
    across single/multi-objective and cost/no-cost variants.
    """

    cmab_class: Type
    strategy_class: Type
    model_types: List[Type[CmabModelType]]

    def _create_actions(
        self,
        action_ids: List[str],
        values: Optional[st.SearchStrategy],
        n_features: PositiveInt,
        hidden_dim_list: List[int],
        rng: np.random.Generator,
        n_objectives: Optional[PositiveInt] = None,
        decay_factor: Optional[Float01] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_types = list(self.model_types)
        if len(model_types) < len(action_ids):
            indices = rng.integers(0, len(model_types), len(action_ids))
            model_types = [model_types[i] for i in indices]
        if all(
            model in [BayesianNeuralNetworkCC, BayesianNeuralNetworkMOCC, QuantitativeBayesianNeuralNetworkCC]
            for model in model_types
        ):
            # Generate random costs
            drawn_values = values.draw(value_strategy(n_actions=len(action_ids)))
            costs = [
                val
                if model_type in [BayesianNeuralNetworkCC, BayesianNeuralNetworkMOCC]
                else partial(_quantitative_callable, value=val)
                for val, model_type in zip(drawn_values, model_types)
            ]
            value_field = "cost"
        elif all(model in [BayesianNeuralNetworkDP, QuantitativeBayesianNeuralNetworkDP] for model in model_types):
            # Generate random prices
            drawn_values = values.draw(value_strategy(n_actions=len(action_ids)))
            costs = [
                val if model_type == BayesianNeuralNetworkDP else partial(_quantitative_callable, value=val)
                for val, model_type in zip(drawn_values, model_types)
            ]
            value_field = "price"
        else:
            costs = None
            value_field = None

        model_cold_start_kwargs = dict()
        if decay_factor is not None:
            model_cold_start_kwargs["decay_factor"] = decay_factor
        base_model_cold_start_kwargs = dict(hidden_dim_list=hidden_dim_list, **model_cold_start_kwargs)

        result: Dict[str, Any] = {}
        if n_objectives is None:
            # Single-objective models
            if costs is not None:
                for action_id, model_type, cost in zip(action_ids, model_types, costs):
                    if issubclass(model_type, (BayesianNeuralNetworkCC, BayesianNeuralNetworkDP)):
                        result[action_id] = model_type.cold_start(
                            n_features=n_features,
                            hidden_dim_list=hidden_dim_list,
                            **{value_field: cost},
                            **model_cold_start_kwargs,
                        )
                    else:
                        # QuantitativeBayesianNeuralNetworkCC / QuantitativeBayesianNeuralNetworkDP
                        result[action_id] = model_type.cold_start(
                            dimension=1,
                            n_features=n_features,
                            base_model_cold_start_kwargs=base_model_cold_start_kwargs,
                            **{value_field: cost},
                        )
            else:
                for action_id, model_type in zip(action_ids, model_types):
                    if issubclass(model_type, BayesianNeuralNetwork):
                        result[action_id] = model_type.cold_start(
                            n_features=n_features,
                            hidden_dim_list=hidden_dim_list,
                            **model_cold_start_kwargs,
                        )
                    else:
                        # QuantitativeBayesianNeuralNetwork
                        result[action_id] = model_type.cold_start(
                            dimension=1,
                            n_features=n_features,
                            base_model_cold_start_kwargs=base_model_cold_start_kwargs,
                        )
        else:
            # Multi-objective models
            if costs is not None:
                for action_id, model_type, cost in zip(action_ids, model_types, costs):
                    result[action_id] = model_type.cold_start(
                        n_objectives=n_objectives,
                        n_features=n_features,
                        hidden_dim_list=hidden_dim_list,
                        cost=cost,
                        **model_cold_start_kwargs,
                    )
            else:
                for action_id, model_type in zip(action_ids, model_types):
                    result[action_id] = model_type.cold_start(
                        n_objectives=n_objectives,
                        n_features=n_features,
                        hidden_dim_list=hidden_dim_list,
                        **model_cold_start_kwargs,
                    )
        return result, base_model_cold_start_kwargs

    def create_cmab_and_actions(
        self,
        action_ids: List[str],
        epsilon: Optional[Float01],
        delta: Optional[PositiveProbability],
        values: st.SearchStrategy,
        n_objectives: st.SearchStrategy[PositiveInt],
        exploit_p: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        subsidy_factor: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        n_features: PositiveInt,
        hidden_dim_list: List[int],
        rng: np.random.Generator,
        decay_factor: Optional[Float01] = None,
        default_action_fraction: Optional[Float01] = None,
        limited_action_fraction: Optional[Float01] = None,
        backbone_hidden_dims: Optional[List[int]] = None,
    ) -> Tuple[BaseCmabBernoulli, Dict[ActionId, CmabModelType], Dict[str, Any]]:
        n_objectives = (
            n_objectives.draw(st.integers(min_value=1, max_value=10))
            if self.cmab_class in [CmabBernoulliMO, CmabBernoulliMOCC]
            else None
        )
        actions, base_model_cold_start_kwargs = self._create_actions(
            action_ids, values, n_features, hidden_dim_list, rng, n_objectives, decay_factor
        )
        default_action = action_ids[0] if epsilon and not delta else None
        if default_action and isinstance(actions[default_action], QuantitativeModel):
            default_action = (default_action, tuple(rng.random(actions[default_action].dimension)))
        epsilon = epsilon if not delta else 0.1
        # default_action_fraction requires a default_action; limited_actions is derived from the
        # action set (excluding default_action) exactly as default_action is derived from epsilon.
        default_action_fraction = default_action_fraction if default_action else None
        limited_actions = {action_ids[-1]} if limited_action_fraction is not None else None
        kwargs = {
            k: v
            for k, v in {
                "epsilon": epsilon,
                "default_action": default_action,
                "delta": delta,
                "default_action_fraction": default_action_fraction,
                "limited_actions": limited_actions,
                "limited_action_fraction": limited_action_fraction,
            }.items()
            if v is not None
        }
        for param, classes in zip(["subsidy_factor", "exploit_p"], [[CmabBernoulliCC], [CmabBernoulliBAI]]):
            if self.cmab_class in classes:
                actual_param = eval(param)
                if isinstance(actual_param, float) or actual_param is None:
                    kwargs[param] = actual_param
                else:
                    kwargs[param] = actual_param.draw(st.floats(min_value=0, max_value=1))

        strategy_kwargs = dict(kwargs)  # epsilon / default_action / delta / subsidy_factor / exploit_p
        kwargs["n_features"] = n_features
        if any(isinstance(model, BaseQuantitativeBayesianNeuralNetwork) for model in actions.values()):
            kwargs["base_model_cold_start_kwargs"] = base_model_cold_start_kwargs
        if any(
            isinstance(model, (BaseBayesianNeuralNetwork, BaseBayesianNeuralNetworkMO)) for model in actions.values()
        ):
            kwargs.update(base_model_cold_start_kwargs)

        # For cold start test
        if self.cmab_class in [CmabBernoulliMO, CmabBernoulliMOCC]:
            kwargs["n_objectives"] = n_objectives

        # A shared backbone is only constructible via cold_start (the manager builds it); it cannot be
        # expressed as a pre-made ``actions`` dict, and it is incompatible with the adaptive window
        # (``delta``). When requested and compatible, cold-start a backbone variant; otherwise build the
        # per-action variant as before.
        if backbone_hidden_dims is not None and delta is None:
            cmab = self.cmab_class.cold_start(
                backbone_hidden_dims=backbone_hidden_dims, **self.cold_start_kwargs(actions, kwargs)
            )
        else:
            cmab = self.cmab_class(actions=actions, **strategy_kwargs)
        return cmab, actions, kwargs

    @staticmethod
    def cold_start_kwargs(actions: Dict[ActionId, CmabModelType], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the ``cold_start`` kwargs equivalent to a pre-built ``actions`` dict (+ strategy kwargs).

        Shared by ``test_cold_start`` (equality check) and the backbone branch of
        ``create_cmab_and_actions``: derives ``action_ids`` / ``quantitative_action_ids`` and any
        per-action cost/price maps from ``actions``, then merges the construction ``kwargs``.
        """
        cold_start_kwargs: Dict[str, Any] = {
            "action_ids": {
                a for a, m in actions.items() if isinstance(m, (BaseBayesianNeuralNetwork, BaseBayesianNeuralNetworkMO))
            },
            "quantitative_action_ids": {a for a, m in actions.items() if isinstance(m, QuantitativeModel)},
        }
        if all(
            isinstance(m, (BayesianNeuralNetworkCC, BayesianNeuralNetworkMOCC, QuantitativeBayesianNeuralNetworkCC))
            for m in actions.values()
        ):
            cold_start_kwargs["action_ids_cost"] = {
                a: m.cost
                for a, m in actions.items()
                if isinstance(m, (BayesianNeuralNetworkCC, BayesianNeuralNetworkMOCC))
            }
            cold_start_kwargs["quantitative_action_ids_cost"] = {
                a: m.cost for a, m in actions.items() if isinstance(m, QuantitativeBayesianNeuralNetworkCC)
            }
        if all(isinstance(m, (BayesianNeuralNetworkDP, QuantitativeBayesianNeuralNetworkDP)) for m in actions.values()):
            cold_start_kwargs["action_ids_price"] = {
                a: m.price for a, m in actions.items() if isinstance(m, BayesianNeuralNetworkDP)
            }
            cold_start_kwargs["quantitative_action_ids_price"] = {
                a: m.price for a, m in actions.items() if isinstance(m, QuantitativeBayesianNeuralNetworkDP)
            }
        cold_start_kwargs.update(kwargs)
        return {k: v for k, v in cold_start_kwargs.items() if v is not None}


TEST_CONFIGS = {
    "cmab": ModelTestConfig(CmabBernoulli, ClassicBandit, [BayesianNeuralNetwork, QuantitativeBayesianNeuralNetwork]),
    "cmab_bai": ModelTestConfig(
        CmabBernoulliBAI, BestActionIdentificationBandit, [BayesianNeuralNetwork, QuantitativeBayesianNeuralNetwork]
    ),
    "cmab_cc": ModelTestConfig(
        CmabBernoulliCC,
        CostControlBandit,
        [BayesianNeuralNetworkCC, QuantitativeBayesianNeuralNetworkCC],
    ),
    "cmab_dp": ModelTestConfig(
        CmabBernoulliDP,
        DynamicPricingBandit,
        [BayesianNeuralNetworkDP, QuantitativeBayesianNeuralNetworkDP],
    ),
    "cmab_mo": ModelTestConfig(
        CmabBernoulliMO,
        MultiObjectiveBandit,
        [BayesianNeuralNetworkMO],
    ),
    "cmab_mocc": ModelTestConfig(
        CmabBernoulliMOCC,
        MultiObjectiveCostControlBandit,
        [BayesianNeuralNetworkMOCC],
    ),
}


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(
        st.text(
            min_size=1,
        ),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    epsilon=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    decay_factor=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
)
def test_cold_start(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    decay_factor: Optional[float],
    rng,
):
    # Create CMAB instance
    cmab, actions, kwargs = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        rng,
        decay_factor,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
    )

    # Cold start comparison logic (shared assembler, covers all model-type variants)
    cold_start_kwargs = config.cold_start_kwargs(actions, kwargs)
    assert config.cmab_class.cold_start(**cold_start_kwargs) == cmab


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    values=st.data(),
    n_objectives=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
)
def test_bad_initialization(
    config: ModelTestConfig,
    action_ids: List[str],
    n_features: int,
    hidden_dim_list: List[PositiveInt],
    values,
    n_objectives,
    exploit_p,
    subsidy_factor,
    rng,
):
    """Test various invalid initialization scenarios for CMAB models"""
    real_n_objectives = n_objectives.draw(st.integers(min_value=1, max_value=10))
    if config.cmab_class in (CmabBernoulliCC, CmabBernoulliMOCC):
        kwargs = {"cost": 1}
    elif config.cmab_class == CmabBernoulliDP:
        kwargs = {"price": 1}
    else:
        kwargs = {}
    kwargs["n_features"] = n_features
    kwargs["hidden_dim_list"] = hidden_dim_list
    if config.cmab_class in [CmabBernoulliMO, CmabBernoulliMOCC]:
        kwargs["n_objectives"] = real_n_objectives

    # Test empty actions
    with pytest.raises(AttributeError):
        config.cmab_class(actions={})

    # Test single action (should warn)
    single_action = {action_ids[0]: config.model_types[0].cold_start(**kwargs)}
    with pytest.warns(UserWarning):
        config.cmab_class(actions=single_action)

    # Test mismatched feature dimensions
    alt_kwargs = deepcopy(kwargs)
    alt_kwargs["n_features"] = n_features + 1
    actions_wrong_dims = {
        action_ids[0]: config.model_types[0].cold_start(**kwargs),
        action_ids[1]: config.model_types[0].cold_start(**alt_kwargs),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_dims)

    # Test mismatched update kwargs
    mismatch_kw1, mismatch_kw2 = ({"num_steps": 100}, {"num_steps": 200})
    actions_wrong_kwargs = {
        action_ids[0]: config.model_types[0].cold_start(
            update_kwargs=mismatch_kw1,
            **kwargs,
        ),
        action_ids[1]: config.model_types[0].cold_start(
            update_kwargs=mismatch_kw2,
            **kwargs,
        ),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_kwargs)

    # Test invalid model types
    actions_wrong_type = {
        action_ids[0]: BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list),
        action_ids[1]: BayesianNeuralNetworkCC.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, cost=1.0
        ),
    }
    with pytest.raises((ValidationError, TypeError)):
        config.cmab_class(actions=actions_wrong_type)

    # Test None actions
    with pytest.raises(ValidationError):
        config.cmab_class(actions={aid: None for aid in action_ids})

    # Test invalid strategy parameters
    if config.cmab_class == CmabBernoulliBAI:
        with pytest.raises(ValidationError):
            config.create_cmab_and_actions(
                action_ids,
                None,
                None,
                values,
                n_objectives,
                exploit_p.draw(st.sampled_from([-0.1, 1.1])),
                subsidy_factor,
                n_features,
                hidden_dim_list,
                rng,
            )
    elif config.cmab_class == CmabBernoulliCC:
        with pytest.raises(ValidationError):
            config.create_cmab_and_actions(
                action_ids,
                None,
                None,
                values,
                n_objectives,
                exploit_p,
                subsidy_factor.draw(st.sampled_from([-0.1, 1.1])),
                n_features,
                hidden_dim_list,
                rng,
            )
    # Test multi-objective specific cases
    if hasattr(config.model_types[0], "models"):
        # Test mismatched number of objectives
        kwargs.pop("n_objectives")
        mo_actions_wrong = {
            action_ids[0]: BayesianNeuralNetworkMO.cold_start(**kwargs, n_objectives=real_n_objectives),
            action_ids[1]: BayesianNeuralNetworkMO.cold_start(**kwargs, n_objectives=real_n_objectives + 1),
        }
        with pytest.raises(AttributeError):
            config.cmab_class(actions=mo_actions_wrong)


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(
        st.text(
            min_size=1,
        ),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    n_samples=st.integers(min_value=1, max_value=5),
    epsilon=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    memory_len=st.integers(min_value=1, max_value=5),
    backbone_hidden_dims=st.one_of(st.none(), st.just([2])),
)
def test_update(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    memory_len,
    rng,
    backbone_hidden_dims,
):
    # rng is a session-scoped fixture shared across every Hypothesis example in this run; reseeding
    # here makes each example's random draws depend only on this call, not on how many draws prior
    # examples (or other tests) already consumed — otherwise Hypothesis's shrink/confirm replay can
    # see different random data for the "same" example and flag a spurious FlakyFailure.
    rng = np.random.default_rng(seed=42)
    # Create CMAB instance (a shared backbone is exercised when backbone_hidden_dims is drawn and delta is None)
    cmab, _, kwargs = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        rng,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
        backbone_hidden_dims=backbone_hidden_dims,
    )
    # create patches

    n_objectives = kwargs.get("n_objectives")
    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    # Generate random rewards
    reward_data = (
        rng.choice([0, 1], size=(n_samples, n_objectives), replace=True)
        if n_objectives
        else rng.choice([0, 1], size=n_samples, replace=True)
    )
    reward_data = reward_data.tolist()
    # Test updates with generated data
    actions_to_update = sample_with_replacement(action_ids, n_samples, rng=rng)
    # Generate quantities only if there are any QuantitativeModel actions
    # Handle multi-objective rewards for MO models

    for_update_kwargs = {"actions": actions_to_update, "rewards": reward_data}
    if any(isinstance(model, BaseQuantitativeBayesianNeuralNetwork) for model in cmab.actions.values()):
        quantity_data = rng.random(size=n_samples).tolist()
        quantity_data = [
            q if isinstance(cmab.actions[action], QuantitativeModel) else None
            for q, action in zip(quantity_data, actions_to_update)
        ]
        for_update_kwargs["quantities"] = quantity_data

    old_cmab = deepcopy(cmab)
    for k, transform in enumerate([lambda x: x, np.array, lambda x: x.copy()]):
        # Skip real SVI on both update paths (joint engine with a backbone, per-head dispatch without);
        # the surrounding update plumbing stays exercised.
        with mocked_cmab_training():
            if k and delta and "actions_memory" not in for_update_kwargs:
                with pytest.warns(UserWarning):
                    cmab.update(context=transform(context), **for_update_kwargs)
            else:
                cmab.update(context=transform(context), **for_update_kwargs)
            if delta and "actions_memory" not in for_update_kwargs:
                for_update_kwargs["actions_memory"] = for_update_kwargs["actions"][-memory_len:]
                for_update_kwargs["rewards_memory"] = for_update_kwargs["rewards"][-memory_len:]
                for_update_kwargs["context_memory"] = context[-memory_len:]
            assert cmab != old_cmab


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(
        st.text(
            min_size=1,
        ),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    n_samples=st.integers(min_value=1, max_value=100),
    epsilon=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    diff=st.data(),
    backbone_hidden_dims=st.one_of(st.none(), st.just([2])),
)
def test_predict(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    diff,
    backbone_hidden_dims,
    monkeymodule,
    rng,
):
    def mock_maximize_by_quantity(quantity_score_func, dimension, constraint=None, n_trials=10000, **kwargs):
        """Mock maximize_by_quantity to return a quick result."""
        return rng.random(dimension)

    if config.cmab_class in (CmabBernoulliMO, CmabBernoulliMOCC):

        def mock_find_pareto_front_normal_constraint(self, func, input_dim, n_objectives, n_divisions, model):
            """Mock _find_pareto_front_normal_constraint to return a quick result."""
            return [rng.random(input_dim) for _ in range(min(3, n_divisions))]

        monkeymodule.setattr(
            pybandits.strategy.MultiObjectiveStrategy,
            "_find_pareto_front_normal_constraint",
            mock_find_pareto_front_normal_constraint,
        )

    with patch.object(pybandits.strategy.single_objective, "maximize_by_quantity", mock_maximize_by_quantity):
        # Create CMAB instance (shared backbone exercised when backbone_hidden_dims is drawn and delta is None)
        cmab = config.create_cmab_and_actions(
            action_ids,
            epsilon,
            delta,
            values,
            n_objectives,
            exploit_p,
            subsidy_factor,
            n_features,
            hidden_dim_list,
            rng,
            default_action_fraction=default_action_fraction,
            limited_action_fraction=limited_action_fraction,
            backbone_hidden_dims=backbone_hidden_dims,
        )[0]
        context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

        # Test predictions with random forbidden actions
        forbidden = (
            set(sample_with_replacement(action_ids, len(action_ids) // 2, rng=rng)) if len(action_ids) > 2 else None
        )
        if cmab.default_action is not None and forbidden is not None:
            da_id = cmab.default_action[0] if isinstance(cmab.default_action, tuple) else cmab.default_action
            if da_id in forbidden:
                forbidden.discard(da_id)
        apply_mock_update(list(cmab.actions.values()))
        best_actions, probs, weights = cmab.predict(context=context, forbidden_actions=forbidden)
        assert len(best_actions) == n_samples
        assert len(probs) == n_samples
        assert len(weights) == n_samples

        if forbidden:
            # Check proper length of probabilities
            for prob in probs:
                action_keys = {action[0] if isinstance(action, tuple) else action for action in prob}
                assert len(action_keys) == len(action_ids) - len(forbidden)

            # Check best actions not in forbidden
            for action in best_actions:
                action_id = action[0] if isinstance(action, tuple) else action
                assert action_id not in forbidden

            # Check prob actions not in forbidden
            for prob in probs:
                for action in prob.keys():
                    action_id = action[0] if isinstance(action, tuple) else action
                    assert action_id not in forbidden

            # Check weight actions not in forbidden
            for weight in weights:
                for action in weight.keys():
                    action_id = action[0] if isinstance(action, tuple) else action
                    assert action_id not in forbidden

        else:
            for prob in probs:
                action_keys = {action[0] if isinstance(action, tuple) else action for action in prob}
                assert len(action_keys) == len(action_ids)


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(
        st.text(
            min_size=1,
        ),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    epsilon=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    diff=st.data(),
    backbone_hidden_dims=st.one_of(st.none(), st.just([2])),
)
def test_serialization(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    diff,
    backbone_hidden_dims,
    monkeymodule,
    rng,
):
    # Create CMAB instance (shared backbone exercised when backbone_hidden_dims is drawn and delta is None)
    cmab = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        rng,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
        backbone_hidden_dims=backbone_hidden_dims,
    )[0]

    pre_update_state = deepcopy(cmab.get_state())
    apply_mock_update(list(cmab.actions.values()))
    post_update_state = cmab.get_state()
    # Verify model updates
    assert pre_update_state != post_update_state

    # Test serialization
    restored_cmab_state = config.cmab_class.from_state(post_update_state[1]).get_state()
    assert restored_cmab_state == post_update_state


_MISSING_FEATURE_CONFIG_TEST_CONFIGS = {
    "cmab": CmabBernoulli,
    "cmab_bai": CmabBernoulliBAI,
}


@pytest.mark.parametrize(
    "CmabClass", _MISSING_FEATURE_CONFIG_TEST_CONFIGS.values(), ids=_MISSING_FEATURE_CONFIG_TEST_CONFIGS.keys()
)
@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_cmab_from_old_state_missing_feature_config(
    CmabClass,
    n_features: int,
    hidden_dim_list: List[int],
) -> None:
    """
    Test the pre-4.4 migration path: model_params present in action states but feature_config absent.

    Verifies that BaseBayesianNeuralNetwork._upgrade_state correctly infers feature_config from
    the first layer's weight shape during from_old_state, and that the reconstructed CMAB equals
    the original.

    Note: CmabBernoulliCC is excluded because cold_start requires per-action costs; the migration
    path itself (elif branch in update_old_state) is class-agnostic, so coverage here is sufficient.
    """
    action_ids = ["a1", "a2"]
    cmab = CmabClass.cold_start(
        action_ids=action_ids,
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
    )

    # Extract current-format state and strip feature_config from each action — simulating pre-4.4 state.
    # Also strip the version key so from_old_state does not reject a version >= 3.0.0.
    _, state_json = cmab.get_state()
    state = json.loads(state_json)
    state.pop("version", None)
    for action_state in state["actions_manager"]["meta_model"]["actions"].values():
        action_state.pop("feature_config", None)

    reconstructed = CmabClass.from_old_state(json.dumps(state))
    assert reconstructed == cmab


_UPDATE_KWARGS_MIGRATION_CASES = [
    # VI: fit.n → num_steps, fit.method → method
    pytest.param({"fit": {"n": 500, "method": "advi"}}, {"num_steps": 500, "method": "advi"}, id="vi_fit_to_flat"),
    # VI: optimizer_kwargs.learning_rate → step_size
    pytest.param(
        {"optimizer_kwargs": {"learning_rate": 0.05}},
        {"optimizer_kwargs": {"step_size": 0.05}},
        id="vi_optimizer_lr_to_step_size",
    ),
    # VI: combined fit + optimizer migration
    pytest.param(
        {"fit": {"n": 200, "method": "advi"}, "optimizer_kwargs": {"learning_rate": 0.01}},
        {"num_steps": 200, "method": "advi", "optimizer_kwargs": {"step_size": 0.01}},
        id="vi_combined",
    ),
    # update_kwargs=None → no migration, left as None
    pytest.param(None, None, id="vi_none_unchanged"),
]


@pytest.mark.parametrize("old_kwargs,expected_kwargs", _UPDATE_KWARGS_MIGRATION_CASES)
@settings(deadline=500)
@given(n_features=st.integers(min_value=1, max_value=5))
def test_cmab_update_kwargs_migration(
    n_features: int, old_kwargs: Optional[dict], expected_kwargs: Optional[dict]
) -> None:
    """
    Test that update_kwargs in old PyMC-format states are correctly migrated to the NumPyro format
    by BaseCmabBernoulli.update_old_state.

    Migration paths covered:
    - VI: ``fit.n`` → ``num_steps``, ``fit.method`` → ``method``
    - VI: ``optimizer_kwargs.learning_rate`` → ``step_size``
    - ``update_kwargs=None`` → left unchanged
    """
    action_state: dict = {
        "n_successes": 1,
        "n_failures": 1,
        "feature_config": {"n_features": n_features, "categorical_features_configs": []},
        "update_kwargs": deepcopy(old_kwargs),
    }

    state = {
        "actions_manager": {"actions": {"a1": deepcopy(action_state), "a2": deepcopy(action_state)}},
        "strategy": {},
    }

    migrated = CmabBernoulli.update_old_state(state)

    for action_id in ("a1", "a2"):
        actual_kwargs = migrated["actions_manager"]["meta_model"]["actions"][action_id].get("update_kwargs")
        assert actual_kwargs == expected_kwargs


@settings(deadline=500)
@given(n_features=st.integers(min_value=1, max_value=5))
def test_cmab_update_old_state_drops_vi_update_method(n_features: int) -> None:
    """A legacy VI-trained state's ``update_method`` field is dropped, not left as an unknown key."""
    action_state: dict = {
        "n_successes": 1,
        "n_failures": 1,
        "feature_config": {"n_features": n_features, "categorical_features_configs": []},
        "update_method": "VI",
    }
    state = {
        "actions_manager": {"actions": {"a1": deepcopy(action_state)}},
        "strategy": {},
    }

    migrated = CmabBernoulli.update_old_state(state)

    migrated_action_state = migrated["actions_manager"]["meta_model"]["actions"]["a1"]
    assert "update_method" not in migrated_action_state


@settings(deadline=500)
@given(n_features=st.integers(min_value=1, max_value=5))
def test_cmab_update_old_state_rejects_mcmc(n_features: int) -> None:
    """A legacy MCMC-trained state is rejected: the MCMC backend was removed, so it cannot be migrated."""
    action_state: dict = {
        "n_successes": 1,
        "n_failures": 1,
        "feature_config": {"n_features": n_features, "categorical_features_configs": []},
        "update_method": "MCMC",
    }
    state = {
        "actions_manager": {"actions": {"a1": deepcopy(action_state)}},
        "strategy": {},
    }

    with pytest.raises(ValueError, match="MCMC"):
        CmabBernoulli.update_old_state(state)


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(
        st.text(
            min_size=1,
        ),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    epsilon=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    diff=st.data(),
)
def test_pickling(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    diff,
    monkeymodule,
    rng,
):
    # Create CMAB instance
    cmab = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        rng,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
    )[0]
    to_temporary_pickle(cmab)
    apply_mock_update(list(cmab.actions.values()))
    to_temporary_pickle(cmab)


@settings(deadline=500)
@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=100),
    st.just([3]),
)
def test_cmab_update_shape_mismatch(rng, n_samples, n_features, hidden_dim_list):
    actions = rng.choice(["a1", "a2"], size=n_samples).tolist()
    actions[0] = "a1"  # ensure the quantitative arm is present so None-quantity checks trigger
    rewards = rng.choice([0, 1], size=n_samples).tolist()
    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    quantities = rng.uniform(low=0, high=1, size=n_samples).tolist()

    mab = CmabBernoulli.cold_start(
        action_ids={"a1", "a2"},
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
    )
    quant_mab = CmabBernoulli.cold_start(
        action_ids={"a2"}, n_features=n_features, hidden_dim_list=hidden_dim_list, quantitative_action_ids={"a1"}
    )
    quantities = [
        q if isinstance(quant_mab.actions[action], QuantitativeModel) else None
        for q, action in zip(quantities, actions)
    ]

    with pytest.raises(AttributeError):  # actions shape mismatch
        mab.update(context=context, actions=actions[1:], rewards=rewards)
        quant_mab.update(context=context, actions=actions[1:], rewards=rewards, quantities=quantities)
    with pytest.raises(AttributeError):  # rewards shape mismatch
        mab.update(context=context, actions=actions, rewards=rewards[1:])
        quant_mab.update(context=context, actions=actions, rewards=rewards[1:], quantities=quantities)
    with pytest.raises(AttributeError):  # context shape mismatch (rows)
        mab.update(context=context[1:, :], actions=actions, rewards=rewards)
        quant_mab.update(context=context[1:, :], actions=actions, rewards=rewards, quantities=quantities)
    with pytest.raises(AttributeError):  # context shape mismatch (columns)
        mab.update(context=context[:, 1:], actions=actions, rewards=rewards)
        quant_mab.update(context=context[:, 1:], actions=actions, rewards=rewards, quantities=quantities)
    with pytest.raises(AttributeError):  # empty context
        mab.update(context=np.empty((0, n_features)), actions=actions, rewards=rewards)
        quant_mab.update(context=np.empty((0, n_features)), actions=actions, rewards=rewards, quantities=quantities)
    with pytest.raises(AttributeError):  # empty quantities
        quant_mab.update(context=context, actions=actions, rewards=rewards, quantities=[])
    with pytest.raises(ValueError):  # None quantities for quantitative action
        quant_mab.update(context=context, actions=actions, rewards=rewards, quantities=None)
    with pytest.raises(ValueError):  # None quantities for non quantitative action
        mab.update(context=context, actions=actions, rewards=rewards, quantities=quantities)
    with pytest.raises(AttributeError):  # mismatch quantities length
        quant_mab.update(context=context, actions=actions, rewards=rewards, quantities=quantities[:-1])

    # None quantities for quantitative action
    if any([q is not None for q in quantities]):
        quant_index = [i for i, q in enumerate(quantities) if q is not None]
        bad_quantities = quantities.copy()
        bad_quantities[quant_index[0]] = None
        with pytest.raises(ValueError):  # None quantities for quantitative action
            quant_mab.update(context=context, actions=actions, rewards=rewards, quantities=bad_quantities)
    else:
        return


@settings(deadline=500)
@given(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=2))
def test_cmab_predict_shape_mismatch(rng, dim_list):
    n_features = dim_list[0]
    hidden_dim_list = dim_list[1:]
    n_features = dim_list[0]
    context = rng.uniform(low=-1.0, high=1.0, size=(100, n_features - 1))
    mab = CmabBernoulli.cold_start(action_ids={"a1", "a2"}, n_features=n_features, hidden_dim_list=hidden_dim_list)
    with pytest.raises(AttributeError):
        mab.predict(context=context)
    with pytest.raises(AttributeError):
        mab.predict(context=np.empty((0, n_features)))


@settings(deadline=500)
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=5),
    st.just([2]),
    st.integers(min_value=2, max_value=3),
)
def test_cmab_mo_update_shape_mismatch(rng, n_samples, n_features, hidden_dim_list, n_objectives):
    actions = rng.choice(["a1", "a2"], size=n_samples).tolist()
    # Multi-objective rewards
    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # Create multi-objective models
    models_a1 = [
        BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list)
        for _ in range(n_objectives)
    ]
    models_a2 = [
        BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list)
        for _ in range(n_objectives)
    ]
    mab = CmabBernoulliMO(
        actions={
            "a1": BayesianNeuralNetworkMO(models=models_a1),
            "a2": BayesianNeuralNetworkMO(models=models_a2),
        }
    )

    # Test with wrong number of objectives in rewards. No backbone here, so update() dispatches
    # to each arm's own BayesianNeuralNetworkMO.update(), whose own shape check raises AttributeError
    # (properly-nested rewards, just the wrong per-row length).
    wrong_rewards = [[rng.choice([0, 1]) for _ in range(n_objectives + 1)] for _ in range(n_samples)]
    with pytest.raises(AttributeError):
        mab.update(context=context, actions=actions, rewards=wrong_rewards)

    # Test with single-objective (flat, not nested) rewards — should fail for MO model. This isn't
    # merely the wrong shape but the wrong structure entirely (List[int] instead of List[List[int]]),
    # so pydantic's own @validate_call type enforcement on BayesianNeuralNetworkMO.update() rejects it
    # before the model's manual shape check ever runs.
    single_rewards = rng.choice([0, 1], size=n_samples).tolist()
    with pytest.raises(ValidationError):
        mab.update(context=context, actions=actions, rewards=single_rewards)


@pytest.mark.parametrize(
    "prob_weight,index,expected",
    [
        # Test with ProbabilityWeight (tuple)
        ((0.7, 150.0), 0, 0.7),
        ((0.7, 150.0), 1, 150.0),
        ((0.7, 150.0), -1, 150.0),
        ((0.7, 150.0), -2, 0.7),
        # Test with MOProbabilityWeight (list of tuples)
        ([(0.7, 150.0), (0.8, 200.0)], 0, [0.7, 0.8]),
        ([(0.7, 150.0), (0.8, 200.0)], 1, [150.0, 200.0]),
        ([(0.7, 150.0), (0.8, 200.0)], -1, [150.0, 200.0]),
        ([(0.7, 150.0), (0.8, 200.0)], -2, [0.7, 0.8]),
        # Test with empty list
        ([], 0, []),
        ([], 1, []),
        # Test with single element list
        ([(0.6, 120.0)], 0, [0.6]),
        ([(0.6, 120.0)], 1, [120.0]),
        # Test with larger list
        ([(0.1, 10.0), (0.2, 20.0), (0.3, 30.0), (0.4, 40.0), (0.5, 50.0)], 0, [0.1, 0.2, 0.3, 0.4, 0.5]),
        ([(0.1, 10.0), (0.2, 20.0), (0.3, 30.0), (0.4, 40.0), (0.5, 50.0)], 1, [10.0, 20.0, 30.0, 40.0, 50.0]),
    ],
)
def test_extract_element_from_probability_weight_valid_cases(
    prob_weight: Union[Tuple[float, float], List[Tuple[float, float]]], index: int, expected: Union[float, List[float]]
) -> None:
    """Test extracting element from probability weight with valid inputs."""
    result = BaseCmabBernoulli._extract_element_from_probability_weight(index, prob_weight)
    assert result == expected


@pytest.mark.parametrize(
    "prob_weight,index",
    [
        # Test invalid index for tuple
        ((0.7, 150.0), 2),
        ((0.7, 150.0), 3),
        ((0.7, 150.0), -3),
        # Test invalid index for list
        ([(0.7, 150.0), (0.8, 200.0)], 2),
        ([(0.7, 150.0), (0.8, 200.0)], 3),
        ([(0.7, 150.0), (0.8, 200.0)], -3),
    ],
)
def test_extract_element_from_probability_weight_invalid_index(
    prob_weight: Union[Tuple[float, float], List[Tuple[float, float]]], index: int
) -> None:
    """Test extracting element with invalid index raises IndexError."""
    with pytest.raises(IndexError):
        BaseCmabBernoulli._extract_element_from_probability_weight(index, prob_weight)


@pytest.mark.parametrize(
    "unsupported_type",
    ["string", 42, 3.14, {"key": "value"}, {1.0, 2.0, 3.0}, None, [1.0, 2.0, 3.0]],
)
def test_extract_element_from_probability_weight_unsupported_type(unsupported_type: Any) -> None:
    """Test that unsupported types raise TypeError."""
    with pytest.raises(TypeError, match=f"Unsupported probability weight type: {type(unsupported_type)}"):
        BaseCmabBernoulli._extract_element_from_probability_weight(0, unsupported_type)


@settings(deadline=None, max_examples=10)
@given(
    action_ids=st.lists(st.text(min_size=1), min_size=1, max_size=4, unique=True),
    n_samples=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=4),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_random_seed_propagates_to_bnn(
    action_ids: List[str], n_samples: int, n_features: int, random_seed: int, rng: np.random.Generator
) -> None:
    """Verify that random_seed set on the CMAB cold_start flows through to every BNN action model.

    The seed must appear on each BNN so that both the MAB-level RNG (epsilon-greedy,
    Thompson sampling) and the BNN-level JAX RNG (VI training) are reproducible
    from a single top-level parameter.
    """
    cmab1 = CmabBernoulli.cold_start(
        action_ids=action_ids,
        strategy=ClassicBandit(),
        n_features=n_features,
        random_seed=random_seed,
    )

    assert cmab1.random_seed == random_seed, "MAB should store the random_seed"
    for action_id, model in cmab1.actions.items():
        assert isinstance(model, BaseBayesianNeuralNetwork)
        assert model.random_seed == random_seed, (
            f"Action '{action_id}': expected BNN random_seed={random_seed}, got {model.random_seed}"
        )

    apply_mock_update(list(cmab1.actions.values()))
    # Deep-copy after mock update so both instances share identical layer params AND rng state.
    cmab2 = CmabBernoulli.from_state(cmab1.get_state()[1])

    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    actions1, probs1, ws1 = cmab1.predict(context=context)
    actions2, probs2, ws2 = cmab2.predict(context=context)

    assert actions1 == actions2
    assert probs1 == probs2
    assert ws1 == ws2


########################################################################################################################
# Region-aware forbidden_actions (forbidding part of a quantitative arm's hypercube) — CMAB edition


class TestForbiddenRegions:
    """Region-aware forbidden_actions: forbidding part of a quantitative arm's hypercube (cMAB edition)."""

    # A forbidden region on a 1-D quantitative arm follows the float signed-margin convention: forbidden where
    # region(x) > 0, i.e. the upper half-interval x[0] > region_split is blocked.
    region_split = 0.5
    region_tolerance = 1e-2
    explore_samples = 50
    hypothesis_exploit_samples = 5  # small enough to stay under hypothesis's default 200ms deadline
    explore_seed = 123
    n_features = 2
    quantitative_dimension = 1
    quantitative_arm_id = "q"
    discrete_arm_id = "d"
    epsilon_full_explore = 1.0  # forces the random explore branch on every step
    context_low = -1.0
    context_high = 1.0
    max_rejection_samples = 1000  # rejection-sampling budget for _constraint_aware_maximize

    @staticmethod
    def _forbid_upper_half(x: np.ndarray, split: float = region_split) -> float:
        """Forbidden-region predicate: x[0] > split is forbidden (region(x) > 0 => forbidden)."""
        return float(x[0]) - split

    @staticmethod
    def _always_forbidden(x: np.ndarray) -> float:
        """Forbidden-region predicate that forbids the entire quantity space (margin always positive)."""
        return 1.0

    @staticmethod
    def _constraint_aware_maximize(
        quantity_score_func,
        dimension: int,
        constraint=None,
        n_trials: int = 10000,
        max_rejection_samples: int = max_rejection_samples,
        rng: np.random.Generator = None,
        **kwargs,
    ) -> np.ndarray:
        """Mock maximize_by_quantity: rejection-samples a feasible quantity; np.zeros fallback is unreachable."""
        constraints = constraint or []
        for _ in range(max_rejection_samples):
            candidate = rng.random(dimension)
            if all(c(candidate) >= 0 for c in constraints):
                return candidate
        return np.zeros(dimension)

    @pytest.fixture(scope="class")
    def make_cmab(self):
        """Factory: builds a cMAB with one continuous (QuantitativeBayesianNeuralNetwork) arm and one BNN arm."""

        def _factory(**kwargs) -> CmabBernoulli:
            return CmabBernoulli(
                actions={
                    self.quantitative_arm_id: QuantitativeBayesianNeuralNetwork.cold_start(
                        dimension=self.quantitative_dimension, n_features=self.n_features
                    ),
                    self.discrete_arm_id: BayesianNeuralNetwork.cold_start(n_features=self.n_features),
                },
                **kwargs,
            )

        return _factory

    def test_normalize_forbidden_actions_forms(
        self,
        make_cmab,
        quantitative_arm_id: str = quantitative_arm_id,
        discrete_arm_id: str = discrete_arm_id,
    ) -> None:
        """_normalize_forbidden_actions handles the Set, dict-None (whole arm) and dict-region (partial) forms."""
        cmab = make_cmab()

        # Legacy Set form: whole-arm blocking, no region constraints.
        valid, regions = cmab._normalize_forbidden_actions({discrete_arm_id})
        assert valid == {quantitative_arm_id} and regions == {}

        # dict with None value is equivalent to whole-arm blocking.
        valid, regions = cmab._normalize_forbidden_actions({discrete_arm_id: None})
        assert valid == {quantitative_arm_id} and regions == {}

        # dict with a region keeps the arm valid (only its quantity space shrinks) and records its constraint.
        valid, regions = cmab._normalize_forbidden_actions({quantitative_arm_id: self._forbid_upper_half})
        assert valid == {quantitative_arm_id, discrete_arm_id}
        assert set(regions) == {quantitative_arm_id} and len(regions[quantitative_arm_id]) == 1

    def test_region_rejected_on_discrete_arm(self, make_cmab, discrete_arm_id: str = discrete_arm_id) -> None:
        """A forbidden region cannot be attached to a non-quantitative arm."""
        cmab = make_cmab()
        with pytest.raises(ValueError, match="quantitative"):
            cmab._normalize_forbidden_actions({discrete_arm_id: self._forbid_upper_half})

    @settings(deadline=None)
    @given(
        quantity=st.floats(min_value=region_split, max_value=1.0, exclude_min=True),
        epsilon=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
        arm_id=st.just(quantitative_arm_id),
    )
    def test_default_action_in_region_raises(self, make_cmab, quantity: float, epsilon: float, arm_id: str) -> None:
        """Any quantitative default action strictly above region_split is rejected when the upper half is forbidden."""
        cmab = make_cmab(epsilon=epsilon, default_action=(arm_id, (quantity,)))
        with pytest.raises(ValueError, match="forbidden region"):
            cmab._normalize_forbidden_actions({arm_id: self._forbid_upper_half})

    @given(
        region_split=st.floats(min_value=0.1, max_value=0.9),
        arm_id=st.just(quantitative_arm_id),
        random_seed=st.just(explore_seed),
    )
    def test_sample_allowed_quantity_respects_and_exhausts_region(
        self,
        make_cmab,
        region_split: float,
        arm_id: str,
        random_seed: int,
    ) -> None:
        """_sample_allowed_quantity returns a point outside the region, or None when the whole space is forbidden."""
        cmab = make_cmab(random_seed=random_seed)
        _, regions = cmab._normalize_forbidden_actions({arm_id: partial(self._forbid_upper_half, split=region_split)})

        allowed = cmab._sample_allowed_quantity(arm_id, regions)
        assert allowed is not None and allowed[0] <= region_split

        # A region that forbids the entire cube yields no allowed point.
        _, full_regions = cmab._normalize_forbidden_actions({arm_id: self._always_forbidden})
        assert cmab._sample_allowed_quantity(arm_id, full_regions) is None

    def test_predict_explore_never_selects_forbidden_region(
        self,
        monkeypatch: MonkeyPatch,
        arm_id: str = quantitative_arm_id,
        dimension: int = quantitative_dimension,
        n_features: int = n_features,
        epsilon: float = epsilon_full_explore,
        random_seed: int = explore_seed,
        n_samples: int = explore_samples,
        context_low: float = context_low,
        context_high: float = context_high,
        region_split: float = region_split,
        tolerance: float = region_tolerance,
        rng: np.random.Generator = None,
    ) -> None:
        """With epsilon=1 (always explore), the random quantitative quantity always avoids the forbidden region."""
        cmab = CmabBernoulli(
            actions={arm_id: QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=n_features)},
            epsilon=epsilon,
            random_seed=random_seed,
        )
        apply_mock_update(list(cmab.actions.values()))
        context = np.random.default_rng(random_seed).uniform(context_low, context_high, size=(n_samples, n_features))
        monkeypatch.setattr(
            pybandits.strategy.single_objective,
            "maximize_by_quantity",
            lambda quantity_score_func, dimension, constraint=None, n_trials=10000: rng.random(dimension),
        )

        selected_actions, _, _ = cmab.predict(context=context, forbidden_actions={arm_id: self._forbid_upper_half})

        assert all(isinstance(action, tuple) and action[0] == arm_id for action in selected_actions)
        assert all(action[1][0] <= region_split + tolerance for action in selected_actions)

    @given(
        region_split=st.floats(min_value=0.1, max_value=0.9),
        arm_id=st.just(quantitative_arm_id),
        dimension=st.just(quantitative_dimension),
        n_features=st.just(n_features),
        random_seed=st.just(explore_seed),
        n_samples=st.just(hypothesis_exploit_samples),
        context_low=st.just(context_low),
        context_high=st.just(context_high),
        tolerance=st.just(region_tolerance),
    )
    def test_predict_exploit_respects_forbidden_region(
        self,
        region_split: float,
        arm_id: str,
        dimension: int,
        n_features: int,
        random_seed: int,
        n_samples: int,
        context_low: float,
        context_high: float,
        tolerance: float,
        rng: np.random.Generator,
    ) -> None:
        """Constrained optimizer stays below any forbidden boundary, not just the default split of 0.5."""

        def forbid_above(x: np.ndarray) -> float:
            return float(x[0]) - region_split

        cmab = CmabBernoulli(
            actions={arm_id: QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=n_features)},
            random_seed=random_seed,
        )
        apply_mock_update(list(cmab.actions.values()))
        context = np.random.default_rng(random_seed).uniform(context_low, context_high, size=(n_samples, n_features))

        with patch.object(
            pybandits.strategy.single_objective,
            "maximize_by_quantity",
            partial(self._constraint_aware_maximize, rng=rng),
        ):
            selected_actions, _, _ = cmab.predict(context=context, forbidden_actions={arm_id: forbid_above})

        assert all(isinstance(action, tuple) and action[0] == arm_id for action in selected_actions)
        assert all(action[1][0] <= region_split + tolerance for action in selected_actions)

    @given(
        seg_lo=st.floats(min_value=0.05, max_value=0.45),
        seg_hi=st.floats(min_value=0.55, max_value=0.95),
        arm_id=st.just(quantitative_arm_id),
        dimension=st.just(quantitative_dimension),
        n_features=st.just(n_features),
        random_seed=st.just(explore_seed),
        n_samples=st.just(hypothesis_exploit_samples),
        context_low=st.just(context_low),
        context_high=st.just(context_high),
        tolerance=st.just(region_tolerance),
    )
    def test_segment_region_outside_end_to_end(
        self,
        seg_lo: float,
        seg_hi: float,
        arm_id: str,
        dimension: int,
        n_features: int,
        random_seed: int,
        n_samples: int,
        context_low: float,
        context_high: float,
        tolerance: float,
        rng: np.random.Generator,
    ) -> None:
        """forbidden_region_outside keeps the exploited quantity inside any valid allowed segment."""
        allowed = Segment(intervals=((seg_lo, seg_hi),))
        cmab = CmabBernoulli(
            actions={arm_id: QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=n_features)},
            random_seed=random_seed,
        )
        apply_mock_update(list(cmab.actions.values()))
        context = np.random.default_rng(random_seed).uniform(context_low, context_high, size=(n_samples, n_features))
        region = allowed.forbidden_region_outside()

        with patch.object(
            pybandits.strategy.single_objective,
            "maximize_by_quantity",
            partial(self._constraint_aware_maximize, rng=rng),
        ):
            selected_actions, _, _ = cmab.predict(context=context, forbidden_actions={arm_id: region})

        assert all(seg_lo - tolerance <= action[1][0] <= seg_hi for action in selected_actions)


########################################################################################################################
# --- shared-backbone (cold_start(backbone_hidden_dims=...)) validation constants ---
# The backbone lifecycle (cold_start / update / predict / serialization) is covered by the
# ``backbone_hidden_dims`` parametrization of test_update / test_predict / test_serialization;
# the tests below cover only backbone-specific construction/validation behaviour.
_BACKBONE_ACTION_IDS = {"a", "b"}
_BACKBONE_N_FEATURES = 8
_BACKBONE_HIDDEN_DIMS = [16, 8]
_BACKBONE_NUM_STEPS = 15


def test_cold_start_without_backbone_builds_independent_heads() -> None:
    """Without ``backbone_hidden_dims`` the factory builds a no-backbone CmabMetaModel of BNN heads."""
    cmab = CmabBernoulli.cold_start(action_ids=_BACKBONE_ACTION_IDS, n_features=_BACKBONE_N_FEATURES, random_seed=1)
    assert cmab.actions_manager.meta_model.backbone is None
    assert all(isinstance(model, BaseBayesianNeuralNetwork) for model in cmab.actions_manager.actions.values())


def test_cold_start_with_backbone_reports_raw_input_dim() -> None:
    """A shared backbone is built and ``input_dim`` reports the raw feature count (not the embedding)."""
    cmab = CmabBernoulli.cold_start(
        action_ids=_BACKBONE_ACTION_IDS,
        n_features=_BACKBONE_N_FEATURES,
        backbone_hidden_dims=_BACKBONE_HIDDEN_DIMS,
        update_kwargs={"num_steps": _BACKBONE_NUM_STEPS},
    )
    assert cmab.actions_manager.meta_model.backbone is not None
    assert cmab.input_dim == _BACKBONE_N_FEATURES


@pytest.mark.parametrize(
    "kwarg_name, value_strategy",
    [
        ("backbone_embedding_dim", st.integers(min_value=1, max_value=64)),
        ("backbone_activation", st.sampled_from(get_args(ActivationFunctions))),
        ("backbone_l2_anchoring", st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)),
        (
            "backbone_lr",
            st.one_of(st.none(), st.floats(min_value=0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        ),
    ],
)
@given(data=st.data())
def test_cold_start_rejects_backbone_only_kwargs_without_backbone(
    kwarg_name: str, value_strategy: st.SearchStrategy, data: st.DataObject
) -> None:
    """A backbone-only knob, at any valid value, raises instead of being silently ignored when no
    backbone is requested (no ``backbone_hidden_dims``)."""
    value = data.draw(value_strategy)
    with pytest.raises(TypeError, match="only apply with a backbone"):
        CmabBernoulli.cold_start(
            action_ids=_BACKBONE_ACTION_IDS, n_features=_BACKBONE_N_FEATURES, **{kwarg_name: value}
        )


@given(delta=st.floats(min_value=0, max_value=1, exclude_min=True, allow_nan=False, allow_infinity=False))
def test_cold_start_rejects_delta_with_backbone(delta: float) -> None:
    """The shared-backbone path does not support the adaptive window (``delta``) yet."""
    with pytest.raises(ValueError, match="adaptive window"):
        CmabBernoulli.cold_start(
            action_ids=_BACKBONE_ACTION_IDS,
            n_features=_BACKBONE_N_FEATURES,
            backbone_hidden_dims=_BACKBONE_HIDDEN_DIMS,
            delta=delta,
        )
