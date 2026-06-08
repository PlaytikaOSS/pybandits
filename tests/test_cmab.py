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
from typing import Any, Dict, List, Optional, Tuple, Type, Union
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
    UpdateMethods,
)
from pybandits.quantitative_model import (
    BaseQuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetworkCC,
    QuantitativeBayesianNeuralNetworkDP,
    QuantitativeModel,
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
    literal_update_methods,
    mock_update,
    sample_with_replacement,
    to_temporary_pickle,
)


def _apply_update_method_to_state(state, update_method):
    for model_state in state["actions_manager"]["meta_model"]["actions"].values():
        model_state["update_method"] = update_method


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
        update_method: UpdateMethods,
        n_objectives: Optional[PositiveInt] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_types = list(self.model_types)
        if len(model_types) < len(action_ids):
            indices = np.random.randint(0, len(model_types), len(action_ids))
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

        model_cold_start_kwargs = dict(update_method=update_method)
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
        update_method: UpdateMethods,
    ) -> Tuple[BaseCmabBernoulli, Dict[ActionId, CmabModelType], Dict[str, Any]]:
        n_objectives = (
            n_objectives.draw(st.integers(min_value=1, max_value=10))
            if self.cmab_class in [CmabBernoulliMO, CmabBernoulliMOCC]
            else None
        )
        actions, base_model_cold_start_kwargs = self._create_actions(
            action_ids, values, n_features, hidden_dim_list, update_method, n_objectives
        )
        default_action = action_ids[0] if epsilon and not delta else None
        if default_action and isinstance(actions[default_action], QuantitativeModel):
            default_action = (default_action, tuple(np.random.random(actions[default_action].dimension)))
        epsilon = epsilon if not delta else 0.1
        kwargs = {
            k: v
            for k, v in {
                "epsilon": epsilon,
                "default_action": default_action,
                "delta": delta,
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

        cmab = self.cmab_class(actions=actions, **kwargs)
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
        return cmab, actions, kwargs


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
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
)
def test_cold_start(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
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
        update_method,
    )

    # Cold start comparison logic (modified for different model types)
    cold_start_kwargs = {
        "action_ids": {
            action
            for action, model in actions.items()
            if isinstance(model, (BaseBayesianNeuralNetwork, BaseBayesianNeuralNetworkMO))
        },
        "quantitative_action_ids": {
            action for action, model in actions.items() if isinstance(model, QuantitativeModel)
        },
    }
    if all(
        isinstance(model, (BayesianNeuralNetworkCC, BayesianNeuralNetworkMOCC, QuantitativeBayesianNeuralNetworkCC))
        for model in actions.values()
    ):
        cold_start_kwargs["action_ids_cost"] = {
            action: model.cost
            for action, model in actions.items()
            if isinstance(model, (BayesianNeuralNetworkCC, BayesianNeuralNetworkMOCC))
        }
        cold_start_kwargs["quantitative_action_ids_cost"] = {
            action: model.cost
            for action, model in actions.items()
            if isinstance(model, QuantitativeBayesianNeuralNetworkCC)
        }
    if all(
        isinstance(model, (BayesianNeuralNetworkDP, QuantitativeBayesianNeuralNetworkDP)) for model in actions.values()
    ):
        cold_start_kwargs["action_ids_price"] = {
            action: model.price for action, model in actions.items() if isinstance(model, BayesianNeuralNetworkDP)
        }
        cold_start_kwargs["quantitative_action_ids_price"] = {
            action: model.price
            for action, model in actions.items()
            if isinstance(model, QuantitativeBayesianNeuralNetworkDP)
        }
    cold_start_kwargs.update(kwargs)  # Add exploit_p or subsidy_factor if needed
    cold_start_kwargs = {k: v for k, v in cold_start_kwargs.items() if v is not None}
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
    update_method=st.sampled_from(literal_update_methods),
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
    update_method,
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

    # Test mismatched update methods
    actions_wrong_update = {
        action_ids[0]: config.model_types[0].cold_start(update_method="VI", **kwargs),
        action_ids[1]: config.model_types[0].cold_start(update_method="MCMC", **kwargs),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_update)

    # Test mismatched update kwargs
    mismatch_kw1, mismatch_kw2 = (
        ({"num_steps": 100}, {"num_steps": 200}) if update_method == "VI" else ({"num_warmup": 10}, {"num_warmup": 20})
    )
    actions_wrong_kwargs = {
        action_ids[0]: config.model_types[0].cold_start(
            update_method=update_method,
            update_kwargs=mismatch_kw1,
            **kwargs,
        ),
        action_ids[1]: config.model_types[0].cold_start(
            update_method=update_method,
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
                update_method,
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
                update_method,
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
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    memory_len=st.integers(min_value=1, max_value=5),
)
def test_update(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    memory_len,
):
    # Create CMAB instance
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
        update_method,
    )
    # create patches

    n_objectives = kwargs.get("n_objectives")
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    # Generate random rewards
    reward_data = (
        np.random.choice([0, 1], size=(n_samples, n_objectives), replace=True)
        if n_objectives
        else np.random.choice([0, 1], size=n_samples, replace=True)
    )
    reward_data = reward_data.tolist()
    # Test updates with generated data
    actions_to_update = sample_with_replacement(action_ids, n_samples)
    # Generate quantities only if there are any QuantitativeModel actions
    # Handle multi-objective rewards for MO models

    for_update_kwargs = {"actions": actions_to_update, "rewards": reward_data}
    if any(isinstance(model, BaseQuantitativeBayesianNeuralNetwork) for model in cmab.actions.values()):
        quantity_data = np.random.random(size=n_samples).tolist()
        quantity_data = [
            q if isinstance(cmab.actions[action], QuantitativeModel) else None
            for q, action in zip(quantity_data, actions_to_update)
        ]
        for_update_kwargs["quantities"] = quantity_data

    old_cmab = deepcopy(cmab)
    for k, transform in enumerate([lambda x: x, np.array, lambda x: x.copy()]):
        with (
            patch.object(BaseBayesianNeuralNetwork, "_update", mock_update),
            patch.object(QuantitativeModel, "_update", mock_update),
        ):
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
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    diff=st.data(),
)
def test_predict(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    diff,
    monkeymodule,
):
    def mock_maximize_by_quantity(quantity_score_func, dimension, constraint=None, n_trials=10000):
        """Mock maximize_by_quantity to return a quick result."""
        return np.random.random(dimension)

    monkeymodule.setattr(pybandits.strategy, "maximize_by_quantity", mock_maximize_by_quantity)

    if config.cmab_class in (CmabBernoulliMO, CmabBernoulliMOCC):

        def mock_find_pareto_front_normal_constraint(self, func, input_dim, n_objectives, n_divisions, model):
            """Mock _find_pareto_front_normal_constraint to return a quick result."""
            return [np.random.random(input_dim) for _ in range(min(3, n_divisions))]

        monkeymodule.setattr(
            pybandits.strategy.MultiObjectiveStrategy,
            "_find_pareto_front_normal_constraint",
            mock_find_pareto_front_normal_constraint,
        )

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
        update_method,
    )[0]
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # Test predictions with random forbidden actions
    forbidden = set(sample_with_replacement(action_ids, len(action_ids) // 2)) if len(action_ids) > 2 else None
    if cmab.default_action is not None and forbidden is not None and cmab.default_action in forbidden:
        forbidden.remove(cmab.default_action)
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
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    diff=st.data(),
)
def test_serialization(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    diff,
    monkeymodule,
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
        update_method,
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
    update_method=st.sampled_from(literal_update_methods),
)
def test_cmab_from_old_state_missing_feature_config(
    CmabClass,
    n_features: int,
    hidden_dim_list: List[int],
    update_method: str,
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
        update_method=update_method,
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
    pytest.param(
        "VI", {"fit": {"n": 500, "method": "advi"}}, {"num_steps": 500, "method": "advi"}, id="vi_fit_to_flat"
    ),
    # VI: optimizer_kwargs.learning_rate → step_size
    pytest.param(
        "VI",
        {"optimizer_kwargs": {"learning_rate": 0.05}},
        {"optimizer_kwargs": {"step_size": 0.05}},
        id="vi_optimizer_lr_to_step_size",
    ),
    # VI: combined fit + optimizer migration
    pytest.param(
        "VI",
        {"fit": {"n": 200, "method": "advi"}, "optimizer_kwargs": {"learning_rate": 0.01}},
        {"num_steps": 200, "method": "advi", "optimizer_kwargs": {"step_size": 0.01}},
        id="vi_combined",
    ),
    # MCMC: trace fields → flat keys
    pytest.param(
        "MCMC",
        {"trace": {"tune": 100, "draws": 200, "chains": 2, "progressbar": True}},
        {"num_warmup": 100, "num_samples": 200, "num_chains": 2, "progress_bar": True},
        id="mcmc_trace_to_flat",
    ),
    # MCMC: target_accept → nuts.target_accept_prob
    pytest.param(
        "MCMC",
        {"trace": {"target_accept": 0.8}},
        {"nuts": {"target_accept_prob": 0.8}},
        id="mcmc_target_accept_to_nuts",
    ),
    # MCMC: PyMC-only keys stripped entirely
    pytest.param(
        "MCMC",
        {"trace": {"init": "auto", "cores": 1, "return_inferencedata": False}},
        {},
        id="mcmc_pymc_only_keys_stripped",
    ),
    # update_kwargs=None → no migration, left as None
    pytest.param("VI", None, None, id="vi_none_unchanged"),
]


@pytest.mark.parametrize("update_method,old_kwargs,expected_kwargs", _UPDATE_KWARGS_MIGRATION_CASES)
@settings(deadline=500)
@given(n_features=st.integers(min_value=1, max_value=5))
def test_cmab_update_kwargs_migration(
    n_features: int, update_method: str, old_kwargs: Optional[dict], expected_kwargs: Optional[dict]
) -> None:
    """
    Test that update_kwargs in old PyMC-format states are correctly migrated to the NumPyro format
    by BaseCmabBernoulli.update_old_state.

    Migration paths covered:
    - VI: ``fit.n`` → ``num_steps``, ``fit.method`` → ``method``
    - VI: ``optimizer_kwargs.learning_rate`` → ``step_size``
    - MCMC: ``trace.tune/draws/chains/progressbar`` → flat keys
    - MCMC: ``trace.target_accept`` → ``nuts.target_accept_prob``
    - MCMC: PyMC-only keys (init, cores, return_inferencedata) stripped
    - ``update_kwargs=None`` → left unchanged
    """
    action_state: dict = {
        "n_successes": 1,
        "n_failures": 1,
        "feature_config": {"n_features": n_features, "categorical_features_configs": []},
        "update_method": update_method,
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
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    diff=st.data(),
)
def test_pickling(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    values,
    n_objectives,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    diff,
    monkeymodule,
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
        update_method,
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
def test_cmab_update_shape_mismatch(n_samples, n_features, hidden_dim_list):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    quantities = np.random.uniform(low=0, high=1, size=n_samples).tolist()

    mab = CmabBernoulli.cold_start(
        action_ids={"a1", "a2"},
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
    )
    quant_mab = CmabBernoulli.cold_start(
        action_ids={"a1", "a2"}, n_features=n_features, hidden_dim_list=hidden_dim_list, quantitative_action_ids={"a1"}
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
def test_cmab_predict_shape_mismatch(dim_list):
    n_features = dim_list[0]
    hidden_dim_list = dim_list[1:]
    n_features = dim_list[0]
    context = np.random.uniform(low=-1.0, high=1.0, size=(100, n_features - 1))
    mab = CmabBernoulli.cold_start(action_ids={"a1", "a2"}, n_features=n_features, hidden_dim_list=hidden_dim_list)
    with pytest.raises(AttributeError):
        mab.predict(context=context)
    with pytest.raises(AttributeError):
        mab.predict(context=np.empty((0, n_features)))


@settings(deadline=500)
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=5),
    st.sampled_from(literal_update_methods),
    st.just([2]),
    st.integers(min_value=2, max_value=3),
)
def test_cmab_mo_update_shape_mismatch(n_samples, n_features, update_method, hidden_dim_list, n_objectives):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    # Multi-objective rewards
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # Create multi-objective models
    models_a1 = [
        BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        for _ in range(n_objectives)
    ]
    models_a2 = [
        BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        for _ in range(n_objectives)
    ]
    mab = CmabBernoulliMO(
        actions={
            "a1": BayesianNeuralNetworkMO(models=models_a1),
            "a2": BayesianNeuralNetworkMO(models=models_a2),
        }
    )

    # Test with wrong number of objectives in rewards
    wrong_rewards = [[np.random.choice([0, 1]) for _ in range(n_objectives + 1)] for _ in range(n_samples)]
    with pytest.raises(AttributeError):
        mab.update(context=context, actions=actions, rewards=wrong_rewards)

    # Test with single-objective rewards (should fail for MO model)
    single_rewards = np.random.choice([0, 1], size=n_samples).tolist()
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
    action_ids: List[str], n_samples: int, n_features: int, random_seed: int
) -> None:
    """Verify that random_seed set on the CMAB cold_start flows through to every BNN action model.

    The seed must appear on each BNN so that both the MAB-level RNG (epsilon-greedy,
    Thompson sampling) and the BNN-level JAX RNG (VI/MCMC training) are reproducible
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

    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    actions1, probs1, ws1 = cmab1.predict(context=context)
    actions2, probs2, ws2 = cmab2.predict(context=context)

    assert actions1 == actions2
    assert probs1 == probs2
    assert ws1 == ws2
