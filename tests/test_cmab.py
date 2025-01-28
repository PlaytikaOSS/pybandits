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
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic.dataclasses import dataclass
from pytest import MonkeyPatch

import pybandits
from pybandits.actions_manager import CmabModelType
from pybandits.base import ActionId, Float01, PositiveProbability, PyBanditsBaseModel
from pybandits.cmab import BaseCmabBernoulli, CmabBernoulli, CmabBernoulliBAI, CmabBernoulliCC
from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BnnLayerParams,
    BnnParams,
    StudentTArray,
    UpdateMethods,
)
from pybandits.pydantic_version_compatibility import (
    PositiveInt,
    ValidationError,
)
from pybandits.quantitative_model import BaseCmabZoomingModel, CmabZoomingModel, CmabZoomingModelCC, QuantitativeModel
from pybandits.strategy import BestActionIdentificationBandit, ClassicBandit, CostControlBandit
from tests.test_utils import (
    FakeApproximation,
    FakePrediction,
    literal_update_methods,
    sample_with_replacement,
    to_temporary_pickle,
)


@st.composite
def diff_strategy(draw):
    return draw(st.floats(min_value=0.001, max_value=0.5))


@st.composite
def cost_strategy(draw, n_actions):
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


def _is_of_relevant_types(member: Any):
    """
    Check if the member is of a relevant type for mocking.

    Parameters
    ----------
    member : Any
        The member to check.

    Returns
    -------
    bool
        True if the member is of a relevant type, False otherwise.
    """
    return isinstance(member, (PyBanditsBaseModel, BnnParams, BnnLayerParams, StudentTArray))


def find_and_update_parameters(obj, diff, monkeymodule, label):
    stack = [obj]

    while stack:
        current = stack.pop()
        if isinstance(current, StudentTArray):
            label = mock_student_t_array(current, diff, monkeymodule, label)
            continue

        for attr in dir(current):
            if not attr.startswith("__"):
                member = getattr(current, attr)
                if _is_of_relevant_types(member):
                    stack.append(member)
                elif isinstance(member, Sequence):
                    stack.extend(item for item in member if _is_of_relevant_types(item))

    return label


def mock_update(models: Union[List[BaseBayesianNeuralNetwork], BaseBayesianNeuralNetwork], diff, monkeymodule, label=0):
    model_list = [models] if isinstance(models, BaseBayesianNeuralNetwork) else models
    for model in model_list:
        label = find_and_update_parameters(model, diff, monkeymodule, label)


@dataclass
class ModelTestConfig:
    cmab_class: Type
    strategy_class: Type
    model_types: List[Type[CmabModelType]]

    @staticmethod
    def _quantitative_cost(x, cost):
        return x**cost

    def _create_actions(
        self,
        action_ids: List[str],
        costs: Optional[st.SearchStrategy],
        n_features: PositiveInt,
        hidden_dim_list: List[int],
        update_method: UpdateMethods,
        update_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if len(self.model_types) < len(action_ids):
            indices = np.random.randint(0, len(self.model_types), len(action_ids))
            self.model_types = [self.model_types[i] for i in indices]
        if all(model in [BayesianNeuralNetworkCC, CmabZoomingModelCC] for model in self.model_types):
            # Generate random costs
            costs = costs.draw(cost_strategy(n_actions=len(action_ids)))
            costs = [
                cost if model_type in [BayesianNeuralNetworkCC] else partial(self._quantitative_cost, cost=cost)
                for cost, model_type in zip(costs, self.model_types)
            ]
        else:
            costs = None

        model_cold_start_kwargs = dict(update_method=update_method, update_kwargs=update_kwargs)
        base_model_cold_start_kwargs = dict(
            n_features=n_features, hidden_dim_list=hidden_dim_list, **model_cold_start_kwargs
        )
        model_params = BaseBayesianNeuralNetwork.create_model_params(
            n_features=n_features, hidden_dim_list=hidden_dim_list
        )

        if costs is not None:
            # Handle models with costs (BayesianNeuralNetworkCC or CmabZoomingModelCC)
            actions_dict = {}
            for action_id, model_type, cost in zip(action_ids, self.model_types, costs):
                if issubclass(model_type, BayesianNeuralNetworkCC):
                    actions_dict[action_id] = model_type(
                        model_params=model_params,
                        **model_cold_start_kwargs,
                        cost=cost,
                    )
                else:  # CmabZoomingModelCC
                    actions_dict[action_id] = model_type.cold_start(
                        dimension=1, base_model_cold_start_kwargs=base_model_cold_start_kwargs, cost=cost
                    )
        else:
            # Handle models without costs (BayesianNeuralNetwork or CmabZoomingModel)
            actions_dict = {}
            for action_id, model_type in zip(action_ids, self.model_types):
                if issubclass(model_type, BayesianNeuralNetwork):
                    actions_dict[action_id] = model_type(model_params=model_params, **model_cold_start_kwargs)
                else:  # CmabZoomingModel
                    actions_dict[action_id] = model_type.cold_start(
                        dimension=1,
                        base_model_cold_start_kwargs=base_model_cold_start_kwargs,
                    )
        return actions_dict, base_model_cold_start_kwargs

    def create_cmab_and_actions(
        self,
        action_ids: List[str],
        epsilon: Optional[Float01],
        delta: Optional[PositiveProbability],
        costs: st.SearchStrategy,
        exploit_p: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        subsidy_factor: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        n_features: PositiveInt,
        hidden_dim_list: List[int],
        update_method: UpdateMethods,
        update_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[BaseCmabBernoulli, Dict[ActionId, CmabModelType], Dict[str, Any]]:
        actions, base_model_cold_start_kwargs = self._create_actions(
            action_ids, costs, n_features, hidden_dim_list, update_method, update_kwargs
        )
        default_action = action_ids[0] if epsilon and not delta else None
        if default_action and isinstance(self.model_types[0], QuantitativeModel):
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
        if any(isinstance(model, BaseCmabZoomingModel) for model in actions.values()):
            kwargs["base_model_cold_start_kwargs"] = base_model_cold_start_kwargs
        if any(isinstance(model, BaseBayesianNeuralNetwork) for model in actions.values()):
            kwargs.update(base_model_cold_start_kwargs)

        return cmab, actions, kwargs


TEST_CONFIGS = {
    "cmab": ModelTestConfig(CmabBernoulli, ClassicBandit, [BayesianNeuralNetwork, CmabZoomingModel]),
    "cmab_bai": ModelTestConfig(
        CmabBernoulliBAI, BestActionIdentificationBandit, [BayesianNeuralNetwork, CmabZoomingModel]
    ),
    "cmab_cc": ModelTestConfig(
        CmabBernoulliCC,
        CostControlBandit,
        [BayesianNeuralNetworkCC, CmabZoomingModelCC],
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
    costs=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    update_kwargs=st.sampled_from([None, {"draws": 500}]),
)
def test_cold_start(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    costs,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    update_kwargs,
):
    # Create CMAB instance
    cmab, actions, kwargs = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        costs,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        update_method,
        update_kwargs,
    )

    # Cold start comparison logic (modified for different model types)
    cold_start_kwargs = {
        "action_ids": {
            action
            for action, model in zip(action_ids, config.model_types)
            if issubclass(model, (BayesianNeuralNetwork))
        },
        "quantitative_action_ids": {
            action for action, model in zip(action_ids, config.model_types) if issubclass(model, QuantitativeModel)
        },
    }
    if all(model in [BayesianNeuralNetworkCC, CmabZoomingModelCC] for model in config.model_types):
        cold_start_kwargs["action_ids_cost"] = {
            action: model.cost for action, model in actions.items() if isinstance(model, (BayesianNeuralNetworkCC))
        }
        cold_start_kwargs["quantitative_action_ids_cost"] = {
            action: model.cost for action, model in actions.items() if isinstance(model, CmabZoomingModelCC)
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
    costs=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    update_kwargs=st.sampled_from([None, {"draws": 500}]),
)
def test_bad_initialization(
    config: ModelTestConfig,
    action_ids: List[str],
    n_features: int,
    hidden_dim_list: List[PositiveInt],
    costs,
    exploit_p,
    subsidy_factor,
    update_method,
    update_kwargs,
):
    """Test various invalid initialization scenarios for CMAB models"""
    kwargs = {"cost": 1} if config.cmab_class == CmabBernoulliCC else {}
    # Test empty actions
    with pytest.raises(AttributeError):
        config.cmab_class(actions={})

    # Test single action (should warn)
    single_action = {
        action_ids[0]: config.model_types[0].cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, **kwargs
        )
    }
    with pytest.warns(UserWarning):
        config.cmab_class(actions=single_action)

    # Test mismatched feature dimensions
    actions_wrong_dims = {
        action_ids[0]: config.model_types[0].cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, **kwargs
        ),
        action_ids[1]: config.model_types[0].cold_start(
            n_features=n_features + 1, hidden_dim_list=hidden_dim_list, **kwargs
        ),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_dims)

    # Test mismatched update methods
    actions_wrong_update = {
        action_ids[0]: config.model_types[0].cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method="VI", **kwargs
        ),
        action_ids[1]: config.model_types[0].cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method="MCMC", **kwargs
        ),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_update)

    # Test mismatched update kwargs
    base_kwargs = {"draws": 500} if update_kwargs else {"draws": 1000}
    actions_wrong_kwargs = {
        action_ids[0]: config.model_types[0].cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs=base_kwargs,
            **kwargs,
        ),
        action_ids[1]: config.model_types[0].cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs={"draws": base_kwargs["draws"] // 2},
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
                costs,
                exploit_p.draw(st.sampled_from([-0.1, 1.1])),
                subsidy_factor,
                n_features,
                hidden_dim_list,
                update_method,
                update_kwargs,
            )
    elif config.cmab_class == CmabBernoulliCC:
        with pytest.raises(ValidationError):
            config.create_cmab_and_actions(
                action_ids,
                None,
                None,
                costs,
                exploit_p,
                subsidy_factor.draw(st.sampled_from([-0.1, 1.1])),
                n_features,
                hidden_dim_list,
                update_method,
                update_kwargs,
            )


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
    costs=st.data(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    update_kwargs=st.sampled_from([None, {"draws": 10}]),
    memory_len=st.integers(min_value=1, max_value=5),
)
def test_update(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    delta,
    costs,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    update_kwargs,
    memory_len,
    monkeymodule,
):
    monkeymodule.setattr(
        pybandits.model,
        "fit",
        lambda *args, **kwargs: FakeApproximation(n_features=n_features, hidden_dim_list=hidden_dim_list),
    )
    monkeymodule.setattr(
        pybandits.model,
        "sample",
        FakeApproximation(n_features=n_features, hidden_dim_list=hidden_dim_list).sample,
    )
    # Create CMAB instance
    cmab, _, kwargs = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        costs,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        update_method,
        update_kwargs,
    )
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    # Generate random rewards
    reward_data = np.random.choice([0, 1], size=n_samples).tolist()
    # Test updates with generated data
    actions_to_update = sample_with_replacement(action_ids, n_samples)
    # Generate quantities only if there are any QuantitativeModel actions
    for_update_kwargs = {"actions": actions_to_update, "rewards": reward_data}
    if any(isinstance(model, BaseCmabZoomingModel) for model in cmab.actions.values()):
        quantity_data = np.random.random(size=n_samples).tolist()
        quantity_data = [
            q if isinstance(cmab.actions[action], QuantitativeModel) else None
            for q, action in zip(quantity_data, actions_to_update)
        ]
        for_update_kwargs["quantities"] = quantity_data

    old_cmab = deepcopy(cmab)
    for k, transform in enumerate([lambda x: x, list, pd.DataFrame]):
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
    costs=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    update_kwargs=st.sampled_from([None, {"draws": 500}]),
    diff=st.data(),
)
def test_predict(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    delta,
    costs,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    update_kwargs,
    diff,
    monkeymodule,
):
    # Create CMAB instance
    cmab = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        costs,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        update_method,
        update_kwargs,
    )[0]
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    monkeymodule.setattr(
        pybandits.model, "sample_prior_predictive", FakePrediction(n_samples=context.shape[0]).sample_prior_predictive
    )

    # Test predictions with random forbidden actions
    forbidden = set(sample_with_replacement(action_ids, len(action_ids) // 2)) if len(action_ids) > 2 else None
    if cmab.default_action is not None and forbidden is not None and cmab.default_action in forbidden:
        forbidden.remove(cmab.default_action)
    mock_update(list(cmab.actions.values()), diff, monkeymodule)
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
    costs=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    update_kwargs=st.sampled_from([None, {"draws": 500}]),
    diff=st.data(),
)
def test_serialization(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    costs,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    update_kwargs,
    diff,
    monkeymodule,
):
    # Create CMAB instance
    cmab = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        costs,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        update_method,
        update_kwargs,
    )[0]

    pre_update_state = deepcopy(cmab.get_state())
    mock_update(list(cmab.actions.values()), diff, monkeymodule)
    post_update_state = cmab.get_state()
    # Verify model updates
    assert pre_update_state != post_update_state

    # Test serialization
    restored_cmab = config.cmab_class.from_state(post_update_state[1])
    assert restored_cmab == cmab

    # Test serialization from old state
    old_post_update_state = post_update_state[1]
    old_post_update_state["actions"] = old_post_update_state.pop("actions_manager")["actions"]
    restored_cmab = config.cmab_class.from_old_state(old_post_update_state, delta=delta)
    assert restored_cmab == cmab


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
    costs=st.data(),
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    update_method=st.sampled_from(literal_update_methods),
    update_kwargs=st.sampled_from([None, {"draws": 500}]),
    diff=st.data(),
)
def test_pickling(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    costs,
    n_features,
    hidden_dim_list,
    exploit_p,
    subsidy_factor,
    update_method,
    update_kwargs,
    diff,
    monkeymodule,
):
    # Create CMAB instance
    cmab = config.create_cmab_and_actions(
        action_ids,
        epsilon,
        delta,
        costs,
        exploit_p,
        subsidy_factor,
        n_features,
        hidden_dim_list,
        update_method,
        update_kwargs,
    )[0]
    to_temporary_pickle(cmab)
    mock_update(list(cmab.actions.values()), diff, monkeymodule)
    to_temporary_pickle(cmab)


@settings(deadline=500)
@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=100),
    st.sampled_from(literal_update_methods),
    st.just([3]),
)
def test_cmab_update_shape_mismatch(n_samples, n_features, update_method, hidden_dim_list):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = CmabBernoulli.cold_start(
        action_ids={"a1", "a2"}, n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
    )

    with pytest.raises(AttributeError):  # actions shape mismatch
        mab.update(context=context, actions=actions[1:], rewards=rewards)
    with pytest.raises(AttributeError):  # rewards shape mismatch
        mab.update(context=context, actions=actions, rewards=rewards[1:])
    with pytest.raises(AttributeError):  # context shape mismatch (rows)
        mab.update(context=context[1:, :], actions=actions, rewards=rewards)
    with pytest.raises(AttributeError):  # context shape mismatch (columns)
        mab.update(context=context[:, 1:], actions=actions, rewards=rewards)
    with pytest.raises(AttributeError):  # empty context
        mab.update(context=[], actions=actions, rewards=rewards)


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
        mab.predict(context=[])
