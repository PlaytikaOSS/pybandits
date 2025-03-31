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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic.dataclasses import dataclass

import pybandits
from pybandits.base import ActionId, Float01, PositiveProbability
from pybandits.cmab import BaseCmabBernoulli, CmabBernoulli, CmabBernoulliBAI, CmabBernoulliCC
from pybandits.model import (
    BaseBayesianLogisticRegression,
    BayesianLogisticRegression,
    BayesianLogisticRegressionCC,
    StudentT,
    UpdateMethods,
)
from pybandits.pydantic_version_compatibility import (
    PositiveInt,
    ValidationError,
)
from pybandits.strategy import BestActionIdentificationBandit, ClassicBandit, CostControlBandit
from tests.test_utils import FakeApproximation, literal_update_methods


@st.composite
def diff_strategy(draw):
    return draw(st.floats(min_value=0.001, max_value=0.5))


@st.composite
def cost_strategy(draw, n_actions):
    return draw(st.lists(st.floats(min_value=0, max_value=2), min_size=n_actions, max_size=n_actions))


@pytest.fixture(scope="module")
def monkeymodule():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def mock_student_t(
    field_value: StudentT,
    diff: Any,
    monkeymodule: Any,
    label: Union[int, str],
) -> int:
    """
    Update the mu and sigma fields of a StudentT object.

    Args:
        field_value: StudentT object to update
        diff: Hypothesis diff object for drawing random values
        monkeymodule: Module for monkey patching
        label: Label for the diff draw

    Returns:
        Updated label value
    """
    for sub_field in ("mu", "sigma"):
        try:
            new_value = getattr(field_value, sub_field) + diff.draw(diff_strategy(), label=f"{label}")
            monkeymodule.setattr(field_value, sub_field, new_value)
            label = int(label) + 1 if isinstance(label, (int, str)) else label + 1
        except AttributeError as e:
            raise ValueError(f"Invalid StudentT field: {sub_field}") from e
    return label


def mock_update(
    models: Union[List[BayesianLogisticRegression], BayesianLogisticRegression], diff, monkeymodule, label=0
):
    model_list = [models] if isinstance(models, BayesianLogisticRegression) else models
    for model in model_list:
        for field in model.model_fields:
            field_value = getattr(model, field)

            # Handle StudentT field
            if isinstance(field_value, StudentT):
                label = mock_student_t(field_value, diff, monkeymodule, label)

            # Handle list of StudentT objects
            elif isinstance(field_value, list) and field_value and isinstance(field_value[0], StudentT):
                for item in field_value:
                    label = mock_student_t(item, diff, monkeymodule, label)

            # Handle list of BayesianLogisticRegression objects
            elif (
                isinstance(field_value, list) and field_value and isinstance(field_value[0], BayesianLogisticRegression)
            ):
                mock_update(field_value, diff, monkeymodule, label)


@dataclass
class ModelTestConfig:
    cmab_class: Type
    strategy_class: Type
    model_types: List[Type[BaseBayesianLogisticRegression]]

    def _create_actions(
        self,
        action_ids: List[str],
        costs: Optional[st.SearchStrategy],
        n_features: PositiveInt,
        update_method: UpdateMethods,
        update_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if len(self.model_types) < len(action_ids):
            indices = np.random.randint(0, len(self.model_types), len(action_ids))
            self.model_types = [self.model_types[i] for i in indices]
        if all(model in [BayesianLogisticRegressionCC] for model in self.model_types):
            # Generate random costs
            costs = costs.draw(cost_strategy(n_actions=len(action_ids)))
            costs = [
                cost if model_type in [BayesianLogisticRegressionCC] else lambda x: x**cost
                for cost, model_type in zip(costs, self.model_types)
            ]
        else:
            costs = None

        model_cold_start_kwargs = dict(update_method=update_method, update_kwargs=update_kwargs)
        base_model_cold_start_kwargs = dict(n_features=n_features, **model_cold_start_kwargs)
        if costs is not None:
            return {
                action_id: model_type(
                    alpha=StudentT(),
                    betas=[StudentT() for _ in range(n_features)],
                    **model_cold_start_kwargs,
                    cost=cost,
                )
                for action_id, model_type, cost in zip(action_ids, self.model_types, costs)
            }, base_model_cold_start_kwargs
        else:
            return {
                action_id: model_type(
                    alpha=StudentT(), betas=[StudentT() for _ in range(n_features)], **model_cold_start_kwargs
                )
                for action_id, model_type in zip(action_ids, self.model_types)
            }, base_model_cold_start_kwargs

    def create_cmab_and_actions(
        self,
        action_ids: List[str],
        epsilon: Optional[Float01],
        delta: Optional[PositiveProbability],
        costs: st.SearchStrategy,
        exploit_p: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        subsidy_factor: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        n_features: PositiveInt,
        update_method: UpdateMethods,
        update_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[BaseCmabBernoulli, Dict[ActionId, BayesianLogisticRegression], Dict[str, Any]]:
        actions, base_model_cold_start_kwargs = self._create_actions(
            action_ids, costs, n_features, update_method, update_kwargs
        )
        default_action = action_ids[0] if epsilon and not delta else None
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
        kwargs.update(base_model_cold_start_kwargs)

        return cmab, actions, kwargs


TEST_CONFIGS = {
    "cmab": ModelTestConfig(CmabBernoulli, ClassicBandit, [BayesianLogisticRegression]),
    "cmab_bai": ModelTestConfig(CmabBernoulliBAI, BestActionIdentificationBandit, [BayesianLogisticRegression]),
    "cmab_cc": ModelTestConfig(
        CmabBernoulliCC,
        CostControlBandit,
        [BayesianLogisticRegressionCC],
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
        update_method,
        update_kwargs,
    )

    # Cold start comparison logic (modified for different model types)
    cold_start_kwargs = {
        "action_ids": {
            action
            for action, model in zip(action_ids, config.model_types)
            if issubclass(model, (BayesianLogisticRegression))
        },
    }
    if all(model in [BayesianLogisticRegressionCC] for model in config.model_types):
        cold_start_kwargs["action_ids_cost"] = {
            action: model.cost for action, model in actions.items() if isinstance(model, (BayesianLogisticRegressionCC))
        }
    cold_start_kwargs.update(kwargs)  # Add exploit_p or subsidy_factor if needed
    cold_start_kwargs = {k: v for k, v in cold_start_kwargs.items() if v is not None}
    cmab.predict_actions_randomly = True
    assert config.cmab_class.cold_start(**cold_start_kwargs) == cmab


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True),
    n_features=st.integers(min_value=1, max_value=5),
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
    single_action = {action_ids[0]: config.model_types[0].cold_start(n_features=n_features, **kwargs)}
    with pytest.warns(UserWarning):
        config.cmab_class(actions=single_action)

    # Test mismatched feature dimensions
    actions_wrong_dims = {
        action_ids[0]: config.model_types[0].cold_start(n_features=n_features, **kwargs),
        action_ids[1]: config.model_types[0].cold_start(n_features=n_features + 1, **kwargs),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_dims)

    # Test mismatched update methods
    actions_wrong_update = {
        action_ids[0]: config.model_types[0].cold_start(n_features=n_features, update_method="VI", **kwargs),
        action_ids[1]: config.model_types[0].cold_start(n_features=n_features, update_method="MCMC", **kwargs),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_update)

    # Test mismatched update kwargs
    base_kwargs = {"draws": 500} if update_kwargs else {"draws": 1000}
    actions_wrong_kwargs = {
        action_ids[0]: config.model_types[0].cold_start(
            n_features=n_features, update_method=update_method, update_kwargs=base_kwargs, **kwargs
        ),
        action_ids[1]: config.model_types[0].cold_start(
            n_features=n_features,
            update_method=update_method,
            update_kwargs={"draws": base_kwargs["draws"] // 2},
            **kwargs,
        ),
    }
    with pytest.raises(AttributeError):
        config.cmab_class(actions=actions_wrong_kwargs)

    # Test invalid model types
    actions_wrong_type = {
        action_ids[0]: BayesianLogisticRegression.cold_start(n_features=n_features),
        action_ids[1]: BayesianLogisticRegressionCC.cold_start(n_features=n_features, cost=1.0),
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
        lambda *args, **kwargs: FakeApproximation(n_features=n_features),
    )
    monkeymodule.setattr(
        pybandits.model,
        "sample",
        FakeApproximation(n_features=n_features).sample,
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
        update_method,
        update_kwargs,
    )
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    # Generate random rewards
    reward_data = np.random.choice([0, 1], size=n_samples).tolist()
    # Test updates with generated data
    actions_to_update = np.random.choice(np.array(action_ids, dtype=str), size=n_samples, replace=True).tolist()

    for_update_kwargs = {"actions": actions_to_update, "rewards": reward_data}

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
        update_method,
        update_kwargs,
    )[0]
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    # Test predictions with random forbidden actions
    forbidden = (
        set(np.random.choice(np.array(action_ids, dtype=str), size=len(action_ids) // 2, replace=False))
        if len(action_ids) > 2
        else None
    )
    if cmab.default_action is not None and forbidden is not None and cmab.default_action in forbidden:
        forbidden.remove(cmab.default_action)

    mock_update(list(cmab.actions.values()), diff, monkeymodule)
    best_actions, probs, weights = cmab.predict(context=context, forbidden_actions=forbidden)
    assert len(best_actions) == n_samples
    assert len(probs) == n_samples
    assert len(weights) == n_samples

    if forbidden:
        assert all(
            len({action[0] if isinstance(action, tuple) else action for action in prob})
            == len(action_ids) - len(forbidden)
            for prob in probs
        )
        assert all(action[0] if isinstance(action, tuple) else action not in forbidden for action in best_actions)
        assert all(
            action[0] if isinstance(action, tuple) else action not in forbidden
            for prob in probs
            for action in prob.keys()
        )
        assert all(
            action[0] if isinstance(action, tuple) else action not in forbidden
            for weight in weights
            for action in weight.keys()
        )
    else:
        assert all(
            len({action[0] if isinstance(action, tuple) else action for action in prob}) == len(action_ids)
            for prob in probs
        )
    if isinstance(cmab, CmabBernoulli) and not cmab.epsilon:
        assert all(prob[best_action] == max(prob.values()) for best_action, prob in zip(best_actions, probs))


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
        update_method,
        update_kwargs,
    )[0]

    pre_update_state = cmab.get_state()
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


@settings(deadline=500)
@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=100),
    st.sampled_from(literal_update_methods),
)
def test_cmab_update_shape_mismatch(n_samples, n_features, update_method):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = CmabBernoulli.cold_start(action_ids={"a1", "a2"}, n_features=n_features, update_method=update_method)

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
@given(st.integers(min_value=1, max_value=10))
def test_cmab_predict_shape_mismatch(a_int):
    context = np.random.uniform(low=-1.0, high=1.0, size=(100, a_int - 1))
    mab = CmabBernoulli.cold_start(action_ids={"a1", "a2"}, n_features=a_int)
    with pytest.raises(AttributeError):
        mab.predict(context=context)
    with pytest.raises(AttributeError):
        mab.predict(context=[])
