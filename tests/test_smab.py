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
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic.dataclasses import dataclass

from pybandits.base import ActionId, Float01, PositiveProbability
from pybandits.model import BaseBeta, Beta, BetaCC, BetaMO, BetaMOCC
from pybandits.pydantic_version_compatibility import PositiveInt, ValidationError
from pybandits.smab import (
    BaseSmabBernoulli,
    SmabBernoulli,
    SmabBernoulliBAI,
    SmabBernoulliCC,
    SmabBernoulliMO,
    SmabBernoulliMOCC,
)
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
)


@st.composite
def diff_strategy(draw):
    return draw(st.integers(min_value=1, max_value=10))


@st.composite
def cost_strategy(draw, n_actions):
    return draw(st.lists(st.floats(min_value=0, max_value=2), min_size=n_actions, max_size=n_actions))


@pytest.fixture(scope="module")
def monkeymodule():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def mock_update(models: List[BaseBeta], diff, monkeymodule, label=0):
    for model in models:
        for field in model.model_fields:
            if field in ("n_successes", "n_failures"):
                monkeymodule.setattr(model, field, getattr(model, field) + diff.draw(diff_strategy(), label=f"{label}"))
                label += 1
            elif isinstance(sub_models := getattr(model, field), list) and isinstance(sub_models[0], BaseBeta):
                mock_update(sub_models, diff, monkeymodule, label)


@dataclass
class ModelTestConfig:
    smab_class: Type
    strategy_class: Type
    model_types: List[Union[Type[BaseBeta], Type[BetaMO]]]

    def _create_actions(
        self, action_ids: List[str], costs: Optional[st.SearchStrategy], n_objectives: Optional[PositiveInt]
    ) -> Dict[str, Any]:
        if len(self.model_types) < len(action_ids):
            indices = np.random.randint(0, len(self.model_types), len(action_ids))
            self.model_types = [self.model_types[i] for i in indices]
        if all(model in [BetaCC, BetaMOCC] for model in self.model_types):
            # Generate random costs
            costs = costs.draw(cost_strategy(n_actions=len(action_ids)))
            costs = [
                cost if model_type in [BetaCC, BetaMOCC] else lambda x: x**cost
                for cost, model_type in zip(costs, self.model_types)
            ]
        else:
            costs = None

        if n_objectives is None:
            if costs is not None:
                return {
                    action_id: model_type(cost=cost)
                    for action_id, model_type, cost in zip(action_ids, self.model_types, costs)
                }
            else:
                return {action_id: model_type() for action_id, model_type in zip(action_ids, self.model_types)}
        else:
            if costs is not None:
                return {
                    action_id: model_type(models=[Beta()] * n_objectives, cost=cost)
                    for action_id, model_type, cost in zip(action_ids, self.model_types, costs)
                }
            else:
                return {
                    action_id: model_type(models=[Beta()] * n_objectives)
                    for action_id, model_type in zip(action_ids, self.model_types)
                }

    def create_smab_and_actions(
        self,
        action_ids: List[str],
        epsilon: Optional[Float01],
        delta: Optional[PositiveProbability],
        costs: st.SearchStrategy,
        n_objectives: st.SearchStrategy[PositiveInt],
        exploit_p: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        subsidy_factor: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
    ) -> Tuple[BaseSmabBernoulli, Dict[ActionId, BaseBeta], Dict[str, Any]]:
        n_objectives = (
            n_objectives.draw(st.integers(min_value=1, max_value=10))
            if self.smab_class in [SmabBernoulliMO, SmabBernoulliMOCC]
            else None
        )
        actions = self._create_actions(action_ids, costs, n_objectives)
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
        for param, classes in zip(["subsidy_factor", "exploit_p"], [[SmabBernoulliCC], [SmabBernoulliBAI]]):
            if self.smab_class in classes:
                actual_param = eval(param)
                if isinstance(actual_param, float) or actual_param is None:
                    kwargs[param] = actual_param
                else:
                    kwargs[param] = actual_param.draw(st.floats(min_value=0, max_value=1))

        smab = self.smab_class(actions=actions, **kwargs)

        # For cold start test
        if self.smab_class in [SmabBernoulliMO, SmabBernoulliMOCC]:
            kwargs["n_objectives"] = n_objectives
        return smab, actions, kwargs


TEST_CONFIGS = {
    "smab": ModelTestConfig(SmabBernoulli, ClassicBandit, [Beta]),
    "smab_bai": ModelTestConfig(SmabBernoulliBAI, BestActionIdentificationBandit, [Beta]),
    "smab_cc": ModelTestConfig(
        SmabBernoulliCC,
        CostControlBandit,
        [BetaCC],
    ),
    "smab_mo": ModelTestConfig(SmabBernoulliMO, MultiObjectiveBandit, [BetaMO]),
    "smab_mocc": ModelTestConfig(SmabBernoulliMOCC, MultiObjectiveCostControlBandit, [BetaMOCC]),
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
    n_objectives=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
)
def test_cold_start(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    costs,
    n_objectives,
    exploit_p,
    subsidy_factor,
):
    # Create SMAB instance
    smab, actions, kwargs = config.create_smab_and_actions(
        action_ids, epsilon, delta, costs, n_objectives, exploit_p, subsidy_factor
    )

    # Cold start comparison logic (modified for different model types)
    cold_start_kwargs = {
        "action_ids": {
            action for action, model in zip(action_ids, config.model_types) if issubclass(model, (Beta, BetaMO))
        }
    }
    if all(model in [BetaCC, BetaMOCC] for model in config.model_types):
        cold_start_kwargs["action_ids_cost"] = {
            action: model.cost for action, model in actions.items() if isinstance(model, (BetaCC, BetaMOCC))
        }
    cold_start_kwargs.update(kwargs)  # Add exploit_p or subsidy_factor if needed
    cold_start_kwargs = {k: v for k, v in cold_start_kwargs.items() if v is not None}
    assert config.smab_class.cold_start(**cold_start_kwargs) == smab


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True),
    n_objectives=st.data(),
    costs=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
)
def test_bad_initialization(
    config: ModelTestConfig,
    action_ids: List[str],
    n_objectives,
    costs,
    exploit_p,
    subsidy_factor,
):
    real_n_objectives = n_objectives.draw(st.integers(min_value=1, max_value=10))
    kwargs = {"cost": 1.0} if config.smab_class in (SmabBernoulliCC, SmabBernoulliMOCC) else {}
    if config.smab_class in [SmabBernoulliMO, SmabBernoulliMOCC]:
        kwargs["models"] = [Beta() for _ in range(real_n_objectives)]

    # Test empty actions
    with pytest.raises(AttributeError):
        config.smab_class(actions={})

    # Test single action (should warn)
    single_action = {action_ids[0]: config.model_types[0](**kwargs)}
    with pytest.warns(UserWarning):
        config.smab_class(actions=single_action)

    # Test mismatched model types
    actions_wrong_type = {
        action_ids[0]: Beta(),
        action_ids[1]: BetaCC(cost=1.0),
    }
    with pytest.raises(ValidationError):
        config.smab_class(actions=actions_wrong_type)

    # Test None actions
    with pytest.raises(ValidationError):
        config.smab_class(actions={aid: None for aid in action_ids})

    # Test invalid strategy parameters
    if config.smab_class == SmabBernoulliBAI:
        with pytest.raises(ValidationError):
            config.create_smab_and_actions(
                action_ids,
                None,
                None,
                costs,
                n_objectives,
                exploit_p.draw(st.sampled_from([-0.1, 1.1])),
                subsidy_factor,
            )
    elif config.smab_class == SmabBernoulliCC:
        with pytest.raises(ValidationError):
            config.create_smab_and_actions(
                action_ids,
                None,
                None,
                costs,
                n_objectives,
                exploit_p,
                subsidy_factor.draw(st.sampled_from([-0.1, 1.1])),
            )

    # Test multi-objective specific cases
    if hasattr(config.model_types[0], "models"):
        # Test mismatched number of objectives
        mo_actions_wrong = {
            action_ids[0]: BetaMO(models=[Beta() for _ in range(real_n_objectives)]),
            action_ids[1]: BetaMO(models=[Beta() for _ in range(real_n_objectives + 1)]),
        }
        with pytest.raises(AttributeError):
            config.smab_class(actions=mo_actions_wrong)


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
    n_objectives=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    memory_len=st.integers(min_value=1, max_value=5),
)
def test_update(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    delta,
    costs,
    n_objectives,
    exploit_p,
    subsidy_factor,
    memory_len,
    monkeymodule,
):
    # Create SMAB instance
    smab, _, kwargs = config.create_smab_and_actions(
        action_ids, epsilon, delta, costs, n_objectives, exploit_p, subsidy_factor
    )
    batched_smab = deepcopy(smab)
    n_objectives = kwargs.get("n_objectives")
    # Generate random rewards
    reward_data = (
        np.random.choice([0, 1], size=(n_samples, n_objectives), replace=True)
        if n_objectives
        else np.random.choice([0, 1], size=n_samples, replace=True)
    )
    reward_data = reward_data.tolist()
    # Test updates with generated data
    actions_to_update = np.random.choice(np.array(action_ids, dtype=str), size=n_samples, replace=True).tolist()

    [smab.update(actions=[action], rewards=[reward]) for action, reward in zip(actions_to_update, reward_data)]
    if delta:
        with pytest.warns(UserWarning):
            batched_smab.update(actions=actions_to_update, rewards=reward_data)
    else:
        batched_smab.update(actions=actions_to_update, rewards=reward_data)

    if delta:
        actions_memory = actions_to_update[-memory_len:]
        rewards_memory = reward_data[-memory_len:]
    else:
        actions_memory = None
        rewards_memory = None

    for action in smab.actions:
        if isinstance(smab.actions[action], Beta):
            assert smab.actions[action] == batched_smab.actions[action]
        relevant_rewards = np.array(reward_data)[[a == action for a in actions_to_update]]
        if hasattr(smab.actions[action], "n_successes"):
            assert (
                smab.actions[action].n_successes
                == batched_smab.actions[action].n_successes
                == sum(relevant_rewards) + 1
            )
            assert (
                smab.actions[action].n_failures
                == batched_smab.actions[action].n_failures
                == sum(1 - relevant_rewards) + 1
            )

    batched_smab.update(
        actions=actions_to_update, rewards=reward_data, actions_memory=actions_memory, rewards_memory=rewards_memory
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
    n_samples=st.integers(min_value=1, max_value=100),
    epsilon=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    costs=st.data(),
    n_objectives=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    diff=st.data(),
)
def test_predict(
    config: ModelTestConfig,
    action_ids: List[str],
    n_samples: int,
    epsilon: Optional[float],
    delta,
    costs,
    n_objectives,
    exploit_p,
    subsidy_factor,
    diff,
    monkeymodule,
):
    # Create SMAB instance
    smab = config.create_smab_and_actions(action_ids, epsilon, delta, costs, n_objectives, exploit_p, subsidy_factor)[0]

    # Test predictions with random forbidden actions
    forbidden = (
        set(np.random.choice(np.array(action_ids, dtype=str), size=len(action_ids) // 2, replace=False))
        if len(action_ids) > 2
        else None
    )
    if smab.default_action is not None and forbidden is not None and smab.default_action in forbidden:
        forbidden.remove(smab.default_action)

    mock_update(list(smab.actions.values()), diff, monkeymodule)
    best_actions, probs = smab.predict(n_samples=n_samples, forbidden_actions=forbidden)
    assert len(best_actions) == n_samples
    assert len(probs) == n_samples

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
    else:
        assert all(
            len({action[0] if isinstance(action, tuple) else action for action in prob}) == len(action_ids)
            for prob in probs
        )
        if isinstance(smab, SmabBernoulli) and not smab.epsilon:
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
    n_objectives=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    diff=st.data(),
)
def test_serialization(
    config: ModelTestConfig,
    action_ids: List[str],
    epsilon: Optional[float],
    delta,
    costs,
    n_objectives,
    exploit_p,
    subsidy_factor,
    diff,
    monkeymodule,
):
    # Create SMAB instance
    smab = config.create_smab_and_actions(action_ids, epsilon, delta, costs, n_objectives, exploit_p, subsidy_factor)[0]

    pre_update_state = smab.get_state()
    mock_update(list(smab.actions.values()), diff, monkeymodule)
    post_update_state = smab.get_state()
    # Verify model updates
    assert pre_update_state != post_update_state

    # Test serialization
    restored_smab = config.smab_class.from_state(post_update_state[1])
    assert restored_smab == smab

    # Test serialization from old state
    old_post_update_state = post_update_state[1]
    old_post_update_state["actions"] = old_post_update_state.pop("actions_manager")["actions"]
    restored_smab = config.smab_class.from_old_state(old_post_update_state, delta=delta)
    assert restored_smab == smab


@given(
    st.integers(min_value=1),
    st.integers(min_value=1),
)
def test_can_instantiate_smab_with_params(a, b):
    s = SmabBernoulli(
        actions={
            "action1": Beta(n_successes=a, n_failures=b),
            "action2": Beta(n_successes=a, n_failures=b),
        },
    )
    assert (s.actions["action1"].n_successes == a) and (s.actions["action1"].n_failures == b)
    assert s.actions["action1"] == s.actions["action2"]


@given(st.integers(max_value=0))
def test_smab_predict_raise_when_samples_low(n_samples):
    s = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
    with pytest.raises(ValidationError):
        s.predict(n_samples=n_samples)


def test_smab_predict_raise_when_all_actions_forbidden():
    s = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
    with pytest.raises(ValueError):
        s.predict(n_samples=10, forbidden_actions=["a1", "a2"])


@given(st.text())
def test_smab_accepts_only_valid_actions(s):
    if s == "":
        with pytest.raises(ValidationError):
            SmabBernoulli(
                actions={
                    s: Beta(),
                    s + "_": Beta(),
                }
            )
    else:
        SmabBernoulli(actions={s: Beta(), s + "_": Beta()})
