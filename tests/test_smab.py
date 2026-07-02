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
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import PositiveInt, ValidationError
from pydantic.dataclasses import dataclass

import pybandits
from pybandits.actions_manager import SmabModelType
from pybandits.base import ActionId, Float01, PositiveProbability
from pybandits.base_model import BaseModel
from pybandits.model import Beta, BetaCC, BetaDP, BetaMO, BetaMOCC
from pybandits.quantitative_model import QuantitativeModel, Segment, Zooming, ZoomingCC, ZoomingDP
from pybandits.smab import (
    BaseSmabBernoulli,
    SmabBernoulli,
    SmabBernoulliBAI,
    SmabBernoulliCC,
    SmabBernoulliDP,
    SmabBernoulliMO,
    SmabBernoulliMOCC,
)
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    DynamicPricingBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
)
from tests.utils import sample_with_replacement, to_temporary_pickle


@st.composite
def diff_strategy(draw):
    return draw(st.integers(min_value=1, max_value=10))


@st.composite
def value_strategy(draw, n_actions):
    return draw(st.lists(st.floats(min_value=0, max_value=2), min_size=n_actions, max_size=n_actions))


def mock_update(models: List[SmabModelType], diff, monkeymodule, label=0):
    for model in models:
        for field in model.model_fields:
            if field in ("n_successes", "n_failures"):
                monkeymodule.setattr(model, field, getattr(model, field) + diff.draw(diff_strategy(), label=f"{label}"))
                label += 1
            elif isinstance(sub_models := getattr(model, field), list) and isinstance(sub_models[0], BaseModel):
                mock_update(sub_models, diff, monkeymodule, label)


def _quantitative_callable(x, value):
    return min(sum(x) ** value, 1000)


@dataclass
class ModelTestConfig:
    smab_class: Type
    strategy_class: Type
    model_types: List[Type[SmabModelType]]

    def _create_actions(
        self,
        action_ids: List[str],
        values: Optional[st.SearchStrategy],
        n_objectives: Optional[PositiveInt],
        rng: np.random.Generator,
        decay_factor: Optional[Float01] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_types = list(self.model_types)
        if len(model_types) < len(action_ids):
            indices = rng.integers(0, len(model_types), len(action_ids))
            model_types = [model_types[i] for i in indices]
        base_model_cold_start_kwargs: Dict[str, Any] = {}
        if decay_factor is not None:
            base_model_cold_start_kwargs["decay_factor"] = decay_factor
        if all(model in [BetaCC, ZoomingCC, BetaMOCC] for model in model_types):
            # Generate random costs
            drawn_values = values.draw(value_strategy(n_actions=len(action_ids)))
            costs = [
                val if model_type in [BetaCC, BetaMOCC] else partial(_quantitative_callable, value=val)
                for val, model_type in zip(drawn_values, model_types)
            ]
            value_field = "cost"
        elif all(model in [BetaDP, ZoomingDP] for model in model_types):
            # Generate random prices
            drawn_values = values.draw(value_strategy(n_actions=len(action_ids)))
            costs = [
                val if model_type == BetaDP else partial(_quantitative_callable, value=val)
                for val, model_type in zip(drawn_values, model_types)
            ]
            value_field = "price"
        else:
            costs = None
            value_field = None

        if n_objectives is None:
            if costs is not None:
                return {
                    action_id: model_type(decay_factor=decay_factor, **{value_field: cost})
                    if issubclass(model_type, (BetaCC, BetaDP))
                    else model_type.cold_start(
                        dimension=1, base_model_cold_start_kwargs=base_model_cold_start_kwargs, **{value_field: cost}
                    )  # ZoomingCC / ZoomingDP
                    for action_id, model_type, cost in zip(action_ids, model_types, costs)
                }, base_model_cold_start_kwargs
            else:
                return {
                    action_id: model_type(decay_factor=decay_factor)
                    if issubclass(model_type, Beta)
                    else model_type.cold_start(
                        dimension=1, base_model_cold_start_kwargs=base_model_cold_start_kwargs
                    )  # Zooming
                    for action_id, model_type in zip(action_ids, model_types)
                }, base_model_cold_start_kwargs
        else:
            if costs is not None:
                return {
                    action_id: model_type(
                        models=[Beta(decay_factor=decay_factor) for _ in range(n_objectives)], cost=cost
                    )
                    for action_id, model_type, cost in zip(action_ids, model_types, costs)
                }, base_model_cold_start_kwargs
            else:
                return {
                    action_id: model_type(models=[Beta(decay_factor=decay_factor) for _ in range(n_objectives)])
                    for action_id, model_type in zip(action_ids, model_types)
                }, base_model_cold_start_kwargs

    def create_smab_and_actions(
        self,
        action_ids: List[str],
        epsilon: Optional[Float01],
        delta: Optional[PositiveProbability],
        values: st.SearchStrategy,
        n_objectives: st.SearchStrategy[PositiveInt],
        exploit_p: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        subsidy_factor: Union[st.SearchStrategy[Optional[Float01]], Optional[float]],
        rng: np.random.Generator,
        decay_factor: Optional[Float01] = None,
        default_action_fraction: Optional[Float01] = None,
        limited_action_fraction: Optional[Float01] = None,
    ) -> Tuple[BaseSmabBernoulli, Dict[ActionId, SmabModelType], Dict[str, Any]]:
        n_objectives = (
            n_objectives.draw(st.integers(min_value=1, max_value=10))
            if self.smab_class in [SmabBernoulliMO, SmabBernoulliMOCC]
            else None
        )
        actions, base_model_cold_start_kwargs = self._create_actions(
            action_ids, values, n_objectives, rng, decay_factor
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
        if any(isinstance(model, QuantitativeModel) for model in actions.values()):
            kwargs["base_model_cold_start_kwargs"] = base_model_cold_start_kwargs
        if decay_factor is not None:
            kwargs["decay_factor"] = decay_factor
        return smab, actions, kwargs


TEST_CONFIGS = {
    "smab": ModelTestConfig(SmabBernoulli, ClassicBandit, [Beta, Zooming]),
    "smab_bai": ModelTestConfig(SmabBernoulliBAI, BestActionIdentificationBandit, [Beta, Zooming]),
    "smab_cc": ModelTestConfig(
        SmabBernoulliCC,
        CostControlBandit,
        [BetaCC, ZoomingCC],
    ),
    "smab_dp": ModelTestConfig(
        SmabBernoulliDP,
        DynamicPricingBandit,
        [BetaDP, ZoomingDP],
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
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
    n_objectives=st.data(),
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
    exploit_p,
    subsidy_factor,
    decay_factor: Optional[float],
    rng: np.random.Generator,
):
    # Create SMAB instance
    smab, actions, kwargs = config.create_smab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        rng,
        decay_factor,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
    )

    # Cold start comparison logic (modified for different model types)
    cold_start_kwargs = {
        "action_ids": {action for action, model in actions.items() if isinstance(model, (Beta, BetaMO))},
        "quantitative_action_ids": {
            action for action, model in actions.items() if isinstance(model, QuantitativeModel)
        },
    }
    if all(isinstance(model, (BetaCC, ZoomingCC, BetaMOCC)) for model in actions.values()):
        cold_start_kwargs["action_ids_cost"] = {
            action: model.cost for action, model in actions.items() if isinstance(model, (BetaCC, BetaMOCC))
        }
        cold_start_kwargs["quantitative_action_ids_cost"] = {
            action: model.cost for action, model in actions.items() if isinstance(model, ZoomingCC)
        }
    if all(isinstance(model, (BetaDP, ZoomingDP)) for model in actions.values()):
        cold_start_kwargs["action_ids_price"] = {
            action: model.price for action, model in actions.items() if isinstance(model, BetaDP)
        }
        cold_start_kwargs["quantitative_action_ids_price"] = {
            action: model.price for action, model in actions.items() if isinstance(model, ZoomingDP)
        }
    cold_start_kwargs.update(kwargs)  # Add exploit_p or subsidy_factor if needed
    cold_start_kwargs = {k: v for k, v in cold_start_kwargs.items() if v is not None}
    # cold_start's default_action argument only accepts a discrete action id; a quantitative
    # (tuple) default_action cannot be expressed through it, so skip the equivalence check there.
    if isinstance(cold_start_kwargs.get("default_action"), tuple):
        return
    assert config.smab_class.cold_start(**cold_start_kwargs) == smab


@settings(deadline=None)
@pytest.mark.parametrize("config", TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
@given(
    action_ids=st.lists(st.text(min_size=1), min_size=2, max_size=5, unique=True),
    n_objectives=st.data(),
    values=st.data(),
    subsidy_factor=st.data(),
    exploit_p=st.data(),
)
def test_bad_initialization(
    config: ModelTestConfig,
    action_ids: List[str],
    n_objectives,
    values,
    exploit_p,
    subsidy_factor,
    rng: np.random.Generator,
):
    real_n_objectives = n_objectives.draw(st.integers(min_value=1, max_value=10))
    if config.smab_class in (SmabBernoulliCC, SmabBernoulliMOCC):
        kwargs = {"cost": 1.0}
    elif config.smab_class == SmabBernoulliDP:
        kwargs = {"price": 1.0}
    else:
        kwargs = {}
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
                values,
                n_objectives,
                exploit_p.draw(st.sampled_from([-0.1, 1.1])),
                subsidy_factor,
                rng,
            )
    elif config.smab_class == SmabBernoulliCC:
        with pytest.raises(ValidationError):
            config.create_smab_and_actions(
                action_ids,
                None,
                None,
                values,
                n_objectives,
                exploit_p,
                subsidy_factor.draw(st.sampled_from([-0.1, 1.1])),
                rng,
            )

    # Test multi-objective specific cases
    if hasattr(config.model_types[0], "models"):
        # Test mismatched number of objectives
        mo_actions_wrong = {
            action_ids[0]: BetaMO(models=[Beta() for _ in range(real_n_objectives)]),
            action_ids[1]: BetaMO(models=[Beta() for _ in range(real_n_objectives + 1)]),
        }
        with pytest.raises(ValidationError):
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
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
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
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    exploit_p,
    subsidy_factor,
    memory_len,
    rng: np.random.Generator,
):
    # Create SMAB instance
    smab, _, kwargs = config.create_smab_and_actions(
        action_ids, epsilon, delta, values, n_objectives, exploit_p, subsidy_factor, rng
    )
    batched_smab = deepcopy(smab)
    n_objectives = kwargs.get("n_objectives")
    # Generate random rewards
    reward_data = (
        rng.choice([0, 1], size=(n_samples, n_objectives), replace=True)
        if n_objectives
        else rng.choice([0, 1], size=n_samples, replace=True)
    )
    reward_data = reward_data.tolist()
    # Test updates with generated data
    actions_to_update = sample_with_replacement(
        action_ids, n_samples
    )  # Generate quantities only if there are any QuantitativeModel actions
    if any(isinstance(model, QuantitativeModel) for model in smab.actions.values()):
        quantity_data = rng.random(size=n_samples).tolist()
        quantity_data = [
            q if isinstance(smab.actions[action], QuantitativeModel) else None
            for q, action in zip(quantity_data, actions_to_update)
        ]
        [
            smab.update(actions=[action], rewards=[reward], quantities=[quantity])
            for action, reward, quantity in zip(actions_to_update, reward_data, quantity_data)
        ]
    else:
        quantity_data = None
        [
            smab.update(actions=[action], rewards=[reward], quantities=quantity_data)
            for action, reward in zip(actions_to_update, reward_data)
        ]

    if delta:
        with pytest.warns(UserWarning):
            batched_smab.update(actions=actions_to_update, rewards=reward_data, quantities=quantity_data)
    else:
        batched_smab.update(actions=actions_to_update, rewards=reward_data, quantities=quantity_data)

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
        actions=actions_to_update,
        rewards=reward_data,
        quantities=quantity_data,
        actions_memory=actions_memory,
        rewards_memory=rewards_memory,
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
    default_action_fraction=st.one_of(st.none(), st.floats(min_value=1e-3, max_value=1)),
    limited_action_fraction=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
    delta=st.one_of(st.none(), st.just(0.1)),
    values=st.data(),
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
    default_action_fraction: Optional[float],
    limited_action_fraction: Optional[float],
    delta,
    values,
    n_objectives,
    exploit_p,
    subsidy_factor,
    diff,
    monkeymodule,
    rng: np.random.Generator,
):
    def mock_maximize_by_quantity(quantity_score_func, dimension, constraint=None, n_trials=10000, **kwargs):
        """Mock maximize_by_quantity to return a quick result."""
        return rng.random(dimension)

    if config.smab_class in (SmabBernoulliMO, SmabBernoulliMOCC):

        def mock_find_pareto_front_normal_constraint(self, func, input_dim, n_objectives, n_divisions, model):
            """Mock _find_pareto_front_normal_constraint to return a quick result."""
            return [rng.random(input_dim) for _ in range(min(3, n_divisions))]

        monkeymodule.setattr(
            pybandits.strategy.MultiObjectiveStrategy,
            "_find_pareto_front_normal_constraint",
            mock_find_pareto_front_normal_constraint,
        )

    with patch.object(pybandits.strategy.single_objective, "maximize_by_quantity", mock_maximize_by_quantity):
        # Create SMAB instance
        smab = config.create_smab_and_actions(
            action_ids,
            epsilon,
            delta,
            values,
            n_objectives,
            exploit_p,
            subsidy_factor,
            rng,
            default_action_fraction=default_action_fraction,
            limited_action_fraction=limited_action_fraction,
        )[0]

        # Test predictions with random forbidden actions
        forbidden = set(sample_with_replacement(action_ids, len(action_ids) // 2)) if len(action_ids) > 2 else None
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
    subsidy_factor=st.data(),
    exploit_p=st.data(),
    diff=st.data(),
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
    exploit_p,
    subsidy_factor,
    diff,
    monkeymodule,
    rng: np.random.Generator,
):
    # Create SMAB instance
    smab = config.create_smab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        rng,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
    )[0]

    pre_update_state = smab.get_state()
    mock_update(list(smab.actions.values()), diff, monkeymodule)
    post_update_state = smab.get_state()
    # Verify model updates
    assert pre_update_state != post_update_state

    # Test serialization
    restored_smab_state = config.smab_class.from_state(post_update_state[1]).get_state()
    assert restored_smab_state == post_update_state


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
    exploit_p,
    subsidy_factor,
    diff,
    monkeymodule,
    rng: np.random.Generator,
):
    # Create SMAB instance
    smab = config.create_smab_and_actions(
        action_ids,
        epsilon,
        delta,
        values,
        n_objectives,
        exploit_p,
        subsidy_factor,
        rng,
        default_action_fraction=default_action_fraction,
        limited_action_fraction=limited_action_fraction,
    )[0]
    to_temporary_pickle(smab)
    mock_update(list(smab.actions.values()), diff, monkeymodule)
    to_temporary_pickle(smab)


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


########################################################################################################################
# Region-aware forbidden_actions (forbidding part of a quantitative arm's hypercube)


class TestForbiddenRegions:
    """Region-aware forbidden_actions: forbidding part of a quantitative arm's hypercube (sMAB edition)."""

    # A forbidden region on a 1-D quantitative arm follows the float signed-margin convention: forbidden where
    # region(x) > 0, i.e. the upper half-interval x[0] > region_split is blocked.
    region_split = 0.5
    region_tolerance = 1e-2
    explore_samples = 50
    hypothesis_exploit_samples = 5  # small enough to stay under hypothesis's default 200ms deadline
    explore_seed = 123
    quantitative_dimension = 1
    quantitative_arm_id = "q"
    discrete_arm_id = "d"
    epsilon_full_explore = 1.0  # forces the random explore branch on every step
    allowed_segment_high = 1.0
    max_rejection_samples = 1000

    def _constraint_aware_maximize(
        self,
        quantity_score_func,
        dimension: int,
        constraint=None,
        n_trials: int = 10000,
        max_rejection_samples: int = max_rejection_samples,
        **kwargs,
    ) -> np.ndarray:
        """Mock maximize_by_quantity: rejection-samples a feasible quantity for hypothesis speed."""
        constraints = constraint or []
        for _ in range(max_rejection_samples):
            candidate = self._rng.random(dimension)
            if all(c(candidate) >= 0 for c in constraints):
                return candidate
        return self._rng.random(dimension)

    @staticmethod
    def _forbid_upper_half(x: np.ndarray, split: float = region_split) -> float:
        """Forbidden-region predicate: x[0] > split is forbidden (region(x) > 0 => forbidden)."""
        return float(x[0]) - split

    @staticmethod
    def _always_forbidden(x: np.ndarray) -> float:
        """Forbidden-region predicate that forbids the entire quantity space (margin always positive)."""
        return 1.0

    @pytest.fixture(scope="class")
    def make_smab(self):
        """Factory: builds a sMAB with one continuous (Zooming) arm and one discrete (Beta) arm."""

        def _factory(**kwargs) -> SmabBernoulli:
            return SmabBernoulli(
                actions={
                    self.quantitative_arm_id: Zooming.cold_start(dimension=self.quantitative_dimension),
                    self.discrete_arm_id: Beta(),
                },
                **kwargs,
            )

        return _factory

    def test_normalize_forbidden_actions_forms(
        self,
        make_smab,
        quantitative_arm_id: str = quantitative_arm_id,
        discrete_arm_id: str = discrete_arm_id,
    ) -> None:
        """_normalize_forbidden_actions handles the Set, dict-None (whole arm) and dict-region (partial) forms."""
        smab = make_smab()

        # Legacy Set form: whole-arm blocking, no region constraints.
        valid, regions = smab._normalize_forbidden_actions({discrete_arm_id})
        assert valid == {quantitative_arm_id} and regions == {}

        # dict with None value is equivalent to whole-arm blocking.
        valid, regions = smab._normalize_forbidden_actions({discrete_arm_id: None})
        assert valid == {quantitative_arm_id} and regions == {}

        # dict with a region keeps the arm valid (only its quantity space shrinks) and records its constraint.
        valid, regions = smab._normalize_forbidden_actions({quantitative_arm_id: self._forbid_upper_half})
        assert valid == {quantitative_arm_id, discrete_arm_id}
        assert set(regions) == {quantitative_arm_id} and len(regions[quantitative_arm_id]) == 1

    def test_region_rejected_on_discrete_arm(self, make_smab, discrete_arm_id: str = discrete_arm_id) -> None:
        """A forbidden region cannot be attached to a non-quantitative arm."""
        smab = make_smab()
        with pytest.raises(ValueError, match="quantitative"):
            smab._normalize_forbidden_actions({discrete_arm_id: self._forbid_upper_half})

    @settings(deadline=None)
    @given(
        quantity=st.floats(min_value=region_split, max_value=1.0, exclude_min=True),
        epsilon=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
        arm_id=st.just(quantitative_arm_id),
    )
    def test_default_action_in_region_raises(self, make_smab, quantity: float, epsilon: float, arm_id: str) -> None:
        """Any quantitative default action strictly above region_split is rejected when the upper half is forbidden."""
        smab = make_smab(epsilon=epsilon, default_action=(arm_id, (quantity,)))
        with pytest.raises(ValueError, match="forbidden region"):
            smab._normalize_forbidden_actions({arm_id: self._forbid_upper_half})

    @settings(deadline=None)
    @given(x0=st.floats(min_value=0.0, max_value=1.0), region_split=st.floats(min_value=0.1, max_value=0.9))
    def test_to_feasibility_constraint_negates_margin(self, x0: float, region_split: float) -> None:
        """Points above region_split are marked forbidden by the converted constraint; at/below are feasible."""
        constraint = SmabBernoulli._to_feasibility_constraint(partial(self._forbid_upper_half, split=region_split))
        point = np.array([x0])
        # By contract: constraint(x) < 0 means forbidden; >= 0 means feasible.
        if x0 > region_split:
            assert constraint(point) < 0
        else:
            assert constraint(point) >= 0

    @given(
        region_split=st.floats(min_value=0.1, max_value=0.9),
        arm_id=st.just(quantitative_arm_id),
        random_seed=st.just(explore_seed),
    )
    def test_sample_allowed_quantity_respects_and_exhausts_region(
        self,
        make_smab,
        region_split: float,
        arm_id: str,
        random_seed: int,
    ) -> None:
        """_sample_allowed_quantity returns a point outside the region, or None when the whole space is forbidden."""
        smab = make_smab(random_seed=random_seed)
        _, regions = smab._normalize_forbidden_actions({arm_id: partial(self._forbid_upper_half, split=region_split)})

        allowed = smab._sample_allowed_quantity(arm_id, regions)
        assert allowed is not None and allowed[0] <= region_split

        # A region that forbids the entire cube yields no allowed point.
        _, full_regions = smab._normalize_forbidden_actions({arm_id: self._always_forbidden})
        assert smab._sample_allowed_quantity(arm_id, full_regions) is None

    def test_predict_explore_never_selects_forbidden_region(
        self,
        arm_id: str = quantitative_arm_id,
        dimension: int = quantitative_dimension,
        epsilon: float = epsilon_full_explore,
        random_seed: int = explore_seed,
        n_samples: int = explore_samples,
        region_split: float = region_split,
        tolerance: float = region_tolerance,
    ) -> None:
        """With epsilon=1 (always explore), the random quantitative quantity always avoids the forbidden region."""
        smab = SmabBernoulli(
            actions={arm_id: Zooming.cold_start(dimension=dimension)},
            epsilon=epsilon,
            random_seed=random_seed,
        )

        selected_actions, _ = smab.predict(n_samples=n_samples, forbidden_actions={arm_id: self._forbid_upper_half})

        assert all(isinstance(action, tuple) and action[0] == arm_id for action in selected_actions)
        assert all(action[1][0] <= region_split + tolerance for action in selected_actions)

    @given(
        region_split=st.floats(min_value=0.1, max_value=0.9),
        arm_id=st.just(quantitative_arm_id),
        dimension=st.just(quantitative_dimension),
        random_seed=st.just(explore_seed),
        n_samples=st.just(hypothesis_exploit_samples),
        tolerance=st.just(region_tolerance),
    )
    def test_predict_exploit_respects_forbidden_region(
        self,
        region_split: float,
        arm_id: str,
        dimension: int,
        random_seed: int,
        n_samples: int,
        tolerance: float,
        rng: np.random.Generator,
    ) -> None:
        """Constrained optimizer stays below any forbidden boundary, not just the default split of 0.5."""
        self._rng = rng

        def forbid_above(x: np.ndarray) -> float:
            return float(x[0]) - region_split

        smab = SmabBernoulli(
            actions={arm_id: Zooming.cold_start(dimension=dimension)},
            random_seed=random_seed,
        )

        with patch.object(pybandits.strategy.single_objective, "maximize_by_quantity", self._constraint_aware_maximize):
            selected_actions, _ = smab.predict(n_samples=n_samples, forbidden_actions={arm_id: forbid_above})

        assert all(isinstance(action, tuple) and action[0] == arm_id for action in selected_actions)
        assert all(action[1][0] <= region_split + tolerance for action in selected_actions)

    @settings(deadline=None)
    @given(
        lo=st.floats(min_value=0.0, max_value=0.4),
        hi=st.floats(min_value=0.6, max_value=1.0),
        x=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_segment_region_outside_1d(self, lo: float, hi: float, x: float) -> None:
        """forbidden_region_outside is non-positive inside [lo, hi] and positive outside (1-D)."""
        region = Segment(intervals=((lo, hi),)).forbidden_region_outside()
        point = np.array([x])
        if lo <= x <= hi:
            assert region(point) <= 0
        else:
            assert region(point) > 0

    @pytest.mark.parametrize(
        "x0, x1, expected_forbidden",
        [
            (0.2, 0.7, False),  # inside [0, 0.5) x [0.5, 1.0] => allowed
            (0.8, 0.7, True),  # x0=0.8 outside [0, 0.5) => forbidden
        ],
    )
    def test_segment_region_outside_2d(
        self,
        x0: float,
        x1: float,
        expected_forbidden: bool,
        region_split: float = region_split,
        segment_high: float = allowed_segment_high,
    ) -> None:
        """forbidden_region_outside extends to 2-D: a point outside any dimension's interval is forbidden."""
        intervals = ((0.0, region_split), (region_split, segment_high))
        region = Segment(intervals=intervals).forbidden_region_outside()
        assert (region(np.array([x0, x1])) > 0) == expected_forbidden

    @given(
        seg_lo=st.floats(min_value=0.05, max_value=0.45),
        seg_hi=st.floats(min_value=0.55, max_value=0.95),
        arm_id=st.just(quantitative_arm_id),
        dimension=st.just(quantitative_dimension),
        random_seed=st.just(explore_seed),
        n_samples=st.just(hypothesis_exploit_samples),
        tolerance=st.just(region_tolerance),
    )
    def test_segment_region_outside_end_to_end(
        self,
        seg_lo: float,
        seg_hi: float,
        arm_id: str,
        dimension: int,
        random_seed: int,
        n_samples: int,
        tolerance: float,
        rng: np.random.Generator,
    ) -> None:
        """forbidden_region_outside keeps the exploited quantity inside any valid allowed segment."""
        self._rng = rng
        allowed = Segment(intervals=((seg_lo, seg_hi),))
        smab = SmabBernoulli(
            actions={arm_id: Zooming.cold_start(dimension=dimension)},
            random_seed=random_seed,
        )
        region = allowed.forbidden_region_outside()

        with patch.object(pybandits.strategy.single_objective, "maximize_by_quantity", self._constraint_aware_maximize):
            selected_actions, _ = smab.predict(n_samples=n_samples, forbidden_actions={arm_id: region})

        assert all(seg_lo - tolerance <= action[1][0] <= seg_hi + tolerance for action in selected_actions)


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
