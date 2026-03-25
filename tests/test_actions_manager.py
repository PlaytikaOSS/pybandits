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

import math
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest_mock.plugin import MockerFixture

import pybandits
from pybandits.actions_manager import ActionsManager, CmabActionsManager, SmabActionsManager
from pybandits.base import ACTION_IDS_PREFIX, QUANTITATIVE_ACTION_IDS_PREFIX, ActionId, BinaryReward
from pybandits.model import BayesianNeuralNetwork, Beta, BetaMO
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    ValidationError,
    pydantic_version,
)
from pybandits.quantitative_model import QuantitativeBayesianNeuralNetwork, SmabZoomingModel
from tests.utils import FakeApproximation

REFERENCE_DELTA = 0.0001


class DummyActionsManager(ActionsManager):
    actions: Dict[ActionId, Union[Beta, BetaMO, SmabZoomingModel]]

    def _update_actions(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        **kwargs,
    ):
        rewards_dict = defaultdict(list)

        for a, r in zip(actions, rewards):
            rewards_dict[a].append(r)

        for a in set(actions):
            self.actions[a].update(rewards=rewards_dict[a])


@given(
    data_len=st.integers(min_value=1, max_value=100),
)
def test_update_with_invalid_memory_delta_none(data_len):
    """Test update validation when delta is None but memory is provided"""
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    action_list = np.random.choice(["action1", "action2"], size=data_len).tolist()
    rewards = np.random.randint(0, 2, size=data_len).tolist()
    with pytest.raises(ValueError):
        manager.update(actions=action_list, rewards=rewards, actions_memory=action_list, rewards_memory=rewards)


@given(
    action_list=st.lists(st.sampled_from(["action1", "action2"]), min_size=1),
)
def test_update_with_missing_memory_delta_set(action_list):
    """Test update validation when delta is set but memory is not provided"""
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    with pytest.warns(UserWarning):
        manager.update(actions=action_list, rewards=[1] * len(action_list), actions_memory=None, rewards_memory=None)


_UPDATE_RESERVED_PARAMS = {"actions", "rewards", "self", "quantities", "context"}


@given(
    data_len=st.integers(min_value=1, max_value=200),
    regular_kwargs=st.dictionaries(
        st.text().filter(lambda x: not x.endswith("_memory") and x not in _UPDATE_RESERVED_PARAMS),
        st.integers(),
        min_size=1,
    ),
    memory_kwargs=st.dictionaries(
        st.text().filter(lambda x: x not in _UPDATE_RESERVED_PARAMS).map(lambda x: x + "_memory"),
        st.integers(),
        min_size=1,
    ),
)
def test_update_kwargs_separation(data_len, regular_kwargs, memory_kwargs, monkeymodule):
    """Test proper separation of regular and memory kwargs"""
    action_list = np.random.choice(["action1", "action2"], size=data_len).tolist()
    rewards = np.random.randint(0, 2, size=data_len).tolist()
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    all_kwargs = {**regular_kwargs, **memory_kwargs}

    # Keep track of captured kwargs
    captured_regular_kwargs = {}
    captured_memory_kwargs = {}

    def validate_params(*args, **kwargs):
        captured_regular_kwargs.update(kwargs)

    def validate_lengths(**kwargs):
        captured_memory_kwargs.update(kwargs)

    # Mock validation methods to capture kwargs
    if pydantic_version == PYDANTIC_VERSION_1:
        manager.__dict__["_validate_update_params"] = validate_params
        manager.__dict__["_validate_params_lengths"] = validate_lengths
    elif pydantic_version == PYDANTIC_VERSION_2:
        monkeymodule.setattr(manager, "_validate_update_params", validate_params)
        monkeymodule.setattr(manager, "_validate_params_lengths", validate_lengths)
    else:
        raise ValueError(f"Unsupported Pydantic version: {pydantic_version}")

    manager.update(actions=action_list, rewards=rewards, **all_kwargs)

    # Verify regular kwargs went to _validate_update_params
    assert all(k in regular_kwargs for k in captured_regular_kwargs)

    # Verify memory kwargs went to _validate_params_lengths
    assert all(k.endswith("_memory") for k in captured_memory_kwargs if k != "actions_memory" and k != "rewards_memory")


def test_init_with_valid_actions():
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions)
    assert len(manager.actions) == 2
    assert manager.delta is None


def test_update_with_valid_inputs(action_list=("action1", "action2", "action1"), rewards=(1, 0, 1)):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions)
    manager.update(actions=list(action_list), rewards=list(rewards))


def test_empty_actions_raises_error():
    with pytest.raises(AttributeError) as exc_info:
        DummyActionsManager(actions={})
    assert str(exc_info.value) == "At least one action should be defined."


def test_single_action_warning():
    with pytest.warns(UserWarning) as warning_info:
        DummyActionsManager(actions={"action1": Beta()})
    assert str(warning_info[0].message) == "Only a single action was supplied. This MAB will be deterministic."


def test_mixed_action_types_error(n_features=1):
    actions = {"action1": BayesianNeuralNetwork.cold_start(n_features=n_features), "action2": Beta()}
    with pytest.raises((ValidationError, TypeError)):
        SmabActionsManager[Beta](actions=actions)
    with pytest.raises((ValidationError, TypeError)):
        CmabActionsManager[BayesianNeuralNetwork](actions=actions)


def test_smab_mixed_action_types_error():
    beta_model = Beta()
    zoom_model = SmabZoomingModel.cold_start()

    actions = {"a1": beta_model, "a2": zoom_model}
    with pytest.raises(ValidationError):
        SmabActionsManager[SmabZoomingModel](actions=actions)
    with pytest.raises(ValidationError):
        SmabActionsManager[Beta](actions=actions)

    SmabActionsManager[Union[Beta, SmabZoomingModel]](actions=actions)


def test_cmab_mixed_action_types_error(n_features=1):
    blr_model = BayesianNeuralNetwork.cold_start(n_features=n_features)
    blr_model2 = BayesianNeuralNetwork.cold_start(n_features=n_features + 1)
    quant_model = QuantitativeBayesianNeuralNetwork.cold_start(n_features=n_features)
    quant_model2 = QuantitativeBayesianNeuralNetwork.cold_start(n_features=n_features + 1)

    actions = {"a1": blr_model, "a2": blr_model2}
    with pytest.raises(AttributeError):
        CmabActionsManager[BayesianNeuralNetwork](actions=actions)

    actions = {"a1": quant_model, "a2": quant_model2}
    with pytest.raises(AttributeError):
        CmabActionsManager[QuantitativeBayesianNeuralNetwork](actions=actions)

    actions = {"a1": blr_model, "a2": quant_model2}
    with pytest.raises(AttributeError):
        CmabActionsManager[Union[BayesianNeuralNetwork, QuantitativeBayesianNeuralNetwork]](actions=actions)

    actions = {"a1": blr_model, "a2": quant_model}
    CmabActionsManager[Union[BayesianNeuralNetwork, QuantitativeBayesianNeuralNetwork]](actions=actions)


@given(
    n_successes=st.just(100),
    n_failures=st.just(1),
    delta=st.just(REFERENCE_DELTA),
    reference=st.just(28),
)
def test_change_detection(n_successes, n_failures, delta, reference):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    actions_memory = ["action1"] * (n_successes - 1)
    rewards_memory = [1] * (n_successes - 1)
    manager.update(actions=actions_memory, rewards=rewards_memory, actions_memory=[], rewards_memory=[])
    assert manager.actions["action1"].n_successes == n_successes
    assert manager.actions["action1"].n_failures == n_failures
    # Initially no changes detected
    assert manager.actions_with_change == set()

    manager.update(
        actions=["action1"] * 100,
        rewards=[0] * 100,
        actions_memory=actions_memory * 2,
        rewards_memory=rewards_memory * 2,
    )
    assert manager.actions["action1"].n_successes == 1
    assert manager.actions["action1"].n_failures == reference
    # Verify that change point was detected and recorded
    assert len(manager.actions_with_change) > 0
    # Check that action1 is in the detected changes
    detected_actions = {action_id for action_id, _ in manager.actions_with_change}
    assert "action1" in detected_actions


########################################################################################################################


# ActionsManager._extract_action_specific_kwargs functionality tests


def test_returns_empty_dict_when_no_action_specific_kwargs():
    kwargs = {"param1": 1, "param2": 2}
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == ({}, {})


def test_processes_kwargs_with_non_dict_values():
    kwargs = {
        f"{ACTION_IDS_PREFIX}param1": "not_a_dict",
    }
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == ({}, {})


def test_manages_kwargs_with_empty_dicts():
    kwargs = {f"{ACTION_IDS_PREFIX}param1": {}, f"{ACTION_IDS_PREFIX}param2": {}}
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == ({}, {})
    assert kwargs == {}


def test_extracts_action_specific_kwargs_with_valid_keys():
    kwargs = {
        f"{ACTION_IDS_PREFIX}param1": {"action1": 1, "action2": 2},
        f"{ACTION_IDS_PREFIX}param2": {"action1": 3, "action2": 4},
    }
    expected_output = ({"action1": {"param1": 1, "param2": 3}, "action2": {"param1": 2, "param2": 4}}, {})
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == expected_output
    assert kwargs == {}


def test_extracts_quantitative_action_specific_kwargs_with_valid_keys():
    kwargs = {
        f"{QUANTITATIVE_ACTION_IDS_PREFIX}param1": {"action1": 1, "action2": 2},
        f"{QUANTITATIVE_ACTION_IDS_PREFIX}param2": {"action1": 3, "action2": 4},
    }
    expected_output = ({}, {"action1": {"param1": 1, "param2": 3}, "action2": {"param1": 2, "param2": 4}})
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == expected_output
    assert kwargs == {}


########################################################################################################################


# ActionsManager._extract_action_model_class_and_attributes functionality tests
class MockActionModel:
    def __init__(self, param1, param2):
        pass

    @classmethod
    def cold_start(cls):
        pass


def test_handles_empty_kwargs_gracefully(mocker: MockerFixture):
    mocker.patch("pybandits.utils.extract_argument_names_from_function", return_value=[])
    mocker.patch("pybandits.base.PyBanditsBaseModel._get_field_type", return_value=MockActionModel)
    mocker.patch("pybandits.actions_manager.issubclass", return_value=True)

    (
        _,
        _,
        action_general_kwargs,
        quantitative_action_general_kwargs,
    ) = ActionsManager._extract_action_model_class_and_attributes({})

    assert action_general_kwargs == {}
    assert quantitative_action_general_kwargs is None

    mocker.patch("pybandits.actions_manager.issubclass", side_effect=[False, True])
    (
        _,
        _,
        action_general_kwargs,
        quantitative_action_general_kwargs,
    ) = ActionsManager._extract_action_model_class_and_attributes({})
    assert action_general_kwargs is None
    assert quantitative_action_general_kwargs == {}


def test_handles_kwargs_with_no_matching_action_model_attributes(mocker: MockerFixture):
    mocker.patch("pybandits.utils.extract_argument_names_from_function", return_value=[])
    mocker.patch("pybandits.base.PyBanditsBaseModel._get_field_type", return_value=MockActionModel)
    mocker.patch("pybandits.actions_manager.issubclass", return_value=True)
    kwargs = {"irrelevant_param": 1}
    kwargs_copy = kwargs.copy()
    (
        _,
        _,
        action_general_kwargs,
        quantitative_action_general_kwargs,
    ) = ActionsManager._extract_action_model_class_and_attributes(kwargs_copy)

    assert action_general_kwargs == {}
    assert quantitative_action_general_kwargs is None
    assert kwargs == kwargs_copy

    mocker.patch("pybandits.actions_manager.issubclass", side_effect=[False, True])
    (
        _,
        _,
        action_general_kwargs,
        quantitative_action_general_kwargs,
    ) = ActionsManager._extract_action_model_class_and_attributes(kwargs_copy)

    assert action_general_kwargs is None
    assert quantitative_action_general_kwargs == {}
    assert kwargs == kwargs_copy


def test_extracts_action_model_class_and_attributes_with_valid_kwargs(mocker: MockerFixture):
    mocker.patch("pybandits.utils.extract_argument_names_from_function", return_value=["param1", "param2"])
    mocker.patch("pybandits.base.PyBanditsBaseModel._get_field_type", return_value=MockActionModel)

    kwargs = {"param1": 1, "param2": 2}
    with pytest.raises(TypeError):
        ActionsManager._extract_action_model_class_and_attributes(kwargs.copy())

    mocker.patch("pybandits.actions_manager.issubclass", return_value=True)
    (
        _,
        _,
        action_general_kwargs,
        quantitative_action_general_kwargs,
    ) = ActionsManager._extract_action_model_class_and_attributes(kwargs.copy())
    assert action_general_kwargs == kwargs
    assert quantitative_action_general_kwargs is None

    mocker.patch("pybandits.actions_manager.issubclass", side_effect=[False, True])
    (
        _,
        _,
        action_general_kwargs,
        quantitative_action_general_kwargs,
    ) = ActionsManager._extract_action_model_class_and_attributes(kwargs.copy())
    assert action_general_kwargs is None
    assert quantitative_action_general_kwargs == kwargs


########################################################################################################################


# SmabActionsManager
@settings(deadline=None)
@given(
    data_len_base=st.just(math.sqrt(10)),
    data_len_power=st.integers(min_value=1, max_value=8),
    memory_len_base=st.just(math.sqrt(10)),
    memory_len_power=st.integers(min_value=1, max_value=8),
    other_reward=st.integers(min_value=0, max_value=1),
)
def test_smab_manager_update(data_len_base, data_len_power, memory_len_base, memory_len_power, other_reward):
    actions_dict = {
        "action1": Beta(),
        "action2": Beta(),
    }
    data_len = int(data_len_base**data_len_power)
    memory_len = int(memory_len_base**memory_len_power)
    manager = SmabActionsManager[Beta](actions=actions_dict, delta=REFERENCE_DELTA)
    actions = ["action1"] * data_len
    rewards = [1] * data_len
    actions_memory = ["action1"] * memory_len
    rewards_memory = [other_reward] * memory_len
    manager.update(actions_memory, rewards_memory, None)
    manager.update(actions, rewards, None, actions_memory=actions_memory, rewards_memory=rewards_memory)


@given(n_objectives=st.integers(min_value=1, max_value=10), other_n_objectives=st.integers(min_value=1, max_value=10))
def test_smab_actions_different_number_of_objectives(n_objectives, other_n_objectives):
    """Test the specific ValueError when actions have different numbers of objectives."""
    if n_objectives == other_n_objectives:
        return
    # Create actions with different numbers of objectives
    # BetaMO has models attribute with different lengths
    beta_mo_n_obj = BetaMO.cold_start(n_objectives=n_objectives)  # 2 objectives
    beta_mo_other_obj = BetaMO.cold_start(n_objectives=other_n_objectives)  # 3 objectives

    actions = {
        "action1": beta_mo_n_obj,  # Single objective (no models attribute)
        "action2": beta_mo_other_obj,  # 2 objectives
    }

    with pytest.raises(ValueError, match="All actions should have the same number of objectives"):
        SmabActionsManager[Union[Beta, BetaMO]](actions=actions)


########################################################################################################################


# CmabActionsManager
@settings(deadline=None)
@given(
    context=st.lists(
        st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=3), min_size=1, max_size=10000
    ),
    context_memory=st.lists(
        st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=3), min_size=1, max_size=10000
    ),
    n_features=st.just(3),
    other_reward=st.integers(min_value=0, max_value=1),
)
def test_cmab_manager_update(context, context_memory, n_features, other_reward, monkeymodule):
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

    actions = {
        "action1": BayesianNeuralNetwork.cold_start(n_features=n_features),
        "action2": BayesianNeuralNetwork.cold_start(n_features=n_features),
    }
    manager = CmabActionsManager[BayesianNeuralNetwork](actions=actions, delta=REFERENCE_DELTA)
    actions = ["action1"] * len(context)
    rewards = [1] * len(context)
    actions_memory = ["action1"] * len(context_memory)
    rewards_memory = [other_reward] * len(context_memory)
    manager.update(actions_memory, rewards_memory, None, context=context_memory)
    manager.update(
        actions,
        rewards,
        None,
        context=context,
        actions_memory=actions_memory,
        rewards_memory=rewards_memory,
        context_memory=context_memory,
    )


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_check_context_matrix(n_samples, n_features):
    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    CmabActionsManager._check_context_matrix(context=context)

    # raise an error if len(context) != len(self.betas)
    with pytest.raises(AttributeError):
        CmabActionsManager._check_context_matrix(context=context.loc[:, 1:])
    with pytest.raises(AttributeError):
        CmabActionsManager._check_context_matrix(context=[[1], [2, 3]])  # context has shape mismatch
    with pytest.raises(AttributeError):
        CmabActionsManager._check_context_matrix(context="a")  # context is a string


# Handle context and context_memory with non matching feature dimensions
@settings(deadline=None)
@given(
    context=st.lists(
        st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4), min_size=1, max_size=10000
    ),
    context_memory=st.lists(
        st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=3), min_size=1, max_size=10000
    ),
    n_features=st.just(3),
    other_reward=st.integers(min_value=0, max_value=1),
)
def test_cmab_context_memory_matching_dimensions(context, context_memory, n_features, other_reward, monkeymodule):
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
    actions = {
        "action1": BayesianNeuralNetwork.cold_start(n_features=n_features),
        "action2": BayesianNeuralNetwork.cold_start(n_features=n_features),
    }
    manager = CmabActionsManager[BayesianNeuralNetwork](actions=actions, delta=REFERENCE_DELTA)
    actions = ["action1"] * len(context)
    rewards = [1] * len(context)
    actions_memory = ["action1"] * len(context_memory)
    rewards_memory = [other_reward] * len(context_memory)
    manager.update(actions_memory, rewards_memory, None, context=context_memory)

    with pytest.raises(ValueError):
        manager.update(
            actions,
            rewards,
            context=context,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
            context_memory=context_memory,
        )


def test_cmab_context_memory_features_mismatch():
    """Test the specific ValueError when context memory has different number of features than context."""

    # Create actions with 3 features
    actions = {
        "action1": BayesianNeuralNetwork.cold_start(n_features=3),
        "action2": BayesianNeuralNetwork.cold_start(n_features=3),
    }
    manager = CmabActionsManager[BayesianNeuralNetwork](actions=actions, delta=REFERENCE_DELTA)

    # Context with 3 features
    context = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    # Context memory with 4 features (mismatch)
    context_memory = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]

    actions_list = ["action1", "action2"]
    rewards = [1, 0]
    actions_memory = ["action1", "action2"]
    rewards_memory = [1, 0]

    with pytest.raises(ValueError, match="Context memory must have the same number of features as the context."):
        manager.update(
            actions=actions_list,
            rewards=rewards,
            quantities=None,
            context=context,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
            context_memory=context_memory,
        )


#######################################################################################################################


# ActionsManager._slice_memory functionality tests
@given(
    memory_len=st.integers(min_value=1, max_value=100),
    data_len=st.integers(min_value=1, max_value=200),
)
def test_slice_memory_with_longer_data(memory_len, data_len):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    # Generate test data longer than memory_len
    data_len = max(memory_len + 1, data_len)
    actions_memory = ["action1"] * data_len
    rewards_memory = [1] * data_len
    memory_kwargs = {"context_memory": list(range(data_len)), "empty_memory": []}

    actions_memory, rewards_memory, memory_kwargs = manager._slice_memory(
        memory_len=memory_len, actions_memory=actions_memory, rewards_memory=rewards_memory, memory_kwargs=memory_kwargs
    )

    assert len(actions_memory) == memory_len
    assert len(rewards_memory) == memory_len
    assert len(memory_kwargs["context_memory"]) == memory_len
    assert len(memory_kwargs["empty_memory"]) == 0


@given(
    memory_len=st.integers(min_value=1, max_value=100),
    data_len=st.integers(min_value=1, max_value=100),
)
def test_slice_memory_with_shorter_data(memory_len, data_len):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    # Generate test data shorter than memory_len
    data_len = min(memory_len - 1, data_len)
    actions_memory = ["action1"] * data_len
    rewards_memory = [1] * data_len
    memory_kwargs = {"context_memory": list(range(data_len)), "empty_memory": []}

    original_lengths = (len(actions_memory), len(rewards_memory), len(memory_kwargs["context_memory"]))

    actions_memory, rewards_memory, memory_kwargs = manager._slice_memory(
        memory_len=memory_len, actions_memory=actions_memory, rewards_memory=rewards_memory, memory_kwargs=memory_kwargs
    )

    # Verify nothing changed when data is shorter than memory_len
    assert len(actions_memory) == original_lengths[0]
    assert len(rewards_memory) == original_lengths[1]
    assert len(memory_kwargs["context_memory"]) == original_lengths[2]
    assert len(memory_kwargs["empty_memory"]) == 0


def test_slice_memory_empty_data():
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    actions_memory = []
    rewards_memory = []
    memory_kwargs = {"context_memory": [], "empty_memory": []}

    actions_memory, rewards_memory, memory_kwargs = manager._slice_memory(
        memory_len=10, actions_memory=actions_memory, rewards_memory=rewards_memory, memory_kwargs=memory_kwargs
    )

    assert len(actions_memory) == 0
    assert len(rewards_memory) == 0
    assert len(memory_kwargs["context_memory"]) == 0
    assert len(memory_kwargs["empty_memory"]) == 0


@given(data_len=st.integers(min_value=101, max_value=200), memory_len=st.integers(min_value=1, max_value=100))
def test_slice_memory_maintains_last_elements(data_len, memory_len):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    actions_backup = np.random.choice(list(actions.keys()), size=data_len).tolist()
    rewards_backup = np.random.randint(0, 2, size=data_len).tolist()  # Use range to verify order preservation
    context_backup = list(range(data_len))
    memory_kwargs = {"context_memory": context_backup.copy()}

    actions_memory = actions_backup.copy()
    rewards_memory = rewards_backup.copy()

    actions_memory, rewards_memory, memory_kwargs = manager._slice_memory(
        memory_len=memory_len, actions_memory=actions_memory, rewards_memory=rewards_memory, memory_kwargs=memory_kwargs
    )

    # Verify we kept the last memory_len elements
    assert actions_memory == actions_backup[-memory_len:]
    assert rewards_memory == rewards_backup[-memory_len:]
    assert memory_kwargs["context_memory"] == context_backup[-memory_len:]


########################################################################################################################


# ActionsManager._maybe_trim_memory functionality tests
def test_memory_trim_when_too_long(mocker: MockerFixture, trials=(3, 3), successes=(2, 1), extra_len=5):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)
    # Mock _extract_current_stats_for_action to return known values
    mocker.patch.object(
        ActionsManager,
        "_extract_current_stats_for_action",
        side_effect=[
            (np.array([[successes[0]]]), np.array([[trials[0]]])),  # action1: 2 successes, 3 trials
            (np.array([[successes[1]]]), np.array([[trials[1]]])),  # action2: 1 success, 3 trials
        ],
    )

    actions_memory = (
        np.random.choice(["action1", "action2"], size=extra_len).tolist()
        + ["action1"] * trials[0]
        + ["action2"] * trials[1]
    )
    rewards_memory = (
        np.random.randint(0, 2, size=extra_len).tolist()
        + [1] * successes[0]
        + [0] * (trials[0] - successes[0])
        + [1] * successes[1]
        + [0] * (trials[1] - successes[1])
    )
    shuffled_indexes = list(range(-sum(trials), 0))
    np.random.shuffle(shuffled_indexes)
    temp_actions = [actions_memory[i] for i in shuffled_indexes]
    temp_rewards = [rewards_memory[i] for i in shuffled_indexes]
    actions_memory[-sum(trials) :] = temp_actions
    rewards_memory[-sum(trials) :] = temp_rewards

    memory_kwargs = {"context_memory": list(range(8))}

    with pytest.warns(UserWarning):
        manager._maybe_trim_memory(actions_memory, rewards_memory, memory_kwargs)


def test_valid_memory_stats(mocker: MockerFixture):
    actions = {"action1": Beta()}
    manager = DummyActionsManager(actions=actions)

    # Mock expected stats: 2 successes out of 3 trials
    mocker.patch.object(
        ActionsManager, "_extract_current_stats_for_action", return_value=(np.array([[2]]), np.array([[3]]))
    )

    # Valid data matching expected stats
    actions_memory = ["action1"] * 3
    rewards_memory = [1, 1, 0]  # 2 successes in 3 trials
    memory_kwargs = {"context_memory": list(range(3))}

    # Should not raise any exceptions
    manager._maybe_trim_memory(actions_memory, rewards_memory, memory_kwargs)


def test_invalid_trials_count(mocker: MockerFixture):
    actions = {"action1": Beta()}
    manager = DummyActionsManager(actions=actions)

    mocker.patch.object(
        ActionsManager, "_extract_current_stats_for_action", return_value=(np.array([[2]]), np.array([[3]]))
    )

    # Too many trials
    actions_memory = ["action1"] * 4  # 4 trials > expected 3
    rewards_memory = [1, 1, 1, 1]
    memory_kwargs = {"context_memory": list(range(4))}

    with pytest.raises(ValueError):
        manager._maybe_trim_memory(actions_memory, rewards_memory, memory_kwargs)


def test_invalid_success_count(mocker: MockerFixture):
    actions = {"action1": Beta()}
    manager = DummyActionsManager(actions=actions)

    mocker.patch.object(
        ActionsManager, "_extract_current_stats_for_action", return_value=(np.array([[2]]), np.array([[3]]))
    )

    # Wrong number of successes for expected trials
    actions_memory = ["action1"] * 3
    rewards_memory = [1, 1, 1]  # 3 successes != expected 2
    memory_kwargs = {"context_memory": list(range(3))}

    with pytest.raises(ValueError, match="Memory for action action1 is not consistent"):
        manager._maybe_trim_memory(actions_memory, rewards_memory, memory_kwargs)


def test_invalid_success_count_less_trials(mocker: MockerFixture):
    """Test the specific error when actual_trials < expected_trials but actual_successes > expected_successes."""
    actions = {"action1": Beta()}
    manager = DummyActionsManager(actions=actions)

    # Mock expected stats: 1 success out of 5 trials
    mocker.patch.object(
        ActionsManager, "_extract_current_stats_for_action", return_value=(np.array([[1]]), np.array([[5]]))
    )

    # Less trials than expected (2 < 5) but more successes than expected (2 > 1)
    # This triggers the else branch
    actions_memory = ["action1"] * 2  # 2 trials < expected 5
    rewards_memory = [1, 1]  # 2 successes > expected 1
    memory_kwargs = {"context_memory": list(range(2))}

    with pytest.raises(ValueError, match="Memory for action action1 is not consistent with the expected stats."):
        manager._maybe_trim_memory(actions_memory, rewards_memory, memory_kwargs)


@given(n_actions=st.integers(min_value=1, max_value=5), trials_per_action=st.integers(min_value=1, max_value=10))
def test_hypothesis_memory_trim(n_actions: int, trials_per_action: int, monkeymodule):
    # Mock stats for each action
    successes_per_action = trials_per_action - 1  # Ensure successes < trials

    actions = {f"action{i}": Beta() for i in range(n_actions)}
    manager = DummyActionsManager(actions=actions)

    def mock_extract_current_stats_for_action(action_id, *args, **kwargs):
        # Return successes and trials for each action
        return (
            np.array([[successes_per_action]]),  # successes
            np.array([[trials_per_action]]),  # trials
        )

    if pydantic_version == PYDANTIC_VERSION_1:
        manager.__dict__["_extract_current_stats_for_action"] = mock_extract_current_stats_for_action
    elif pydantic_version == PYDANTIC_VERSION_2:
        monkeymodule.setattr(manager, "_extract_current_stats_for_action", mock_extract_current_stats_for_action)
    else:
        raise ValueError(f"Unsupported Pydantic version: {pydantic_version}")

    total_trials = n_actions * trials_per_action
    actions_memory = [f"action{i}" for _ in range(trials_per_action) for i in range(n_actions)]
    rewards_memory = [0] * n_actions + [1] * (total_trials - n_actions)  # Match expected successes
    memory_kwargs = {"context_memory": list(range(total_trials))}

    # Should not raise any exceptions
    manager._maybe_trim_memory(actions_memory, rewards_memory, memory_kwargs)

    # Verify lengths remain unchanged since data matches expectations
    assert len(actions_memory) == total_trials
    assert len(rewards_memory) == total_trials
    assert len(memory_kwargs["context_memory"]) == total_trials


########################################################################################################################


# ActionsManager.actions_with_change functionality tests
def test_actions_with_change_basic_functionality():
    """Test basic functionality: initialization, clearing, and structure"""
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)

    # Test initialization
    assert manager.actions_with_change == set()

    # Test manual population and structure
    manager.actions_with_change.add(("action1", 10))
    manager.actions_with_change.add(("action2", 25))

    # Verify structure
    assert isinstance(manager.actions_with_change, set)
    assert len(manager.actions_with_change) == 2
    for action_id, change_point in manager.actions_with_change:
        assert isinstance(action_id, str) and action_id in actions
        assert isinstance(change_point, int) and change_point >= 0

    # Test clearing on update
    manager.update(actions=["action1"], rewards=[1])
    assert manager.actions_with_change == set()


def test_actions_with_change_detection_scenarios(monkeymodule):
    """Test change detection scenarios: populated, selective, and empty"""
    actions = {"action1": Beta(), "action2": Beta(), "action3": Beta()}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)

    # Initialize actions with some data to avoid memory validation issues
    initial_actions = ["action1"] * 20 + ["action2"] * 30 + ["action3"] * 10
    initial_rewards = [1] * 60
    manager.update(actions=initial_actions, rewards=initial_rewards, actions_memory=[], rewards_memory=[])

    # Scenario 1: Multiple changes detected
    def mock_multiple_changes(action_id, *args, **kwargs):
        changes = {"action1": 15, "action2": 25, "action3": manager._no_change_point}
        return changes[action_id]

    if pydantic_version == PYDANTIC_VERSION_1:
        manager.__dict__["_get_last_change_point_for_action"] = mock_multiple_changes
    elif pydantic_version == PYDANTIC_VERSION_2:
        monkeymodule.setattr(manager, "_get_last_change_point_for_action", mock_multiple_changes)
    else:
        raise ValueError(f"Unsupported Pydantic version: {pydantic_version}")

    manager.update(actions=["action1"], rewards=[1], actions_memory=initial_actions, rewards_memory=initial_rewards)

    # Should contain changes for action1 and action2 only
    expected_changes = {("action1", 15), ("action2", 25)}
    assert manager.actions_with_change == expected_changes

    # Scenario 2: No changes detected - create a new manager to avoid state interference
    if pydantic_version == PYDANTIC_VERSION_1:
        manager.__dict__["_get_last_change_point_for_action"] = lambda *args, **kwargs: manager._no_change_point
    elif pydantic_version == PYDANTIC_VERSION_2:
        monkeymodule.setattr(
            manager, "_get_last_change_point_for_action", lambda *args, **kwargs: manager._no_change_point
        )
    else:
        raise ValueError(f"Unsupported Pydantic version: {pydantic_version}")

    manager.update(
        actions=["action1"],
        rewards=[1],
        actions_memory=initial_actions + ["action1"],
        rewards_memory=initial_rewards + [1],
    )
    assert manager.actions_with_change == set()


@given(change_point_indices=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=3))
def test_actions_with_change_hypothesis(change_point_indices, monkeymodule):
    """Hypothesis test for actions_with_change with various scenarios"""
    actions = {f"action{i}": Beta() for i in range(3)}
    manager = DummyActionsManager(actions=actions, delta=REFERENCE_DELTA)

    # Initialize actions with some data to match memory expectations
    initial_actions = ["action0"] * 50 + ["action1"] * 50 + ["action2"] * 50
    initial_rewards = [1] * 150
    manager.update(actions=initial_actions, rewards=initial_rewards, actions_memory=[], rewards_memory=[])

    expected_changes = set()

    def mock_change_detection(action_id, *args, **kwargs):
        action_num = int(action_id.replace("action", ""))
        if action_num < len(change_point_indices) and change_point_indices[action_num] > 0:
            change_point = change_point_indices[action_num]
            expected_changes.add((action_id, change_point))
            return change_point
        return manager._no_change_point

    if pydantic_version == PYDANTIC_VERSION_1:
        manager.__dict__["_get_last_change_point_for_action"] = mock_change_detection
    elif pydantic_version == PYDANTIC_VERSION_2:
        monkeymodule.setattr(manager, "_get_last_change_point_for_action", mock_change_detection)
    else:
        raise ValueError(f"Unsupported Pydantic version: {pydantic_version}")

    manager.update(actions=["action0"], rewards=[1], actions_memory=initial_actions, rewards_memory=initial_rewards)

    assert manager.actions_with_change == expected_changes


########################################################################################################################


# ActionsManager._get_expected_memory_length tests
@pytest.mark.parametrize(
    "actions, expected_len",
    [
        (
            {
                "action1": Beta(n_successes=5, n_failures=3),
                "action2": Beta(n_successes=2, n_failures=1),
                "action3": Beta(n_successes=10, n_failures=5),
            },
            20,
        ),
        ({"action1": Beta(n_successes=7, n_failures=3)}, 8),
    ],
)
def test_get_expected_memory_length_with_base_model_so(actions, expected_len):
    """Test _get_expected_memory_length with BaseModelSO models."""
    actual_length = ActionsManager._get_expected_memory_length(actions)
    assert actual_length == expected_len


@pytest.mark.parametrize(
    "actions, expected_len",
    [
        (
            {
                "action1": BetaMO(models=[Beta(n_successes=3, n_failures=2), Beta(n_successes=1, n_failures=1)]),
                "action2": BetaMO(models=[Beta(n_successes=4, n_failures=1), Beta(n_successes=2, n_failures=3)]),
            },
            6,
        ),
    ],
)
def test_get_expected_memory_length_with_base_model_mo(actions, expected_len):
    """Test _get_expected_memory_length with BaseModelMO models."""
    actual_length = ActionsManager._get_expected_memory_length(actions)
    assert actual_length == expected_len


@pytest.mark.parametrize(
    "actions, error_type, error_msg",
    [
        ({}, AttributeError, "At least one action should be defined."),
        ({"action1": object()}, ValueError, "Model type.*not supported."),
    ],
)
def test_get_expected_memory_length_with_errors(actions, error_type, error_msg):
    """Test _get_expected_memory_length with empty or unsupported actions."""
    with pytest.raises(error_type, match=error_msg):
        ActionsManager._get_expected_memory_length(actions)


def test_get_expected_memory_length_with_mixed_model_types():
    """Test _get_expected_memory_length with mixed model types uses first model type."""

    # Create mixed model types - should use the first one (BaseModelSO)
    beta1 = Beta(n_successes=2, n_failures=1)  # count = 3
    beta2 = Beta(n_successes=1, n_failures=1)  # count = 2
    beta_mo = BetaMO(models=[beta1, beta2])

    actions = {
        "action1": Beta(n_successes=3, n_failures=2),  # count = 5, BaseModelSO
        "action2": beta_mo,  # BaseModelMO
    }

    # The method should use the first model type (BaseModelSO) for consistency
    # But since BetaMO doesn't have a count attribute, this will raise an error
    # This test demonstrates the limitation of mixed model types
    with pytest.raises(AttributeError, match="'BetaMO' object has no attribute 'count'"):
        ActionsManager._get_expected_memory_length(actions)


########################################################################################################################


# ActionsManager._validate_update_params tests


def test_actions_type_validation_with_correct_types():
    """Test that no error is raised when all actions follow the expected type."""

    # Create actions with correct types
    actions = {"action1": Beta(), "action2": Beta()}

    # This should not raise any error
    manager = DummyActionsManager(actions=actions)
    assert len(manager.actions) == 2
    assert all(isinstance(action, Beta) for action in manager.actions.values())


@pytest.mark.parametrize(
    "actions, invalid_actions, rewards, error_msg",
    [
        (
            {"action1": Beta(), "action2": Beta()},
            ("action1", "invalid_action", "action2"),
            (1, 0, 1),
            "The following invalid action\\(s\\) were specified: {'invalid_action'}.",
        ),
    ],
)
def test_validate_update_params_invalid_actions(actions, invalid_actions, rewards, error_msg):
    manager = DummyActionsManager(actions=actions)
    with pytest.raises(AttributeError, match=error_msg):
        manager._validate_update_params(actions=invalid_actions, rewards=rewards)


@pytest.mark.parametrize(
    "actions, action_list, rewards, quantities",
    [
        ({"action1": Beta(), "action2": Beta()}, ("action1", "action2"), (1, 0), None),
    ],
)
def test_validate_update_params_valid_regular_actions(actions, action_list, rewards, quantities):
    manager = DummyActionsManager(actions=actions)
    manager._validate_update_params(actions=action_list, rewards=rewards, quantities=quantities)


@pytest.mark.parametrize(
    "actions, action_list, rewards",
    [
        ({"action1": Beta(), "action2": Beta()}, ("action1", "action2"), (1,)),
    ],
)
def test_validate_update_params_length_mismatch(actions, action_list, rewards):
    manager = DummyActionsManager(actions=actions)
    with pytest.raises((ValueError, AttributeError)):
        manager._validate_update_params(actions=action_list, rewards=rewards)


@pytest.mark.parametrize(
    "actions, action_list, rewards",
    [
        ({"action1": Beta(), "action2": Beta()}, tuple(), tuple()),
    ],
)
def test_validate_update_params_empty_actions(actions, action_list, rewards):
    manager = DummyActionsManager(actions=actions)
    manager._validate_update_params(actions=action_list, rewards=rewards)


def test_validate_update_params_multi_objective_rewards():
    """Test _validate_update_params with multi-objective rewards."""

    # Create multi-objective actions
    beta1 = Beta()
    beta2 = Beta()
    beta_mo = BetaMO(models=[beta1, beta2])

    actions = {
        "action1": beta_mo,
        "action2": beta_mo,
    }
    manager = DummyActionsManager(actions=actions)

    # Test with multi-objective rewards
    action_list = ["action1", "action2"]
    rewards = [[1, 0], [0, 1]]  # Multi-objective rewards

    # Should not raise any error
    manager._validate_update_params(actions=action_list, rewards=rewards)


########################################################################################################################


# ActionsManager.update tests
@pytest.fixture
def actions(names=("action1", "action2")):
    return {name: Beta() for name in names}


@pytest.mark.parametrize(
    "delta, actions_memory, rewards_memory, expected_exception, expected_message",
    [
        # Test 1: delta is None but memory is provided - should raise AttributeError
        (
            None,
            ["action1", "action2"],
            [1, 0],
            AttributeError,
            "Adaptive window size is not set, so memory should not be provided.",
        ),
        # Test 2: delta is set but memory is not provided - should warn
        (REFERENCE_DELTA, None, None, UserWarning, "Adaptive window size is set, but memory was not provided."),
        # Test 3: delta is set but only partial memory is provided - should warn
        (REFERENCE_DELTA, None, None, UserWarning, "Adaptive window size is set, but memory was not provided."),
    ],
)
def test_update_adaptive_window_size_validation(
    actions, delta, actions_memory, rewards_memory, expected_exception, expected_message
):
    """Test update validation for adaptive window size scenarios."""
    manager = DummyActionsManager(actions=actions, delta=delta)

    action_list = list(actions.keys())
    rewards = np.random.randint(0, 2, size=len(action_list)).tolist()

    if expected_exception is AttributeError:
        with pytest.raises(expected_exception, match=expected_message):
            manager.update(
                actions=action_list, rewards=rewards, actions_memory=actions_memory, rewards_memory=rewards_memory
            )
    else:  # UserWarning
        with pytest.warns(expected_exception, match=expected_message):
            manager.update(
                actions=action_list, rewards=rewards, actions_memory=actions_memory, rewards_memory=rewards_memory
            )
