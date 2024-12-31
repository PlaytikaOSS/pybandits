from typing import List, Union

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from pybandits.actions_manager import ActionsManager, CmabActionsManager, SmabActionsManager
from pybandits.base import ACTION_IDS_PREFIX, ActionId, BinaryReward
from pybandits.model import BayesianLogisticRegression, Beta
from pybandits.pydantic_version_compatibility import NonNegativeInt, ValidationError

REFERENCE_DELTA = 0.0001


class DummyActionsManager(ActionsManager):
    def _update_actions(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], *args, **kwargs
    ):
        pass

    def _get_relative_change_point(self, last_change_point: NonNegativeInt) -> NonNegativeInt:
        return len(self.actions_memory) - last_change_point


def test_init_with_valid_actions():
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions)
    assert len(manager.actions) == 2
    assert manager.adaptive_window_size is None
    assert manager.delta is None


def test_update_with_valid_inputs(action_list=("action1", "action2", "action1"), rewards=(1, 0, 1)):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = DummyActionsManager(actions=actions, adaptive_window_size="inf")

    manager.update(actions=list(action_list), rewards=list(rewards))
    assert list(manager.actions_memory) == list(action_list)
    assert list(manager.rewards_memory) == list(rewards)


def test_empty_actions_raises_error():
    with pytest.raises(AttributeError) as exc_info:
        DummyActionsManager(actions={})
    assert str(exc_info.value) == "At least one action should be defined."


def test_single_action_warning():
    with pytest.warns(UserWarning) as warning_info:
        DummyActionsManager(actions={"action1": Beta()})
    assert str(warning_info[0].message) == "Only a single action was supplied. This MAB will be deterministic."


def test_mixed_action_types_error(n_features=1):
    actions = {"action1": BayesianLogisticRegression.cold_start(n_features=n_features), "action2": Beta()}
    with pytest.raises((ValidationError, TypeError)):
        SmabActionsManager[Beta](actions=actions)
    with pytest.raises((ValidationError, TypeError)):
        CmabActionsManager[BayesianLogisticRegression](actions=actions)


def test_invalid_memory_initialization(n_actions=1, int_adaptive_window_size=5):
    actions = {f"action{i}": Beta() for i in range(n_actions)}
    with pytest.raises(AttributeError):
        DummyActionsManager(actions=actions, adaptive_window_size="inf", actions_memory=["action1"], rewards_memory=[])
    with pytest.raises(AttributeError):
        DummyActionsManager(actions=actions, adaptive_window_size="inf", actions_memory=[], rewards_memory=[0])

    with pytest.raises(AttributeError):  # memory length should be 0 as action models are cold started
        DummyActionsManager(actions=actions, adaptive_window_size="inf", actions_memory=[0], rewards_memory=[0])

    with pytest.raises(AttributeError):
        DummyActionsManager(
            actions=actions,
            adaptive_window_size=int_adaptive_window_size,
            actions_memory=[0] * (int_adaptive_window_size + 1),
            rewards_memory=[0] * (int_adaptive_window_size + 1),
        )


@given(
    n_successes=st.just(100),
    n_failures=st.just(1),
    adaptive_window_size=st.sampled_from([100]),
    delta=st.just(REFERENCE_DELTA),
    reference=st.just(100),
)
def test_change_detection(n_successes, n_failures, adaptive_window_size, delta, reference):
    actions = {"action1": Beta(), "action2": Beta()}
    manager = SmabActionsManager[Beta](actions=actions, adaptive_window_size=adaptive_window_size, delta=delta)
    manager.update(actions=["action1"] * (n_successes - 1), rewards=[1] * (n_successes - 1))
    assert manager.actions["action1"].n_successes == n_successes
    assert manager.actions["action1"].n_failures == n_failures
    manager.update(actions=["action1"] * 100, rewards=[0] * 100)
    assert manager.actions["action1"].n_successes == 1
    assert manager.actions["action1"].n_failures == reference
    assert list(manager.actions_memory) == ["action1"] * (reference - 1)


########################################################################################################################


# ActionsManager._extract_action_specific_kwargs functionality tests


def test_returns_empty_dict_when_no_action_specific_kwargs():
    kwargs = {"param1": 1, "param2": 2}
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == {}


def test_processes_kwargs_with_non_dict_values():
    kwargs = {
        f"{ACTION_IDS_PREFIX}param1": "not_a_dict",
    }
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == {}


def test_manages_kwargs_with_empty_dicts():
    kwargs = {f"{ACTION_IDS_PREFIX}param1": {}, f"{ACTION_IDS_PREFIX}param2": {}}
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == {}


def test_extracts_action_specific_kwargs_with_valid_keys():
    kwargs = {
        f"{ACTION_IDS_PREFIX}param1": {"action1": 1, "action2": 2},
        f"{ACTION_IDS_PREFIX}param2": {"action1": 3, "action2": 4},
    }
    expected_output = {"action1": {"param1": 1, "param2": 3}, "action2": {"param1": 2, "param2": 4}}
    result = ActionsManager._extract_action_specific_kwargs(kwargs)
    assert result == expected_output


########################################################################################################################


# ActionsManager._extract_action_model_class_and_attributes functionality tests
class MockActionModel:
    def __init__(self, param1, param2):
        pass

    @classmethod
    def cold_start(cls):
        pass


def test_extracts_action_model_class_and_attributes_with_valid_kwargs(mocker: MockerFixture):
    mocker.patch("pybandits.utils.extract_argument_names_from_function", return_value=["param1", "param2"])

    kwargs = {"param1": 1, "param2": 2}
    action_general_kwargs = ActionsManager._extract_action_model_class_and_attributes(kwargs, MockActionModel.__init__)

    assert action_general_kwargs == {"param1": 1, "param2": 2}


def test_returns_callable_for_action_model_cold_start_instantiation(mocker: MockerFixture):
    mocker.patch("pybandits.base.PyBanditsBaseModel._get_field_type", return_value=MockActionModel)

    action_model_cold_start = ActionsManager._get_action_model_start_method(cold_start_mode=True)

    assert callable(action_model_cold_start)


def test_handles_empty_kwargs_gracefully(mocker: MockerFixture):
    mocker.patch("pybandits.utils.extract_argument_names_from_function", return_value=[])

    kwargs = {}
    action_general_kwargs = ActionsManager._extract_action_model_class_and_attributes(kwargs, MockActionModel.__init__)

    assert action_general_kwargs == {}


def test_handles_kwargs_with_no_matching_action_model_attributes(mocker: MockerFixture):
    mocker.patch("pybandits.utils.extract_argument_names_from_function", return_value=[])

    kwargs = {"irrelevant_param": 1}
    action_general_kwargs = ActionsManager._extract_action_model_class_and_attributes(kwargs, MockActionModel.__init__)

    assert action_general_kwargs == {}
