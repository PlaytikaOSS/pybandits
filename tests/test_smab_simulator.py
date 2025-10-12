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

import os
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest_mock.plugin import MockerFixture

import pybandits
from pybandits.actions_manager import SmabModelType
from pybandits.base_model import BaseModel
from pybandits.model import Beta
from pybandits.quantitative_model import QuantitativeModel, SmabZoomingModel
from pybandits.smab import SmabBernoulli
from pybandits.smab_simulator import SmabSimulator
from tests.utils import mock_update, sample_with_replacement, to_unified_action_id


def test_mismatched_probs_reward_columns(mocker: MockerFixture):
    """
    Test that SmabSimulator raises ValueError when probs_reward keys don't match action keys.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest mock fixture for mocking objects.
    """
    smab = mocker.Mock(spec=SmabBernoulli)
    smab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    smab.epsilon = 0.0
    smab.default_action = None
    probs_reward = {str(i): {"a1": 0.5, "a2": 0.5} for i in range(2)}
    with pytest.raises(ValueError):
        SmabSimulator(mab=smab, probs_reward=probs_reward)


@pytest.mark.parametrize(
    "probs_reward_config, expected_values",
    [
        # Standard actions with float probabilities
        ({"a1": 0.7, "a2": 0.3}, {"a1": 0.7, "a2": 0.3}),
        # Quantitative actions with callable functions
        ({"a1": lambda q: 0.8, "a2": lambda q: 0.4}, {"a1": 0.8, "a2": 0.4}),
        # Mixed actions (standard + quantitative)
        ({"a1": 0.6, "a2": lambda q: 0.9}, {"a1": 0.6, "a2": 0.9}),
    ],
)
def test_smab_simulator_with_explicit_probs_reward(
    mocker: MockerFixture,
    probs_reward_config: Dict[str, Union[float, Callable]],
    expected_values: Dict[str, float],
    dimension: int = 2,
    test_quantity=np.array([0.5, 0.3]),
):
    """
    Test SmabSimulator when probs_reward is explicitly provided (not None).

    This test ensures that the if condition in replace_null_and_validate_probs_reward
    is false, improving test coverage for that branch. Covers standard actions,
    quantitative actions, and mixed scenarios.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest mock fixture for mocking objects.
    probs_reward_config : Dict[str, Union[float, Callable]]
        Explicit probs_reward configuration with float values or callable functions.
    expected_values : Dict[str, float]
        Expected probability values for verification.
    """
    # Create mock MAB with actions that match the probs_reward types
    smab = mocker.Mock(spec=SmabBernoulli)
    smab.actions = {}

    for action_id, prob_value in probs_reward_config.items():
        if callable(prob_value):
            # Quantitative action - needs a model with dimension
            mock_model = mocker.Mock(spec=QuantitativeModel)
            mock_model.dimension = dimension
            smab.actions[action_id] = mock_model
        else:
            # Standard action - regular mock
            smab.actions[action_id] = mocker.Mock()

    smab.epsilon = 0.0
    smab.default_action = None

    # # Mock isinstance to return True for quantitative actions
    # def mock_isinstance(obj, cls):
    #     if cls.__name__ == 'QuantitativeModel':
    #         return True
    #     return False

    # with mocker.patch('builtins.isinstance', side_effect=mock_isinstance):
    # Create simulator with explicit probs_reward
    simulator = SmabSimulator(mab=smab, probs_reward=probs_reward_config)

    # Verify that the provided probs_reward was used (not generated)
    assert simulator.probs_reward == probs_reward_config

    # Test that the values work correctly

    for action_id, expected_value in expected_values.items():
        prob_value = simulator.probs_reward[action_id]
        if callable(prob_value):
            # For quantitative actions, test the callable function
            actual_value = prob_value(test_quantity)
            assert actual_value == expected_value
        else:
            # For standard actions, test the float value directly
            assert prob_value == expected_value


@pytest.mark.parametrize(
    "probability, is_quantitative_action, should_pass",
    [
        # Valid non-quantitative actions
        (0.5, False, True),
        (0.0, False, True),
        (1.0, False, True),
        # Invalid non-quantitative actions - not float
        ("0.5", False, False),
        (None, False, False),
        (lambda x: 0.5, False, False),
        # Invalid non-quantitative actions - out of range
        (-0.1, False, False),
        (1.1, False, False),
        # Valid quantitative actions
        (lambda x: 0.5, True, True),
        (lambda quantity: 0.7, True, True),
        (lambda *args: 0.5, True, True),
        # Invalid quantitative actions - not callable
        (0.5, True, False),
        (None, True, False),
        # Invalid quantitative actions - wrong argument count
        (lambda: 0.5, True, False),
        (lambda x, y: 0.5, True, False),
    ],
)
def test_validate_probs_reward_values(
    probability: Union[float, Callable], is_quantitative_action: bool, should_pass: bool
):
    """
    Test the _validate_probs_reward_values method with various combinations
    of probability values and action types.

    Parameters
    ----------
    probability : Union[float, callable]
        The probability value to test
    is_quantitative_action : bool
        Whether the action is quantitative
    should_pass : bool
        Whether the validation should pass
    """
    if should_pass:
        # Should not raise any exception
        SmabSimulator._validate_probs_reward_values(probability, is_quantitative_action)
    else:
        # Should raise ValueError
        with pytest.raises(ValueError):
            SmabSimulator._validate_probs_reward_values(probability, is_quantitative_action)


def mock_predict(self, n_samples, *args, **kwargs):
    action_ids = [to_unified_action_id(action_id, model) for action_id, model in self.actions.items()]
    return (
        sample_with_replacement(action_ids, n_samples),
        [{action_id: np.random.random() for action_id in action_ids} for _ in range(n_samples)],
    )


@settings(deadline=None)
@given(
    action_ids=st.just(["a1", "a2"]),
    models=st.lists(st.sampled_from([Beta(), SmabZoomingModel.cold_start()]), min_size=2, max_size=2),
)
def test_smab_e2e_simulation_with_default_args(
    action_ids: List[str], models: List[BaseModel], monkeymodule: pytest.MonkeyPatch
):
    """
    Test end-to-end simulation with default arguments.

    Parameters
    ----------
    action_ids : List[str]
        List of action IDs for the MAB.
    models : List[BaseModel]
        List of models for the actions.
    monkeymodule : MonkeyPatch
        Pytest monkeypatch fixture for modifying module attributes.
    """
    monkeymodule.setattr(pybandits.utils, "maximize_by_quantity", lambda *args, **kwargs: np.random.random())
    monkeymodule.setattr(pybandits.smab_simulator, "maximize_by_quantity", lambda *args, **kwargs: np.random.random())
    monkeymodule.setattr(pybandits.smab.SmabBernoulli, "predict", mock_predict)
    monkeymodule.setattr(pybandits.smab.SmabBernoulli, "update", mock_update)

    mab = SmabBernoulli(actions=dict(zip(action_ids, models)))
    with TemporaryDirectory() as path:
        simulator = SmabSimulator(mab=mab, visualize=True, save=True, path=path)
        simulator.run()
        assert not simulator.results.empty
        dir_list = os.listdir(path)
        assert "simulation_results.csv" in dir_list
        assert "selected_actions_count.csv" in dir_list
        assert "positive_reward_proportion.csv" in dir_list
        assert "simulation_results.html" in dir_list


@settings(deadline=None)
@given(
    action_ids=st.just(["a1", "a2"]),
    models=st.lists(st.sampled_from([Beta(), SmabZoomingModel.cold_start()]), min_size=2, max_size=2),
    n_updates=st.integers(min_value=1, max_value=10),
    batch_size=st.integers(min_value=1, max_value=10),
    save=st.booleans(),
    random_seed=st.sampled_from([None, 0, 42]),
    verbose=st.booleans(),
    visualize=st.booleans(),
    file_prefix=st.sampled_from(["", "unit_test"]),
)
def test_smab_e2e_simulation_with_non_default_args(
    action_ids: List[str],
    models: List[SmabModelType],
    n_updates: int,
    batch_size: int,
    save: bool,
    random_seed: Optional[int],
    verbose: bool,
    visualize: bool,
    file_prefix: str,
    monkeymodule: MonkeyPatch,
):
    """
    Test end-to-end simulation with non-default arguments.

    Parameters
    ----------
    action_ids : List[str]
        List of action IDs for the MAB.
    models : List[BaseModel]
        List of models for the actions.
    n_updates : int
        Number of updates for the simulation.
    batch_size : int
        Batch size for the simulation.
    save : bool
        Whether to save results.
    random_seed : Optional[int]
        Random seed for reproducibility.
    verbose : bool
        Whether to enable verbose output.
    visualize : bool
        Whether to enable visualization.
    file_prefix : str
        Prefix for saved files.
    monkeymodule : MonkeyPatch
        Pytest monkeypatch fixture for modifying module attributes.
    """
    monkeymodule.setattr(pybandits.utils, "maximize_by_quantity", lambda *args, **kwargs: np.random.random())
    monkeymodule.setattr(pybandits.smab_simulator, "maximize_by_quantity", lambda *args, **kwargs: np.random.random())
    monkeymodule.setattr(pybandits.smab.SmabBernoulli, "predict", mock_predict)
    monkeymodule.setattr(pybandits.smab.SmabBernoulli, "update", mock_update)

    mab = SmabBernoulli(actions=dict(zip(action_ids, models)))
    if visualize and not save:
        with pytest.raises(ValueError):
            SmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=None,
                verbose=verbose,
                file_prefix=file_prefix,
            )
    else:
        with TemporaryDirectory() as path:
            simulator = SmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                path=path,
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=None,
                verbose=verbose,
                file_prefix=file_prefix,
            )
            simulator.run()
            if save:
                assert not simulator.results.empty
                dir_list = os.listdir(path)
                if file_prefix:
                    assert f"{file_prefix}_simulation_results.csv" in dir_list
                    assert f"{file_prefix}_selected_actions_count.csv" in dir_list
                    assert f"{file_prefix}_positive_reward_proportion.csv" in dir_list
                    if visualize:
                        assert f"{file_prefix}_simulation_results.html" in dir_list
                else:
                    assert "simulation_results.csv" in dir_list
                    assert "selected_actions_count.csv" in dir_list
                    assert "positive_reward_proportion.csv" in dir_list
                    if visualize:
                        assert "simulation_results.html" in dir_list
