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
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

import pybandits
from pybandits.cmab import CmabBernoulli
from pybandits.cmab_simulator import CmabSimulator
from pybandits.model import BayesianLogisticRegression
from pybandits.quantitative_model import CmabZoomingModel
from tests.test_utils import FakeApproximation


def test_mismatched_probs_reward_columns(mocker: MockerFixture, groups=(0, 1)):
    def check_value_error(probs_reward, context):
        with pytest.raises(ValueError):
            CmabSimulator(mab=cmab, probs_reward=probs_reward, group=list(groups), context=context)

    num_groups = len(groups)
    cmab = mocker.Mock(spec=CmabBernoulli)
    cmab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    cmab.epsilon = 0.0
    cmab.default_action = None
    context = pd.DataFrame({"a1": [0.5] * num_groups, "a2": [0.5] * num_groups})
    probs_reward = pd.DataFrame({"a1": [0.5], "a2": [0.5]})
    check_value_error(probs_reward, context)
    probs_reward = pd.DataFrame({"a1": [0.5] * num_groups, "a2": [0.5] * num_groups})
    check_value_error(probs_reward, context[:1])


@pytest.mark.parametrize(
    "probability, is_quantitative, should_pass",
    [
        # Valid functions
        (lambda x: 0.5, False, True),  # Valid non-quantitative function
        (lambda x, y: 0.5, True, True),  # Valid quantitative function
        # Invalid functions - not callable
        (0.5, False, False),
        (None, True, False),
        # Invalid functions - wrong argument count
        (lambda x, y: 0.5, False, False),
        (lambda x: 0.5, True, False),
        (lambda x, y, z: 0.5, False, False),
        (lambda x, y, z: 0.5, True, False),
    ],
)
def test_validate_probs_reward_values(probability, is_quantitative, should_pass):
    """
    Test the _validate_probs_reward_values method with various combinations
    of probability values and action types.

    Parameters
    ----------
    probability : Union[float, callable]
        The probability value to test
    is_quantitative : bool
        Whether the action is quantitative
    should_pass : bool
        Whether the validation should pass
    """
    if should_pass:
        # Should not raise any exception
        CmabSimulator._validate_probs_reward_values(probability, is_quantitative)
    else:
        # Should raise ValueError
        with pytest.raises(ValueError):
            CmabSimulator._validate_probs_reward_values(probability, is_quantitative)


@pytest.mark.parametrize(
    "probs_reward_config, expected_values, groups, context_shape",
    [
        # Standard actions with single-argument functions (context only)
        (
            {"0": {"a1": lambda x: 0.7, "a2": lambda x: 0.3}, "1": {"a1": lambda x: 0.8, "a2": lambda x: 0.2}},
            {"0": {"a1": 0.7, "a2": 0.3}, "1": {"a1": 0.8, "a2": 0.2}},
            ["0", "1"],
            (1000, 3),  # batch_size * n_updates = 100 * 10 = 1000
        ),
        # Different probability values for different groups
        (
            {"0": {"a1": lambda x: 0.6, "a2": lambda x: 0.4}, "1": {"a1": lambda x: 0.9, "a2": lambda x: 0.1}},
            {"0": {"a1": 0.6, "a2": 0.4}, "1": {"a1": 0.9, "a2": 0.1}},
            ["0", "1"],
            (1000, 3),
        ),
        # Single group scenario
        (
            {"0": {"a1": lambda x: 0.5, "a2": lambda x: 0.5}},
            {"0": {"a1": 0.5, "a2": 0.5}},
            ["0"],
            (1000, 3),
        ),
    ],
)
def test_cmab_simulator_with_explicit_probs_reward(
    mocker: MockerFixture,
    probs_reward_config: Dict[str, Dict[str, Union[Callable, float]]],
    expected_values: Dict[str, Dict[str, float]],
    groups: List[str],
    context_shape: tuple,
    dimension: int = 2,
    test_context=np.array([0.5, 0.3, 0.7]),
):
    """
    Test CmabSimulator when probs_reward is explicitly provided (not None).

    This test ensures that the if condition in replace_nulls_and_validate_sizes_and_dtypes
    is false, improving test coverage for that branch. Covers different group scenarios
    with context-based probability functions.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest mock fixture for mocking objects.
    probs_reward_config : Dict[str, Dict[str, Union[Callable, float]]]
        Explicit probs_reward configuration with group-based dictionaries containing
        callable functions that take context as input.
    expected_values : Dict[str, Dict[str, float]]
        Expected probability values for verification per group.
    groups : List[str]
        List of group identifiers.
    context_shape : tuple
        Shape of the context array (n_samples, n_features).
    dimension : int
        Dimension for quantitative models.
    test_context : np.ndarray
        Context for testing the probability values.
    """
    # Create mock MAB with actions
    cmab = mocker.Mock(spec=CmabBernoulli)
    cmab.actions = {}

    # Create mock actions for all action IDs
    first_group_probs = next(iter(probs_reward_config.values()))
    for action_id in first_group_probs.keys():
        cmab.actions[action_id] = mocker.Mock()

    cmab.epsilon = 0.0
    cmab.default_action = None

    # Create context and group arrays
    n_samples, n_features = context_shape
    context = np.random.random(context_shape)
    group_list = groups * (n_samples // len(groups)) + groups[: n_samples % len(groups)]

    # Create simulator with explicit probs_reward
    simulator = CmabSimulator(mab=cmab, probs_reward=probs_reward_config, context=context, group=group_list)

    # Verify that the provided probs_reward was used (not generated)
    assert simulator.probs_reward == probs_reward_config

    # Test that the values work correctly for each group

    for group_id, group_probs in expected_values.items():
        for action_id, expected_value in group_probs.items():
            prob_value = simulator.probs_reward[group_id][action_id]
            if callable(prob_value):
                # For context-based actions, test with context only
                actual_value = prob_value(test_context)
                assert actual_value == expected_value
            else:
                # For non-callable values, test directly
                assert prob_value == expected_value


@settings(deadline=None)
@given(
    st.just(["a1", "a2"]),
    st.lists(
        st.sampled_from(
            [
                BayesianLogisticRegression.cold_start(n_features=2, update_method="VI"),
                CmabZoomingModel.cold_start(base_model_cold_start_kwargs={"n_features": 2, "update_method": "VI"}),
            ]
        ),
        min_size=2,
        max_size=2,
    ),
    st.just(2),
    st.just(2),
)
def test_cmab_e2e_simulation_with_default_arguments(monkeymodule, action_ids, models, n_features, num_groups):
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
    monkeymodule.setattr(
        CmabSimulator,
        "_maximize_prob_reward",
        lambda *args, **kwargs: np.random.random(),
    )
    mab = CmabBernoulli(actions=dict(zip(action_ids, models)))
    base_groups = list(range(num_groups))
    n_updates = CmabSimulator.model_fields["n_updates"].default
    batch_size = CmabSimulator.model_fields["batch_size"].default
    group = base_groups * (n_updates * batch_size // num_groups) + base_groups[: (n_updates * batch_size % num_groups)]
    context = (
        np.repeat(np.arange(n_features).reshape(1, -1), n_updates * batch_size, axis=0).T
        * (np.array(group) - np.mean(group))
    ).T
    with TemporaryDirectory() as path:
        simulator = CmabSimulator(
            mab=mab,
            visualize=True,
            save=True,
            path=path,
            group=[str(g) for g in group],
            batch_size=batch_size,
            n_updates=n_updates,
            context=context,
        )
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
    models=st.lists(
        st.sampled_from(
            [
                BayesianLogisticRegression.cold_start(n_features=3, update_method="VI"),
                CmabZoomingModel.cold_start(base_model_cold_start_kwargs={"n_features": 3, "update_method": "VI"}),
            ]
        ),
        min_size=2,
        max_size=2,
    ),
    n_features=st.just(3),
    n_updates=st.integers(min_value=1, max_value=2),
    batch_size=st.integers(min_value=1, max_value=5),
    save=st.booleans(),
    random_seed=st.sampled_from([None, 0, 42]),
    verbose=st.booleans(),
    visualize=st.booleans(),
    file_prefix=st.sampled_from(["", "unit_test"]),
    num_groups=st.integers(min_value=1, max_value=3),
)
def test_cmab_e2e_simulation_with_non_default_args(
    action_ids,
    models,
    n_features,
    n_updates,
    batch_size,
    save,
    random_seed,
    verbose,
    visualize,
    file_prefix,
    num_groups,
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

    monkeymodule.setattr(
        CmabSimulator,
        "_maximize_prob_reward",
        lambda *args, **kwargs: np.random.random(),
    )
    base_groups = list(range(num_groups))
    group = base_groups * (n_updates * batch_size // num_groups) + base_groups[: (n_updates * batch_size % num_groups)]
    context = (
        np.repeat(np.arange(n_features).reshape(1, -1), n_updates * batch_size, axis=0).T
        * (np.array(group) - np.mean(group))
    ).T
    mab = CmabBernoulli(actions=dict(zip(action_ids, models)))
    if visualize and not save:
        with pytest.raises(ValueError):
            CmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                group=[str(g) for g in group],
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=None,
                verbose=verbose,
                file_prefix=file_prefix,
                context=context,
            )
    else:
        with TemporaryDirectory() as path:
            simulator = CmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                path=path,
                group=[str(g) for g in group],
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=None,
                verbose=verbose,
                file_prefix=file_prefix,
                context=context,
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
