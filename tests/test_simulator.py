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

from random import choice
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from pybandits.base import ActionId, BinaryReward, Probability
from pybandits.mab import BaseMab
from pybandits.simulator import ProbabilityValue, Simulator


class DummySimulator(Simulator):
    def _initialize_results(self):
        self._results = pd.DataFrame()

    @classmethod
    def _validate_probs_reward_values(cls, probability: ProbabilityValue, is_quantitative_action: bool):
        pass

    def _draw_rewards(self, actions: List[ActionId], metadata: Dict[str, List]) -> List[BinaryReward]:
        return choice([0, 1], k=len(actions))

    def _get_batch_step_kwargs_and_metadata(self, batch_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        return {}, {}

    def _finalize_step(self, batch_results: pd.DataFrame, update_kwargs: Dict[str, np.ndarray]) -> pd.DataFrame:
        return batch_results

    def _extract_ground_truth(self, *args, **kwargs) -> Probability:
        return np.random.random()


def test_mismatched_probs_reward_columns(mocker: MockerFixture):
    def check_value_error(probs_reward):
        with pytest.raises(ValueError):
            DummySimulator(mab=mab, probs_reward=probs_reward)

    mab = mocker.Mock(spec=BaseMab)
    mab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    mab.epsilon = 0.0
    mab.default_action = None
    probs_reward = {"a3": [0.5]}
    check_value_error(probs_reward)
    probs_reward = {"a1": [0.5], "a2": [2]}
    check_value_error(probs_reward)
    probs_reward = {"a1": [0.5], "a2": [0.5], "a3": [0.5]}
    check_value_error(probs_reward)


# Test _generate_prob_reward


# Returns spline function for single dimension input when second_dimension=0
@given(first_dimension=st.integers(min_value=1, max_value=10))
def test_single_dimension_spline(first_dimension):
    spline_fn = Simulator._generate_prob_reward(first_dimension=first_dimension)
    test_input = np.random.random(first_dimension)
    result = spline_fn(test_input)
    assert isinstance(result, float)
    assert 0 <= result <= 1


# Returns spline function for two dimension inputs when second_dimension>0
@given(first_dim=st.integers(min_value=1, max_value=5), second_dim=st.integers(min_value=1, max_value=5))
def test_two_dimension_spline(first_dim, second_dim):
    spline_fn = Simulator._generate_prob_reward(first_dimension=first_dim, second_dimension=second_dim)
    input1 = np.random.random(first_dim)
    input2 = np.random.random(second_dim)
    result = spline_fn(input1, input2)
    assert isinstance(result, float)
    assert 0 <= result <= 1


# Generates n_points random uniform values between -1 and 1 for spline interpolation
def test_random_points_generation(mocker):
    random_mock = mocker.patch("numpy.random.uniform")
    with pytest.raises(ValueError):
        Simulator._generate_prob_reward(first_dimension=1, n_points=5)
    random_mock.assert_called_with(-1, 1, 5)


# Raises ValueError if spline_degree >= n_points
def test_invalid_spline_degree():
    with pytest.raises(ValueError):
        Simulator._generate_prob_reward(first_dimension=1, n_points=3, spline_degree=3)


# Validates that n_points is positive integer
@given(n_points=st.integers(max_value=0))
def test_n_points_validation(n_points):
    with pytest.raises(ValueError):
        Simulator._generate_prob_reward(first_dimension=1, n_points=n_points)
