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

from typing import Tuple
from unittest import mock

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pybandits import offline_policy_estimator
from pybandits.offline_policy_estimator import BaseOfflinePolicyEstimator, DoublyRobustWithOptimisticShrinkage
from pybandits.utils import get_non_abstract_classes


@st.composite
def invalid_inputs(draw, n_samples: int = 10, n_actions: int = 2):
    """Generate invalid inputs for testing with configurable sample and action counts."""
    reward = None
    propensity_score = None
    estimated_policy = None
    expected_reward = None
    expected_importance_weight = None
    bad_argument = draw(
        st.sampled_from(
            [
                "action",
                "reward",
                "propensity_score",
                "estimated_policy",
                "expected_reward",
                "expected_importance_weight",
            ]
        )
    )
    if bad_argument == "action":
        action = draw(arrays(dtype=int, shape=(n_samples, 2), elements=st.integers(0, n_actions - 1)))
    else:
        action = draw(arrays(dtype=int, shape=(n_samples,), elements=st.integers(0, n_actions - 1)))
        assume(np.unique(action).size == n_actions)
        if bad_argument == "reward":
            reward = draw(
                st.one_of(
                    arrays(dtype=int, shape=(n_samples, 2), elements=st.integers(0, 1)),
                    arrays(dtype=float, shape=(n_samples,), elements=st.floats(0, 1)),
                    arrays(
                        dtype=int,
                        shape=(n_samples - 1,),
                        elements=st.integers(0, 1),
                    ),
                    arrays(
                        dtype=int,
                        shape=(n_samples + 1,),
                        elements=st.integers(0, 1),
                    ),
                )
            )
        elif bad_argument in ("propensity_score", "expected_importance_weight"):
            random_value = draw(
                st.one_of(
                    arrays(dtype=float, shape=(n_samples, 2), elements=st.floats(0, 1)),
                    arrays(dtype=float, shape=(n_samples,), elements=st.floats(0, 0)),
                    arrays(dtype=int, shape=(n_samples,), elements=st.integers(0, 1)),
                    arrays(
                        dtype=float,
                        shape=(n_samples - 1,),
                        elements=st.floats(0, 1),
                    ),
                    arrays(
                        dtype=float,
                        shape=(n_samples + 1,),
                        elements=st.floats(0, 1),
                    ),
                )
            )

            if bad_argument == "propensity_score":
                propensity_score = random_value
            elif bad_argument == "expected_importance_weight":
                expected_importance_weight = random_value
        elif bad_argument == "estimated_policy":
            estimated_policy = draw(
                st.one_of(
                    arrays(dtype=float, shape=(n_samples,), elements=st.floats(0, 1)),
                    arrays(dtype=float, shape=(n_samples, 2), elements=st.floats(0, 0)),
                    arrays(dtype=int, shape=(n_samples, 2), elements=st.integers(0, 1)),
                    arrays(
                        dtype=float,
                        shape=(n_samples - 1, 1),
                        elements=st.floats(0, 1),
                    ),
                    arrays(
                        dtype=float,
                        shape=(n_samples + 1, 1),
                        elements=st.floats(0, 1),
                    ),
                )
            )
        elif bad_argument == "expected_reward":
            expected_reward = draw(
                st.one_of(
                    arrays(dtype=float, shape=(n_samples,), elements=st.floats(0, 1)),
                    arrays(dtype=int, shape=(n_samples, 2), elements=st.integers(0, 1)),
                    arrays(
                        dtype=float,
                        shape=(n_samples - 1, 1),
                        elements=st.floats(0, 1),
                    ),
                    arrays(
                        dtype=float,
                        shape=(n_samples + 1, 1),
                        elements=st.floats(0, 1),
                    ),
                )
            )
        else:
            raise ValueError(f"Invalid bad_argument: {bad_argument}")
    return action, reward, propensity_score, estimated_policy, expected_reward, expected_importance_weight


@mock.patch.multiple(BaseOfflinePolicyEstimator, __abstractmethods__=set())
@given(invalid_inputs())
def test_shape_mismatches(
    inputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    """Test shape mismatches with configurable inputs."""
    action, reward, propensity_score, estimated_policy, expected_reward, expected_importance_weight = inputs
    estimator = BaseOfflinePolicyEstimator()
    kwargs = {}
    if reward is not None:
        kwargs["reward"] = reward
    if propensity_score is not None:
        kwargs["propensity_score"] = propensity_score
    if estimated_policy is not None:
        kwargs["estimated_policy"] = estimated_policy
    if expected_reward is not None:
        kwargs["expected_reward"] = expected_reward
    if expected_importance_weight is not None:
        kwargs["expected_importance_weight"] = expected_importance_weight
    with pytest.raises(ValueError):
        estimator._check_inputs(action=action, **kwargs)


def test_check_array_mismatched_actions(n_samples: int = 5, n_actions: int = 3, wrong_actions: int = 2):
    """Test _check_array raises ValueError when 2D array's second dimension does not match number of actions."""
    # estimated_policy with wrong number of actions (should be n_actions, but is wrong_actions)
    estimated_policy = np.ones((n_samples, wrong_actions), dtype=float)
    data = {"estimated_policy": estimated_policy}
    with pytest.raises(ValueError):
        BaseOfflinePolicyEstimator._check_array(
            name="estimated_policy",
            data=data,
            ndim=2,
            dtype=float,
            n_samples=n_samples,
            n_actions=n_actions,
        )


def test_check_inputs_action_not_integer(n_samples: int = 5):
    """Test _check_inputs raises ValueError when action array is not of integer dtype."""
    # action array with float dtype
    action = np.array([0.0, 1.0, 2.0, 0.0, 1.0][:n_samples], dtype=float)
    with pytest.raises(ValueError, match="action must be an integer array."):
        BaseOfflinePolicyEstimator._check_inputs(action=action)


@given(
    arrays(dtype=int, shape=(10,), elements=st.integers(0, 1)),
    arrays(dtype=int, shape=(10,), elements=st.integers(0, 1)),
    arrays(dtype=float, shape=(10,), elements=st.floats(0.01, 1)),
    arrays(dtype=float, shape=(10, 2), elements=st.floats(0.01, 1)),
    arrays(dtype=float, shape=(10, 2), elements=st.floats(0, 1)),
    arrays(dtype=float, shape=(10,), elements=st.floats(0.01, 1)),
)
def test_default_estimators(
    action, reward, propensity_score, estimated_policy, expected_reward, expected_importance_weight
):
    """Test default estimators with configurable inputs."""
    if np.unique(action).size > 1:
        estimators = [class_() for class_ in get_non_abstract_classes(offline_policy_estimator)]
        for estimator in estimators:
            estimator.estimate_policy_value_with_confidence_interval(
                action=action,
                reward=reward,
                propensity_score=propensity_score,
                estimated_policy=estimated_policy,
                expected_reward=expected_reward,
                expected_importance_weight=expected_importance_weight,
            )


class TestDoublyRobustWithOptimisticShrinkage:
    """Test cases for DoublyRobustWithOptimisticShrinkage._shrink_weights method."""

    @pytest.mark.parametrize(
        "shrinkage_factor,importance_weight_values,expected_result,test_description",
        [
            (0.0, [1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0], "zero shrinkage factor"),
            (float("inf"), [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], "infinite shrinkage factor"),
            (2.0, [1.0, 2.0, 3.0, 4.0, 5.0], None, "finite shrinkage factor"),  # Will be calculated
            (
                1.5,
                [0.0, 1.0, 0.0, 2.0, 0.0],
                [0.0, 1.5 * 1.0 / (1.0**2 + 1.5), 0.0, 1.5 * 2.0 / (2.0**2 + 1.5), 0.0],
                "with zeros",
            ),
            (0.1, [10.0, 100.0, 1000.0], None, "large values"),  # Will be calculated
            (10.0, [0.1, 0.01, 0.001], None, "small values"),  # Will be calculated
            (1.0, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], None, "2D array"),  # Will be calculated
            (2.0, [-1.0, -2.0, -3.0], None, "negative values"),  # Will be calculated
            (1.5, [-2.0, 0.0, 1.0, 3.0, -1.0], None, "mixed values"),  # Will be calculated
            (0.5, [2.0], [0.5 * 2.0 / (2.0**2 + 0.5)], "single value"),
            (1.0, [], [], "empty array"),
            (1e-10, [1e10, 1e-10, 1e5], None, "numerical stability"),  # Will be calculated
        ],
    )
    def test_shrink_weights_parametrized(
        self, shrinkage_factor: float, importance_weight_values: list, expected_result: list, test_description: str
    ) -> None:
        """Parametrized test for _shrink_weights method covering various scenarios."""
        estimator = DoublyRobustWithOptimisticShrinkage(shrinkage_factor=shrinkage_factor)
        importance_weight = np.array(importance_weight_values)

        result = estimator._shrink_weights(importance_weight)

        # Calculate expected result if not provided
        if expected_result is None:
            expected_result = shrinkage_factor * importance_weight / (importance_weight**2 + shrinkage_factor)
        else:
            expected_result = np.array(expected_result)

        # For numerical stability test, check for finite results
        if test_description == "numerical stability":
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
            assert np.all(np.isfinite(result))
        else:
            np.testing.assert_array_almost_equal(result, expected_result)

    @given(
        st.floats(min_value=0.0, max_value=10.0),
        arrays(dtype=float, shape=st.integers(1, 5), elements=st.floats(min_value=-5.0, max_value=5.0)),
    )
    def test_shrink_weights_property_based(self, shrinkage_factor: float, importance_weight: np.ndarray) -> None:
        """Property-based test for _shrink_weights method."""
        estimator = DoublyRobustWithOptimisticShrinkage(shrinkage_factor=shrinkage_factor)

        result = estimator._shrink_weights(importance_weight)

        # Test that result has same shape as input
        assert result.shape == importance_weight.shape

        # Test that result has same dtype as input
        assert result.dtype == importance_weight.dtype

        # Test that when importance_weight is 0, result is 0
        zero_mask = importance_weight == 0.0
        if zero_mask.any():
            np.testing.assert_array_equal(result[zero_mask], 0.0)

        # Test that when shrinkage_factor is 0, result is all zeros
        if shrinkage_factor == 0.0:
            np.testing.assert_array_equal(result, np.zeros_like(importance_weight))

        # Test that when shrinkage_factor is infinity, result equals importance_weight
        if np.isinf(shrinkage_factor) and shrinkage_factor > 0:
            np.testing.assert_array_equal(result, importance_weight)
