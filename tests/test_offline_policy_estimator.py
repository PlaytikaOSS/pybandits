from typing import Tuple
from unittest import mock

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pybandits import offline_policy_estimator
from pybandits.offline_policy_estimator import BaseOfflinePolicyEstimator
from pybandits.utils import get_non_abstract_classes


@st.composite
def invalid_inputs(draw, n_samples: int = 10, n_actions: int = 2):
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
