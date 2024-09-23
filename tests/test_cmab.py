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
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import NonNegativeFloat, ValidationError

from pybandits.base import Float01
from pybandits.cmab import (
    CmabBernoulli,
    CmabBernoulliBAI,
    CmabBernoulliCC,
    create_cmab_bernoulli_bai_cold_start,
    create_cmab_bernoulli_cc_cold_start,
    create_cmab_bernoulli_cold_start,
)
from pybandits.model import (
    BayesianLogisticRegression,
    BayesianLogisticRegressionCC,
    StudentT,
    UpdateMethods,
    create_bayesian_logistic_regression_cc_cold_start,
    create_bayesian_logistic_regression_cold_start,
)
from pybandits.strategy import (
    BestActionIdentification,
    ClassicBandit,
    CostControlBandit,
)
from pybandits.utils import to_serializable_dict
from tests.test_utils import is_serializable

literal_update_methods = get_args(UpdateMethods)


def _apply_update_method_to_state(state, update_method):
    for action in state["actions"]:
        state["actions"][action]["update_method"] = update_method


########################################################################################################################


# CmabBernoulli with strategy=ClassicBandit()


@settings(deadline=500)
@given(st.integers(max_value=100))
def test_create_cmab_bernoulli_cold_start(a_int):
    # n_features must be > 0
    if a_int <= 0:
        with pytest.raises(ValidationError):
            create_cmab_bernoulli_cold_start(action_ids={"a1", "a2"}, n_features=a_int)
    else:
        mab1 = create_cmab_bernoulli_cold_start(action_ids={"a1", "a2"}, n_features=a_int)
        mab2 = CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
            }
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=10))
def test_cmab_can_instantiate(n_features):
    with pytest.raises(TypeError):
        CmabBernoulli()
    with pytest.raises(AttributeError):
        CmabBernoulli(actions={})
    with pytest.raises(AttributeError):
        CmabBernoulli(actions={"a1": create_bayesian_logistic_regression_cold_start(n_betas=2)})
    with pytest.raises(TypeError):  # strategy is not an argument of init
        CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            },
            strategy=ClassicBandit(),
        )
    with pytest.raises(TypeError):  # predict_with_proba is not an argument of init
        CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            },
            predict_with_proba=True,
        )
    with pytest.raises(ValidationError):
        CmabBernoulli(
            actions={
                "a1": None,
                "a2": None,
            },
        )
    mab = CmabBernoulli(
        actions={
            "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
        }
    )

    assert mab.actions["a1"] == create_bayesian_logistic_regression_cold_start(n_betas=n_features)
    assert mab.actions["a2"] == create_bayesian_logistic_regression_cold_start(n_betas=n_features)
    assert not mab.predict_actions_randomly
    assert not mab.predict_with_proba
    mab.predict_with_proba = True
    mab.predict_actions_randomly = True
    assert mab.predict_actions_randomly
    assert mab.predict_with_proba


@settings(deadline=500)
@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=6, max_value=10),
    st.integers(min_value=0, max_value=1),
    st.just("draws"),
    st.just(2),
)
def test_cmab_init_with_wrong_blr_models(
    first_n_betas, second_n_betas, first_update_method_index, kwarg_to_alter, factor
):
    with pytest.raises(AttributeError):
        CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=first_n_betas),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=first_n_betas),
                "a3": create_bayesian_logistic_regression_cold_start(n_betas=second_n_betas),
            }
        )
    first_update_method = literal_update_methods[first_update_method_index]
    second_update_method = literal_update_methods[1 - first_update_method_index]
    with pytest.raises(AttributeError):
        CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(
                    n_betas=first_n_betas, update_method=first_update_method
                ),
                "a2": create_bayesian_logistic_regression_cold_start(
                    n_betas=first_n_betas, update_method=second_update_method
                ),
            }
        )
    first_model = create_bayesian_logistic_regression_cold_start(
        n_betas=first_n_betas, update_method=first_update_method
    )
    altered_kwarg = first_model.update_kwargs[kwarg_to_alter] // factor
    with pytest.raises(AttributeError):
        CmabBernoulli(
            actions={
                "a1": first_model,
                "a2": create_bayesian_logistic_regression_cold_start(
                    n_betas=first_n_betas,
                    update_method=first_update_method,
                    update_kwargs={kwarg_to_alter: altered_kwarg},
                ),
            }
        )


@settings(deadline=60000)
@given(st.just(100), st.just(3), st.sampled_from(literal_update_methods))
def test_cmab_update(n_samples, n_features, update_method):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    def run_update(context):
        mab = create_cmab_bernoulli_cold_start(
            action_ids={"a1", "a2"}, n_features=n_features, update_method=update_method
        )
        assert all(
            [
                mab.actions[a]
                == create_bayesian_logistic_regression_cold_start(n_betas=n_features, update_method=update_method)
                for a in set(actions)
            ]
        )
        mab.update(context=context, actions=actions, rewards=rewards)
        assert all(
            [
                mab.actions[a]
                != create_bayesian_logistic_regression_cold_start(n_betas=n_features, update_method=update_method)
                for a in set(actions)
            ]
        )
        assert not mab.predict_actions_randomly

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    run_update(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) is list
    run_update(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) is pd.DataFrame
    run_update(context=context)


@settings(deadline=10000)
@given(st.just(100), st.just(3), st.sampled_from(literal_update_methods))
def test_cmab_update_not_all_actions(n_samples, n_feat, update_method):
    actions = np.random.choice(["a3", "a4"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_feat))
    mab = create_cmab_bernoulli_cold_start(
        action_ids={"a1", "a2", "a3", "a4"}, n_features=n_feat, update_method=update_method
    )

    mab.update(context=context, actions=actions, rewards=rewards)
    assert mab.actions["a1"] == create_bayesian_logistic_regression_cold_start(
        n_betas=n_feat, update_method=update_method
    )
    assert mab.actions["a2"] == create_bayesian_logistic_regression_cold_start(
        n_betas=n_feat, update_method=update_method
    )
    assert mab.actions["a3"] != create_bayesian_logistic_regression_cold_start(
        n_betas=n_feat, update_method=update_method
    )
    assert mab.actions["a4"] != create_bayesian_logistic_regression_cold_start(
        n_betas=n_feat, update_method=update_method
    )


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
    mab = create_cmab_bernoulli_cold_start(action_ids={"a1", "a2"}, n_features=n_features, update_method=update_method)

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
@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_cmab_predict_cold_start(n_samples, n_features):
    def run_predict(context):
        mab = create_cmab_bernoulli_cold_start(action_ids={"a1", "a2"}, n_features=n_features)
        selected_actions, probs, weighted_sums = mab.predict(context=context)
        assert mab.predict_actions_randomly
        assert all([a in ["a1", "a2"] for a in selected_actions])
        assert len(selected_actions) == n_samples
        assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
        assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    run_predict(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) is list
    run_predict(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) is pd.DataFrame
    run_predict(context=context)


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=3))
def test_cmab_predict_not_cold_start(n_samples, n_features):
    def run_predict(context):
        mab = CmabBernoulli(
            actions={
                "a1": BayesianLogisticRegression(alpha=StudentT(mu=1, sigma=2), betas=n_features * [StudentT()]),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            },
        )
        assert not mab.predict_actions_randomly
        selected_actions, probs, weighted_sums = mab.predict(context=context)
        assert len(selected_actions) == len(probs) == len(weighted_sums) == n_samples

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    run_predict(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) is list
    run_predict(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) is pd.DataFrame
    run_predict(context=context)


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=10))
def test_cmab_predict_shape_mismatch(a_int):
    context = np.random.uniform(low=-1.0, high=1.0, size=(100, a_int - 1))
    mab = create_cmab_bernoulli_cold_start(action_ids={"a1", "a2"}, n_features=a_int)
    with pytest.raises(AttributeError):
        mab.predict(context=context)
    with pytest.raises(AttributeError):
        mab.predict(context=[])


def test_cmab_predict_with_forbidden_actions(n_features=3):
    def run_predict(mab):
        context = np.random.uniform(low=-1.0, high=1.0, size=(1000, n_features))
        assert set(mab.predict(context=context, forbidden_actions={"a2", "a3", "a4", "a5"})[0]) == {"a1"}
        assert set(mab.predict(context=context, forbidden_actions={"a1", "a3"})[0]) == {"a2", "a4", "a5"}
        assert set(mab.predict(context=context, forbidden_actions={"a1"})[0]) == {"a2", "a3", "a4", "a5"}
        assert set(mab.predict(context=context, forbidden_actions=set())[0]) == {"a1", "a2", "a3", "a4", "a5"}

        with pytest.raises(ValidationError):  # not a list
            assert set(mab.predict(context=context, forbidden_actions={1})[0])
        with pytest.raises(ValueError):  # invalid action_ids
            assert set(mab.predict(context=context, forbidden_actions={"a1", "a9999", "a", 5})[0])
        with pytest.raises(ValueError):  # all actions forbidden
            assert set(mab.predict(context=context, forbidden_actions={"a1", "a2", "a3", "a4", "a5"})[0])
        with pytest.raises(ValueError):  # all actions forbidden (unordered)
            assert set(mab.predict(n_samples=1000, forbidden_actions={"a5", "a4", "a2", "a3", "a1"})[0])

    # cold start mab
    mab = create_cmab_bernoulli_cold_start(action_ids={"a1", "a2", "a3", "a4", "a5"}, n_features=n_features)
    run_predict(mab=mab)

    # not cold start mab
    mab = CmabBernoulli(
        actions={
            "a1": BayesianLogisticRegression(alpha=StudentT(mu=1, sigma=2), betas=[StudentT(), StudentT(), StudentT()]),
            "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            "a3": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            "a4": BayesianLogisticRegression(alpha=StudentT(mu=4, sigma=5), betas=[StudentT(), StudentT(), StudentT()]),
            "a5": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
        },
    )
    assert mab != create_cmab_bernoulli_cold_start(action_ids={"a1", "a2", "a3", "a4", "a5"}, n_features=n_features)
    run_predict(mab=mab)


@settings(deadline=500)
@given(st.integers(min_value=1), st.integers(min_value=1), st.integers(min_value=2, max_value=100))
def test_cmab_get_state(mu, sigma, n_features):
    actions: dict = {
        "a1": BayesianLogisticRegression(alpha=StudentT(mu=mu, sigma=sigma), betas=n_features * [StudentT()]),
        "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
    }

    cmab = CmabBernoulli(actions=actions)
    expected_state = to_serializable_dict(
        {
            "actions": actions,
            "strategy": {},
            "predict_with_proba": False,
            "predict_actions_randomly": False,
            "epsilon": None,
            "default_action": None,
        }
    )

    class_name, cmab_state = cmab.get_state()
    assert class_name == "CmabBernoulli"
    assert cmab_state == expected_state

    assert is_serializable(cmab_state), "Internal state is not serializable"


@settings(deadline=500)
@given(
    state=st.fixed_dictionaries(
        {
            "actions": st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.fixed_dictionaries(
                    {
                        "alpha": st.fixed_dictionaries(
                            {
                                "mu": st.floats(min_value=-100, max_value=100),
                                "nu": st.floats(min_value=0, max_value=100),
                                "sigma": st.floats(min_value=0, max_value=100),
                            }
                        ),
                        "betas": st.lists(
                            st.fixed_dictionaries(
                                {
                                    "mu": st.floats(min_value=-100, max_value=100),
                                    "nu": st.floats(min_value=0, max_value=100),
                                    "sigma": st.floats(min_value=0, max_value=100),
                                }
                            ),
                            min_size=3,
                            max_size=3,
                        ),
                    },
                ),
                min_size=2,
            ),
            "strategy": st.fixed_dictionaries({}),
        }
    ),
    update_method=st.sampled_from(literal_update_methods),
)
def test_cmab_from_state(state, update_method):
    _apply_update_method_to_state(state, update_method)
    cmab = CmabBernoulli.from_state(state)
    assert isinstance(cmab, CmabBernoulli)

    actual_actions = to_serializable_dict(cmab.actions)  # Normalize the dict
    expected_actions = {k: {**v, **state["actions"][k]} for k, v in actual_actions.items()}
    assert expected_actions == actual_actions

    # Ensure get_state and from_state compatibility
    new_cmab = globals()[cmab.get_state()[0]].from_state(state=cmab.get_state()[1])
    assert new_cmab == cmab


########################################################################################################################


# CmabBernoulli with strategy=BestActionIdentification()


@settings(deadline=500)
@given(st.integers(max_value=100))
def test_create_cmab_bernoulli_bai_cold_start(a_int):
    # n_features must be > 0
    if a_int <= 0:
        with pytest.raises(ValidationError):
            create_cmab_bernoulli_bai_cold_start(action_ids={"a1", "a2"}, n_features=a_int)
    else:
        # default exploit_p
        mab1 = create_cmab_bernoulli_bai_cold_start(action_ids={"a1", "a2"}, n_features=a_int)
        mab2 = CmabBernoulliBAI(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
            }
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2

        # set exploit_p
        mab1 = create_cmab_bernoulli_bai_cold_start(action_ids={"a1", "a2"}, n_features=a_int, exploit_p=0.42)
        mab2 = CmabBernoulliBAI(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
            },
            exploit_p=0.42,
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=10))
def test_cmab_bai_can_instantiate(n_features):
    with pytest.raises(TypeError):
        CmabBernoulliBAI()
    with pytest.raises(AttributeError):
        CmabBernoulliBAI(actions={})
    with pytest.raises(AttributeError):
        CmabBernoulliBAI(actions={"a1": create_bayesian_logistic_regression_cold_start(n_betas=2)})
    with pytest.raises(TypeError):  # strategy is not an argument of init
        CmabBernoulliBAI(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            },
            strategy=BestActionIdentification(),
        )
    with pytest.raises(TypeError):  # predict_with_proba is not an argument of init
        CmabBernoulliBAI(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            },
            predict_with_proba=True,
        )
    with pytest.raises(ValidationError):
        CmabBernoulliBAI(
            actions={
                "a1": None,
                "a2": None,
            },
        )
    mab = CmabBernoulliBAI(
        actions={
            "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
        }
    )
    assert mab.actions["a1"] == create_bayesian_logistic_regression_cold_start(n_betas=n_features)
    assert mab.actions["a2"] == create_bayesian_logistic_regression_cold_start(n_betas=n_features)
    assert not mab.predict_actions_randomly
    assert not mab.predict_with_proba
    assert mab.strategy == BestActionIdentification()

    mab = CmabBernoulliBAI(
        actions={
            "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
        },
        exploit_p=0.42,
    )
    assert mab.actions["a1"] == create_bayesian_logistic_regression_cold_start(n_betas=n_features)
    assert mab.actions["a2"] == create_bayesian_logistic_regression_cold_start(n_betas=n_features)
    assert not mab.predict_actions_randomly
    assert not mab.predict_with_proba
    assert mab.strategy == BestActionIdentification(exploit_p=0.42)


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=3))
def test_cmab_bai_predict(n_samples, n_features):
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # cold start
    mab = create_cmab_bernoulli_bai_cold_start(action_ids={"a1", "a2"}, n_features=n_features)
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert mab.predict_actions_randomly
    assert all([a in ["a1", "a2"] for a in selected_actions])
    assert len(selected_actions) == n_samples
    assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
    assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]

    # not cold start
    mab = CmabBernoulliBAI(
        actions={
            "a1": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
            "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
        },
        exploit_p=0.42,
    )
    assert not mab.predict_actions_randomly
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert len(selected_actions) == len(probs) == len(weighted_sums) == n_samples


@settings(deadline=10000)
@given(st.just(100), st.just(3), st.sampled_from(literal_update_methods))
def test_cmab_bai_update(n_samples, n_features, update_method):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = create_cmab_bernoulli_bai_cold_start(
        action_ids={"a1", "a2"}, n_features=n_features, update_method=update_method
    )
    assert mab.predict_actions_randomly
    assert all(
        [
            mab.actions[a]
            == create_bayesian_logistic_regression_cold_start(n_betas=n_features, update_method=update_method)
            for a in set(actions)
        ]
    )
    mab.update(context=context, actions=actions, rewards=rewards)
    assert all(
        [
            mab.actions[a]
            != create_bayesian_logistic_regression_cold_start(n_betas=n_features, update_method=update_method)
            for a in set(actions)
        ]
    )
    assert not mab.predict_actions_randomly


@settings(deadline=500)
@given(
    st.integers(min_value=1),
    st.integers(min_value=1),
    st.integers(min_value=2, max_value=100),
    st.floats(min_value=0, max_value=1),
)
def test_cmab_bai_get_state(mu, sigma, n_features, exploit_p: Float01):
    actions: dict = {
        "a1": BayesianLogisticRegression(alpha=StudentT(mu=mu, sigma=sigma), betas=n_features * [StudentT()]),
        "a2": create_bayesian_logistic_regression_cold_start(n_betas=n_features),
    }

    cmab = CmabBernoulliBAI(actions=actions, exploit_p=exploit_p)
    expected_state = to_serializable_dict(
        {
            "actions": actions,
            "strategy": {"exploit_p": exploit_p},
            "predict_with_proba": False,
            "predict_actions_randomly": False,
            "epsilon": None,
            "default_action": None,
        }
    )

    class_name, cmab_state = cmab.get_state()
    assert class_name == "CmabBernoulliBAI"
    assert cmab_state == expected_state

    assert is_serializable(cmab_state), "Internal state is not serializable"


@settings(deadline=500)
@given(
    state=st.fixed_dictionaries(
        {
            "actions": st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.fixed_dictionaries(
                    {
                        "alpha": st.fixed_dictionaries(
                            {
                                "mu": st.floats(min_value=-100, max_value=100),
                                "nu": st.floats(min_value=0, max_value=100),
                                "sigma": st.floats(min_value=0, max_value=100),
                            }
                        ),
                        "betas": st.lists(
                            st.fixed_dictionaries(
                                {
                                    "mu": st.floats(min_value=-100, max_value=100),
                                    "nu": st.floats(min_value=0, max_value=100),
                                    "sigma": st.floats(min_value=0, max_value=100),
                                }
                            ),
                            min_size=3,
                            max_size=3,
                        ),
                    },
                ),
                min_size=2,
            ),
            "strategy": st.one_of(
                st.just({}),
                st.just({"exploit_p": None}),
                st.builds(lambda x: {"exploit_p": x}, st.floats(min_value=0, max_value=1)),
            ),
        }
    ),
    update_method=st.sampled_from(literal_update_methods),
)
def test_cmab_bai_from_state(state, update_method):
    _apply_update_method_to_state(state, update_method)
    cmab = CmabBernoulliBAI.from_state(state)
    assert isinstance(cmab, CmabBernoulliBAI)

    actual_actions = to_serializable_dict(cmab.actions)  # Normalize the dict
    expected_actions = {k: {**v, **state["actions"][k]} for k, v in actual_actions.items()}
    assert expected_actions == actual_actions

    expected_exploit_p = (
        state["strategy"].get("exploit_p", 0.5) if state["strategy"].get("exploit_p") is not None else 0.5
    )  # Covers both not existing and existing + None
    actual_exploit_p = cmab.strategy.exploit_p
    assert expected_exploit_p == actual_exploit_p

    # Ensure get_state and from_state compatibility
    new_cmab = globals()[cmab.get_state()[0]].from_state(state=cmab.get_state()[1])
    assert new_cmab == cmab


########################################################################################################################


# CmabBernoulli with strategy=CostControlBandit()


@settings(deadline=500)
@given(st.integers(max_value=100))
def test_create_cmab_bernoulli_cc_cold_start(a_int):
    action_ids_cost = {"a1": 10, "a2": 20.5}
    # n_features must be > 0
    if a_int <= 0:
        with pytest.raises(ValidationError):
            create_cmab_bernoulli_cc_cold_start(action_ids_cost=action_ids_cost, n_features=a_int)
    else:
        # default subsidy_factor
        mab1 = create_cmab_bernoulli_cc_cold_start(action_ids_cost=action_ids_cost, n_features=a_int)
        mab2 = CmabBernoulliCC(
            actions={
                "a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=a_int, cost=action_ids_cost["a1"]),
                "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=a_int, cost=action_ids_cost["a2"]),
            }
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2

        # set subsidy_factor
        mab1 = create_cmab_bernoulli_cc_cold_start(
            action_ids_cost=action_ids_cost, n_features=a_int, subsidy_factor=0.42
        )
        mab2 = CmabBernoulliCC(
            actions={
                "a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=a_int, cost=action_ids_cost["a1"]),
                "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=a_int, cost=action_ids_cost["a2"]),
            },
            subsidy_factor=0.42,
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=10))
def test_cmab_cc_can_instantiate(n_features):
    with pytest.raises(TypeError):
        CmabBernoulliCC()
    with pytest.raises(AttributeError):
        CmabBernoulliCC(actions={})
    with pytest.raises(AttributeError):
        CmabBernoulliCC(actions={"a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)})
    with pytest.raises(TypeError):  # strategy is not an argument of init
        CmabBernoulliCC(
            actions={
                create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
                create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
            },
            strategy=CostControlBandit(),
        )
    with pytest.raises(TypeError):  # predict_with_proba is not an argument of init
        CmabBernoulliCC(
            actions={
                "a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
                "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
            },
            predict_with_proba=True,
        )
    with pytest.raises(ValidationError):
        CmabBernoulliCC(
            actions={
                "a1": None,
                "a2": None,
            },
        )
    mab = CmabBernoulliCC(
        actions={
            "a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
            "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
        }
    )
    assert mab.actions["a1"] == create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)
    assert mab.actions["a2"] == create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)
    assert not mab.predict_actions_randomly
    assert mab.predict_with_proba
    assert mab.strategy == CostControlBandit()

    mab = CmabBernoulliCC(
        actions={
            "a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
            "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
        },
        subsidy_factor=0.42,
    )
    assert mab.actions["a1"] == create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)
    assert mab.actions["a2"] == create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)
    assert not mab.predict_actions_randomly
    assert mab.predict_with_proba
    assert mab.strategy == CostControlBandit(subsidy_factor=0.42)


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=3))
def test_cmab_cc_predict(n_samples, n_features):
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # cold start
    mab = create_cmab_bernoulli_cc_cold_start(action_ids_cost={"a1": 10, "a2": 20.5}, n_features=n_features)
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert mab.predict_actions_randomly
    assert all([a in ["a1", "a2"] for a in selected_actions])
    assert len(selected_actions) == n_samples
    assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
    assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]

    # not cold start
    mab = CmabBernoulliCC(
        actions={
            "a1": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10),
            "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=20.5),
        },
        subsidy_factor=0.42,
    )
    assert not mab.predict_actions_randomly
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert len(selected_actions) == len(probs) == len(weighted_sums) == n_samples


@settings(deadline=10000)
@given(st.just(100), st.just(3), st.sampled_from(literal_update_methods))
def test_cmab_cc_update(n_samples, n_features, update_method):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = create_cmab_bernoulli_cc_cold_start(
        action_ids_cost={"a1": 10, "a2": 10}, n_features=n_features, update_method=update_method
    )
    assert mab.predict_actions_randomly
    assert all(
        [
            mab.actions[a]
            == create_bayesian_logistic_regression_cc_cold_start(
                n_betas=n_features, cost=10, update_method=update_method
            )
            for a in set(actions)
        ]
    )
    mab.update(context=context, actions=actions, rewards=rewards)
    assert all(
        [
            mab.actions[a]
            != create_bayesian_logistic_regression_cc_cold_start(
                n_betas=n_features, cost=10, update_method=update_method
            )
            for a in set(actions)
        ]
    )
    assert not mab.predict_actions_randomly


@settings(deadline=500)
@given(
    st.integers(min_value=1),
    st.integers(min_value=1),
    st.integers(min_value=2, max_value=100),
    st.floats(min_value=0),
    st.floats(min_value=0),
    st.floats(min_value=0, max_value=1),
)
def test_cmab_cc_get_state(
    mu, sigma, n_features, cost_1: NonNegativeFloat, cost_2: NonNegativeFloat, subsidy_factor: Float01
):
    actions: dict = {
        "a1": BayesianLogisticRegressionCC(
            alpha=StudentT(mu=mu, sigma=sigma), betas=n_features * [StudentT()], cost=cost_1
        ),
        "a2": create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=cost_2),
    }

    cmab = CmabBernoulliCC(actions=actions, subsidy_factor=subsidy_factor)
    expected_state = to_serializable_dict(
        {
            "actions": actions,
            "strategy": {"subsidy_factor": subsidy_factor},
            "predict_with_proba": True,
            "predict_actions_randomly": False,
            "epsilon": None,
            "default_action": None,
        }
    )

    class_name, cmab_state = cmab.get_state()
    assert class_name == "CmabBernoulliCC"
    assert cmab_state == expected_state

    assert is_serializable(cmab_state), "Internal state is not serializable"


@settings(deadline=500)
@given(
    state=st.fixed_dictionaries(
        {
            "actions": st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.fixed_dictionaries(
                    {
                        "alpha": st.fixed_dictionaries(
                            {
                                "mu": st.floats(min_value=-100, max_value=100),
                                "nu": st.floats(min_value=0, max_value=100),
                                "sigma": st.floats(min_value=0, max_value=100),
                            }
                        ),
                        "betas": st.lists(
                            st.fixed_dictionaries(
                                {
                                    "mu": st.floats(min_value=-100, max_value=100),
                                    "nu": st.floats(min_value=0, max_value=100),
                                    "sigma": st.floats(min_value=0, max_value=100),
                                }
                            ),
                            min_size=3,
                            max_size=3,
                        ),
                        "cost": st.floats(min_value=0),
                    },
                ),
                min_size=2,
            ),
            "strategy": st.one_of(
                st.just({}),
                st.just({"subsidy_factor": None}),
                st.builds(lambda x: {"subsidy_factor": x}, st.floats(min_value=0, max_value=1)),
            ),
        }
    ),
    update_method=st.sampled_from(literal_update_methods),
)
def test_cmab_cc_from_state(state, update_method):
    _apply_update_method_to_state(state, update_method)
    cmab = CmabBernoulliCC.from_state(state)
    assert isinstance(cmab, CmabBernoulliCC)

    actual_actions = to_serializable_dict(cmab.actions)  # Normalize the dict
    expected_actions = {k: {**v, **state["actions"][k]} for k, v in actual_actions.items()}
    assert expected_actions == actual_actions

    expected_subsidy_factor = (
        state["strategy"].get("subsidy_factor", 0.5) if state["strategy"].get("subsidy_factor") is not None else 0.5
    )  # Covers both not existing and existing + None
    actual_subsidy_factor = cmab.strategy.subsidy_factor
    assert expected_subsidy_factor == actual_subsidy_factor

    # Ensure get_state and from_state compatibility
    new_cmab = globals()[cmab.get_state()[0]].from_state(state=cmab.get_state()[1])
    assert new_cmab == cmab


########################################################################################################################


# Cmab with epsilon-greedy super strategy


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_epsilon_greedy_cmab_predict_cold_start(n_samples, n_features):
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    mab = create_cmab_bernoulli_cold_start(
        action_ids={"a1", "a2"}, n_features=n_features, epsilon=0.1, default_action="a1"
    )
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert mab.predict_actions_randomly
    assert all([a in ["a1", "a2"] for a in selected_actions])
    assert len(selected_actions) == n_samples
    assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
    assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=3))
def test_epsilon_greedy_cmab_bai_predict(n_samples, n_features):
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    mab = create_cmab_bernoulli_bai_cold_start(
        action_ids={"a1", "a2"}, n_features=n_features, epsilon=0.1, default_action="a1"
    )
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert mab.predict_actions_randomly
    assert all([a in ["a1", "a2"] for a in selected_actions])
    assert len(selected_actions) == n_samples
    assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
    assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]


@settings(deadline=500)
@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=3))
def test_epsilon_greedy_cmab_cc_predict(n_samples, n_features):
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # cold start
    mab = create_cmab_bernoulli_cc_cold_start(
        action_ids_cost={"a1": 10, "a2": 20.5}, n_features=n_features, epsilon=0.1, default_action="a1"
    )
    selected_actions, probs, weighted_sums = mab.predict(context=context)
    assert mab.predict_actions_randomly
    assert all([a in ["a1", "a2"] for a in selected_actions])
    assert len(selected_actions) == n_samples
    assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
    assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]
