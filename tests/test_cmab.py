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

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

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
    StudentT,
    create_bayesian_logistic_regression_cc_cold_start,
    create_bayesian_logistic_regression_cold_start,
)
from pybandits.strategy import (
    BestActionIdentification,
    ClassicBandit,
    CostControlBandit,
)

########################################################################################################################


# CmabBernoulli with strategy=ClassicBandit()


@given(st.integers(max_value=100))
def test_create_cmab_bernoulli_cold_start(a_int):
    # n_features must be > 0
    if a_int <= 0:
        with pytest.raises(ValidationError):
            create_cmab_bernoulli_cold_start(action_ids=["a1", "a2"], n_features=a_int)
    else:
        mab1 = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2"], n_features=a_int)
        mab2 = CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
            }
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2


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


@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
def test_cmab_init_with_wrong_blr_models(a, b):
    # all blr models must have the same n_betas. If not raise a ValueError.
    if a != b:
        with pytest.raises(AttributeError):
            CmabBernoulli(
                actions={
                    "a1": create_bayesian_logistic_regression_cold_start(n_betas=a),
                    "a2": create_bayesian_logistic_regression_cold_start(n_betas=a),
                    "a3": create_bayesian_logistic_regression_cold_start(n_betas=b),
                }
            )
    else:
        CmabBernoulli(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=b),
                "a3": create_bayesian_logistic_regression_cold_start(n_betas=b),
            }
        )


def test_cmab_update(n_samples=100, n_features=3):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    def run_update(context):
        mab = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2"], n_features=n_features)
        assert all(
            [mab.actions[a] == create_bayesian_logistic_regression_cold_start(n_betas=n_features) for a in set(actions)]
        )
        mab.update(context=context, actions=actions, rewards=rewards)
        assert all(
            [mab.actions[a] != create_bayesian_logistic_regression_cold_start(n_betas=n_features) for a in set(actions)]
        )
        assert not mab.predict_actions_randomly

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) == np.ndarray
    run_update(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) == list
    run_update(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) == pd.DataFrame
    run_update(context=context)


def test_cmab_update_not_all_actions(n_samples=100, n_feat=3):
    actions = np.random.choice(["a3", "a4"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_feat))
    mab = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2", "a3", "a4"], n_features=n_feat)

    mab.update(context=context, actions=actions, rewards=rewards)
    mab.actions["a1"] == create_bayesian_logistic_regression_cold_start(n_betas=n_feat)
    mab.actions["a2"] == create_bayesian_logistic_regression_cold_start(n_betas=n_feat)
    mab.actions["a3"] != create_bayesian_logistic_regression_cold_start(n_betas=n_feat)
    mab.actions["a4"] != create_bayesian_logistic_regression_cold_start(n_betas=n_feat)


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_cmab_update_shape_mismatch(n_samples, n_features):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2"], n_features=n_features)

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


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_cmab_predict_cold_start(n_samples, n_features):
    def run_predict(context):
        mab = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2"], n_features=n_features)
        selected_actions, probs, weighted_sums = mab.predict(context=context)
        assert mab.predict_actions_randomly
        assert all([a in ["a1", "a2"] for a in selected_actions])
        assert len(selected_actions) == n_samples
        assert probs == n_samples * [{"a1": 0.5, "a2": 0.5}]
        assert weighted_sums == n_samples * [{"a1": 0, "a2": 0}]

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) == np.ndarray
    run_predict(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) == list
    run_predict(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) == pd.DataFrame
    run_predict(context=context)


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
    assert type(context) == np.ndarray
    run_predict(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) == list
    run_predict(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) == pd.DataFrame
    run_predict(context=context)


@given(st.integers(min_value=1, max_value=10))
def test_cmab_predict_shape_mismatch(a_int):
    context = np.random.uniform(low=-1.0, high=1.0, size=(100, a_int - 1))
    mab = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2"], n_features=a_int)
    with pytest.raises(AttributeError):
        mab.predict(context=context)
    with pytest.raises(AttributeError):
        mab.predict(context=[])


def test_cmab_predict_with_forbidden_actions(n_features=3):
    def run_predict(mab):
        context = np.random.uniform(low=-1.0, high=1.0, size=(1000, n_features))
        assert set(mab.predict(context=context, forbidden_actions=["a2", "a3", "a4", "a5"])[0]) == {"a1"}
        assert set(mab.predict(context=context, forbidden_actions=["a1", "a3"])[0]) == {"a2", "a4", "a5"}
        assert set(mab.predict(context=context, forbidden_actions=["a1"])[0]) == {"a2", "a3", "a4", "a5"}
        assert set(mab.predict(context=context, forbidden_actions=[])[0]) == {"a1", "a2", "a3", "a4", "a5"}

        with pytest.raises(ValidationError):  # not a list
            assert set(mab.predict(context=context, forbidden_actions=1)[0])
        with pytest.raises(ValueError):  # invalid action_ids
            assert set(mab.predict(context=context, forbidden_actions=["a1", "a9999", "a", 5])[0])
        with pytest.raises(ValueError):  # all actions forbidden
            assert set(mab.predict(context=context, forbidden_actions=["a1", "a2", "a3", "a4", "a5"])[0])
        with pytest.raises(ValueError):  # all actions forbidden (unordered)
            assert set(mab.predict(n_samples=1000, forbidden_actions=["a5", "a4", "a2", "a3", "a1"])[0])

    # cold start mab
    mab = create_cmab_bernoulli_cold_start(action_ids=["a1", "a2", "a3", "a4", "a5"], n_features=n_features)
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
    assert mab != create_cmab_bernoulli_cold_start(action_ids=["a1", "a2", "a3", "a4", "a5"], n_features=n_features)
    run_predict(mab=mab)


########################################################################################################################


# CmabBernoulli with strategy=BestActionIdentification()


@given(st.integers(max_value=100))
def test_create_cmab_bernoulli_bai_cold_start(a_int):
    # n_features must be > 0
    if a_int <= 0:
        with pytest.raises(ValidationError):
            create_cmab_bernoulli_bai_cold_start(action_ids=["a1", "a2"], n_features=a_int)
    else:
        # default exploit_p
        mab1 = create_cmab_bernoulli_bai_cold_start(action_ids=["a1", "a2"], n_features=a_int)
        mab2 = CmabBernoulliBAI(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
            }
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2

        # set exploit_p
        mab1 = create_cmab_bernoulli_bai_cold_start(action_ids=["a1", "a2"], n_features=a_int, exploit_p=0.42)
        mab2 = CmabBernoulliBAI(
            actions={
                "a1": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
                "a2": create_bayesian_logistic_regression_cold_start(n_betas=a_int),
            },
            exploit_p=0.42,
        )
        mab2.predict_actions_randomly = True
        assert mab1 == mab2


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


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=3))
def test_cmab_bai_predict(n_samples, n_features):
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))

    # cold start
    mab = create_cmab_bernoulli_bai_cold_start(action_ids=["a1", "a2"], n_features=n_features)
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


def test_cmab_bai_update(n_samples=100, n_features=3):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = create_cmab_bernoulli_bai_cold_start(action_ids=["a1", "a2"], n_features=n_features)
    assert mab.predict_actions_randomly
    assert all(
        [mab.actions[a] == create_bayesian_logistic_regression_cold_start(n_betas=n_features) for a in set(actions)]
    )
    mab.update(context=context, actions=actions, rewards=rewards)
    assert all(
        [mab.actions[a] != create_bayesian_logistic_regression_cold_start(n_betas=n_features) for a in set(actions)]
    )
    assert not mab.predict_actions_randomly


########################################################################################################################


# SmabBernoulli with strategy=CostControlBandit()


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


def test_cmab_cc_update(n_samples=100, n_features=3):
    actions = np.random.choice(["a1", "a2"], size=n_samples).tolist()
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    mab = create_cmab_bernoulli_cc_cold_start(action_ids_cost={"a1": 10, "a2": 10}, n_features=n_features)
    assert mab.predict_actions_randomly
    assert all(
        [
            mab.actions[a] == create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)
            for a in set(actions)
        ]
    )
    mab.update(context=context, actions=actions, rewards=rewards)
    assert all(
        [
            mab.actions[a] != create_bayesian_logistic_regression_cc_cold_start(n_betas=n_features, cost=10)
            for a in set(actions)
        ]
    )
    assert not mab.predict_actions_randomly
