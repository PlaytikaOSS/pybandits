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

from pybandits.model import (
    BayesianLogisticRegression,
    BayesianLogisticRegressionCC,
    Beta,
    BetaCC,
    BetaMO,
    BetaMOCC,
    StudentT,
)
from pybandits.pydantic_version_compatibility import ValidationError

########################################################################################################################

# Beta


@given(st.integers(), st.integers())
def test_can_init_beta(success_counter, failure_counter):
    if (success_counter <= 0) or (failure_counter <= 0):
        with pytest.raises(ValidationError):
            Beta(n_successes=success_counter, n_failures=failure_counter)
    else:
        b = Beta(n_successes=success_counter, n_failures=failure_counter)
        assert (b.n_successes, b.n_failures) == (success_counter, failure_counter)

        b = Beta()
        assert (b.n_successes, b.n_failures) == (1, 1)


def test_both_or_neither_counters_are_defined():
    with pytest.raises(ValidationError):
        Beta(n_successes=0)
    with pytest.raises(ValidationError):
        Beta(n_failures=0)
    with pytest.raises(ValidationError):
        Beta(n_successes=1, n_failures=None)
    with pytest.raises(ValidationError):
        Beta(n_successes=None, n_failures=0)


@given(st.lists(st.integers(min_value=0, max_value=1)))
def test_beta_update(rewards):
    b = Beta(n_successes=1, n_failures=2)
    b.update(rewards=rewards)
    assert b == Beta(n_successes=1 + sum(rewards), n_failures=2 + (len(rewards) - sum(rewards)))


@given(st.builds(Beta))
def test_beta_get_stats_is_working(e: Beta):
    assert e.mean >= 0, "Mean negative"
    assert e.std >= 0, "Std negative"
    assert e.count >= 2, "Count too low"


def test_beta_sample_proba():
    b = Beta(n_successes=1, n_failures=2)

    for _ in range(1000):
        prob = b.sample_proba()
        assert prob >= 0 and prob <= 1


########################################################################################################################


# BetaCC


@given(st.floats())
def test_can_init_betaCC(a_float):
    if a_float < 0 or np.isnan(a_float):
        with pytest.raises(ValidationError):
            BetaCC(cost=a_float)
    else:
        b = BetaCC(cost=a_float)
        assert b.cost == a_float


########################################################################################################################


# BetaMO


def test_can_init_base_beta_mo():
    # init with default params
    b = BetaMO(counters=[Beta(), Beta()])
    assert b.counters[0].n_successes == 1 and b.counters[0].n_failures == 1
    assert b.counters[1].n_successes == 1 and b.counters[1].n_failures == 1

    # init with empty dict
    b = BetaMO(counters=[{}, {}])
    assert b.counters[0] == Beta()

    # invalid init with BetaCC instead of Beta
    with pytest.raises(ValidationError):
        BetaMO(counters=[BetaCC(cost=1), BetaCC(cost=1)])


def test_calculate_proba_beta_mo():
    b = BetaMO(counters=[Beta(), Beta()])
    b.sample_proba()


@given(
    st.lists(st.integers(min_value=0, max_value=1)),
    st.lists(st.integers(min_value=0, max_value=1)),
)
def test_beta_update_mo(rewards1, rewards2):
    min_len = min([len(rewards1), len(rewards2)])
    rewards1, rewards2 = rewards1[:min_len], rewards2[:min_len]
    rewards = [[a, b] for a, b in zip(rewards1, rewards2)]

    b = BetaMO(counters=[Beta(n_successes=11, n_failures=22), Beta(n_successes=33, n_failures=44)])

    b.update(rewards=rewards)

    assert b == BetaMO(
        counters=[
            Beta(n_successes=11 + sum(rewards1), n_failures=22 + len(rewards1) - sum(rewards1)),
            Beta(n_successes=33 + sum(rewards2), n_failures=44 + len(rewards2) - sum(rewards2)),
        ]
    )

    with pytest.raises(AttributeError):
        b.update(rewards=[[1, 1], [1], [0, 1]])


########################################################################################################################


# BetaMO


def test_can_init_beta_mo():
    # init with default params
    b = BetaMO(counters=[Beta(), Beta()])
    assert b.counters == [Beta(), Beta()]

    # init with empty dict
    b = BetaMO(counters=[{}, {}])
    assert b.counters == [Beta(), Beta()]

    # invalid init with BetaCC instead of Beta
    with pytest.raises(ValidationError):
        BetaMO(counters=[BetaCC(cost=1), BetaCC(cost=1)])


########################################################################################################################


# BetaMOCC


@given(st.floats())
def test_can_init_beta_mo_cc(a_float):
    if a_float < 0 or np.isnan(a_float):
        with pytest.raises(ValidationError):
            BetaMOCC(counters=[Beta(), Beta()], cost=a_float)
    else:
        # init with default params
        b = BetaMOCC(counters=[Beta(), Beta()], cost=a_float)
        assert b.counters == [Beta(), Beta()]
        assert b.cost == a_float

        # init with empty dict
        b = BetaMOCC(counters=[{}, {}], cost=a_float)
        assert b.counters == [Beta(), Beta()]
        assert b.cost == a_float

        # invalid init with BetaCC instead of Beta
        with pytest.raises(ValidationError):
            BetaMOCC(counters=[BetaCC(cost=1), BetaCC(cost=1)], cost=a_float)


########################################################################################################################


# StudentT


@given(st.floats(), st.floats(), st.floats())
def test_can_init_studentt(mu, sigma, nu):
    # init with default args
    s = StudentT()
    assert (s.mu, s.sigma, s.nu) == (0, 10, 5)

    # init with args
    if np.isnan(mu) or np.isinf(mu) or np.isnan(sigma) or np.isinf(sigma) or np.isnan(nu) or np.isinf(nu):
        with pytest.raises(ValidationError):
            StudentT(mu=mu, sigma=sigma, nu=nu)
    else:
        s = StudentT(mu=mu, sigma=sigma, nu=nu)
        assert (s.mu, s.sigma, s.nu) == (mu, sigma, nu)


########################################################################################################################


# BayesianLogisticRegression


@given(st.integers(max_value=100))
def test_can_init_bayesian_logistic_regression(a_int):
    # at least one beta must be specified
    if a_int <= 0:
        with pytest.raises(ValidationError):
            BayesianLogisticRegression(alpha=StudentT(), betas=[StudentT() for _ in range(a_int)])
    else:
        blr = BayesianLogisticRegression(alpha=StudentT(), betas=[StudentT() for _ in range(a_int)])
        assert (blr.alpha, blr.betas) == (StudentT(), [StudentT() for _ in range(a_int)])


@given(st.integers(max_value=100))
def test_create_default_instance_bayesian_logistic_regression(a_int):
    # at least one beta must be specified
    if a_int <= 0:
        with pytest.raises(ValidationError):
            BayesianLogisticRegression.cold_start(n_features=a_int)
    else:
        blr = BayesianLogisticRegression.cold_start(n_features=a_int)
        assert blr == BayesianLogisticRegression(alpha=StudentT(), betas=[StudentT() for _ in range(a_int)])


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_check_context_matrix(n_samples, n_features):
    blr = BayesianLogisticRegression.cold_start(n_features=n_features)

    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    blr.check_context_matrix(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) is list
    blr.check_context_matrix(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) is pd.DataFrame
    blr.check_context_matrix(context=context)

    # raise an error if len(context) != len(self.betas)
    with pytest.raises(AttributeError):
        blr.check_context_matrix(context=context.loc[:, 1:])

    blr = BayesianLogisticRegression.cold_start(n_features=2)

    with pytest.raises(AttributeError):
        blr.check_context_matrix(context=[[1], [2, 3]])  # context has shape mismatch
    with pytest.raises(AttributeError):
        blr.check_context_matrix(context=1.0)  # context is a number
    with pytest.raises(AttributeError):
        blr.check_context_matrix(context="a")  # context is a string
    with pytest.raises(AttributeError):
        blr.check_context_matrix(context=[1.0])  # context is a 1-dim list


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=100))
def test_blr_sample_proba(n_samples, n_features):
    def sample_proba(context):
        prob, weighted_sum = blr.sample_proba(context=context)

        assert type(prob) is type(weighted_sum) is np.ndarray  # type of the returns must be np.ndarray
        assert len(prob) == len(weighted_sum) == n_samples  # return 1 sampled probability and ws per each sample
        assert all([0 <= p <= 1 for p in prob])  # probs must be in the interval [0, 1]

    blr = BayesianLogisticRegression.cold_start(n_features=n_features)

    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    sample_proba(context=context)

    # context is python list
    context = context.tolist()
    assert type(context) is list
    sample_proba(context=context)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) is pd.DataFrame
    sample_proba(context=context)


def test_blr_update(n_samples=100, n_features=3):
    def update(context, rewards):
        blr = BayesianLogisticRegression.cold_start(n_features=n_features)
        assert blr.alpha == StudentT(mu=0.0, sigma=10.0, nu=5.0)
        assert blr.betas == [
            StudentT(mu=0.0, sigma=10.0, nu=5.0),
            StudentT(mu=0.0, sigma=10.0, nu=5.0),
            StudentT(mu=0.0, sigma=10.0, nu=5.0),
        ]

        blr.update(context=context, rewards=rewards)

        assert blr.alpha != StudentT(mu=0.0, sigma=10.0, nu=5.0)
        assert blr.betas != [
            StudentT(mu=0.0, sigma=10.0, nu=5.0),
            StudentT(mu=0.0, sigma=10.0, nu=5.0),
            StudentT(mu=0.0, sigma=10.0, nu=5.0),
        ]

    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    update(context=context, rewards=rewards)

    # context is python list
    context = context.tolist()
    assert type(context) is list
    update(context=context, rewards=rewards)

    # context is pandas DataFrame
    context = pd.DataFrame(context)
    assert type(context) is pd.DataFrame
    update(context=context, rewards=rewards)

    # raise an error if len(context) != len(rewards)
    with pytest.raises(ValueError):
        blr = BayesianLogisticRegression.cold_start(n_features=n_features)
        blr.update(context=context, rewards=rewards[1:])


########################################################################################################################


# BayesianLogisticRegressionCC


@given(st.integers(max_value=100), st.floats(allow_nan=False, allow_infinity=False))
def test_can_init_bayesian_logistic_regression_cc(n_betas, cost):
    # at least one beta must be specified
    if n_betas <= 0 or cost < 0:
        with pytest.raises(ValidationError):
            BayesianLogisticRegressionCC(alpha=StudentT(), betas=[StudentT() for _ in range(n_betas)], cost=cost)
    else:
        blr = BayesianLogisticRegressionCC(alpha=StudentT(), betas=[StudentT() for _ in range(n_betas)], cost=cost)
        assert (blr.alpha, blr.betas) == (StudentT(), [StudentT() for _ in range(n_betas)])


@given(st.integers(max_value=100), st.floats(allow_nan=False, allow_infinity=False))
def test_create_default_instance_bayesian_logistic_regression_cc(n_betas, cost):
    # at least one beta must be specified
    if n_betas <= 0 or cost < 0:
        with pytest.raises(ValidationError):
            BayesianLogisticRegressionCC.cold_start(n_features=n_betas, cost=cost)
    else:
        blr = BayesianLogisticRegressionCC.cold_start(n_features=n_betas, cost=cost)
        assert blr == BayesianLogisticRegressionCC(
            alpha=StudentT(), betas=[StudentT() for _ in range(n_betas)], cost=cost
        )
