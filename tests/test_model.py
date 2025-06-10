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
from hypothesis import given, settings
from hypothesis import strategies as st

from pybandits.model import (
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    Beta,
    BetaCC,
    BetaMO,
    BetaMOCC,
    StudentTArray,
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


def test_both_or_neither_models_are_defined():
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


def test_beta_sample_proba(n_samples=100):
    b = Beta(n_successes=1, n_failures=2)
    prob = b.sample_proba(n_samples=n_samples)
    assert len(prob) == n_samples
    assert all([p >= 0 and p <= 1 for p in prob])


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
    b = BetaMO(models=[Beta(), Beta()])
    assert b.models[0].n_successes == 1 and b.models[0].n_failures == 1
    assert b.models[1].n_successes == 1 and b.models[1].n_failures == 1

    # init with empty dict
    b = BetaMO(models=[{}, {}])
    assert b.models[0] == Beta()

    # invalid init with BetaCC instead of Beta
    with pytest.raises(ValidationError):
        BetaMO(models=[BetaCC(cost=1), BetaCC(cost=1)])


def test_calculate_proba_beta_mo(n_samples=100):
    b = BetaMO(models=[Beta(), Beta()])
    b.sample_proba(n_samples=n_samples)


@given(
    st.lists(st.integers(min_value=0, max_value=1)),
    st.lists(st.integers(min_value=0, max_value=1)),
)
def test_beta_update_mo(rewards1, rewards2):
    min_len = min([len(rewards1), len(rewards2)])
    rewards1, rewards2 = rewards1[:min_len], rewards2[:min_len]
    rewards = [[a, b] for a, b in zip(rewards1, rewards2)]

    b = BetaMO(models=[Beta(n_successes=11, n_failures=22), Beta(n_successes=33, n_failures=44)])

    b.update(rewards=rewards)

    assert b == BetaMO(
        models=[
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
    b = BetaMO(models=[Beta(), Beta()])
    assert b.models == [Beta(), Beta()]

    # init with empty dict
    b = BetaMO(models=[{}, {}])
    assert b.models == [Beta(), Beta()]

    # invalid init with BetaCC instead of Beta
    with pytest.raises(ValidationError):
        BetaMO(models=[BetaCC(cost=1), BetaCC(cost=1)])


########################################################################################################################


# BetaMOCC


@given(st.floats())
def test_can_init_beta_mo_cc(a_float):
    if a_float < 0 or np.isnan(a_float):
        with pytest.raises(ValidationError):
            BetaMOCC(models=[Beta(), Beta()], cost=a_float)
    else:
        # init with default params
        b = BetaMOCC(models=[Beta(), Beta()], cost=a_float)
        assert b.models == [Beta(), Beta()]
        assert b.cost == a_float

        # init with empty dict
        b = BetaMOCC(models=[{}, {}], cost=a_float)
        assert b.models == [Beta(), Beta()]
        assert b.cost == a_float

        # invalid init with BetaCC instead of Beta
        with pytest.raises(ValidationError):
            BetaMOCC(models=[BetaCC(cost=1), BetaCC(cost=1)], cost=a_float)


########################################################################################################################


# StudentTArray
@settings(deadline=500)
@given(
    st.one_of(
        st.integers(min_value=1, max_value=10),
        st.tuples(st.integers(min_value=1, max_value=10)),
        st.tuples(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10)),
    ),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.001, allow_nan=False, allow_infinity=False),
)
def test_can_init_studenttarray(shape, mu, sigma, nu):
    # init with default args
    s = StudentTArray.cold_start(shape=shape)
    assert s.mu == np.full(shape, 0.0).tolist()
    assert s.sigma == np.full(shape, 10.0).tolist()
    assert s.nu == np.full(shape, 5.0).tolist()
    # assert s.shape == shape

    s = StudentTArray.cold_start(shape=shape, mu=mu, sigma=sigma, nu=nu)
    assert s.mu == np.full(shape, mu).tolist()
    assert s.sigma == np.full(shape, sigma).tolist()
    assert s.nu == np.full(shape, nu).tolist()
    # assert s.shape == shape


########################################################################################################################


# BayesianNeuralNetwork and BayesianLogisticRegression
@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_can_init_bayesian_neural_network(n_features, hidden_dim_list):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list):
        with pytest.raises((ValidationError, ValueError)):
            model_params = BayesianNeuralNetwork.create_model_params(n_features, hidden_dim_list)
            bnn = BayesianNeuralNetwork(model_params=model_params)
    else:
        model_params = BayesianNeuralNetwork.create_model_params(n_features, hidden_dim_list)
        bnn = BayesianNeuralNetwork(model_params=model_params)
        assert bnn.model_params == model_params


@settings(deadline=500)
@given(
    n_samples=st.integers(min_value=1, max_value=1000),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_check_context_matrix(n_samples, n_features, hidden_dim_list):
    bnn = BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list)

    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    bnn.check_context_matrix(context=context)

    # raise an error if len(context) != len(self.betas)
    with pytest.raises(AttributeError):
        bnn.check_context_matrix(context=context.loc[:, 1:])

    # check that context is a numeric numpy array
    context_str = context.copy()
    context_str = context_str.astype(object)
    context_str[:, 0] = context_str[:, 0].astype(str)
    with pytest.raises(ValueError):
        bnn.check_context_matrix(context=context_str)


@settings(deadline=20000)
@given(
    n_samples=st.integers(min_value=1, max_value=1000),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_bnn_sample_proba(n_samples, n_features, hidden_dim_list):
    def sample_proba(context):
        prob_and_weighted_sum = bnn.sample_proba(context=np.array(context))
        prob, weighted_sum = zip(*prob_and_weighted_sum)
        assert len(prob) == len(weighted_sum) == n_samples  # return 1 sampled probability and ws per each sample
        assert all([0 <= p <= 1 for p in prob])  # probs must be in the interval [0, 1]

    bnn = BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list)

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


@settings(deadline=None)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(100),
    update_method=st.just("VI"),
)
def test_bnn_vi_update(n_features, hidden_dim_list, n_samples, update_method):
    def update(context, rewards):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        init_params = dict(mu=0.0, sigma=10.0, nu=5.0)
        dim_list = [n_features] + hidden_dim_list
        for param in init_params.keys():
            for layer_ind in range(len(dim_list)):
                layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
                layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params

                assert all(w_val == init_params[param] for w_val in np.array(layer_w[param]).flatten())
                assert all(b_val == init_params[param] for b_val in np.array(layer_b[param]).flatten())

        bnn.update(context=context, rewards=rewards)

        for param in ["mu", "sigma"]:  # nu is not updated:
            for layer_ind in range(len(dim_list)):
                layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
                layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params

                assert all(w_val != init_params[param] for w_val in np.array(layer_w[param]).flatten())
                assert all(b_val != init_params[param] for b_val in np.array(layer_b[param]).flatten())

    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    update(context=context, rewards=rewards)

    # raise an error if len(context) != len(rewards)
    with pytest.raises(AttributeError):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        bnn.update(context=context, rewards=rewards[1:])


@pytest.mark.parametrize("n_features", [1, 2])
def test_bnn_mcmc_update(n_features, hidden_dim_list=(2,), n_samples=100, update_method="MCMC"):
    hidden_dim_list = list(hidden_dim_list)

    def update(context, rewards):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        init_params = dict(mu=0.0, sigma=10.0, nu=5.0)
        dim_list = [n_features] + hidden_dim_list
        for param in init_params.keys():
            for layer_ind in range(len(dim_list)):
                layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
                layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params

                assert all(w_val == init_params[param] for w_val in np.array(layer_w[param]).flatten())
                assert all(b_val == init_params[param] for b_val in np.array(layer_b[param]).flatten())

            bnn.update(context=context, rewards=rewards)

            for param in ["mu", "sigma"]:  # nu is not updated:
                for layer_ind in range(len(dim_list)):
                    layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
                    layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params

                    assert all(w_val != init_params[param] for w_val in np.array(layer_w[param]).flatten())
                    assert all(b_val != init_params[param] for b_val in np.array(layer_b[param]).flatten())

        rewards = np.random.choice([0, 1], size=n_samples).tolist()

        # context is numpy array
        context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
        assert type(context) is np.ndarray
        update(context=context, rewards=rewards)

        # raise an error if len(context) != len(rewards)
        with pytest.raises(AttributeError):
            bnn = BayesianNeuralNetwork.cold_start(
                n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
            )
            bnn.update(context=context, rewards=rewards[1:])


########################################################################################################################


# BayesianNeuralNetworkCC
@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    cost=st.floats(allow_nan=False, allow_infinity=False),
)
def test_can_init_bayesian_neural_network_cc(n_features, hidden_dim_list, cost):
    # at least one beta must be specified
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or (cost < 0):
        with pytest.raises((ValidationError, ValueError)):
            model_params = BayesianNeuralNetwork.create_model_params(n_features, hidden_dim_list)
            bnn = BayesianNeuralNetworkCC(model_params=model_params, cost=cost)
    else:
        model_params = BayesianNeuralNetwork.create_model_params(n_features, hidden_dim_list)
        bnn = BayesianNeuralNetworkCC(model_params=model_params, cost=cost)
        assert bnn.model_params == model_params


@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    cost=st.floats(allow_nan=False, allow_infinity=False),
)
def test_create_default_instance_bayesian_neural_network_cc(n_features, hidden_dim_list, cost):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or (cost < 0):
        with pytest.raises((ValidationError, ValueError)):
            BayesianNeuralNetworkCC.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list, cost=cost)
    else:
        bnn_cold_start = BayesianNeuralNetworkCC.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, cost=cost
        )
        model_params = BayesianNeuralNetwork.create_model_params(n_features=n_features, hidden_dim_list=hidden_dim_list)
        bnn_init = BayesianNeuralNetworkCC(model_params=model_params, cost=cost)
        assert bnn_cold_start == bnn_init
