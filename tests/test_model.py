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
    BayesianLogisticRegression,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
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


# BaseBetaMO


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
            BayesianNeuralNetwork(model_params=model_params)
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


@pytest.mark.parametrize(
    "invalid_context",
    [
        "not_an_array",
        None,
        42,
        [["not_numeric", 1], [2, "also_not_numeric"]],
        {1: 2, 3: 4},
        True,
        [1, 2, 3],
        [[1]],
    ],
)
def test_check_context_matrix_bad_input_type(invalid_context) -> None:
    """Test error handling in check_context_matrix method for non-ArrayLike inputs."""
    bnn = BayesianNeuralNetwork.cold_start(n_features=2, hidden_dim_list=[])

    with pytest.raises(ValidationError):
        bnn.check_context_matrix(context=invalid_context)


@given(
    n_features=st.integers(min_value=1, max_value=5),
    n_rows=st.integers(min_value=1, max_value=3),
    invalid_col_delta=st.integers(min_value=1, max_value=3),
)
def test_check_context_matrix_error_handling(n_features: int, n_rows: int, invalid_col_delta: int) -> None:
    """
    Test error handling in check_context_matrix method for ArrayLike inputs with invalid number of columns.

    This test generates numpy arrays with a number of columns different from n_features,
    which should trigger a ValidationError.
    """
    # Generate invalid context arrays with too many or too few columns
    for invalid_n_features in [n_features + invalid_col_delta, max(1, n_features - invalid_col_delta)]:
        if invalid_n_features == n_features:
            continue  # skip if by chance delta is 0
        invalid_context = np.random.rand(n_rows, invalid_n_features)
        bnn = BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=[])
        with pytest.raises(AttributeError):
            bnn.check_context_matrix(context=invalid_context)


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

    # check that the model is working with multi-sample prediction
    context = np.repeat(np.random.uniform(low=-1.0, high=1.0, size=(1, n_features)), n_samples, axis=0)
    assert type(context) is np.ndarray
    prob_and_weighted_sum = bnn.sample_proba(context=np.array(context))
    prob, weighted_sum = zip(*prob_and_weighted_sum)
    is_all_different = len(np.unique(weighted_sum)) == len(weighted_sum)
    assert is_all_different


@settings(deadline=None)
@given(
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(5),
    update_method=st.just("VI"),
)
def test_bnn_vi_update(n_features, hidden_dim_list, n_samples, update_method):
    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        init_params = dict(mu=0.0, sigma=10.0, nu=5.0)
        dim_list = [n_features] + hidden_dim_list

        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            for param in init_params.keys():
                assert np.all(np.array(layer_w[param]) == init_params[param])
                assert np.all(np.array(layer_b[param]) == init_params[param])

        bnn.update(context=context, rewards=rewards)

        # nu is not updated:
        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            for param in ["mu", "sigma"]:
                assert np.all(np.array(layer_w[param]) != init_params[param])
                assert np.all(np.array(layer_b[param]) != init_params[param])

    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    update(context=context, rewards=rewards)

    # raise an error if len(context) != len(rewards)
    with pytest.raises(AttributeError):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        bnn.update(context=context, rewards=rewards[1:])


@pytest.mark.parametrize("n_features", [1, 2])
def test_bnn_mcmc_update(n_features, hidden_dim_list=(2,), n_samples=5, update_method="MCMC"):
    hidden_dim_list = list(hidden_dim_list)

    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, update_method=update_method
        )
        init_params = dict(mu=0.0, sigma=10.0, nu=5.0)
        dim_list = [n_features] + hidden_dim_list
        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            for param in init_params.keys():
                assert np.all(np.array(layer_w[param]) == init_params[param])
                assert np.all(np.array(layer_b[param]) == init_params[param])

        bnn.update(context=context, rewards=rewards)

        # nu is not updated:
        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            for param in ["mu", "sigma"]:
                assert np.all(np.array(layer_w[param]) != init_params[param])
                assert np.all(np.array(layer_b[param]) != init_params[param])

    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
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


########################################################################################################################


# BayesianLogisticRegression


@given(
    n_features=st.integers(min_value=1, max_value=10),
)
def test_bayesian_logistic_regression_valid_init(n_features: int) -> None:
    """Test that BayesianLogisticRegression can be initialized with valid single-layer configurations."""

    # Should not raise any validation errors
    blr = BayesianLogisticRegression.cold_start(n_features=n_features, hidden_dim_list=[])
    assert len(blr.model_params.bnn_layer_params) == 1
    assert len(blr.model_params.bnn_layer_params_init) == 1


@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=5),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
)
def test_bayesian_logistic_regression_invalid_init(n_features: int, hidden_dim_list: list) -> None:
    """Test that BayesianLogisticRegression raises ValueError for multi-layer configurations."""

    with pytest.raises(ValueError, match="The Bayesian Logistic Regression model should have only one layer."):
        BayesianLogisticRegression.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list)


########################################################################################################################


# BaseBayesianNeuralNetworkMO


@settings(deadline=500)
@given(
    cost=st.just(1.0),
    n_features=st.integers(min_value=1, max_value=3),
    extra_n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_validate_models_n_features_raises_value_error(
    cost: float, n_features: int, extra_n_features: int, hidden_dim_list: list
) -> None:
    """
    Test that validate_models_n_features raises ValueError when models have different input dimensions.

    This test creates BayesianNeuralNetwork models with different input dimensions
    and verifies that the validation method correctly raises a ValueError.
    """
    # Create models with different input dimensions
    model_1 = BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=hidden_dim_list)
    model_2 = BayesianNeuralNetwork.cold_start(
        n_features=n_features + extra_n_features, hidden_dim_list=hidden_dim_list
    )

    # Attempt to create a multi-objective model with different input dimensions
    with pytest.raises(ValueError):
        BayesianNeuralNetworkMO(models=[model_1, model_2])

    # Test with cost control version as well
    with pytest.raises(ValueError):
        BayesianNeuralNetworkMOCC(models=[model_1, model_2], cost=cost)


########################################################################################################################


# BayesianNeuralNetworkMO


@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_objectives=st.integers(min_value=1, max_value=3),
)
def test_can_init_bayesian_neural_network_mo(n_features, hidden_dim_list, n_objectives):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or n_objectives <= 0:
        with pytest.raises((ValidationError, ValueError)):
            models = [BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list) for _ in range(n_objectives)]
            BayesianNeuralNetworkMO(models=models)
    else:
        models = [BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list) for _ in range(n_objectives)]
        bnn_mo = BayesianNeuralNetworkMO(models=models)
        assert len(bnn_mo.models) == n_objectives
        assert all(isinstance(model, BayesianNeuralNetwork) for model in bnn_mo.models)


@settings(deadline=500)
@given(
    n_samples=st.integers(min_value=1, max_value=1000),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    n_objectives=st.integers(min_value=1, max_value=3),
)
def test_bayesian_neural_network_mo_sample_proba(n_samples, n_features, hidden_dim_list, n_objectives):
    models = [BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list) for _ in range(n_objectives)]
    bnn_mo = BayesianNeuralNetworkMO(models=models)

    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    prob_weights = bnn_mo.sample_proba(context=context)

    assert len(prob_weights) == n_samples

    for prob_weight in prob_weights:
        assert len(prob_weight) == n_objectives
        for objective in range(n_objectives):
            prob, weight = prob_weight[objective]
            assert 0 <= prob <= 1


@given(
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(5),
    update_method=st.just("VI"),
    n_objectives=st.integers(min_value=1, max_value=2),
)
def test_bayesian_neural_network_mo_update(n_features, hidden_dim_list, n_samples, update_method, n_objectives):
    models = [
        BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list, update_method=update_method)
        for _ in range(n_objectives)
    ]
    bnn_mo = BayesianNeuralNetworkMO(models=models)

    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    rewards = [[np.random.randint(0, 2) for _ in range(n_objectives)] for _ in range(n_samples)]

    # Should not raise any exceptions
    bnn_mo.update(context=context, rewards=rewards)

    # Test with invalid rewards shape
    invalid_rewards = [[1] * (n_objectives + 1) for _ in range(n_samples)]
    with pytest.raises((ValueError, AttributeError)):
        bnn_mo.update(context=context, rewards=invalid_rewards)


########################################################################################################################


# BayesianNeuralNetworkMOCC


@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    n_objectives=st.integers(min_value=1, max_value=3),
    cost=st.floats(allow_nan=False, allow_infinity=False),
)
def test_can_init_bayesian_neural_network_mo_cc(n_features, hidden_dim_list, n_objectives, cost):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or n_objectives <= 0 or cost < 0:
        with pytest.raises((ValidationError, ValueError)):
            models = [BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list) for _ in range(n_objectives)]
            BayesianNeuralNetworkMOCC(models=models, cost=cost)
    else:
        models = [BayesianNeuralNetwork.cold_start(n_features, hidden_dim_list) for _ in range(n_objectives)]
        bnn_mo_cc = BayesianNeuralNetworkMOCC(models=models, cost=cost)
        assert len(bnn_mo_cc.models) == n_objectives
        assert bnn_mo_cc.cost == cost
        assert all(isinstance(model, BayesianNeuralNetwork) for model in bnn_mo_cc.models)
