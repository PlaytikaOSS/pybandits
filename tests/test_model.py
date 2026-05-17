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

from typing import Literal, Optional
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpyro.distributions import Normal as NumpyroNormal
from numpyro.distributions import StudentT as NumpyroStudentT
from numpyro.infer import Predictive
from pydantic import ValidationError

from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BaseLocationScaleArray,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
    Beta,
    BetaCC,
    BetaMO,
    BetaMOCC,
    CategoricalFeatureConfig,
    EarlyStopping,
    EmbeddingParams,
    FeaturesConfig,
    NormalArray,
    StudentTArray,
    UpdateMethods,
)

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


def test_beta_sample_proba(rng, n_samples=100):
    b = Beta(n_successes=1, n_failures=2)
    prob = b.sample_proba(n_samples=n_samples, rng=rng)
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


def test_calculate_proba_beta_mo(rng, n_samples=100):
    b = BetaMO(models=[Beta(), Beta()])
    b.sample_proba(n_samples=n_samples, rng=rng)


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


# NormalArray and StudentTArray
@settings(deadline=500)
@pytest.mark.parametrize("array_class", [NormalArray, StudentTArray])
@given(
    shape=st.one_of(
        st.integers(min_value=1, max_value=10),
        st.tuples(st.integers(min_value=1, max_value=10)),
        st.tuples(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10)),
    ),
    mu=st.floats(allow_nan=False, allow_infinity=False),
    sigma=st.floats(min_value=0, exclude_min=True, allow_nan=False, allow_infinity=False),
    nu=st.floats(min_value=0.001, allow_nan=False, allow_infinity=False),
)
def test_can_init_location_scale_array(array_class, shape, mu, sigma, nu):
    # Test with default args
    a = array_class.cold_start(shape=shape)
    expected_shape = shape if isinstance(shape, tuple) else (shape,)
    assert a.shape == expected_shape
    if array_class == NormalArray:
        assert not hasattr(a, "nu")
    else:
        assert hasattr(a, "nu")

    # Test with custom args
    if array_class == NormalArray:
        a = array_class.cold_start(shape=shape, mu=mu, sigma=sigma)
        assert a.mu == np.full(shape, mu).tolist()
        assert a.sigma == np.full(shape, sigma).tolist()
    else:
        a = array_class.cold_start(shape=shape, mu=mu, sigma=sigma, nu=nu)
        assert a.mu == np.full(shape, mu).tolist()
        assert a.sigma == np.full(shape, sigma).tolist()
        assert a.nu == np.full(shape, nu).tolist()


@settings(deadline=500)
@pytest.mark.parametrize("array_class,other_class", [(NormalArray, StudentTArray), (StudentTArray, NormalArray)])
@given(shape=st.one_of(st.integers(min_value=1, max_value=10), st.tuples(st.integers(min_value=1, max_value=10))))
def test_location_scale_array_inherits_from_base(array_class, other_class, shape):
    a = array_class.cold_start(shape=shape)
    assert isinstance(a, BaseLocationScaleArray)
    assert not isinstance(a, other_class)


@settings(deadline=500)
@pytest.mark.parametrize(
    "array_class,expected_numpyro_dist_class",
    [(NormalArray, NumpyroNormal), (StudentTArray, NumpyroStudentT)],
)
@given(
    shape=st.one_of(
        st.tuples(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5)),
    ),
    mu=st.floats(allow_nan=False, allow_infinity=False),
    sigma=st.floats(min_value=0.001, allow_nan=False, allow_infinity=False),
    nu=st.floats(min_value=0.001, allow_nan=False, allow_infinity=False),
)
def test_location_scale_array_to_numpyro_distribution(array_class, expected_numpyro_dist_class, shape, mu, sigma, nu):
    if array_class == NormalArray:
        a = array_class.cold_start(shape=shape, mu=mu, sigma=sigma)
    else:
        a = array_class.cold_start(shape=shape, mu=mu, sigma=sigma, nu=nu)

    dist = a.to_numpyro_distribution()
    assert isinstance(dist, expected_numpyro_dist_class)


########################################################################################################################


# BayesianNeuralNetwork
@settings(deadline=500)
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_can_init_bayesian_neural_network(n_features, hidden_dim_list):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list):
        with pytest.raises((ValidationError, ValueError)):
            fc = FeaturesConfig(n_features=n_features)
            model_params = BayesianNeuralNetwork.create_model_params(fc, hidden_dim_list)
            BayesianNeuralNetwork(model_params=model_params, feature_config=fc)
    else:
        fc = FeaturesConfig(n_features=n_features)
        model_params = BayesianNeuralNetwork.create_model_params(fc, hidden_dim_list)
        bnn = BayesianNeuralNetwork(model_params=model_params, feature_config=fc)
        assert bnn.model_params == model_params


@settings(deadline=500)
@given(
    n_samples=st.integers(min_value=1, max_value=1000),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_check_context_matrix(n_samples, n_features, hidden_dim_list):
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
    )

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

    with pytest.raises((AttributeError, ValueError)):
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
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=[],
        )
        with pytest.raises(AttributeError):
            bnn.check_context_matrix(context=invalid_context)


@settings(deadline=20000)
@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_samples=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
)
def test_bnn_sample_proba(
    rng, activation, use_residual_connections, use_layerwise_scaling, dist_type, n_samples, n_features, hidden_dim_list
):
    def sample_proba(context):
        prob_and_weighted_sum = bnn.sample_proba(context=np.array(context), rng=rng)
        prob, weighted_sum = zip(*prob_and_weighted_sum)
        assert len(prob) == len(weighted_sum) == n_samples  # return 1 sampled probability and ws per each sample
        assert all([0 <= p <= 1 for p in prob])  # probs must be in the interval [0, 1]

    bnn = BayesianNeuralNetwork.cold_start(
        n_features,
        hidden_dim_list,
        activation=activation,
        use_residual_connections=use_residual_connections,
        use_layerwise_scaling=use_layerwise_scaling,
        dist_type=dist_type,
    )

    # Verify distribution type
    expected_array = NormalArray if dist_type == "normal" else StudentTArray
    for layer_params in bnn.model_params.bnn_layer_params:
        assert isinstance(layer_params.weight, expected_array)
        assert isinstance(layer_params.bias, expected_array)

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    sample_proba(context=context)

    # check that the model is working with multi-sample prediction
    context = np.repeat(np.random.uniform(low=-1.0, high=1.0, size=(1, n_features)), n_samples, axis=0)
    assert type(context) is np.ndarray
    prob_and_weighted_sum = bnn.sample_proba(context=np.array(context), rng=rng)
    prob, weighted_sum = zip(*prob_and_weighted_sum)
    is_all_different = len(np.unique(weighted_sum)) == len(weighted_sum)
    assert is_all_different


@settings(deadline=None, max_examples=20)
@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(2),
    update_method=st.just("VI"),
    epochs=st.just(1),
)
def test_bnn_vi_update(
    activation,
    use_residual_connections,
    use_layerwise_scaling,
    dist_type,
    n_features,
    hidden_dim_list,
    n_samples,
    update_method,
    epochs,
):
    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            activation=activation,
            use_residual_connections=use_residual_connections,
            use_layerwise_scaling=use_layerwise_scaling,
            dist_type=dist_type,
            update_kwargs={"epochs": epochs},  # Use minimal iterations for faster tests
        )

        expected_array = NormalArray if dist_type == "normal" else StudentTArray
        dim_list = [n_features] + hidden_dim_list

        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_w_init = bnn.model_params.bnn_layer_params_init[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            layer_b_init = bnn.model_params.bnn_layer_params_init[layer_ind].bias.params
            for param in expected_array.model_fields.keys():
                assert np.all(layer_w[param] == layer_w_init[param])
                assert np.all(layer_b[param] == layer_b_init[param])

            assert isinstance(bnn.model_params.bnn_layer_params[layer_ind].weight, expected_array)
            assert isinstance(bnn.model_params.bnn_layer_params[layer_ind].bias, expected_array)

        bnn.update(context=context, rewards=rewards)

        # mu and sigma are updated:
        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight
            layer_w_init = bnn.model_params.bnn_layer_params_init[layer_ind].weight
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias
            layer_b_init = bnn.model_params.bnn_layer_params_init[layer_ind].bias
            # mu necessarily changes, but nor sigma, due to high prior value and low sample rate
            for param in ["mu"]:
                assert np.all(layer_w.params[param] != layer_w_init.params[param])
                assert np.all(layer_b.params[param] != layer_b_init.params[param])

            for param in expected_array.model_fields.keys():
                assert layer_w.params[param].tolist() == getattr(layer_w, param)
                assert layer_b.params[param].tolist() == getattr(layer_b, param)

        # Verify distribution type is preserved after update
        for layer_ind in range(len(dim_list)):
            assert isinstance(bnn.model_params.bnn_layer_params[layer_ind].weight, expected_array)
            assert isinstance(bnn.model_params.bnn_layer_params[layer_ind].bias, expected_array)

    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    # context is numpy array
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    update(context=context, rewards=rewards)

    # raise an error if len(context) != len(rewards)
    with pytest.raises(AttributeError):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            use_residual_connections=use_residual_connections,
            use_layerwise_scaling=use_layerwise_scaling,
            dist_type=dist_type,
            update_kwargs={"epochs": epochs},  # Use minimal iterations for faster tests
        )
        bnn.update(context=context, rewards=rewards[1:])


def _create_update_kwargs(
    batch_size: Optional[int] = None,
    optimizer_type: Optional[str] = None,
    lr: Optional[float] = None,
    early_stopping_diff: Optional[Literal["absolute", "relative"]] = None,
    early_stopping_tol: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    method: str = "advi",
) -> dict:
    """Create update_kwargs dictionary with vi sub-dict from individual parameters."""
    vi_kwargs: dict = {"method": method}
    if batch_size is not None:
        vi_kwargs["batch_size"] = batch_size
    if optimizer_type is not None:
        vi_kwargs["optimizer_type"] = optimizer_type
        if lr is not None:
            vi_kwargs["optimizer_kwargs"] = {"step_size": lr}

    if early_stopping_diff is not None or early_stopping_tol is not None or early_stopping_patience is not None:
        early_stopping_kwargs: dict = {}
        if early_stopping_diff is not None:
            early_stopping_kwargs["diff_type"] = early_stopping_diff
        if early_stopping_tol is not None:
            early_stopping_kwargs["tolerance"] = early_stopping_tol
        if early_stopping_patience is not None:
            early_stopping_kwargs["patience"] = early_stopping_patience
        if len(early_stopping_kwargs):
            vi_kwargs["early_stopping_kwargs"] = early_stopping_kwargs

    return vi_kwargs


@given(
    n_features=st.just(2),
    hidden_dim_list=st.just([1]),
    batch_size=st.sampled_from((None, 2, 4)),
    optimizer_type=st.sampled_from((None, "adam")),
    lr=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
    early_stopping_diff=st.sampled_from((None, "absolute", "relative")),
    early_stopping_tol=st.one_of(st.none(), st.floats(min_value=1e-6, max_value=1e-1)),
    early_stopping_patience=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    update_method=st.just("VI"),
    restore_best_svi_state=st.one_of(st.none(), st.booleans()),
)
def test_bnn_vi_update_parameters(
    n_features: int,
    hidden_dim_list: list[int],
    batch_size: Optional[int],
    optimizer_type: Optional[str],
    lr: Optional[float],
    early_stopping_diff: Optional[Literal["absolute", "relative"]],
    early_stopping_tol: Optional[float],
    early_stopping_patience: Optional[int],
    update_method: UpdateMethods,
    restore_best_svi_state: Optional[bool],
) -> None:
    """Test BNN VI update with various valid parameters."""
    update_kwargs = _create_update_kwargs(
        batch_size, optimizer_type, lr, early_stopping_diff, early_stopping_tol, early_stopping_patience
    )
    if restore_best_svi_state is not None:
        update_kwargs["restore_best_svi_state"] = restore_best_svi_state

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method=update_method,
        update_kwargs=update_kwargs,
    )

    model_fn = bnn.create_update_model()
    assert callable(model_fn)

    # Optimizer is always set for VI (built from defaults or user override)
    assert bnn._obj_optimizer is not None

    if "early_stopping_kwargs" in update_kwargs:
        assert bnn._get_early_stopping_callback() is not None
        assert isinstance(bnn._get_early_stopping_callback(), EarlyStopping)
    else:
        assert bnn._get_early_stopping_callback() is None

    # restore_best_svi_state defaults to True; explicit value is preserved
    expected = restore_best_svi_state if restore_best_svi_state is not None else True
    assert bnn._update_kwargs.get("restore_best_svi_state") is expected


@given(
    n_features=st.just(2),
    hidden_dim_list=st.just([1]),
    n_samples=st.just(5),
    update_method=st.just("MCMC"),
)
def test_bnn_mcmc_update_parameters(
    n_features: int,
    hidden_dim_list: list[int],
    n_samples: int,
    update_method: UpdateMethods,
) -> None:
    """Test BNN MCMC update with valid parameters (no batch_size, optimizer, or early_stopping)."""

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method=update_method,
        update_kwargs={},
    )

    model_fn = bnn.create_update_model()
    assert callable(model_fn)


@given(
    n_features=st.just(2),
    hidden_dim_list=st.just([1]),
    batch_size=st.sampled_from((2, 4)),
    optimizer_type=st.sampled_from((None, "adam")),
    lr=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
    early_stopping_diff=st.sampled_from(("absolute", "relative")),
    early_stopping_tol=st.floats(min_value=1e-6, max_value=1e-1),
    early_stopping_patience=st.integers(min_value=1, max_value=10),
    update_method=st.just("MCMC"),
)
def test_bnn_mcmc_update_parameters_failures(
    n_features: int,
    hidden_dim_list: list[int],
    batch_size: int,
    optimizer_type: Optional[str],
    lr: Optional[float],
    early_stopping_diff: str,
    early_stopping_tol: float,
    early_stopping_patience: int,
    update_method: UpdateMethods,
) -> None:
    """Test that MCMC update raises ValueError when invalid parameters are provided."""
    # Test with batch_size
    update_kwargs_with_batch = _create_update_kwargs(batch_size)
    with pytest.raises(ValueError):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs=update_kwargs_with_batch,
        )

    # Test with optimizer_type
    if optimizer_type is not None:
        update_kwargs_with_optimizer = _create_update_kwargs(None, optimizer_type, lr, None, None, None)
        with pytest.raises(ValueError):
            BayesianNeuralNetwork.cold_start(
                n_features=n_features,
                hidden_dim_list=hidden_dim_list,
                update_method="MCMC",
                update_kwargs=update_kwargs_with_optimizer,
            )

    # Test with early_stopping_kwargs
    update_kwargs_with_early_stopping = _create_update_kwargs(
        early_stopping_diff=early_stopping_diff,
        early_stopping_tol=early_stopping_tol,
        early_stopping_patience=early_stopping_patience,
    )
    with pytest.raises(ValueError):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs=update_kwargs_with_early_stopping,
        )


@given(
    n_features=st.just(2),
    hidden_dim_list=st.just([1]),
    batch_size=st.sampled_from((None, 2, 4)),
    optimizer_type=st.just("dummy_optimizer"),
    update_method=st.just("VI"),
)
def test_bnn_vi_update_parameters_dummy_optimizer_failure(
    n_features: int,
    hidden_dim_list: list[int],
    batch_size: Optional[int],
    optimizer_type: Optional[str],
    update_method: UpdateMethods,
) -> None:
    """Test that dummy_optimizer raises ValueError for BNN VI update."""
    update_kwargs = _create_update_kwargs(batch_size, optimizer_type)

    with pytest.raises(ValueError):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs=update_kwargs,
        )


@pytest.mark.parametrize(
    "optimizer_type,optimizer_kwargs",
    [
        ("adam", {"invalid_param": 123, "step_size": 0.01}),
        ("sgd", {"invalid_param": 123}),
    ],
)
def test_invalid_optimizer_kwargs(
    optimizer_type: str, optimizer_kwargs: dict, n_features: int = 2, update_method: UpdateMethods = "VI"
) -> None:
    """Test that invalid optimizer kwargs raise TypeError or ValueError."""

    with pytest.raises((TypeError, ValueError), match="Invalid optimizer kwargs"):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_method=update_method,
            update_kwargs={
                "optimizer_type": optimizer_type,
                "optimizer_kwargs": optimizer_kwargs,
            },
        )


@pytest.mark.parametrize(
    "early_stopping_kwargs",
    [
        {"invalid_param": 123, "tolerance": 1e-3},
        {"diff": "invalid", "tolerance": 1e-3},
    ],
)
def test_invalid_early_stopping_kwargs(
    early_stopping_kwargs: dict, n_features: int = 2, update_method: UpdateMethods = "VI"
) -> None:
    """Test that invalid early stopping kwargs raise TypeError or ValueError."""

    with pytest.raises((TypeError, ValueError, KeyError), match="Invalid early stopping kwargs"):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_method=update_method,
            update_kwargs={
                "early_stopping_kwargs": early_stopping_kwargs,
            },
        )


@given(
    n_features=st.just(2),
    update_method=st.just("VI"),
    epochs=st.integers(min_value=1, max_value=100),
    num_steps=st.integers(min_value=1, max_value=100),
)
def test_epochs_and_num_steps_warns(n_features: int, update_method: UpdateMethods, epochs: int, num_steps: int) -> None:
    """Test that specifying both 'epochs' and 'num_steps' raises a UserWarning (epochs takes precedence)."""
    with pytest.warns(UserWarning, match="'epochs' takes precedence"):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_method=update_method,
            update_kwargs={"epochs": epochs, "num_steps": num_steps},
        )


@pytest.mark.parametrize("n_features", [1, 2])
def test_bnn_mcmc_update(
    n_features,
    hidden_dim_list=(1,),
    n_samples=3,
    update_method="MCMC",
    update_kwargs={"num_warmup": 2, "num_samples": 2, "num_chains": 1},
):
    hidden_dim_list = list(hidden_dim_list)

    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs=update_kwargs,
        )
        dim_list = [n_features] + hidden_dim_list
        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_w_init = bnn.model_params.bnn_layer_params_init[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            layer_b_init = bnn.model_params.bnn_layer_params_init[layer_ind].bias.params
            for param in ["mu", "sigma", "nu"]:
                assert np.all(layer_w[param] == layer_w_init[param])
                assert np.all(layer_b[param] == layer_b_init[param])

        bnn.update(context=context, rewards=rewards)

        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight
            layer_w_init = bnn.model_params.bnn_layer_params_init[layer_ind].weight
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias
            layer_b_init = bnn.model_params.bnn_layer_params_init[layer_ind].bias
            for param in ["mu", "sigma"]:
                assert np.all(layer_w.params[param] != layer_w_init.params[param])
                assert np.all(layer_b.params[param] != layer_b_init.params[param])

            for param in ["mu", "sigma", "nu"]:
                assert layer_w.params[param].tolist() == getattr(layer_w, param)
                assert layer_b.params[param].tolist() == getattr(layer_b, param)

    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    update(context=context, rewards=rewards)

    # raise an error if len(context) != len(rewards)
    with pytest.raises(AttributeError):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method=update_method,
            update_kwargs=update_kwargs,
        )
        bnn.update(context=context, rewards=rewards[1:])


@pytest.mark.parametrize("n_features", [1, 2])
def test_bnn_fullrank_advi_update(n_features, hidden_dim_list=(1,), n_samples=5, method="fullrank_advi", num_steps=1):
    hidden_dim_list = list(hidden_dim_list)

    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_method="VI",
            update_kwargs={"method": method, "num_steps": num_steps},
        )
        dim_list = [n_features] + hidden_dim_list
        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight.params
            layer_w_init = bnn.model_params.bnn_layer_params_init[layer_ind].weight.params
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias.params
            layer_b_init = bnn.model_params.bnn_layer_params_init[layer_ind].bias.params
            for param in ["mu", "sigma", "nu"]:
                assert np.all(layer_w[param] == layer_w_init[param])
                assert np.all(layer_b[param] == layer_b_init[param])

        bnn.update(context=context, rewards=rewards)

        for layer_ind in range(len(dim_list)):
            layer_w = bnn.model_params.bnn_layer_params[layer_ind].weight
            layer_w_init = bnn.model_params.bnn_layer_params_init[layer_ind].weight
            layer_b = bnn.model_params.bnn_layer_params[layer_ind].bias
            layer_b_init = bnn.model_params.bnn_layer_params_init[layer_ind].bias
            for param in ["mu", "sigma"]:
                assert np.all(layer_w.params[param] != layer_w_init.params[param])
                assert np.all(layer_b.params[param] != layer_b_init.params[param])

            for param in ["mu", "sigma", "nu"]:
                assert layer_w.params[param].tolist() == getattr(layer_w, param)
                assert layer_b.params[param].tolist() == getattr(layer_b, param)

    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    update(context=context, rewards=rewards)


########################################################################################################################


# BayesianNeuralNetwork - SVI NaN protection and restore_best_svi_state


@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(2),
    update_method=st.just("VI"),
)
def test_bnn_svi_nan_loss_raises_error(
    n_features: int, hidden_dim_list: list[int], n_samples: int, update_method: UpdateMethods
) -> None:
    """Test that a NaN loss during SVI training raises a ValueError immediately."""

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method=update_method,
    )

    # Patch np.mean to return NaN on the first epoch to simulate divergence.
    original_mean = np.mean
    call_count = {"n": 0}

    def nan_mean(a, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:  # call 1 is avg_sigma in _build_svi_guide_init; call 2 is epoch loss
            return float("nan")
        return original_mean(a, *args, **kwargs)

    context = np.random.uniform(size=(n_samples, n_features))
    rewards = _make_random_rewards(n_samples)

    with patch("pybandits.model.np.mean", nan_mean):
        with pytest.raises(ValueError, match="SVI training diverged.*NaN"):
            bnn.update(context=context, rewards=rewards)


########################################################################################################################


# BayesianNeuralNetwork - Activation Functions


@pytest.mark.parametrize("invalid_activation", ["invalid", "elu", "selu", "leaky_relu", ""])
@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("hidden_dim_list", [[], [3], [2, 3]])
def test_bnn_invalid_activation_raises_error(invalid_activation, n_features, hidden_dim_list):
    """Test that invalid activation functions raise ValueError."""
    with pytest.raises(ValidationError) as exc_info:
        BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=hidden_dim_list, activation=invalid_activation
        )
    # Check that the error message contains information about invalid activation
    error_str = str(exc_info.value)
    assert "Invalid activation function" in error_str or "activation" in error_str.lower()


def test_bnn_jax_and_numpy_activation_keys_match():
    """Test that the keys of _jax_activations and _numpy_activations dictionaries match."""
    jax_keys = set(BayesianNeuralNetwork._jax_activations.keys())
    numpy_keys = set(BayesianNeuralNetwork._numpy_activations.keys())

    assert jax_keys == numpy_keys, (
        f"Keys mismatch between _jax_activations and _numpy_activations. JAX keys: {jax_keys}, NumPy keys: {numpy_keys}"
    )


@settings(deadline=500)
@pytest.mark.parametrize("use_layerwise_scaling", [True, False])
@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    sigma=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_bnn_layerwise_scaling(use_layerwise_scaling, n_features, hidden_dim_list, sigma):
    """Test that use_layerwise_scaling correctly scales weight sigma by sqrt(input_dim)."""
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        use_layerwise_scaling=use_layerwise_scaling,
        dist_params_init={"sigma": sigma},
    )

    dim_list = [n_features] + hidden_dim_list
    dim_list.append(1)

    for layer_ind, layer_params in enumerate(bnn.model_params.bnn_layer_params):
        input_dim = dim_list[layer_ind]
        weight_sigma = np.array(layer_params.weight.sigma)
        bias_sigma = np.array(layer_params.bias.sigma)

        if use_layerwise_scaling:
            expected_weight_sigma = sigma / np.sqrt(input_dim)
        else:
            expected_weight_sigma = sigma

        # Check that weight sigma is scaled correctly
        assert np.allclose(weight_sigma, expected_weight_sigma), (
            f"Layer {layer_ind}: weight sigma should be {expected_weight_sigma} but got {weight_sigma.mean()}"
        )

        # Check that bias sigma is not scaled
        assert np.allclose(bias_sigma, sigma), (
            f"Layer {layer_ind}: bias sigma should be {sigma} but got {bias_sigma.mean()}"
        )


########################################################################################################################


# BayesianNeuralNetworkCC
@settings(deadline=500)
@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    cost=st.floats(allow_nan=False, allow_infinity=False),
)
def test_can_init_bayesian_neural_network_cc(
    activation, use_residual_connections, use_layerwise_scaling, n_features, hidden_dim_list, cost
):
    # at least one beta must be specified
    dim_list = [n_features] + hidden_dim_list
    fc = FeaturesConfig(n_features=n_features)
    if any(layer_dim <= 0 for layer_dim in dim_list) or (cost < 0):
        with pytest.raises((ValidationError, ValueError)):
            model_params = BayesianNeuralNetwork.create_model_params(
                fc, hidden_dim_list, use_layerwise_scaling=use_layerwise_scaling
            )
            bnn = BayesianNeuralNetworkCC(
                model_params=model_params,
                cost=cost,
                activation=activation,
                use_residual_connections=use_residual_connections,
                feature_config=fc,
            )
    else:
        model_params = BayesianNeuralNetwork.create_model_params(
            fc, hidden_dim_list, use_layerwise_scaling=use_layerwise_scaling
        )
        bnn = BayesianNeuralNetworkCC(
            model_params=model_params,
            cost=cost,
            activation=activation,
            use_residual_connections=use_residual_connections,
            feature_config=fc,
        )
        assert bnn.model_params == model_params
        assert bnn.activation == activation
        assert bnn.use_residual_connections == use_residual_connections


@settings(deadline=500)
@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    cost=st.floats(allow_nan=False, allow_infinity=False),
)
def test_create_default_instance_bayesian_neural_network_cc(
    rng, activation, use_residual_connections, use_layerwise_scaling, n_features, hidden_dim_list, cost
):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or (cost < 0):
        with pytest.raises((ValidationError, ValueError)):
            BayesianNeuralNetworkCC.cold_start(
                n_features=n_features,
                hidden_dim_list=hidden_dim_list,
                cost=cost,
                activation=activation,
                use_residual_connections=use_residual_connections,
                use_layerwise_scaling=use_layerwise_scaling,
            )
    else:
        bnn_cold_start = BayesianNeuralNetworkCC.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            cost=cost,
            activation=activation,
            use_residual_connections=use_residual_connections,
            use_layerwise_scaling=use_layerwise_scaling,
        )
        fc = FeaturesConfig(n_features=n_features)
        model_params = BayesianNeuralNetwork.create_model_params(
            fc, hidden_dim_list=hidden_dim_list, use_layerwise_scaling=use_layerwise_scaling
        )
        bnn_init = BayesianNeuralNetworkCC(
            model_params=model_params,
            cost=cost,
            activation=activation,
            use_residual_connections=use_residual_connections,
            feature_config=fc,
        )
        assert bnn_cold_start == bnn_init
        assert bnn_cold_start.activation == activation
        assert bnn_cold_start.use_residual_connections == use_residual_connections

        # Test sample_proba works
        context = np.random.uniform(low=-1.0, high=1.0, size=(5, n_features))
        prob_and_weighted_sum = bnn_cold_start.sample_proba(context=context, rng=rng)
        prob, weighted_sum = zip(*prob_and_weighted_sum)
        assert len(prob) == 5
        assert all([0 <= p <= 1 for p in prob])


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
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    n_samples=st.integers(min_value=1, max_value=1000),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    n_objectives=st.integers(min_value=1, max_value=3),
)
def test_bayesian_neural_network_mo_sample_proba(
    rng,
    activation,
    use_residual_connections,
    use_layerwise_scaling,
    n_samples,
    n_features,
    hidden_dim_list,
    n_objectives,
):
    models = [
        BayesianNeuralNetwork.cold_start(
            n_features,
            hidden_dim_list,
            activation=activation,
            use_residual_connections=use_residual_connections,
            use_layerwise_scaling=use_layerwise_scaling,
        )
        for _ in range(n_objectives)
    ]
    bnn_mo = BayesianNeuralNetworkMO(models=models)
    # Verify all models have the same activation and residual connections setting
    for model in bnn_mo.models:
        assert model.activation == activation
        assert model.use_residual_connections == use_residual_connections

    context = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    prob_weights = bnn_mo.sample_proba(context=context, rng=rng)

    assert len(prob_weights) == n_samples

    for prob_weight in prob_weights:
        assert len(prob_weight) == n_objectives
        for objective in range(n_objectives):
            prob, weight = prob_weight[objective]
            assert 0 <= prob <= 1


@settings(deadline=None, max_examples=25)
@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(3),
    update_method=st.just("VI"),
    n_objectives=st.integers(min_value=1, max_value=2),
    epochs=st.just(1),
)
def test_bayesian_neural_network_mo_update(
    activation, n_features, hidden_dim_list, n_samples, update_method, n_objectives, epochs
):
    models = [
        BayesianNeuralNetwork.cold_start(
            n_features,
            hidden_dim_list,
            update_method=update_method,
            activation=activation,
            update_kwargs={"epochs": epochs},  # Use minimal iterations for faster tests
        )
        for _ in range(n_objectives)
    ]
    bnn_mo = BayesianNeuralNetworkMO(models=models)
    # Verify all models have the same activation
    for model in bnn_mo.models:
        assert model.activation == activation

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
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    n_objectives=st.integers(min_value=1, max_value=3),
    cost=st.floats(allow_nan=False, allow_infinity=False),
)
def test_can_init_bayesian_neural_network_mo_cc(
    activation, use_residual_connections, use_layerwise_scaling, n_features, hidden_dim_list, n_objectives, cost
):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or n_objectives <= 0 or cost < 0:
        with pytest.raises((ValidationError, ValueError)):
            models = [
                BayesianNeuralNetwork.cold_start(
                    n_features,
                    hidden_dim_list,
                    activation=activation,
                    use_residual_connections=use_residual_connections,
                    use_layerwise_scaling=use_layerwise_scaling,
                )
                for _ in range(n_objectives)
            ]
            BayesianNeuralNetworkMOCC(models=models, cost=cost)
    else:
        models = [
            BayesianNeuralNetwork.cold_start(
                n_features,
                hidden_dim_list,
                activation=activation,
                use_residual_connections=use_residual_connections,
                use_layerwise_scaling=use_layerwise_scaling,
            )
            for _ in range(n_objectives)
        ]
        bnn_mo_cc = BayesianNeuralNetworkMOCC(models=models, cost=cost)
        assert len(bnn_mo_cc.models) == n_objectives
        assert bnn_mo_cc.cost == cost
        assert all(isinstance(model, BayesianNeuralNetwork) for model in bnn_mo_cc.models)
        # Verify all models have the same activation and residual connections setting
        for model in bnn_mo_cc.models:
            assert model.activation == activation
            assert model.use_residual_connections == use_residual_connections


########################################################################################################################
# Embedding layer tests
########################################################################################################################


# ---------------------------------------------------------------------------
# CategoricalFeatureConfig
# ---------------------------------------------------------------------------


@settings(deadline=500)
@given(
    column_index=st.integers(),
    cardinality=st.integers(),
    embedding_dim=st.integers(),
)
def test_can_init_categorical_feature_config(column_index, cardinality, embedding_dim):
    if column_index < 0 or cardinality <= 0 or embedding_dim <= 0:
        with pytest.raises(ValidationError):
            CategoricalFeatureConfig(column_index=column_index, cardinality=cardinality, embedding_dim=embedding_dim)
    else:
        cfg = CategoricalFeatureConfig(column_index=column_index, cardinality=cardinality, embedding_dim=embedding_dim)
        assert cfg.column_index == column_index
        assert cfg.cardinality == cardinality
        assert cfg.embedding_dim == embedding_dim

    if column_index >= 0 and cardinality > 0:
        cfg = CategoricalFeatureConfig(column_index=column_index, cardinality=cardinality, embedding_dim=8)
        assert cfg.embedding_dim == 8


# ---------------------------------------------------------------------------
# FeaturesConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_features,cat_configs,expected_n_numerical,expected_indices,expected_output_dim",
    [
        (4, [(2, 3, 4), (3, 2, 6)], 2, [0, 1], 2 + 4 + 6),
        (5, [(2, 3, 4), (4, 2, 4)], 3, [0, 1, 3], 3 + 4 + 4),
    ],
)
def test_feature_config_properties(
    n_features, cat_configs, expected_n_numerical, expected_indices, expected_output_dim
):
    """Derived properties are consistent: n_numerical, numerical_indices, total_output_dim."""
    fc = FeaturesConfig(
        n_features=n_features,
        categorical_features_configs=[
            CategoricalFeatureConfig(column_index=col, cardinality=card, embedding_dim=emb)
            for col, card, emb in cat_configs
        ],
    )
    assert fc.n_features == n_features
    assert fc.n_numerical == expected_n_numerical
    assert fc.numerical_indices == expected_indices
    assert fc.total_output_dim == expected_output_dim


@pytest.mark.parametrize("n_features,n_numerical,total_output_dim", [(0, 0, 0)])
def test_feature_config_defaults(n_features, n_numerical, total_output_dim):
    fc = FeaturesConfig()
    assert fc.n_numerical == n_numerical
    assert fc.categorical_features_configs == []
    assert fc.n_features == n_features
    assert fc.total_output_dim == total_output_dim


@pytest.mark.parametrize(
    "categorical_configs,error_match",
    [
        (
            [
                CategoricalFeatureConfig(column_index=1, cardinality=3, embedding_dim=4),
                CategoricalFeatureConfig(column_index=1, cardinality=2, embedding_dim=4),
            ],
            "Duplicate",
        ),
        (
            [CategoricalFeatureConfig(column_index=5, cardinality=3, embedding_dim=4)],
            "out of range",
        ),
    ],
)
@pytest.mark.parametrize("n_features", [3])
def test_feature_config_invalid_column_indices(categorical_configs, error_match, n_features):
    with pytest.raises(ValidationError, match=error_match):
        FeaturesConfig(n_features=n_features, categorical_features_configs=categorical_configs)


# ---------------------------------------------------------------------------
# EmbeddingParams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist_type,dist_class", [("studentt", StudentTArray), ("normal", NormalArray)])
@pytest.mark.parametrize(
    "n_features,cat_configs,expected_shapes",
    [(2, [(0, 3, 4), (1, 2, 6)], [(3, 4), (2, 6)])],
)
def test_embedding_params_cold_start_shapes(dist_type, dist_class, n_features, cat_configs, expected_shapes):
    fc = FeaturesConfig(
        n_features=n_features,
        categorical_features_configs=[
            CategoricalFeatureConfig(column_index=col, cardinality=card, embedding_dim=emb)
            for col, card, emb in cat_configs
        ],
    )
    ep = EmbeddingParams.cold_start(fc, dist_class=dist_class)
    assert len(ep.embeddings) == len(cat_configs)
    for emb, expected in zip(ep.embeddings, expected_shapes):
        assert emb.shape == expected
    assert isinstance(ep.embeddings[0], dist_class)


@pytest.mark.parametrize("n_features,cardinality,embedding_dim", [(1, 2, 2)])
def test_embedding_params_init_is_frozen_copy(n_features, cardinality, embedding_dim):
    fc = FeaturesConfig(
        n_features=n_features,
        categorical_features_configs=[
            CategoricalFeatureConfig(column_index=0, cardinality=cardinality, embedding_dim=embedding_dim)
        ],
    )
    ep = EmbeddingParams.cold_start(fc)
    original_mu = [row[:] for row in ep.embeddings[0].mu]

    # Mutate current embeddings
    new_emb = ep.embeddings[0].with_dist_parameters(mu=[[1.0, 2.0], [3.0, 4.0]], sigma=[[0.1, 0.1], [0.1, 0.1]])
    ep.embeddings[0] = new_emb
    assert ep.embeddings[0].mu != original_mu

    # embeddings_init should still have original values
    assert ep.embeddings_init[0].mu == original_mu


# ---------------------------------------------------------------------------
# BNN with feature_config: cold_start and check_context_matrix
# ---------------------------------------------------------------------------


def _make_bnn_with_categoricals(n_features=2, categorical_features=None, dist_type="studentt", hidden_dim_list=None):
    return BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        categorical_features=categorical_features or {1: 3},
        hidden_dim_list=hidden_dim_list or [8],
        dist_type=dist_type,
        update_kwargs={"epochs": 1},
    )


def test_cold_start_requires_n_features():
    with pytest.raises((TypeError, Exception)):
        BayesianNeuralNetwork.cold_start()


def _make_categorical_context(n_samples: int, n_features: int, categorical_features: dict) -> np.ndarray:
    """Generate random context with valid numerical and categorical columns."""
    context = np.random.uniform(-1.0, 1.0, size=(n_samples, n_features))
    for col_idx, cardinality in categorical_features.items():
        context[:, col_idx] = np.random.randint(0, cardinality, size=n_samples)
    return context


def _make_random_rewards(n_samples: int) -> list:
    return np.random.choice([0, 1], size=n_samples).tolist()


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=2, max_value=5),
    cardinality=st.integers(min_value=2, max_value=8),
    hidden_dim_list=st.lists(st.integers(min_value=2, max_value=8), min_size=1, max_size=2),
)
def test_cold_start_with_feature_config(n_features, cardinality, hidden_dim_list):
    """cold_start creates correct embedding params and first-layer weight shape."""
    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(
        n_features=n_features, categorical_features=categorical_features, hidden_dim_list=hidden_dim_list
    )
    embedding_dim = bnn.model_params.embedding_params.embeddings[0].shape[1]
    n_numerical = n_features - len(categorical_features)
    expected_input_dim = n_numerical + embedding_dim

    assert bnn.model_params.embedding_params is not None
    assert len(bnn.model_params.embedding_params.embeddings) == len(categorical_features)
    assert bnn.model_params.embedding_params.embeddings[0].shape[0] == cardinality
    assert bnn.model_params.bnn_layer_params[0].weight.shape[0] == expected_input_dim


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=2, max_value=5),
    cardinality=st.integers(min_value=2, max_value=8),
    n_samples=st.integers(min_value=2, max_value=10),
)
def test_check_context_matrix_with_categorical(n_features, cardinality, n_samples):
    """check_context_matrix validates column count and categorical range for feature_config models."""
    cat_col = n_features - 1
    bnn = _make_bnn_with_categoricals(n_features=n_features, categorical_features={cat_col: cardinality})
    # valid context
    context = _make_categorical_context(n_samples, n_features, {cat_col: cardinality})
    bnn.check_context_matrix(context)
    # too few columns
    with pytest.raises(AttributeError, match="Shape mismatch"):
        bnn.check_context_matrix(np.random.uniform(size=(n_samples, 1)))
    # out-of-range category
    bad_context = _make_categorical_context(1, n_features, {cat_col: cardinality})
    bad_context[0, cat_col] = cardinality + 2
    with pytest.raises(ValueError, match="out of range"):
        bnn.check_context_matrix(bad_context)


# ---------------------------------------------------------------------------
# _prepare_context_arrays
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=5)
@given(
    n_samples=st.integers(min_value=2, max_value=10),
    n_features=st.integers(min_value=2, max_value=5),
    cardinality=st.integers(min_value=2, max_value=8),
)
def test_prepare_context_arrays_splits_correctly(n_samples, n_features, cardinality):
    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(n_features=n_features, categorical_features=categorical_features)
    context = _make_categorical_context(n_samples, n_features, categorical_features)
    num_arr, cat_idx = bnn._prepare_context_arrays(context)

    n_numerical = n_features - 1
    assert num_arr.shape == (n_samples, n_numerical)
    np.testing.assert_allclose(num_arr, context[:, :n_numerical])
    assert cat_idx[0].tolist() == context[:, cat_col].astype(np.int32).tolist()


# ---------------------------------------------------------------------------
# sample_proba with feature_config
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=5)
@given(
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_samples=st.integers(min_value=1, max_value=10),
    n_features=st.integers(min_value=2, max_value=4),
    cardinality=st.integers(min_value=2, max_value=6),
)
def test_sample_proba_with_feature_config(rng, dist_type, n_samples, n_features, cardinality):
    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(
        n_features=n_features, categorical_features=categorical_features, dist_type=dist_type
    )
    context = _make_categorical_context(n_samples, n_features, categorical_features)
    results = bnn.sample_proba(context=context, rng=rng)

    assert len(results) == n_samples
    for prob, ws in results:
        assert 0.0 <= prob <= 1.0
        assert isinstance(ws, float)


# ---------------------------------------------------------------------------
# create_update_model with feature_config
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=2, max_value=4),
    cardinality=st.integers(min_value=2, max_value=6),
    n_samples=st.integers(min_value=2, max_value=10),
)
def test_create_update_model_contains_embedding_variable(n_features, cardinality, n_samples):
    """Verify the NumPyro model function samples embedding variables."""
    import jax.numpy as jnp

    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(n_features=n_features, categorical_features=categorical_features)
    model_fn = bnn.create_update_model()
    context = jnp.array(_make_categorical_context(n_samples, n_features, categorical_features), dtype=jnp.float32)
    rewards = jnp.array(_make_random_rewards(n_samples), dtype=jnp.int32)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model_fn, rng_seed=0)).get_trace(context, rewards)
    assert any("embedding_0" in name for name in trace.keys())


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=2, max_value=4),
    cardinality=st.integers(min_value=2, max_value=6),
    n_samples=st.integers(min_value=3, max_value=10),
    batch_size=st.integers(min_value=2, max_value=3),
)
def test_create_update_model_minibatch_with_categoricals(n_features, cardinality, n_samples, batch_size):
    """Verify the NumPyro model function with minibatching samples embedding variables."""
    import jax.numpy as jnp

    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(n_features=n_features, categorical_features=categorical_features)
    model_fn = bnn.create_update_model(batch_size=batch_size)
    context = jnp.array(_make_categorical_context(n_samples, n_features, categorical_features), dtype=jnp.float32)
    rewards = jnp.array(_make_random_rewards(n_samples), dtype=jnp.int32)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model_fn, rng_seed=0)).get_trace(context, rewards)
    assert any("embedding_0" in name for name in trace.keys())


# ---------------------------------------------------------------------------
# _reset with embeddings
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=2, max_value=4),
    cardinality=st.integers(min_value=2, max_value=6),
)
def test_reset_restores_embedding_params(n_features, cardinality):
    cat_col = n_features - 1
    bnn = _make_bnn_with_categoricals(n_features=n_features, categorical_features={cat_col: cardinality})
    emb = bnn.model_params.embedding_params.embeddings[0]
    original_mu = [row[:] for row in emb.mu]
    embedding_dim = emb.shape[1]

    # Manually mutate embeddings
    new_emb = emb.with_dist_parameters(
        mu=[[1.0] * embedding_dim] * cardinality, sigma=[[0.1] * embedding_dim] * cardinality
    )
    bnn.model_params.embedding_params.embeddings[0] = new_emb
    assert bnn.model_params.embedding_params.embeddings[0].mu != original_mu

    # Reset should restore
    bnn._reset()
    assert bnn.model_params.embedding_params.embeddings[0].mu == original_mu


# ---------------------------------------------------------------------------
# End-to-end VI update with categorical embeddings
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=5)
@given(
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_features=st.integers(min_value=2, max_value=4),
    cardinality=st.integers(min_value=2, max_value=6),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(2),
    update_method=st.just("VI"),
    epochs=st.just(1),
)
def test_bnn_vi_update_with_categorical_features_updates_embeddings(
    dist_type, n_features, cardinality, hidden_dim_list, n_samples, update_method, epochs
):
    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        categorical_features=categorical_features,
        hidden_dim_list=hidden_dim_list,
        dist_type=dist_type,
        update_method=update_method,
        update_kwargs={"epochs": epochs},
    )
    context = _make_categorical_context(n_samples, n_features, categorical_features)
    rewards = _make_random_rewards(n_samples)

    init_mu = [row[:] for row in bnn.model_params.embedding_params.embeddings[0].mu]
    bnn._update(context=context, rewards=rewards)
    updated_mu = bnn.model_params.embedding_params.embeddings[0].mu

    assert updated_mu != init_mu

    # Reset should restore initial embeddings
    bnn._reset()
    assert bnn.model_params.embedding_params.embeddings[0].mu == init_mu


@pytest.mark.parametrize(
    "update_method, update_kwargs",
    [("VI", {"num_steps": 2}), ("MCMC", {"num_warmup": 2, "num_samples": 2})],
)
def test_bnn_sample_proba_and_update_both_use_forward_layers(
    rng, update_method: str, update_kwargs: dict, n_features: int = 1, n_samples: int = 1, ref: int = 1
) -> None:
    """Verify that both sample_proba and update call _forward_layers."""
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        update_method=update_method,
        update_kwargs=update_kwargs,
    )
    context = np.random.rand(n_samples, n_features).astype(np.float32)
    rewards = _make_random_rewards(n_samples)

    original_forward_layers = BaseBayesianNeuralNetwork._forward_layers
    call_count = 0

    def tracking_forward_layers(self_inner, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward_layers(self_inner, *args, **kwargs)

    # Check sample_proba calls _forward_layers
    call_count = 0
    with patch.object(BaseBayesianNeuralNetwork, "_forward_layers", tracking_forward_layers):
        bnn.sample_proba(context=context, rng=rng)
    assert call_count == ref, "sample_proba must call _forward_layers"

    # Check update calls _forward_layers
    call_count = 0
    with patch.object(BaseBayesianNeuralNetwork, "_forward_layers", tracking_forward_layers):
        bnn.update(context=context, rewards=rewards)
    assert call_count >= ref, "update must call _forward_layers"


########################################################################################################################


# ParameterizedScaleAutoNormal / ADVI guide consistency


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=1),
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_samples=st.integers(min_value=5, max_value=10),
    sigma_init=st.floats(min_value=0.1, max_value=1.0),
    nu=st.just(5.0),
    epochs=st.just(2),
    n_predictive_samples=st.just(4000),
    n_sigma_tolerance=st.just(5),
)
def test_advi_extracted_params_match_predictive_moments(
    n_features, hidden_dim_list, dist_type, n_samples, sigma_init, nu, epochs, n_predictive_samples, n_sigma_tolerance
):
    """Stored (mu, sigma) from _extract_advi_params must match guide posterior sample moments.

    AutoNormal draws each site from Normal(loc, scale), so posterior predictive
    sample mean/std must recover the extracted loc/scale up to sampling noise.
    sigma_init is bounded so SE = sigma/sqrt(n_predictive_samples) stays far below tolerance.
    """
    dist_params_init = {"sigma": sigma_init} if dist_type == "normal" else {"sigma": sigma_init, "nu": nu}
    context = np.random.default_rng(0).standard_normal((n_samples, n_features)).astype(np.float32)
    rewards = np.random.choice([0, 1], size=n_samples).tolist()

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method="VI",
        update_kwargs={"epochs": epochs, "method": "advi"},
        dist_type=dist_type,
        dist_params_init=dist_params_init,
    )

    x_jnp = jnp.array(context)
    y_jnp = jnp.array(rewards, dtype=jnp.int32)
    _, guide, params = bnn._run_svi_training_loop(x_jnp, y_jnp, n_samples)
    site_mu, site_sigma = bnn._extract_advi_params(params)

    samples = Predictive(guide, params=params, num_samples=n_predictive_samples)(jax.random.PRNGKey(0), x_jnp, y_jnp)

    # tolerances are n_sigma_tolerance * SE; atol_mu is per-site since ADVI can widen sigma above sigma_init
    rtol_sigma = n_sigma_tolerance * np.sqrt(2.0 / n_predictive_samples)

    for layer_ind in range(len(bnn.model_params.bnn_layer_params)):
        for name in bnn.get_layer_params_name(layer_ind):
            draw = np.array(samples[name])
            atol_mu = n_sigma_tolerance * float(np.max(site_sigma[name])) / np.sqrt(n_predictive_samples)
            np.testing.assert_allclose(site_mu[name], np.mean(draw, axis=0), atol=atol_mu)
            np.testing.assert_allclose(site_sigma[name], np.std(draw, axis=0), rtol=rtol_sigma)


@settings(deadline=None, max_examples=5)
@given(
    n_features=st.integers(min_value=1, max_value=2),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_samples=st.integers(min_value=2, max_value=5),
    mu_init=st.floats(min_value=0.1, max_value=2.0),
    sigma_init=st.floats(min_value=0.5, max_value=3.0),
    nu=st.just(5.0),
    num_steps=st.just(3),
    lr=st.just(0.0),
)
def test_advi_zero_lr_posterior_equals_prior(
    n_features, hidden_dim_list, dist_type, n_samples, mu_init, sigma_init, nu, num_steps, lr
):
    """With SGD step_size=0, ADVI must leave mu and sigma identical to the prior.

    mu_init > 0 ensures init_to_value is used (not the stochastic init_to_median
    triggered for all-zero priors), so the guide loc is seeded exactly at prior_mu
    and stays there with zero gradient steps.
    """
    dist_params_init = (
        {"mu": mu_init, "sigma": sigma_init}
        if dist_type == "normal"
        else {"mu": mu_init, "sigma": sigma_init, "nu": nu}
    )
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_method="VI",
        update_kwargs={
            "num_steps": num_steps,
            "method": "advi",
            "optimizer_type": "sgd",
            "optimizer_kwargs": {"step_size": lr},
        },
        dist_type=dist_type,
        dist_params_init=dist_params_init,
    )

    context = np.random.standard_normal((n_samples, n_features)).astype(np.float32)
    rewards = np.random.choice([0, 1], size=n_samples).tolist()
    bnn.update(context=context, rewards=rewards)

    # float32 rounding on mu; softplus(softplus_inverse(sigma)) rounding on sigma
    atol_mu = max(mu_init, sigma_init) * np.finfo(np.float32).eps * 100
    rtol_sigma = np.finfo(np.float32).eps * 100

    for layer_ind in range(len(bnn.model_params.bnn_layer_params)):
        prior_w = bnn.model_params.bnn_layer_params_init[layer_ind].weight
        prior_b = bnn.model_params.bnn_layer_params_init[layer_ind].bias
        post_w = bnn.model_params.bnn_layer_params[layer_ind].weight
        post_b = bnn.model_params.bnn_layer_params[layer_ind].bias

        np.testing.assert_allclose(post_w.params["mu"], prior_w.params["mu"], atol=atol_mu)
        np.testing.assert_allclose(post_b.params["mu"], prior_b.params["mu"], atol=atol_mu)
        np.testing.assert_allclose(post_w.params["sigma"], prior_w.params["sigma"], rtol=rtol_sigma)
        np.testing.assert_allclose(post_b.params["sigma"], prior_b.params["sigma"], rtol=rtol_sigma)
