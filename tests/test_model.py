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

from typing import List, Literal, Optional, Tuple, get_args
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
from numpyro.infer.autoguide import AutoNormal
from pydantic import ValidationError
from utils import make_binary_rewards

from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BaseLocationScaleArray,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkDP,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
    Beta,
    BetaCC,
    BetaDP,
    BetaMO,
    BetaMOCC,
    CategoricalFeatureConfig,
    EarlyStopping,
    EmbeddingParams,
    FeaturesConfig,
    NormalArray,
    OptaxKind,
    StudentTArray,
    _wrap_guide_with_kl_scale,
)


@pytest.fixture(scope="module")
def make_bnn():
    """Factory fixture: returns a callable that builds a BayesianNeuralNetwork via cold_start."""

    def _factory(
        n_features: int, hidden_dim_list: list[int], calibrate_output_bias: bool = True
    ) -> BayesianNeuralNetwork:
        return BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            calibrate_output_bias=calibrate_output_bias,
        )

    return _factory


@pytest.fixture(scope="module")
def calibration_bnn(make_bnn) -> BayesianNeuralNetwork:
    """Single BNN instance for calibration tests; call ``_reset()`` before each use to restore fresh state."""
    return make_bnn(n_features=2, hidden_dim_list=[4])


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


# Beta / BNN decay factor (temperature scaling)


class TestBetaDecayFactor:
    """Per-update forgetting (decay_factor) on the Beta model: raw counts stay pristine, sampling uses
    the separate effective counts decayed towards the Beta(1, 1) prior."""

    MIN_DECAY_FACTOR = 1e-3
    MAX_DECAY_FACTOR = 1.0
    NO_DECAY_FACTOR = 1.0
    PRIOR = 1.0
    MAX_COUNT = 1000
    MAX_EMPTY_UPDATES = 30
    N_SAMPLES = 50

    counts = st.integers(min_value=1, max_value=MAX_COUNT)
    diverging_counts = st.integers(min_value=2, max_value=MAX_COUNT)  # > prior, so decayed != raw
    binary_rewards = st.lists(st.integers(min_value=0, max_value=1), min_size=1)
    # Highest decay factor that still forgets *measurably*: at 1 - 1e-16 the decayed counts are within
    # one float of the raw ones, so the two Beta(a, b) draws come out bit-identical and any
    # "decayed sampling differs from raw sampling" assertion is vacuously false.
    MAX_FORGETTING_DECAY_FACTOR = 0.999
    decay_factors = st.floats(min_value=MIN_DECAY_FACTOR, max_value=MAX_DECAY_FACTOR)
    forgetting_decay_factors = st.floats(min_value=MIN_DECAY_FACTOR, max_value=MAX_FORGETTING_DECAY_FACTOR)

    @given(
        decay_factor=st.one_of(st.none(), decay_factors),
        n_successes=counts,
        n_failures=counts,
    )
    def test_seeds_effective_counts_only_when_enabled(
        self, decay_factor: Optional[float], n_successes: int, n_failures: int
    ) -> None:
        """Effective counts are seeded from the raw counts only when decay is enabled."""
        b = Beta(n_successes=n_successes, n_failures=n_failures, decay_factor=decay_factor)
        if decay_factor is None:
            assert b.decayed_n_successes is None and b.decayed_n_failures is None
        else:
            assert b.decayed_n_successes == n_successes and b.decayed_n_failures == n_failures

    @given(
        rewards=binary_rewards,
        decay_factor=decay_factors,
        n_successes=counts,
        n_failures=counts,
        prior=st.just(PRIOR),
    )
    def test_update_keeps_raw_counts_and_decays_effective_counts(
        self, rewards: List[int], decay_factor: float, n_successes: int, n_failures: int, prior: float
    ) -> None:
        """Update accumulates raw counts unchanged and decays the effective counts towards the prior."""
        b = Beta(n_successes=n_successes, n_failures=n_failures, decay_factor=decay_factor)
        b.update(rewards=rewards)
        new_successes = sum(rewards)
        new_failures = len(rewards) - new_successes
        assert b.n_successes == n_successes + new_successes
        assert b.n_failures == n_failures + new_failures
        assert b.decayed_n_successes == pytest.approx(prior + decay_factor * (n_successes - prior) + new_successes)
        assert b.decayed_n_failures == pytest.approx(prior + decay_factor * (n_failures - prior) + new_failures)

    @given(rewards=binary_rewards, decay_factor=st.just(NO_DECAY_FACTOR), n_successes=counts, n_failures=counts)
    def test_factor_one_tracks_raw_counts(
        self, rewards: List[int], decay_factor: float, n_successes: int, n_failures: int
    ) -> None:
        """With decay_factor == 1 the effective counts equal the raw counts (no forgetting)."""
        b = Beta(n_successes=n_successes, n_failures=n_failures, decay_factor=decay_factor)
        b.update(rewards=rewards)
        assert b.decayed_n_successes == b.n_successes
        assert b.decayed_n_failures == b.n_failures

    @given(
        rewards=binary_rewards,
        decay_factor=forgetting_decay_factors,
        n_successes=diverging_counts,
        n_failures=diverging_counts,
        n_samples=st.just(N_SAMPLES),
    )
    def test_sampling_uses_effective_counts(
        self, rng, rewards: List[int], decay_factor: float, n_successes: int, n_failures: int, n_samples: int
    ) -> None:
        """Sampling draws from the effective counts, not the raw counts, when decay is enabled."""
        b = Beta(n_successes=n_successes, n_failures=n_failures, decay_factor=decay_factor)
        b.update(rewards=rewards)  # effective counts diverge from raw counts
        state = rng.bit_generator.state
        samples = b.sample_proba(n_samples=n_samples, rng=rng)
        rng.bit_generator.state = state
        from_effective = list(rng.beta(b.decayed_n_successes, b.decayed_n_failures, size=n_samples))
        rng.bit_generator.state = state
        from_raw = list(rng.beta(b.n_successes, b.n_failures, size=n_samples))
        assert samples == from_effective
        assert samples != from_raw

    @given(
        decay_factor=decay_factors,
        count=counts,
        n_updates=st.integers(min_value=1, max_value=MAX_EMPTY_UPDATES),
        prior=st.just(PRIOR),
    )
    def test_empty_updates_decay_effective_counts_geometrically(
        self, decay_factor: float, count: int, n_updates: int, prior: float
    ) -> None:
        """With no new data, the effective counts decay geometrically: prior + decay**k * (count - prior)."""
        b = Beta(n_successes=count, n_failures=count, decay_factor=decay_factor)
        for _ in range(n_updates):
            b.update(rewards=[])
        expected = prior + decay_factor**n_updates * (count - prior)
        assert b.decayed_n_successes == pytest.approx(expected)
        assert b.decayed_n_failures == pytest.approx(expected)

    @given(
        rewards=binary_rewards,
        decay_factor=decay_factors,
        n_successes=counts,
        n_failures=counts,
        prior=st.just(PRIOR),
    )
    def test_reset_restores_prior_effective_counts(
        self, rewards: List[int], decay_factor: float, n_successes: int, n_failures: int, prior: float
    ) -> None:
        """Resetting the model restores both the raw and effective counts to the prior."""
        b = Beta(n_successes=n_successes, n_failures=n_failures, decay_factor=decay_factor)
        b.update(rewards=rewards)
        b.reset()
        assert (b.n_successes, b.n_failures) == (int(prior), int(prior))
        assert (b.decayed_n_successes, b.decayed_n_failures) == (prior, prior)

    @given(rewards=binary_rewards, decay_factor=decay_factors, n_successes=counts, n_failures=counts)
    def test_serialization_roundtrip(
        self, rewards: List[int], decay_factor: float, n_successes: int, n_failures: int
    ) -> None:
        """get_state/from_state preserves the decay factor and the (non-recomputable) effective counts."""
        b = Beta(n_successes=n_successes, n_failures=n_failures, decay_factor=decay_factor)
        b.update(rewards=rewards)
        assert Beta.model_validate_json(b.model_dump_json()) == b

    @given(n_successes=counts, n_failures=counts)
    def test_state_without_decay_fields_loads_with_decay_disabled(self, n_successes: int, n_failures: int) -> None:
        """A serialized state predating the decay feature loads with decay disabled."""
        b = Beta.model_validate({"n_successes": n_successes, "n_failures": n_failures})
        assert b.decay_factor is None
        assert b.decayed_n_successes is None and b.decayed_n_failures is None

    @given(
        decay_factor=st.one_of(
            st.floats(max_value=0), st.floats(min_value=MAX_DECAY_FACTOR, exclude_min=True), st.just(float("nan"))
        )
    )
    def test_factor_out_of_range_raises(self, decay_factor: float) -> None:
        """decay_factor must lie in (0, 1]."""
        with pytest.raises(ValidationError):
            Beta(decay_factor=decay_factor)


class TestBnnDecayFactor:
    """Per-update prior-variance inflation (decay_factor) on the BayesianNeuralNetwork: weight, bias, and
    embedding sigmas are scaled by 1 / decay_factor before each re-fit, while means stay untouched.

    These build a network per case, so they stay plain (no @given) with a single representative factor."""

    DECAY_FACTOR = 0.5
    N_FEATURES = 2
    HIDDEN = (3,)
    CAT_COLUMN = 0
    CARDINALITY = 4

    def test_inflates_weight_and_bias_variance(
        self,
        decay_factor: float = DECAY_FACTOR,
        n_features: int = N_FEATURES,
        hidden_dim_list: Tuple[int, ...] = HIDDEN,
    ) -> None:
        """Weight/bias sigma are scaled by 1 / decay_factor while the means (mu) are left unchanged."""
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features, hidden_dim_list=list(hidden_dim_list), decay_factor=decay_factor
        )
        before = [
            (np.array(lp.weight.params["sigma"]), np.array(lp.bias.params["sigma"]), np.array(lp.weight.params["mu"]))
            for lp in bnn.model_params.bnn_layer_params
        ]
        bnn._inflate_prior_variance()
        for (w_sigma, b_sigma, w_mu), lp in zip(before, bnn.model_params.bnn_layer_params):
            assert np.allclose(np.array(lp.weight.params["sigma"]), w_sigma / decay_factor)
            assert np.allclose(np.array(lp.bias.params["sigma"]), b_sigma / decay_factor)
            assert np.allclose(np.array(lp.weight.params["mu"]), w_mu)

    def test_disabled_leaves_variance_unchanged(
        self,
        make_bnn,
        n_features: int = N_FEATURES,
        hidden_dim_list: Tuple[int, ...] = HIDDEN,
    ) -> None:
        """With decay disabled, _inflate_prior_variance is a no-op."""
        bnn = make_bnn(n_features=n_features, hidden_dim_list=list(hidden_dim_list))
        before = [np.array(lp.weight.params["sigma"]) for lp in bnn.model_params.bnn_layer_params]
        bnn._inflate_prior_variance()
        for w_sigma, lp in zip(before, bnn.model_params.bnn_layer_params):
            assert np.allclose(np.array(lp.weight.params["sigma"]), w_sigma)

    def test_inflates_embedding_variance(
        self,
        decay_factor: float = DECAY_FACTOR,
        n_features: int = N_FEATURES,
        hidden_dim_list: Tuple[int, ...] = HIDDEN,
        cat_column: int = CAT_COLUMN,
        cardinality: int = CARDINALITY,
    ) -> None:
        """Categorical-embedding sigma is scaled by 1 / decay_factor."""
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=list(hidden_dim_list),
            categorical_features={cat_column: cardinality},
            decay_factor=decay_factor,
        )
        before = [np.array(emb.params["sigma"]) for emb in bnn.model_params.embedding_params.embeddings]
        bnn._inflate_prior_variance()
        for emb_sigma, emb in zip(before, bnn.model_params.embedding_params.embeddings):
            assert np.allclose(np.array(emb.params["sigma"]), emb_sigma / decay_factor)


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


# BetaDP


@given(st.floats())
def test_can_init_betaDP(a_float):
    if a_float < 0 or np.isnan(a_float):
        with pytest.raises(ValidationError):
            BetaDP(price=a_float)
    else:
        b = BetaDP(price=a_float)
        assert b.price == a_float


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


class TestBiasStd:
    """Test suite for the ``bias_std`` cold-start parameter of the BNN."""

    @staticmethod
    def _dist_params(dist_type: Literal["studentt", "normal"], sigma: float, nu: float) -> dict:
        params = {"mu": 0.0, "sigma": sigma}
        if dist_type == "studentt":
            params["nu"] = nu
        return params

    @settings(deadline=500)
    @given(
        bias_std=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        nu=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dist_type=st.sampled_from(["studentt", "normal"]),
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    )
    def test_bias_std_overrides_bias_sigma_only(
        self,
        bias_std: float,
        sigma: float,
        nu: float,
        dist_type: Literal["studentt", "normal"],
        n_features: int,
        hidden_dim_list: list,
    ) -> None:
        """``bias_std`` sets the bias prior sigma on every layer while weight priors keep ``sigma``."""
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            dist_type=dist_type,
            dist_params_init=self._dist_params(dist_type, sigma, nu),
            bias_std=bias_std,
        )
        assert len(bnn.model_params.bnn_layer_params) == len(hidden_dim_list) + 1
        for layer_params in bnn.model_params.bnn_layer_params:
            np.testing.assert_allclose(layer_params.bias.params["sigma"], bias_std)
            np.testing.assert_allclose(layer_params.weight.params["sigma"], sigma)

    @settings(deadline=500)
    @given(
        sigma=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        nu=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dist_type=st.sampled_from(["studentt", "normal"]),
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
        random_seed=st.integers(min_value=0, max_value=2**16),
    )
    def test_bias_std_none_matches_default(
        self,
        sigma: float,
        nu: float,
        dist_type: Literal["studentt", "normal"],
        n_features: int,
        hidden_dim_list: list,
        random_seed: int,
    ) -> None:
        """Passing ``bias_std=None`` produces the same ``model_params`` as omitting it."""
        kwargs = dict(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            dist_type=dist_type,
            dist_params_init=self._dist_params(dist_type, sigma, nu),
            random_seed=random_seed,
        )
        bnn_default = BayesianNeuralNetwork.cold_start(**kwargs)
        bnn_none = BayesianNeuralNetwork.cold_start(bias_std=None, **kwargs)
        assert bnn_default.model_params == bnn_none.model_params

    @settings(deadline=500)
    @given(
        bias_std=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        nu=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dist_type=st.sampled_from(["studentt", "normal"]),
        n_objectives=st.integers(min_value=1, max_value=3),
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    )
    def test_bias_std_propagates_through_mo_cold_start(
        self,
        bias_std: float,
        sigma: float,
        nu: float,
        dist_type: Literal["studentt", "normal"],
        n_objectives: int,
        n_features: int,
        hidden_dim_list: list,
    ) -> None:
        """``bias_std`` is forwarded to each per-objective BNN in the MO variant."""
        mo_bnn = BayesianNeuralNetworkMO.cold_start(
            n_objectives=n_objectives,
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            dist_type=dist_type,
            dist_params=self._dist_params(dist_type, sigma, nu),
            bias_std=bias_std,
        )
        assert len(mo_bnn.models) == n_objectives
        for model in mo_bnn.models:
            for layer_params in model.model_params.bnn_layer_params:
                np.testing.assert_allclose(layer_params.bias.params["sigma"], bias_std)

    @settings(deadline=500)
    @given(
        bias_std=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        nu=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dist_type=st.sampled_from(["studentt", "normal"]),
        n_features=st.integers(min_value=1, max_value=5),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    )
    def test_bias_std_independent_of_layerwise_scaling(
        self,
        bias_std: float,
        sigma: float,
        nu: float,
        dist_type: Literal["studentt", "normal"],
        n_features: int,
        hidden_dim_list: list,
    ) -> None:
        """``bias_std`` applies verbatim to bias sigma even when ``use_layerwise_scaling`` shrinks weight sigma."""
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            dist_type=dist_type,
            dist_params_init=self._dist_params(dist_type, sigma, nu),
            use_layerwise_scaling=True,
            bias_std=bias_std,
        )
        fan_ins = [n_features] + hidden_dim_list
        for layer_params, fan_in in zip(bnn.model_params.bnn_layer_params, fan_ins):
            np.testing.assert_allclose(layer_params.bias.params["sigma"], bias_std)
            np.testing.assert_allclose(layer_params.weight.params["sigma"], sigma / np.sqrt(fan_in))


class TestCalibrateOutputBias:
    """Test suite for ``calibrate_output_bias`` and the ``bias_calibrated`` flag on the BNN."""

    @pytest.mark.parametrize(
        "calibrate_output_bias, bias_calibrated, should_raise",
        [
            (False, True, True),
            (True, True, False),
            (True, False, False),
        ],
    )
    def test_validator_bias_calibrated_consistency(
        self, calibrate_output_bias: bool, bias_calibrated: bool, should_raise: bool, n_features: int = 2
    ) -> None:
        """Validator rejects bias_calibrated=True when calibrate_output_bias=False; allows all other combos."""
        fc = FeaturesConfig(n_features=n_features)
        model_params = BayesianNeuralNetwork.create_model_params(fc, [])
        kwargs = dict(
            model_params=model_params,
            feature_config=fc,
            calibrate_output_bias=calibrate_output_bias,
            bias_calibrated=bias_calibrated,
        )
        if should_raise:
            with pytest.raises(ValidationError):
                BayesianNeuralNetwork(**kwargs)
        else:
            BayesianNeuralNetwork(**kwargs)

    @given(
        n_rewards=st.integers(min_value=5, max_value=50),
        reward_rate=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
        rtol=st.just(1e-5),
    )
    def test_output_bias_mu_set_to_logit_of_reward_rate(
        self, calibration_bnn: BayesianNeuralNetwork, n_rewards: int, reward_rate: float, rtol
    ) -> None:
        """After ``_calibrate_output_bias``, the output-layer bias mu equals ``logit(empirical_reward_rate)``."""
        calibration_bnn._reset()
        rewards = make_binary_rewards(n_rewards, reward_rate)
        calibration_bnn._calibrate_output_bias(rewards)

        eps = BaseBayesianNeuralNetwork._numerical_eps
        empirical_rate = float(np.clip(np.mean(rewards), eps, 1 - eps))
        expected_intercept = float(np.log(empirical_rate / (1 - empirical_rate)))
        np.testing.assert_allclose(
            np.array(calibration_bnn.model_params.bnn_layer_params[-1].bias.params["mu"]),
            expected_intercept,
            rtol=rtol,
        )

    @given(
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
        n_rewards=st.integers(min_value=5, max_value=50),
        reward_rate=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
    )
    def test_calibration_fires_only_once(
        self, make_bnn, n_features: int, hidden_dim_list: list[int], n_rewards: int, reward_rate: float
    ) -> None:
        """A second call to ``_calibrate_output_bias`` is a no-op (bias_calibrated guard)."""
        bnn = make_bnn(n_features, hidden_dim_list)
        rewards = make_binary_rewards(n_rewards, reward_rate)

        bnn._calibrate_output_bias(rewards)
        mu_after_first = list(bnn.model_params.bnn_layer_params[-1].bias.params["mu"])

        bnn._calibrate_output_bias([1 - r for r in rewards])
        assert list(bnn.model_params.bnn_layer_params[-1].bias.params["mu"]) == mu_after_first

    @given(
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
        rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
    )
    def test_calibration_flag_lifecycle(
        self, make_bnn, n_features: int, hidden_dim_list: list[int], rewards: list[int]
    ) -> None:
        """``bias_calibrated`` is set by ``_calibrate_output_bias`` and cleared by ``_reset``."""
        bnn = make_bnn(n_features, hidden_dim_list)
        bnn._calibrate_output_bias(rewards)
        assert bnn.bias_calibrated is True
        bnn._reset()
        assert bnn.bias_calibrated is False

    @given(
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
        rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
    )
    def test_calibration_only_mutates_output_layer_bias_mu(
        self, make_bnn, n_features: int, hidden_dim_list: list[int], rewards: list[int]
    ) -> None:
        """Calibration touches only output-layer bias mu; all sigmas and hidden-layer mus are unchanged."""
        bnn = make_bnn(n_features, hidden_dim_list)
        output_sigma_before = list(bnn.model_params.bnn_layer_params[-1].bias.params["sigma"])
        hidden_mus_before = [list(layer.bias.params["mu"]) for layer in bnn.model_params.bnn_layer_params[:-1]]

        bnn._calibrate_output_bias(rewards)

        assert list(bnn.model_params.bnn_layer_params[-1].bias.params["sigma"]) == output_sigma_before
        assert [list(layer.bias.params["mu"]) for layer in bnn.model_params.bnn_layer_params[:-1]] == hidden_mus_before

    @pytest.mark.parametrize("constant_reward", [0, 1])
    @given(
        n_features=st.integers(min_value=1, max_value=3),
        hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
        n_rewards=st.integers(min_value=5, max_value=50),
    )
    def test_constant_rewards_clip_to_finite_logit(
        self,
        make_bnn,
        constant_reward: int,
        n_features: int,
        hidden_dim_list: list[int],
        n_rewards: int,
    ) -> None:
        """All-identical rewards clip to eps / 1-eps, keeping the logit finite."""
        bnn = make_bnn(n_features, hidden_dim_list)
        bnn._calibrate_output_bias([constant_reward] * n_rewards)
        assert all(np.isfinite(bnn.model_params.bnn_layer_params[-1].bias.params["mu"]))


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
    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    sample_proba(context=context)

    # check that the model is working with multi-sample prediction
    context = np.repeat(rng.uniform(low=-1.0, high=1.0, size=(1, n_features)), n_samples, axis=0)
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
    epochs=st.just(1),
)
def test_bnn_vi_update(
    rng,
    activation,
    use_residual_connections,
    use_layerwise_scaling,
    dist_type,
    n_features,
    hidden_dim_list,
    n_samples,
    epochs,
):
    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
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

    rewards = _make_random_rewards(n_samples)

    # context is numpy array
    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    update(context=context, rewards=rewards)

    # raise an error if len(context) != len(rewards)
    with pytest.raises(AttributeError):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
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
        update_kwargs=update_kwargs,
    )

    model_fn = bnn._create_update_model()
    assert callable(model_fn)

    # Optimizer is always set for VI (built from defaults or user override)
    assert bnn.obj_optimizer is not None

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
    batch_size=st.sampled_from((None, 2, 4)),
    optimizer_type=st.just("dummy_optimizer"),
)
def test_bnn_vi_update_parameters_dummy_optimizer_failure(
    n_features: int,
    hidden_dim_list: list[int],
    batch_size: Optional[int],
    optimizer_type: Optional[str],
) -> None:
    """Test that dummy_optimizer raises ValueError for BNN VI update."""
    update_kwargs = _create_update_kwargs(batch_size, optimizer_type)

    with pytest.raises(ValueError):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            update_kwargs=update_kwargs,
        )


@pytest.mark.parametrize(
    "optimizer_type,optimizer_kwargs",
    [
        ("adam", {"invalid_param": 123, "step_size": 0.01}),
        ("sgd", {"invalid_param": 123}),
    ],
)
def test_invalid_optimizer_kwargs(optimizer_type: str, optimizer_kwargs: dict, n_features: int = 2) -> None:
    """Test that invalid optimizer kwargs raise TypeError or ValueError."""

    with pytest.raises((TypeError, ValueError), match="Invalid optimizer kwargs"):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_kwargs={
                "optimizer_type": optimizer_type,
                "optimizer_kwargs": optimizer_kwargs,
            },
        )


def test_optax_kind_literal_matches_optax_return_types_keys() -> None:
    """Verify that OptaxKind Literal values exactly match _optax_return_types keys."""

    literal_values = set(get_args(OptaxKind))
    classvar_keys = set(BayesianNeuralNetwork._optax_return_types.keys())
    assert literal_values == classvar_keys, (
        f"OptaxKind Literal {literal_values} does not match _optax_return_types keys {classvar_keys}"
    )


def test_optax_kind_literal_matches_optax_required_kwargs_keys() -> None:
    """Verify that OptaxKind Literal values exactly match _optax_required_kwargs keys."""

    literal_values = set(get_args(OptaxKind))
    classvar_keys = set(BayesianNeuralNetwork._optax_required_kwargs.keys())
    assert literal_values == classvar_keys, (
        f"OptaxKind Literal {literal_values} does not match _optax_required_kwargs keys {classvar_keys}"
    )


@pytest.mark.parametrize(
    "kind,fn_name",
    [
        # optax.scale returns GradientTransformation but accepts step_size, not learning_rate
        ("optimizer", "scale"),
        # optax.constant_schedule returns Schedule but accepts value, not init_value
        ("lr_scheduler", "constant_schedule"),
    ],
)
def test_resolve_optax_fn_rejects_missing_required_kwarg(
    kind: str,
    fn_name: str,
    n_features: int = 2,
) -> None:
    """Test that _resolve_optax_fn raises ValueError when the required kwarg is absent."""
    bnn = BayesianNeuralNetwork.cold_start(n_features=n_features)
    with pytest.raises(ValueError, match="does not accept the required keyword argument"):
        bnn._resolve_optax_fn(fn_name, kind)


@pytest.mark.parametrize(
    "early_stopping_kwargs",
    [
        {"invalid_param": 123, "tolerance": 1e-3},
        {"diff": "invalid", "tolerance": 1e-3},
    ],
)
def test_invalid_early_stopping_kwargs(early_stopping_kwargs: dict, n_features: int = 2) -> None:
    """Test that invalid early stopping kwargs raise TypeError or ValueError."""

    with pytest.raises((TypeError, ValueError, KeyError), match="Invalid early stopping kwargs"):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_kwargs={
                "early_stopping_kwargs": early_stopping_kwargs,
            },
        )


@pytest.mark.parametrize(
    "optimizer_type,optimizer_kwargs,lr_scheduler_type,lr_scheduler_kwargs",
    [
        ("sgd", {"step_size": 0.01}, "exponential_decay", {"transition_steps": 100, "decay_rate": 0.9}),
        ("adam", {"step_size": 0.01}, "exponential_decay", {"transition_steps": 50, "decay_rate": 0.95}),
        ("sgd", {"step_size": 0.01, "momentum": 0.9}, "cosine_decay_schedule", {"decay_steps": 100}),
        (
            "adam",
            {"step_size": 0.01},
            "exponential_decay",
            {"transition_steps": 100, "decay_rate": 0.9, "staircase": True},
        ),
        (
            "sgd",
            {"step_size": 0.01},
            "warmup_cosine_decay_schedule",
            {"peak_value": 0.05, "warmup_steps": 10, "decay_steps": 100},
        ),
        ("adam", {"step_size": 0.01}, "linear_schedule", {"end_value": 1e-5, "transition_steps": 200}),
    ],
)
def test_lr_scheduler_valid(
    optimizer_type: str,
    optimizer_kwargs: dict,
    lr_scheduler_type: str,
    lr_scheduler_kwargs: dict,
    n_features: int = 2,
) -> None:
    """Test that valid lr_scheduler_type/lr_scheduler_kwargs build an optimizer successfully."""
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        update_kwargs={
            "optimizer_type": optimizer_type,
            "optimizer_kwargs": optimizer_kwargs,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
        },
    )
    assert bnn.obj_optimizer is not None


@pytest.mark.parametrize(
    "lr_scheduler_type,lr_scheduler_kwargs",
    [
        (
            "dummy_scheduler",
            {},
        ),
        ("exponential_decay", {"invalid_kwarg": 1}),
    ],
)
def test_lr_scheduler_invalid_type_or_kwargs(
    lr_scheduler_type: str,
    lr_scheduler_kwargs: dict,
    n_features: int = 2,
) -> None:
    """Test that invalid lr_scheduler_type or lr_scheduler_kwargs raise ValueError."""
    with pytest.raises((TypeError, ValueError)):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_kwargs={
                "lr_scheduler_type": lr_scheduler_type,
                "lr_scheduler_kwargs": lr_scheduler_kwargs,
            },
        )


@settings(max_examples=20)
@given(
    optimizer_type=st.sampled_from(("sgd", "adam")),
    num_particles=st.sampled_from((1, 2)),
    gradient_clip_norm=st.one_of(st.none(), st.floats(min_value=0.1, max_value=10.0)),
    lr_scheduler_type=st.sampled_from((None, "exponential_decay")),
    kl_annealing_fraction=st.one_of(st.none(), st.floats(min_value=0.05, max_value=1.0)),
    num_steps=st.just(5),
    n_features=st.just(2),
    n_samples=st.just(5),
    decay_rate=st.just(0.9),
    transition_steps_factor=st.just(2),
)
def test_vi_training_options(
    rng,
    optimizer_type: str,
    num_particles: int,
    gradient_clip_norm: Optional[float],
    lr_scheduler_type: Optional[str],
    kl_annealing_fraction: Optional[float],
    num_steps: int,
    n_features: int,
    n_samples: int,
    decay_rate: float,
    transition_steps_factor: int,
) -> None:
    """Test that VI training options (num_particles, gradient_clip_norm, lr_scheduler) compose correctly."""
    update_kwargs: dict = {
        "num_steps": num_steps,
        "optimizer_type": optimizer_type,
        "num_particles": num_particles,
    }
    if gradient_clip_norm is not None:
        update_kwargs["gradient_clip_norm"] = gradient_clip_norm
    if lr_scheduler_type is not None:
        update_kwargs["lr_scheduler_type"] = lr_scheduler_type
        update_kwargs["lr_scheduler_kwargs"] = {
            "transition_steps": num_steps // transition_steps_factor,
            "decay_rate": decay_rate,
        }
    if kl_annealing_fraction is not None:
        update_kwargs["kl_annealing_fraction"] = kl_annealing_fraction

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        update_kwargs=update_kwargs,
    )
    context = rng.random((n_samples, n_features)).astype(np.float32)
    rewards = _make_random_rewards(n_samples)
    bnn.update(context=context, rewards=rewards)
    result = bnn.sample_proba(context=context, rng=rng)
    assert len(result) == n_samples
    assert all(0 <= p[0] <= 1 for p in result), f"Probabilities out of [0,1]: {result}"
    assert all(np.isfinite(p[1]) for p in result), f"Non-finite weights: {result}"


# ---------------------------------------------------------------------------
# KL annealing
# ---------------------------------------------------------------------------


class TestKLAnnealing:
    """Tests for the optional `kl_annealing_fraction` VI training kwarg.

    The feature is implemented by wrapping the prior sample sites in
    ``numpyro.handlers.scale(scale=kl_annealing_factor)`` and threading a per-step
    factor array through ``svi.update``. These tests verify the handler-level
    annotation (which is what the feature actually controls) via trace introspection,
    plus the schedule formula at the helper level. The likelihood ``out`` site must
    remain unscaled in every case.
    """

    @staticmethod
    def _build_bnn(kl_annealing_fraction: Optional[float], num_steps: int = 4, n_features: int = 2):
        update_kwargs: dict = {"num_steps": num_steps}
        if kl_annealing_fraction is not None:
            update_kwargs["kl_annealing_fraction"] = kl_annealing_fraction
        return BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_kwargs=update_kwargs,
        )

    @pytest.mark.parametrize(
        "epoch_steps_list",
        [
            [1],
            [5],
            [3, 3, 1],
            [2, 2, 2, 2],
        ],
    )
    def test_inactive_factor_array_is_all_ones(self, epoch_steps_list: list) -> None:
        """With no kl_annealing_fraction the per-step factor schedule is a neutral all-ones array,
        regardless of how the total steps are split into epochs."""
        total_steps = sum(epoch_steps_list)
        bnn = self._build_bnn(kl_annealing_fraction=None, num_steps=total_steps)
        epoch_chunks = bnn.build_kl_annealing_factors(epoch_steps_list)
        # Assert the per-epoch split contract directly: one chunk per epoch, each chunk
        # holding exactly that epoch's step count. The flattened check below cannot catch a
        # regression that splits at the wrong boundaries (lax.scan consumes one chunk per epoch).
        assert len(epoch_chunks) == len(epoch_steps_list)
        assert [len(chunk) for chunk in epoch_chunks] == epoch_steps_list
        factor_array = np.concatenate([np.asarray(chunk) for chunk in epoch_chunks])
        assert factor_array.shape == (total_steps,)
        np.testing.assert_allclose(factor_array, np.ones(total_steps), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "kl_annealing_fraction, epoch_steps_list",
        [
            (0.5, [10]),
            (0.25, [4, 4, 4]),
            (1.0, [5, 5]),
            (0.1, [3, 3, 3, 1]),
        ],
    )
    def test_active_factor_array_matches_linear_ramp_formula(
        self, kl_annealing_fraction: float, epoch_steps_list: list
    ) -> None:
        """The active schedule is min(1, (step+1)/W) with W = max(1, ceil(fraction * total_steps))."""
        total_steps = sum(epoch_steps_list)
        bnn = self._build_bnn(kl_annealing_fraction=kl_annealing_fraction, num_steps=total_steps)
        epoch_chunks = bnn.build_kl_annealing_factors(epoch_steps_list)
        # Assert the per-epoch split contract directly (see the inactive-schedule test)
        assert len(epoch_chunks) == len(epoch_steps_list)
        assert [len(chunk) for chunk in epoch_chunks] == epoch_steps_list
        actual = np.concatenate([np.asarray(chunk) for chunk in epoch_chunks])

        warmup_steps = max(1, int(np.ceil(kl_annealing_fraction * total_steps)))
        # Match JAX's default float32 dtype so identity-level comparison stays exact.
        expected = np.minimum(1.0, (np.arange(total_steps, dtype=np.float32) + 1.0) / warmup_steps)

        assert actual.shape == (total_steps,)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        # Endpoint sanity: strictly positive at step 0, exactly 1.0 by the last warmup step.
        assert actual[0] > 0.0
        assert actual[warmup_steps - 1] == pytest.approx(1.0)

    @pytest.mark.parametrize("kl_annealing_factor", [1.0, 0.3])
    @pytest.mark.parametrize("n_features, n_samples", [(1, 3), (3, 7)])
    def test_model_trace_scale_annotation_on_priors_and_not_on_likelihood(
        self, rng: np.random.Generator, n_features: int, n_samples: int, kl_annealing_factor: float
    ) -> None:
        """Tracing the inner model surfaces handlers.scale on every prior sample site (weights, biases)
        with the supplied factor, while the likelihood ``out`` site is left unscaled.

        This is a pure-trace assertion: it does not run SVI, so it is deterministic and
        independent of XLA layout. It tests the actual surface (handler-level scale annotation)
        the feature controls.
        """
        bnn = self._build_bnn(kl_annealing_fraction=None, n_features=n_features)
        model_fn = bnn._create_update_model()

        x = jnp.asarray(rng.random((n_samples, n_features)).astype(np.float32))
        y = jnp.asarray(_make_random_rewards(n_samples), dtype=jnp.int32)

        tr = numpyro.handlers.trace(numpyro.handlers.seed(model_fn, rng_seed=0)).get_trace(x, y, kl_annealing_factor)

        prior_site_names = [name for name in tr if name.startswith(("weight_", "bias_"))]
        assert prior_site_names, "expected at least one prior site in the model trace"
        for name in prior_site_names:
            site = tr[name]
            assert site["type"] == "sample"
            assert site["scale"] == kl_annealing_factor, (
                f"prior site {name!r} expected scale={kl_annealing_factor}, got {site['scale']!r}"
            )

        out_site = tr["out"]
        assert out_site["type"] == "sample"
        # The likelihood site must remain outside the scale context. handlers.scale leaves
        # site["scale"] as None when no wrap is in effect.
        assert out_site["scale"] is None, (
            f"likelihood site expected scale=None (outside annealing context), got {out_site['scale']!r}"
        )

    @settings(deadline=None, max_examples=5)
    @given(
        n_features=st.integers(min_value=1, max_value=5),
        n_samples=st.integers(min_value=2, max_value=8),
        kl_annealing_factor=st.floats(
            min_value=0.0, max_value=1.0, exclude_min=True, allow_nan=False, allow_infinity=False
        ),
    )
    def test_symmetric_guide_wrap_scales_guide_sample_sites(
        self, rng: np.random.Generator, n_features: int, n_samples: int, kl_annealing_factor: float
    ) -> None:
        """The guide wrap installed in `_run_svi_training_loop` must scale the guide's sample
        sites symmetrically with the model's prior sites. Without it the per-site KL contribution
        (log p - log q) would not scale uniformly.

        Traces the production wrapper (`_wrap_guide_with_kl_scale`) directly. The asserted
        `scale == factor` identity holds for any factor in (0, 1], so the factor and model
        shape are drawn from Hypothesis strategies.
        """
        # The BNN's kl_annealing_fraction is irrelevant here: this test does not run SVI and
        # never consults the schedule. The factor under test is fed directly to get_trace(...)
        # via the kl_annealing_factor parameter below.
        bnn = self._build_bnn(kl_annealing_fraction=None, n_features=n_features)
        model_fn = bnn._create_update_model()

        # Build a bare AutoNormal guide over the same model (matches the advi setup in
        # `_run_svi_training_loop`) without relying on the per-site init_scale_fn details,
        # then wrap it with the exact production helper.
        guide = AutoNormal(model_fn)
        scaled_guide = _wrap_guide_with_kl_scale(guide)

        x = jnp.asarray(rng.random((n_samples, n_features)).astype(np.float32))
        y = jnp.asarray(_make_random_rewards(n_samples), dtype=jnp.int32)

        tr = numpyro.handlers.trace(numpyro.handlers.seed(scaled_guide, rng_seed=0)).get_trace(
            x, y, kl_annealing_factor
        )

        guide_sample_sites = [name for name, site in tr.items() if site["type"] == "sample"]
        assert guide_sample_sites, "expected at least one sample site in the guide trace"
        for name in guide_sample_sites:
            site = tr[name]
            assert site["scale"] == kl_annealing_factor, (
                f"guide site {name!r} expected scale={kl_annealing_factor}, got {site['scale']!r}"
            )

    # invalid_value is the axis under test; n_features/num_steps are scaffolding, pinned via st.just.
    @pytest.mark.parametrize("invalid_value", [0.0, -0.1, 1.5])
    @given(n_features=st.just(2), num_steps=st.just(5))
    def test_invalid_values_rejected_at_construction(self, invalid_value, n_features: int, num_steps: int) -> None:
        """`kl_annealing_fraction` must lie in the half-open interval (0, 1] (enforced by PositiveFloat01)."""
        with pytest.raises(ValueError, match="kl_annealing_fraction"):
            BayesianNeuralNetwork.cold_start(
                n_features=n_features,
                update_kwargs={"num_steps": num_steps, "kl_annealing_fraction": invalid_value},
            )

    @given(
        n_features=st.just(2),
        valid_numpy_fraction=st.sampled_from([np.float64(0.5), np.float32(0.5), np.int64(1), np.int32(1)]),
    )
    def test_valid_numpy_scalar_fractions_accepted_at_construction(self, n_features: int, valid_numpy_fraction) -> None:
        """A NumPy real scalar in (0, 1] is a valid `kl_annealing_fraction` and must construct."""
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_kwargs={"num_steps": 5, "kl_annealing_fraction": valid_numpy_fraction},
        )
        assert bnn is not None


@given(
    n_features=st.just(2),
    epochs=st.integers(min_value=1, max_value=100),
    num_steps=st.integers(min_value=1, max_value=100),
)
def test_epochs_and_num_steps_warns(n_features: int, epochs: int, num_steps: int) -> None:
    """Test that specifying both 'epochs' and 'num_steps' raises a UserWarning (epochs takes precedence)."""
    with pytest.warns(UserWarning, match="'epochs' takes precedence"):
        BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            update_kwargs={"epochs": epochs, "num_steps": num_steps},
        )


@pytest.mark.parametrize("n_features", [1, 2])
def test_bnn_fullrank_advi_update(
    rng, n_features, hidden_dim_list=(1,), n_samples=5, method="fullrank_advi", num_steps=1
):
    hidden_dim_list = list(hidden_dim_list)

    def update(context: np.ndarray, rewards: list):
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
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

    rewards = _make_random_rewards(n_samples)
    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    update(context=context, rewards=rewards)


########################################################################################################################


# BayesianNeuralNetwork - SVI NaN protection and restore_best_svi_state


@given(
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(2),
)
def test_bnn_svi_nan_loss_raises_error(
    rng: np.random.Generator, n_features: int, hidden_dim_list: list[int], n_samples: int
) -> None:
    """Test that a NaN loss during SVI training raises a ValueError immediately."""

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
    )

    # Patch np.mean to return NaN on the first epoch to simulate divergence.
    original_mean = np.mean
    call_count = {"n": 0}

    def nan_mean(a, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:  # call 1 is avg_sigma in _build_svi_guide_init; call 2 is epoch loss
            return float("nan")
        return original_mean(a, *args, **kwargs)

    context = rng.uniform(size=(n_samples, n_features))
    rewards = _make_random_rewards(n_samples)

    with patch("pybandits.model.bnn.network.np.mean", nan_mean):
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
    n_samples=st.integers(min_value=1, max_value=100),
)
def test_create_default_instance_bayesian_neural_network_cc(
    rng, activation, use_residual_connections, use_layerwise_scaling, n_features, hidden_dim_list, cost, n_samples
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
        context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
        prob_and_weighted_sum = bnn_cold_start.sample_proba(context=context, rng=rng)
        prob, weighted_sum = zip(*prob_and_weighted_sum)
        assert len(prob) == n_samples
        assert all([0 <= p <= 1 for p in prob])


########################################################################################################################


# BayesianNeuralNetworkDP
@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    price=st.floats(allow_nan=False, allow_infinity=False),
)
def test_can_init_bayesian_neural_network_dp(
    activation, use_residual_connections, use_layerwise_scaling, n_features, hidden_dim_list, price
):
    # at least one beta must be specified
    dim_list = [n_features] + hidden_dim_list
    fc = FeaturesConfig(n_features=n_features)
    if any(layer_dim <= 0 for layer_dim in dim_list) or (price < 0):
        with pytest.raises((ValidationError, ValueError)):
            model_params = BayesianNeuralNetwork.create_model_params(
                fc, hidden_dim_list, use_layerwise_scaling=use_layerwise_scaling
            )
            BayesianNeuralNetworkDP(
                model_params=model_params,
                price=price,
                activation=activation,
                use_residual_connections=use_residual_connections,
                feature_config=fc,
            )
    else:
        model_params = BayesianNeuralNetwork.create_model_params(
            fc, hidden_dim_list, use_layerwise_scaling=use_layerwise_scaling
        )
        bnn = BayesianNeuralNetworkDP(
            model_params=model_params,
            price=price,
            activation=activation,
            use_residual_connections=use_residual_connections,
            feature_config=fc,
        )
        assert bnn.model_params == model_params
        assert bnn.price == price
        assert bnn.activation == activation
        assert bnn.use_residual_connections == use_residual_connections


@given(
    activation=st.sampled_from(["tanh", "relu", "sigmoid", "gelu"]),
    use_residual_connections=st.booleans(),
    use_layerwise_scaling=st.booleans(),
    n_features=st.integers(min_value=1, max_value=3),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=2),
    price=st.floats(allow_nan=False, allow_infinity=False),
    n_samples=st.integers(min_value=1, max_value=100),
)
def test_create_default_instance_bayesian_neural_network_dp(
    rng, activation, use_residual_connections, use_layerwise_scaling, n_features, hidden_dim_list, price, n_samples
):
    dim_list = [n_features] + hidden_dim_list
    if any(layer_dim <= 0 for layer_dim in dim_list) or (price < 0):
        with pytest.raises((ValidationError, ValueError)):
            BayesianNeuralNetworkDP.cold_start(
                n_features=n_features,
                hidden_dim_list=hidden_dim_list,
                price=price,
                activation=activation,
                use_residual_connections=use_residual_connections,
                use_layerwise_scaling=use_layerwise_scaling,
            )
    else:
        bnn_cold_start = BayesianNeuralNetworkDP.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
            price=price,
            activation=activation,
            use_residual_connections=use_residual_connections,
            use_layerwise_scaling=use_layerwise_scaling,
        )
        fc = FeaturesConfig(n_features=n_features)
        model_params = BayesianNeuralNetwork.create_model_params(
            fc, hidden_dim_list=hidden_dim_list, use_layerwise_scaling=use_layerwise_scaling
        )
        bnn_init = BayesianNeuralNetworkDP(
            model_params=model_params,
            price=price,
            activation=activation,
            use_residual_connections=use_residual_connections,
            feature_config=fc,
        )
        assert bnn_cold_start == bnn_init
        assert bnn_cold_start.price == price
        assert bnn_cold_start.activation == activation
        assert bnn_cold_start.use_residual_connections == use_residual_connections

        # Test sample_proba works
        context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
        prob_and_weighted_sum = bnn_cold_start.sample_proba(context=context, rng=rng)
        prob, weighted_sum = zip(*prob_and_weighted_sum)
        assert len(prob) == n_samples
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

    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
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
    n_objectives=st.integers(min_value=1, max_value=2),
    epochs=st.just(1),
)
def test_bayesian_neural_network_mo_update(
    rng, activation, n_features, hidden_dim_list, n_samples, n_objectives, epochs
):
    models = [
        BayesianNeuralNetwork.cold_start(
            n_features,
            hidden_dim_list,
            activation=activation,
            update_kwargs={"epochs": epochs},  # Use minimal iterations for faster tests
        )
        for _ in range(n_objectives)
    ]
    bnn_mo = BayesianNeuralNetworkMO(models=models)
    # Verify all models have the same activation
    for model in bnn_mo.models:
        assert model.activation == activation

    context = rng.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    rewards = [[rng.integers(0, 2) for _ in range(n_objectives)] for _ in range(n_samples)]

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
# BNN with feature_config: cold_start (context validation lives in tests/test_dnn.py)
# ---------------------------------------------------------------------------


def _make_bnn_with_categoricals(
    n_features=2, categorical_features=None, dist_type="studentt", hidden_dim_list=None, update_kwargs=None
):
    kwargs = {"epochs": 1}
    if update_kwargs:
        kwargs.update(update_kwargs)
    return BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        categorical_features=categorical_features or {1: 3},
        hidden_dim_list=hidden_dim_list or [8],
        dist_type=dist_type,
        update_kwargs=kwargs,
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

    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(n_features=n_features, categorical_features=categorical_features)
    model_fn = bnn._create_update_model()
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

    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = _make_bnn_with_categoricals(
        n_features=n_features, categorical_features=categorical_features, update_kwargs={"batch_size": batch_size}
    )
    model_fn = bnn._create_update_model()
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


@settings(max_examples=20)
@given(
    dist_type=st.sampled_from(["studentt", "normal"]),
    n_features=st.integers(min_value=2, max_value=4),
    cardinality=st.integers(min_value=2, max_value=6),
    hidden_dim_list=st.lists(st.integers(min_value=1, max_value=2), min_size=0, max_size=1),
    n_samples=st.just(2),
    epochs=st.just(1),
)
def test_bnn_vi_update_with_categorical_features_updates_embeddings(
    dist_type, n_features, cardinality, hidden_dim_list, n_samples, epochs
):
    cat_col = n_features - 1
    categorical_features = {cat_col: cardinality}
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        categorical_features=categorical_features,
        hidden_dim_list=hidden_dim_list,
        dist_type=dist_type,
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
    "update_kwargs",
    [{"num_steps": 2}],
)
def test_bnn_sample_proba_and_update_both_use_forward_layers(
    rng, update_kwargs: dict, n_features: int = 1, n_samples: int = 1, ref: int = 1
) -> None:
    """Verify that both sample_proba and update call _forward_layers."""
    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        update_kwargs=update_kwargs,
    )
    context = rng.random((n_samples, n_features)).astype(np.float32)
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
    rng,
    n_features,
    hidden_dim_list,
    dist_type,
    n_samples,
    sigma_init,
    nu,
    epochs,
    n_predictive_samples,
    n_sigma_tolerance,
):
    """Stored (mu, sigma) from extract_advi_params must match guide posterior sample moments.

    AutoNormal draws each site from Normal(loc, scale), so posterior predictive
    sample mean/std must recover the extracted loc/scale up to sampling noise.
    sigma_init is bounded so SE = sigma/sqrt(n_predictive_samples) stays far below tolerance.
    """
    dist_params_init = {"sigma": sigma_init} if dist_type == "normal" else {"sigma": sigma_init, "nu": nu}
    context = np.random.default_rng(0).standard_normal((n_samples, n_features)).astype(np.float32)
    rewards = rng.choice([0, 1], size=n_samples).tolist()

    bnn = BayesianNeuralNetwork.cold_start(
        n_features=n_features,
        hidden_dim_list=hidden_dim_list,
        update_kwargs={"epochs": epochs, "method": "advi"},
        dist_type=dist_type,
        dist_params_init=dist_params_init,
    )

    x_jnp = jnp.array(context)
    y_jnp = jnp.array(rewards, dtype=jnp.int32)
    _, guide, params = bnn._run_svi_training_loop(x_jnp, y_jnp, n_samples)
    site_mu, site_sigma = bnn.extract_advi_params(params)

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
    rng, n_features, hidden_dim_list, dist_type, n_samples, mu_init, sigma_init, nu, num_steps, lr
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
        update_kwargs={
            "num_steps": num_steps,
            "method": "advi",
            "optimizer_type": "sgd",
            "optimizer_kwargs": {"step_size": lr},
        },
        dist_type=dist_type,
        dist_params_init=dist_params_init,
    )

    context = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    rewards = rng.choice([0, 1], size=n_samples).tolist()
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
