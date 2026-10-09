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
"""Tests for the Gaussian (continuous-reward) BNN, its meta-model wiring and the CmabGaussian bandit."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pybandits.cmab import CmabBernoulli, CmabGaussian
from pybandits.meta_model import CmabMetaModel
from pybandits.model import BayesianNeuralNetwork, GaussianBayesianNeuralNetwork

# --- test configuration constants ---
N_FEATURES = 3
HIDDEN_DIMS = [8]
OUTPUT_DIM = 2
RANDOM_SEED = 0
N_TRAIN = 2000
N_POSTERIOR_SAMPLES = 200
# Narrow prior + larger step size: the setting the class docstring recommends for a fast fit.
FAST_FIT_KWARGS: Dict[str, Any] = {
    "dist_type": "normal",
    "dist_params_init": {"mu": 0, "sigma": 0.1},
    "update_kwargs": {"num_steps": 1000, "optimizer_kwargs": {"step_size": 3e-3}},
}
SHORT_FIT_KWARGS: Dict[str, Any] = {"update_kwargs": {"num_steps": 5}}
# Synthetic reward: REWARD_OFFSET + REWARD_SLOPE * x0 + noise, noise std LOW / HIGH for x1 <= 0 / x1 > 0.
REWARD_OFFSET = 100.0
REWARD_SLOPE = 30.0
NOISE_STD_LOW = 5.0
NOISE_STD_HIGH = 15.0
# Relative tolerance on the recovered mean / noise std at the probe points.
MEAN_REL_TOL = 0.1
SIGMA_REL_TOL = 0.5
ACTION_IDS = {"a", "b", "c"}
ARM_MEANS = {"a": -1.0, "b": 0.0, "c": 2.0}
BEST_ARM = "c"
N_ROUNDS = 4
N_PER_ROUND = 200
MIN_BEST_ARM_SHARE = 0.8
# Probe contexts: (x0, x1) in {-1, 1}^2 hit both mean levels and both noise regions.
PROBE_CONTEXTS = [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]
ROUND_TRIP_TOL = 1e-6
MAX_ABS_REWARD = 1e6
MAX_REWARDS = 50
BACKBONE_KWARGS: Dict[str, Any] = {"backbone_hidden_dims": [8], "backbone_embedding_dim": 3}
MINIBATCH_SIZE = 64
BAD_REWARDS = [float("nan"), float("inf"), -float("inf")]
OUT_OF_RANGE_SOFT_REWARD = 2.5
DELTA = 0.1
OTHER_SIGMA_MIN = 0.2
EPSILON = 0.1
HOMOSCEDASTIC_OUTPUT_DIM = 1
NOISE_SIGMA = 3.0
DECAY_FACTOR = 0.5
# Zero-inflated, skewed reward (as revenue): a buy with probability BUY_PROB, then a log-normal amount, so the mean
# is SKEW_OFFSET + SKEW_SLOPE * x0. The noise grows with the mean, which a learned sigma(x) uses to shrink mu.
BUY_PROB = 0.15
SKEW_OFFSET = 2.0
SKEW_SLOPE = 1.5
SKEW_LOG_SD = 1.0
LEVEL_REL_TOL = 0.15


def _skewed_rewards(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Zero-inflated log-normal rewards with mean SKEW_OFFSET + SKEW_SLOPE * x0 (x0 clipped to keep it positive)."""
    mean = SKEW_OFFSET + SKEW_SLOPE * np.clip(x[:, 0], -1.0, 2.0)
    amount = rng.lognormal(np.log(mean / BUY_PROB) - SKEW_LOG_SD**2 / 2, SKEW_LOG_SD)
    return (rng.uniform(size=len(x)) < BUY_PROB) * amount


def _heteroscedastic_rewards(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Linear-in-x0 rewards whose noise std switches with the sign of x1."""
    noise_std = np.where(x[:, 1] > 0, NOISE_STD_HIGH, NOISE_STD_LOW)
    return REWARD_OFFSET + REWARD_SLOPE * x[:, 0] + rng.normal(size=len(x)) * noise_std


@pytest.fixture(scope="module")
def make_gaussian_bnn() -> Callable[..., GaussianBayesianNeuralNetwork]:
    """Factory: cold-starts a GaussianBayesianNeuralNetwork with the module's shape and extra kwargs."""

    def _factory(**kwargs: Any) -> GaussianBayesianNeuralNetwork:
        return GaussianBayesianNeuralNetwork.cold_start(
            n_features=N_FEATURES, hidden_dim_list=HIDDEN_DIMS, random_seed=RANDOM_SEED, **kwargs
        )

    return _factory


@pytest.fixture(scope="module")
def fitted_gaussian_bnn(
    make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
) -> GaussianBayesianNeuralNetwork:
    """A Gaussian BNN fitted on heteroscedastic synthetic rewards."""
    data_rng = np.random.default_rng(RANDOM_SEED)
    bnn = make_gaussian_bnn(**FAST_FIT_KWARGS)
    x = data_rng.normal(size=(N_TRAIN, N_FEATURES))
    bnn.update(context=x, rewards=_heteroscedastic_rewards(x, data_rng).tolist())
    return bnn


@pytest.fixture(scope="module")
def make_cmab_gaussian() -> Callable[..., CmabGaussian]:
    """Factory: cold-starts a CmabGaussian over ACTION_IDS with the fast-fit prior and extra kwargs."""

    def _factory(**kwargs: Any) -> CmabGaussian:
        return CmabGaussian.cold_start(
            action_ids=ACTION_IDS,
            n_features=N_FEATURES,
            hidden_dim_list=HIDDEN_DIMS,
            random_seed=RANDOM_SEED,
            **{**FAST_FIT_KWARGS, **kwargs},
        )

    return _factory


class TestGaussianBayesianNeuralNetwork:
    """Tests for the standalone Gaussian BNN."""

    def test_output_layer_has_mean_and_noise_units(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        """The output layer has one unit for the mean and one for the noise std."""
        bnn = make_gaussian_bnn()
        output_layer = bnn.model_params.bnn_layer_params[-1]
        assert output_layer.weight.shape[-1] == output_dim
        assert output_layer.bias.shape == (output_dim,)

    def test_sample_proba_returns_mean_and_positive_noise(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
    ) -> None:
        """Each sample is a (mu, sigma) pair with sigma above the configured floor."""
        bnn = make_gaussian_bnn()
        samples = bnn.sample_proba(context=rng.normal(size=(n_samples, n_features)), rng=rng)
        assert len(samples) == n_samples
        sigmas = np.array([sigma for _, sigma in samples])
        assert np.all(sigmas >= bnn.sigma_min)

    def test_fit_recovers_mean_and_heteroscedastic_noise(
        self,
        fitted_gaussian_bnn: GaussianBayesianNeuralNetwork,
        rng: np.random.Generator,
        n_posterior_samples: int = N_POSTERIOR_SAMPLES,
        offset: float = REWARD_OFFSET,
        slope: float = REWARD_SLOPE,
        noise_low: float = NOISE_STD_LOW,
        noise_high: float = NOISE_STD_HIGH,
        mean_rel_tol: float = MEAN_REL_TOL,
        sigma_rel_tol: float = SIGMA_REL_TOL,
        probe_contexts: List[List[float]] = PROBE_CONTEXTS,
    ) -> None:
        """The posterior mean tracks the linear trend and the noise head separates low / high noise regions."""
        probes = np.array(probe_contexts)
        true_mu = offset + slope * probes[:, 0]
        true_sigma = np.where(probes[:, 1] > 0, noise_high, noise_low)
        draws = np.array(
            [fitted_gaussian_bnn.sample_proba(context=probes, rng=rng) for _ in range(n_posterior_samples)]
        )
        mu, sigma = draws[..., 0].mean(axis=0), draws[..., 1].mean(axis=0)
        np.testing.assert_allclose(mu, true_mu, rtol=mean_rel_tol)
        high_noise = probes[:, 1] > 0
        np.testing.assert_allclose(sigma[high_noise], true_sigma[high_noise], rtol=sigma_rel_tol)
        # VI pulls the noise head towards the pooled noise level, so the low-noise region is only required to
        # sit clearly below it (a homoscedastic fit would put every probe at the pooled std).
        pooled_std = np.sqrt((noise_low**2 + noise_high**2) / 2)
        assert np.all(sigma[~high_noise] < pooled_std)

    def test_counters_track_observations(
        self,
        fitted_gaussian_bnn: GaussianBayesianNeuralNetwork,
        n_train: int = N_TRAIN,
    ) -> None:
        """count / mean come from the observed rewards; the Beta pseudo-counts are untouched."""
        assert fitted_gaussian_bnn.count == n_train
        assert fitted_gaussian_bnn.mean == pytest.approx(fitted_gaussian_bnn.reward_sum / n_train)
        assert fitted_gaussian_bnn.n_successes == fitted_gaussian_bnn._prior_pseudo_count
        assert fitted_gaussian_bnn.n_failures == fitted_gaussian_bnn._prior_pseudo_count

    @given(
        rewards=st.lists(
            st.floats(min_value=-MAX_ABS_REWARD, max_value=MAX_ABS_REWARD, allow_nan=False),
            min_size=2,
            max_size=MAX_REWARDS,
            unique=True,
        ),
        n_features=st.just(N_FEATURES),
        tol=st.just(ROUND_TRIP_TOL),
    )
    def test_standardization_is_fitted_once(self, rewards: List[float], n_features: int, tol: float) -> None:
        """The first batch fixes reward_loc / reward_scale; later batches are mapped with the same values."""
        bnn = GaussianBayesianNeuralNetwork.cold_start(n_features=n_features)
        targets = bnn.prepare_rewards(rewards)
        loc, scale = bnn.reward_loc, bnn.reward_scale
        assert loc == pytest.approx(np.mean(rewards))
        np.testing.assert_allclose(targets * scale + loc, rewards, rtol=tol, atol=tol * scale)
        bnn.prepare_rewards([value + scale for value in rewards])
        assert (bnn.reward_loc, bnn.reward_scale) == (loc, scale)

    @given(
        reward=st.floats(min_value=-MAX_ABS_REWARD, max_value=MAX_ABS_REWARD, allow_nan=False),
        n_features=st.just(N_FEATURES),
    )
    def test_single_reward_standardization_has_unit_or_larger_scale(self, reward: float, n_features: int) -> None:
        """A zero-spread first batch falls back to max(|reward|, 1) instead of dividing by ~0."""
        bnn = GaussianBayesianNeuralNetwork.cold_start(n_features=n_features)
        bnn.prepare_rewards([reward])
        assert bnn.reward_scale == max(abs(reward), 1.0)

    @pytest.mark.parametrize("bad_reward", BAD_REWARDS)
    def test_update_rejects_non_finite_rewards(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        bad_reward: float,
        n_features: int = N_FEATURES,
    ) -> None:
        """NaN / inf rewards are rejected."""
        bnn = make_gaussian_bnn()
        with pytest.raises(ValidationError):
            bnn.update(context=np.zeros((1, n_features)), rewards=[bad_reward])

    @pytest.mark.parametrize("reward_loc, reward_scale", [(1.0, None), (None, 1.0)])
    def test_rejects_half_specified_standardization(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        reward_loc: Optional[float],
        reward_scale: Optional[float],
    ) -> None:
        """reward_loc and reward_scale must be given together."""
        with pytest.raises(ValidationError):
            make_gaussian_bnn(reward_loc=reward_loc, reward_scale=reward_scale)

    def test_calibrate_output_bias_sets_mean_and_noise_bias(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        tol: float = ROUND_TRIP_TOL,
    ) -> None:
        """Calibration puts the output biases at the standardized target mean and inverse-softplus std."""
        bnn = make_gaussian_bnn(calibrate_output_bias=True)
        rewards = rng.normal(size=n_samples).tolist()
        bnn._calibrate_output_bias(rewards)
        targets = bnn.prepare_rewards(rewards)
        mu_bias, raw_sigma_bias = bnn.model_params.bnn_layer_params[-1].bias.params["mu"]
        assert bnn.bias_calibrated
        assert mu_bias == pytest.approx(targets.mean(), abs=tol)
        assert np.logaddexp(0, raw_sigma_bias) + bnn.sigma_min == pytest.approx(targets.std(), rel=tol)

    def test_reset_clears_counters_and_keeps_standardization(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """reset() zeroes the reward counters and the network, but keeps the fitted standardization."""
        bnn = make_gaussian_bnn(**short_fit_kwargs)
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        loc, scale = bnn.reward_loc, bnn.reward_scale
        bnn.reset()
        assert bnn.n_observations == bnn.reward_sum == 0
        assert (bnn.reward_loc, bnn.reward_scale) == (loc, scale)
        assert bnn.model_params.bnn_layer_params == bnn.model_params.bnn_layer_params_init

    def test_serialization_round_trip(self, fitted_gaussian_bnn: GaussianBayesianNeuralNetwork) -> None:
        """The JSON state round-trips, including the standardization and counters."""
        restored = GaussianBayesianNeuralNetwork.model_validate_json(fitted_gaussian_bnn.model_dump_json())
        assert restored.reward_loc == fitted_gaussian_bnn.reward_loc
        assert restored.reward_scale == fitted_gaussian_bnn.reward_scale
        assert restored.n_observations == fitted_gaussian_bnn.n_observations
        assert restored.model_params == fitted_gaussian_bnn.model_params


class TestGaussianMetaModel:
    """Tests for Gaussian heads in CmabMetaModel."""

    def test_rejects_mixed_likelihoods(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        n_features: int = N_FEATURES,
        hidden_dims: List[int] = HIDDEN_DIMS,
    ) -> None:
        """Bernoulli and Gaussian heads cannot share one meta-model."""
        bernoulli = BayesianNeuralNetwork.cold_start(n_features=n_features, hidden_dim_list=hidden_dims)
        with pytest.raises(AttributeError):
            CmabMetaModel(actions={"a": make_gaussian_bnn(), "b": bernoulli})

    def test_rejects_different_sigma_min(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        sigma_min: float = OTHER_SIGMA_MIN,
    ) -> None:
        """Gaussian heads must share sigma_min, which the joint likelihood takes from one head."""
        with pytest.raises(AttributeError):
            CmabMetaModel(actions={"a": make_gaussian_bnn(), "b": make_gaussian_bnn(sigma_min=sigma_min)})


class TestCmabGaussian:
    """Tests for the CmabGaussian bandit."""

    @pytest.mark.parametrize(
        "extra_kwargs",
        [
            {},
            {**BACKBONE_KWARGS, "update_kwargs": {**FAST_FIT_KWARGS["update_kwargs"], "batch_size": MINIBATCH_SIZE}},
        ],
        ids=["no_backbone", "backbone_minibatch"],
    )
    def test_learns_best_arm(
        self,
        make_cmab_gaussian: Callable[..., CmabGaussian],
        extra_kwargs: Dict[str, Any],
        rng: np.random.Generator,
        arm_means: Dict[str, float] = ARM_MEANS,
        best_arm: str = BEST_ARM,
        n_rounds: int = N_ROUNDS,
        n_per_round: int = N_PER_ROUND,
        n_features: int = N_FEATURES,
        min_share: float = MIN_BEST_ARM_SHARE,
    ) -> None:
        """After a few rounds of real-valued feedback (incl. negative rewards) the best-mean arm dominates."""
        mab = make_cmab_gaussian(**extra_kwargs)
        for _ in range(n_rounds):
            context = rng.normal(size=(n_per_round, n_features))
            actions, mus, sigmas = mab.predict(context=context)
            rewards = [arm_means[a] + rng.normal() for a in actions]
            mab.update(actions=actions, rewards=rewards, context=context)
        actions, mus, sigmas = mab.predict(context=rng.normal(size=(n_per_round, n_features)))
        assert np.mean(np.array(actions) == best_arm) >= min_share
        assert all(sigma > 0 for row in sigmas for sigma in row.values())

    def test_state_round_trip(
        self,
        make_cmab_gaussian: Callable[..., CmabGaussian],
        rng: np.random.Generator,
        n_per_round: int = N_PER_ROUND,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """The bandit state round-trips through get_state / from_state."""
        mab = make_cmab_gaussian(**short_fit_kwargs)
        context = rng.normal(size=(n_per_round, n_features))
        actions, _, _ = mab.predict(context=context)
        mab.update(actions=actions, rewards=rng.normal(size=n_per_round).tolist(), context=context)
        restored = CmabGaussian.from_state(mab.get_state()[1])
        assert restored.actions == mab.actions

    @pytest.mark.parametrize("bad_reward", BAD_REWARDS)
    def test_rejects_non_finite_rewards(
        self,
        make_cmab_gaussian: Callable[..., CmabGaussian],
        bad_reward: float,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """NaN / inf rewards are rejected at the bandit entry point."""
        mab = make_cmab_gaussian(**short_fit_kwargs)
        with pytest.raises(ValueError):
            mab.update(actions=[next(iter(mab.actions))], rewards=[bad_reward], context=np.zeros((1, n_features)))

    def test_rejects_adaptive_window(
        self,
        action_ids: set = ACTION_IDS,
        n_features: int = N_FEATURES,
        delta: float = DELTA,
        epsilon: float = EPSILON,
    ) -> None:
        """The adaptive window assumes binary rewards, so delta is refused."""
        with pytest.raises(ValueError):
            CmabGaussian.cold_start(action_ids=action_ids, n_features=n_features, delta=delta, epsilon=epsilon)


@pytest.mark.parametrize("use_soft_rewards", [False, True])
def test_bernoulli_bandit_still_rejects_out_of_range_rewards(
    use_soft_rewards: bool,
    action_ids: set = ACTION_IDS,
    n_features: int = N_FEATURES,
    reward: float = OUT_OF_RANGE_SOFT_REWARD,
    short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
) -> None:
    """Widening the entry-point reward type must not let a Bernoulli bandit accept rewards outside [0, 1]."""
    mab = CmabBernoulli.cold_start(
        action_ids=action_ids, n_features=n_features, use_soft_rewards=use_soft_rewards, **short_fit_kwargs
    )
    with pytest.raises(ValueError):
        mab.update(actions=[next(iter(mab.actions))], rewards=[reward], context=np.zeros((1, n_features)))


class TestHomoscedasticGaussian:
    """Tests for the homoscedastic Gaussian BNN (one mean unit, a learned scalar noise latent)."""

    def test_output_layer_has_only_the_mean_unit(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        output_dim: int = HOMOSCEDASTIC_OUTPUT_DIM,
    ) -> None:
        """The output layer has a single unit and the noise latent is unset before the first update."""
        bnn = make_gaussian_bnn(homoscedastic=True)
        assert bnn.model_params.bnn_layer_params[-1].weight.shape[-1] == output_dim
        assert bnn.noise_log_sigma is None

    @given(
        rewards=st.lists(
            st.floats(min_value=-MAX_ABS_REWARD, max_value=MAX_ABS_REWARD, allow_nan=False),
            min_size=2,
            max_size=MAX_REWARDS,
            unique=True,
        ),
        n_features=st.just(N_FEATURES),
        tol=st.just(ROUND_TRIP_TOL),
    )
    def test_noise_initialized_from_first_batch(self, rewards: List[float], n_features: int, tol: float) -> None:
        """Without noise_sigma, the log-sigma prior is centered on the first batch's std (1 if it has no spread)."""
        bnn = GaussianBayesianNeuralNetwork.cold_start(n_features=n_features, homoscedastic=True)
        targets = bnn.prepare_rewards(rewards)
        expected = targets.std() if targets.std() > bnn._numerical_eps else 1.0
        assert np.exp(bnn.noise_log_sigma.params["mu"][0]) == pytest.approx(expected, rel=tol)

    def test_noise_initialized_from_noise_sigma(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        noise_sigma: float = NOISE_SIGMA,
        n_samples: int = N_TRAIN,
        tol: float = ROUND_TRIP_TOL,
    ) -> None:
        """noise_sigma (reward units) sets the prior once the reward scale is known."""
        bnn = make_gaussian_bnn(homoscedastic=True, noise_sigma=noise_sigma)
        bnn.prepare_rewards(rng.normal(size=n_samples).tolist())
        prior_sigma = np.exp(bnn.noise_log_sigma.params["mu"][0]) * bnn.reward_scale
        assert prior_sigma == pytest.approx(noise_sigma, rel=tol)
        assert bnn.noise_log_sigma.params["sigma"][0] == pytest.approx(bnn.noise_log_sigma_prior_std, rel=tol)

    def test_noise_posterior_is_carried_across_updates(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """Each update replaces the stored noise posterior, and the next update starts from it."""
        bnn = make_gaussian_bnn(homoscedastic=True, **short_fit_kwargs)
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        first = bnn.noise_log_sigma
        assert first is not None
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        assert bnn.noise_log_sigma != first
        _, site_sigmas, _, _ = bnn.collect_guide_init_arrays()
        np.testing.assert_array_equal(site_sigmas[bnn._noise_var_name], bnn.noise_log_sigma.params["sigma"])

    def test_sample_proba_noise_is_shared_across_contexts(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """The predicted noise std doesn't depend on the context: its spread comes only from the log-sigma posterior."""
        bnn = make_gaussian_bnn(homoscedastic=True, **short_fit_kwargs)
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        sigmas = np.array([s for _, s in bnn.sample_proba(context=rng.normal(size=(n_samples, n_features)), rng=rng)])
        log_sigma_sd = bnn.noise_log_sigma.params["sigma"][0]
        assert np.all(sigmas > 0)
        assert np.log(sigmas).std() == pytest.approx(log_sigma_sd, rel=SIGMA_REL_TOL)

    def test_mean_is_level_calibrated_on_skewed_rewards(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        n_posterior_samples: int = N_POSTERIOR_SAMPLES,
        rel_tol: float = LEVEL_REL_TOL,
        fast_fit_kwargs: Dict[str, Any] = FAST_FIT_KWARGS,
    ) -> None:
        """On zero-inflated skewed rewards, the average posterior mean matches the average reward (least squares)."""
        data_rng = np.random.default_rng(RANDOM_SEED)
        x = data_rng.normal(size=(n_samples, n_features))
        y = _skewed_rewards(x, data_rng)
        bnn = make_gaussian_bnn(homoscedastic=True, **fast_fit_kwargs)
        bnn.update(context=x, rewards=y.tolist())
        mu = np.mean([[m for m, _ in bnn.sample_proba(context=x, rng=rng)] for _ in range(n_posterior_samples)], axis=0)
        assert mu.mean() == pytest.approx(y.mean(), rel=rel_tol)

    def test_reset_clears_noise_and_keeps_standardization(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """reset() clears the noise latent (re-initialized on the next update) but keeps loc / scale."""
        bnn = make_gaussian_bnn(homoscedastic=True, **short_fit_kwargs)
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        loc, scale = bnn.reward_loc, bnn.reward_scale
        bnn.reset()
        assert bnn.noise_log_sigma is None
        assert (bnn.reward_loc, bnn.reward_scale) == (loc, scale)

    def test_decay_inflates_noise_posterior(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        decay_factor: float = DECAY_FACTOR,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
        tol: float = ROUND_TRIP_TOL,
    ) -> None:
        """Forgetting widens the noise posterior by 1 / decay_factor, like the weights."""
        bnn = make_gaussian_bnn(homoscedastic=True, decay_factor=decay_factor, **short_fit_kwargs)
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        before = bnn.noise_log_sigma.params["sigma"][0]
        bnn._inflate_prior_variance()
        assert bnn.noise_log_sigma.params["sigma"][0] == pytest.approx(before / decay_factor, rel=tol)

    def test_serialization_round_trip(
        self,
        make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork],
        rng: np.random.Generator,
        n_samples: int = N_TRAIN,
        n_features: int = N_FEATURES,
        short_fit_kwargs: Dict[str, Any] = SHORT_FIT_KWARGS,
    ) -> None:
        """The noise posterior survives the JSON round trip."""
        bnn = make_gaussian_bnn(homoscedastic=True, **short_fit_kwargs)
        bnn.update(context=rng.normal(size=(n_samples, n_features)), rewards=rng.normal(size=n_samples).tolist())
        restored = GaussianBayesianNeuralNetwork.model_validate_json(bnn.model_dump_json())
        assert restored.noise_log_sigma == bnn.noise_log_sigma
        assert restored.homoscedastic

    @pytest.mark.parametrize(
        "kwargs",
        [{"noise_sigma": NOISE_SIGMA}, {"noise_log_sigma": {"mu": [0.0], "sigma": [1.0]}}],
        ids=["noise_sigma", "noise_log_sigma"],
    )
    def test_heteroscedastic_rejects_noise_settings(
        self, make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork], kwargs: Dict[str, Any]
    ) -> None:
        """noise_sigma / noise_log_sigma only make sense for the homoscedastic model."""
        with pytest.raises(ValidationError):
            make_gaussian_bnn(**kwargs)

    def test_rejects_output_width_mismatch(
        self, make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork]
    ) -> None:
        """A heteroscedastic network (2 output units) can't be flagged homoscedastic."""
        bnn = make_gaussian_bnn()
        with pytest.raises(ValidationError):
            GaussianBayesianNeuralNetwork(**{**bnn.model_dump(), "homoscedastic": True})

    def test_meta_model_rejects_mixed_noise_models(
        self, make_gaussian_bnn: Callable[..., GaussianBayesianNeuralNetwork]
    ) -> None:
        """Homoscedastic and heteroscedastic heads can't share one meta-model."""
        with pytest.raises(AttributeError):
            CmabMetaModel(actions={"a": make_gaussian_bnn(), "b": make_gaussian_bnn(homoscedastic=True)})

    @pytest.mark.parametrize(
        "extra_kwargs",
        [
            {},
            {**BACKBONE_KWARGS, "update_kwargs": {**FAST_FIT_KWARGS["update_kwargs"], "batch_size": MINIBATCH_SIZE}},
        ],
        ids=["no_backbone", "backbone_minibatch"],
    )
    def test_cmab_learns_best_arm_and_per_arm_noise(
        self,
        make_cmab_gaussian: Callable[..., CmabGaussian],
        extra_kwargs: Dict[str, Any],
        rng: np.random.Generator,
        arm_means: Dict[str, float] = ARM_MEANS,
        best_arm: str = BEST_ARM,
        n_rounds: int = N_ROUNDS,
        n_per_round: int = N_PER_ROUND,
        n_features: int = N_FEATURES,
        min_share: float = MIN_BEST_ARM_SHARE,
    ) -> None:
        """The homoscedastic bandit finds the best arm, and every played arm carries its own noise posterior."""
        mab = make_cmab_gaussian(homoscedastic=True, **extra_kwargs)
        for _ in range(n_rounds):
            context = rng.normal(size=(n_per_round, n_features))
            actions, _, _ = mab.predict(context=context)
            mab.update(actions=actions, rewards=[arm_means[a] + rng.normal() for a in actions], context=context)
        actions, _, _ = mab.predict(context=rng.normal(size=(n_per_round, n_features)))
        assert np.mean(np.array(actions) == best_arm) >= min_share
        assert mab.actions[best_arm].noise_log_sigma is not None
        assert CmabGaussian.from_state(mab.get_state()[1]).actions == mab.actions
