# MIT License
#
# Copyright (c) 2024 Playtika Ltd.
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

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pybandits.strategy.single_objective
from pybandits.cmab import CmabBernoulliCC
from pybandits.mab import BaseMab
from pybandits.smab import SmabBernoulli, SmabBernoulliCC, SmabBernoulliMOCC
from pybandits.subsidy_factor import SubsidyFactorTuningResult, _detect_knee, tune_subsidy_factor

# Three actions with a reward/cost trade-off: the priciest action is best, a much cheaper action is
# almost as good (creates an early knee), and the free action is poor.
ACTION_COSTS: dict[str, float] = {"best": 3.0, "cheap_good": 1.0, "cheap_bad": 0.0}
# action_id -> (n_successes, n_failures) fed via update, so the posteriors differ markedly.
ACTION_REWARDS: dict[str, tuple[int, int]] = {"best": (60, 40), "cheap_good": (55, 45), "cheap_bad": (20, 80)}
DEFAULT_SUBSIDY_FACTOR: float = 0.5

N_SAMPLES: int = 10
N_POINTS: int = 5
N_BOOTSTRAP: int = 3
RANDOM_SEED: int = 123
DETERMINISM_SEED: int = 7

SF_MIN: float = 0.0
SF_MAX: float = 1.0
FULL_PCT: float = 100.0
UNIT_SUM: float = 1.0
REWARD_GAP_MIN: float = 0.1
EDGE_WINDOW: int = 3

# Analytic flat-then-declining reward curve used to probe the knee detector.
PLATEAU_REWARD: float = 1.0
FLOOR_REWARD: float = 0.0
KNEE_MIN: float = 0.1
KNEE_MAX: float = 0.9
EPS: float = 1e-9

# Hypothesis ranges for the knee detector: vary grid size and reward scale.
# PLATEAU_MIN must exceed FLOOR_REWARD so the curve has non-zero y_range.
N_POINTS_MIN: int = 10
N_POINTS_MAX: int = 50
PLATEAU_MIN: float = 0.1
PLATEAU_MAX: float = 2.0

# Cheap configuration for guard / smoke tests that must not pay for a full sweep.
CC_COSTS_AB: dict[str, float] = {"a": 1.0, "b": 2.0}
NON_CC_ACTION_IDS = {"a", "b"}
N_OBJECTIVES: int = 2
CMAB_N_FEATURES: int = 2
CMAB_CONTEXT_ROWS: int = 4
CMAB_SAMPLES: int = 3
CMAB_POINTS: int = 5
CMAB_BOOTSTRAP: int = 10
CMAB_SEED: int = 0
FAST_TUNE_KWARGS: dict[str, int] = {"n_samples": 10, "n_points": 5, "n_bootstrap": 5}
MIN_POINTS_INVALID: int = 2
MISUSE_CONTEXT = np.zeros((3, CMAB_N_FEATURES))

# Quantitative action smoke-test constants.
# QUANT_SEGMENT_QUANTITIES covers all 4 initial ZoomingCC segments so failures suppress the whole space.
QUANT_ACTION_ID: str = "q"
QUANT_SEGMENT_QUANTITIES: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
QUANT_N_FAILURES: int = 5
QUANT_N_SUCCESSES_PER_ACTION: int = 50
QUANT_FAST_N_POINTS: int = FAST_TUNE_KWARGS["n_points"]
QUANT_SEED: int = 42


def QUANT_COST_CALLABLE(x: float | np.ndarray) -> float:
    """Cost function for the quantitative ZoomingCC action (average quantity)."""
    return float(np.asarray(x).mean())


@pytest.fixture
def fitted_smab_cc() -> Callable[..., SmabBernoulliCC]:
    """Factory building a cold-started ``SmabBernoulliCC`` updated into distinct per-action posteriors."""

    def _build(
        costs: dict[str, float] = ACTION_COSTS,
        rewards: dict[str, tuple[int, int]] = ACTION_REWARDS,
        subsidy_factor: float = DEFAULT_SUBSIDY_FACTOR,
        seed: int = RANDOM_SEED,
    ) -> SmabBernoulliCC:
        mab = SmabBernoulliCC.cold_start(action_ids_cost=dict(costs), subsidy_factor=subsidy_factor, random_seed=seed)
        actions, reward_list = [], []
        for action_id, (n_successes, n_failures) in rewards.items():
            actions += [action_id] * (n_successes + n_failures)
            reward_list += [1] * n_successes + [0] * n_failures
        mab.update(actions=actions, rewards=reward_list)
        return mab

    return _build


class TestDetectKnee:
    """Knee detection on a reward-vs-subsidy_factor curve."""

    @given(
        knee_x=st.floats(min_value=KNEE_MIN, max_value=KNEE_MAX),
        n_points=st.integers(min_value=N_POINTS_MIN, max_value=N_POINTS_MAX),
        plateau=st.floats(min_value=PLATEAU_MIN, max_value=PLATEAU_MAX, allow_nan=False, allow_infinity=False),
        floor=st.just(FLOOR_REWARD),
        sf_max=st.just(SF_MAX),
        eps=st.just(EPS),
    )
    def test_finds_analytic_elbow(
        self, knee_x: float, n_points: int, plateau: float, floor: float, sf_max: float, eps: float
    ) -> None:
        """Recovers the corner of a flat-then-declining reward curve (within one grid step)."""
        subsidy_factors = np.linspace(floor, sf_max, n_points)
        rewards = np.where(
            subsidy_factors <= knee_x,
            plateau,
            plateau - (subsidy_factors - knee_x) / (sf_max - knee_x) * (plateau - floor),
        )
        detected = _detect_knee(subsidy_factors, rewards)
        grid_step = (sf_max - floor) / (n_points - 1)
        assert abs(detected - knee_x) <= grid_step + eps

    @given(
        n_points=st.integers(min_value=N_POINTS_MIN, max_value=N_POINTS_MAX),
        constant=st.floats(min_value=0.0, max_value=PLATEAU_MAX, allow_nan=False, allow_infinity=False),
        floor=st.just(FLOOR_REWARD),
        sf_max=st.just(SF_MAX),
    )
    def test_flat_curve_returns_first(self, n_points: int, constant: float, floor: float, sf_max: float) -> None:
        """A flat reward curve at any constant value has no knee, so the smallest factor is returned."""
        subsidy_factors = np.linspace(floor, sf_max, n_points)
        assert _detect_knee(subsidy_factors, np.full(n_points, constant)) == subsidy_factors[0]


class TestTuneSubsidyFactor:
    """End-to-end discovery of the cost-control subsidy factor."""

    def test_end_to_end_smab_cc(
        self,
        fitted_smab_cc: Callable[..., SmabBernoulliCC],
        n_samples: int = N_SAMPLES,
        n_points: int = N_POINTS,
        n_bootstrap: int = N_BOOTSTRAP,
        seed: int = RANDOM_SEED,
        sf_min: float = SF_MIN,
        sf_max: float = SF_MAX,
        unit_sum: float = UNIT_SUM,
        reward_gap_min: float = REWARD_GAP_MIN,
        full_pct: float = FULL_PCT,
        edge_window: int = EDGE_WINDOW,
    ) -> None:
        """The full sweep yields a knee in range, a decreasing reward frontier, and an applied copy."""
        mab = fitted_smab_cc()
        original_subsidy_factor = mab.strategy.subsidy_factor

        result = tune_subsidy_factor(
            mab, n_samples=n_samples, n_points=n_points, n_bootstrap=n_bootstrap, random_seed=seed
        )

        assert isinstance(result, SubsidyFactorTuningResult)
        assert sf_min <= result.subsidy_factor <= sf_max
        lower, upper = result.subsidy_factor_ci
        assert sf_min <= lower <= result.subsidy_factor <= upper <= sf_max

        frontier = result.frontier
        assert len(frontier) == n_points
        assert frontier["subsidy_factor"].iloc[0] == sf_min
        assert frontier["subsidy_factor"].iloc[-1] == sf_max
        # One selection per draw: per-row selection probabilities sum to 1.
        select_cols = [c for c in frontier.columns if c.startswith("p_select[")]
        assert np.allclose(frontier[select_cols].sum(axis=1), unit_sum)

        rewards = frontier["mean_reward"].to_numpy()
        # Classic (sf=0) maximises reward; cost minimisation (sf=1) is materially worse here.
        assert rewards[0] == pytest.approx(rewards.max())
        assert rewards[0] - rewards[-1] > reward_gap_min
        assert rewards[:edge_window].mean() >= rewards[-edge_window:].mean()
        assert frontier["mean_reward_pct"].iloc[0] == pytest.approx(full_pct)

        # The discovered factor is applied to the returned copy; the input bandit is untouched.
        assert result.mab.strategy.subsidy_factor == result.subsidy_factor
        assert mab.strategy.subsidy_factor == original_subsidy_factor

    def test_deterministic_with_seed(
        self,
        fitted_smab_cc: Callable[..., SmabBernoulliCC],
        n_samples: int = N_SAMPLES,
        n_points: int = N_POINTS,
        n_bootstrap: int = N_BOOTSTRAP,
        seed: int = DETERMINISM_SEED,
    ) -> None:
        """A fixed ``random_seed`` makes the discovered factor and frontier reproducible."""
        mab = fitted_smab_cc()
        kwargs = dict(n_samples=n_samples, n_points=n_points, n_bootstrap=n_bootstrap, random_seed=seed)

        first = tune_subsidy_factor(mab, **kwargs)
        second = tune_subsidy_factor(mab, **kwargs)

        assert first.subsidy_factor == second.subsidy_factor
        assert first.subsidy_factor_ci == second.subsidy_factor_ci
        pd.testing.assert_frame_equal(first.frontier, second.frontier)

    def test_smoke_cmab_cc(
        self,
        costs: dict[str, float] = CC_COSTS_AB,
        n_features: int = CMAB_N_FEATURES,
        n_rows: int = CMAB_CONTEXT_ROWS,
        n_samples: int = CMAB_SAMPLES,
        n_points: int = CMAB_POINTS,
        n_bootstrap: int = CMAB_BOOTSTRAP,
        seed: int = CMAB_SEED,
        sf_min: float = SF_MIN,
        sf_max: float = SF_MAX,
    ) -> None:
        """A contextual cost-control bandit produces a valid frontier when given a context."""
        cmab = CmabBernoulliCC.cold_start(action_ids_cost=dict(costs), n_features=n_features, random_seed=seed)
        context = np.random.default_rng(seed).normal(size=(n_rows, n_features))

        result = tune_subsidy_factor(
            cmab, context=context, n_samples=n_samples, n_points=n_points, n_bootstrap=n_bootstrap, random_seed=seed
        )

        assert sf_min <= result.subsidy_factor <= sf_max
        assert len(result.frontier) == n_points

    def test_smoke_quantitative_action(
        self,
        quant_action_id: str = QUANT_ACTION_ID,
        discrete_costs: dict[str, float] = CC_COSTS_AB,
        quant_cost: Callable[[float | np.ndarray], float] = QUANT_COST_CALLABLE,
        segment_quantities: list[float] = QUANT_SEGMENT_QUANTITIES,
        n_failures: int = QUANT_N_FAILURES,
        n_successes_per_action: int = QUANT_N_SUCCESSES_PER_ACTION,
        n_fast_points: int = QUANT_FAST_N_POINTS,
        seed: int = QUANT_SEED,
        sf_min: float = SF_MIN,
        sf_max: float = SF_MAX,
    ) -> None:
        """A cost-control bandit with a quantitative action produces a valid result."""
        mab = SmabBernoulliCC.cold_start(
            action_ids_cost=dict(discrete_costs),
            quantitative_action_ids_cost={quant_action_id: quant_cost},
            subsidy_factor=DEFAULT_SUBSIDY_FACTOR,
            random_seed=seed,
        )
        # Ground posteriors: cover all initial ZoomingCC segments with failures so the
        # discrete actions dominate. Cold-start ZoomingCC over-optimism otherwise makes the
        # cost-control threshold unreachable for discrete actions at subsidy_factor=0.
        discrete_action_ids = list(discrete_costs.keys())
        mab.update(actions=[quant_action_id] * n_failures, rewards=[0] * n_failures, quantities=segment_quantities)
        mab.update(
            actions=discrete_action_ids * n_successes_per_action,
            rewards=[1] * (len(discrete_action_ids) * n_successes_per_action),
            quantities=[None] * (len(discrete_action_ids) * n_successes_per_action),
        )
        with patch.object(
            pybandits.strategy.single_objective,
            "maximize_by_quantity",
            lambda score_func, dimension, constraint=None, **kwargs: np.zeros(dimension),
        ):
            result = tune_subsidy_factor(mab, **FAST_TUNE_KWARGS, random_seed=seed)
        assert sf_min <= result.subsidy_factor <= sf_max
        assert len(result.frontier) == n_fast_points

    @pytest.mark.parametrize(
        "build_mab, call_kwargs, expected_exception",
        [
            pytest.param(
                lambda: SmabBernoulli.cold_start(action_ids=NON_CC_ACTION_IDS),
                FAST_TUNE_KWARGS,
                TypeError,
                id="non_cost_control",
            ),
            pytest.param(
                lambda: SmabBernoulliMOCC.cold_start(action_ids_cost=CC_COSTS_AB, n_objectives=N_OBJECTIVES),
                FAST_TUNE_KWARGS,
                NotImplementedError,
                id="multi_objective_cost_control",
            ),
            pytest.param(
                lambda: CmabBernoulliCC.cold_start(action_ids_cost=CC_COSTS_AB, n_features=CMAB_N_FEATURES),
                {},
                ValueError,
                id="contextual_without_context",
            ),
            pytest.param(
                lambda: SmabBernoulliCC.cold_start(action_ids_cost=CC_COSTS_AB),
                {**FAST_TUNE_KWARGS, "context": MISUSE_CONTEXT},
                ValueError,
                id="context_for_non_contextual",
            ),
            pytest.param(
                lambda: SmabBernoulliCC.cold_start(action_ids_cost=CC_COSTS_AB),
                {**FAST_TUNE_KWARGS, "n_points": MIN_POINTS_INVALID},
                ValueError,
                id="too_few_grid_points",
            ),
            pytest.param(
                lambda: CmabBernoulliCC.cold_start(action_ids_cost=CC_COSTS_AB, n_features=CMAB_N_FEATURES),
                {**FAST_TUNE_KWARGS, "context": np.zeros((0, CMAB_N_FEATURES))},
                ValueError,
                id="contextual_empty_context",
            ),
        ],
    )
    def test_rejects_invalid_inputs(
        self, build_mab: Callable[[], BaseMab], call_kwargs: dict[str, object], expected_exception: type[Exception]
    ) -> None:
        """Misconfigured bandits and arguments raise before any sampling work is done."""
        mab = build_mab()
        with pytest.raises(expected_exception):
            tune_subsidy_factor(mab, **call_kwargs)
