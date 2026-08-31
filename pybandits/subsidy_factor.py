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

"""
Automatic discovery of the cost-control ``subsidy_factor``.

A :class:`~pybandits.strategy.single_objective.CostControlBandit` selects, per round, the
cheapest action whose sampled reward is within ``(1 - subsidy_factor)`` of the best sampled
reward. ``subsidy_factor`` therefore trades reward for cost: ``0`` behaves like a classic bandit
(maximum reward), ``1`` minimises cost. Picking it by hand means sweeping the factor on the
current model and eyeballing the knee of the mean-reward curve - the point past which reward
damage accelerates. This module automates that eyeball.

The mean reward as a function of ``subsidy_factor`` is estimated by replaying the fitted
posterior: for every candidate factor each Monte-Carlo draw is fed to the strategy's own
``select_action``, and the selection is credited with an *independent* posterior draw of its
expected reward. Using an independent draw for the credit (rather than the draw that drove the
selection) removes Thompson-sampling optimism bias and yields a single code path that works for
discrete, quantitative, and mixed action sets - cost only ever enters through ``select_action``,
never the decision. The knee of that curve is then located robustly by bootstrapping the draws
and taking the median knee.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from pydantic import NonNegativeInt, PositiveInt, confloat, validate_call

from pybandits.base import ActionId, UnifiedActionId
from pybandits.cmab import BaseCmabBernoulli
from pybandits.mab import BaseMab
from pybandits.strategy import CostControlBandit, MultiObjectiveCostControlBandit

# A single Monte-Carlo draw maps each action to its sampled reward: a scalar for discrete actions
# or a callable ``quantity -> reward`` for quantitative actions.
ProbDraw = dict[ActionId, float | Callable[[np.ndarray], float]]


class SubsidyFactorTuningResult(NamedTuple):
    """
    Result of :func:`tune_subsidy_factor`.

    Attributes
    ----------
    subsidy_factor : float
        The discovered subsidy factor (bootstrap-median knee of the mean-reward curve).
    subsidy_factor_ci : tuple[float, float]
        Bootstrap confidence interval for the knee, at the requested confidence level.
    mab : BaseMab
        A copy of the input bandit with ``strategy.with_subsidy_factor(subsidy_factor)`` applied.
    frontier : pd.DataFrame
        The full sweep, one row per candidate factor, with columns ``subsidy_factor``,
        ``mean_reward``, ``mean_reward_pct`` (reward as a percentage of the ``subsidy_factor=0``
        reward) and one ``p_select[<action_id>]`` column per action giving its selection
        frequency. Lets the caller reproduce the wiki-style plot and override the auto-pick.
    """

    subsidy_factor: float
    subsidy_factor_ci: tuple[float, float]
    mab: BaseMab
    frontier: pd.DataFrame


def _detect_knee(subsidy_factors: np.ndarray, rewards: np.ndarray) -> float:
    """
    Locate the knee of a reward-vs-subsidy_factor curve (Kneedle, distance-from-chord variant).

    Both axes are normalised to ``[0, 1]`` and the knee is the point of maximum distance *above*
    the chord joining the curve's endpoints - the corner of the flat-then-declining reward curve.

    Parameters
    ----------
    subsidy_factors : np.ndarray
        Candidate subsidy factors, ascending.
    rewards : np.ndarray
        Mean reward at each subsidy factor.

    Returns
    -------
    float
        The subsidy factor at the knee. Falls back to ``subsidy_factors[0]`` for a flat curve.
    """
    x = np.asarray(subsidy_factors, dtype=float)
    y = np.asarray(rewards, dtype=float)
    x_range = x[-1] - x[0]
    y_range = y.max() - y.min()
    if x_range == 0 or y_range == 0:  # degenerate / flat reward - no knee to speak of
        return float(x[0])
    x_n = (x - x[0]) / x_range
    y_n = (y - y.min()) / y_range
    chord = y_n[0] + (y_n[-1] - y_n[0]) * x_n
    distance = y_n - chord
    return float(x[int(np.argmax(distance))])


def _sample_prob_draws(mab: BaseMab, context: np.ndarray | None, n_samples: int, contextual: bool) -> list[ProbDraw]:
    """
    Draw Monte-Carlo posterior samples as a flat list of per-action reward dicts.

    For a non-contextual bandit this is ``n_samples`` draws. For a contextual bandit each posterior
    pass produces one draw per context row, so the result holds ``n_samples * len(context)`` draws;
    the contextual ``(probability, weight)`` tuples are reduced to the probability element, matching
    what ``predict`` feeds to the strategy.

    Parameters
    ----------
    mab : BaseMab
        The bandit to sample from (its ``_rng`` is advanced).
    context : np.ndarray | None
        Context matrix for contextual bandits; ignored otherwise.
    n_samples : int
        Number of posterior draws (contextual: posterior passes over the whole context).
    contextual : bool
        Whether ``mab`` is contextual.

    Returns
    -------
    list[ProbDraw]
        One dict per draw mapping each action to its sampled reward (scalar or callable).
    """
    valid_actions = set(mab.actions.keys())
    if not contextual:
        return list(mab._get_action_probabilities(valid_actions=valid_actions, n_samples=n_samples))
    draws: list[ProbDraw] = []
    for _ in range(n_samples):
        for row in mab._get_action_probabilities(valid_actions=valid_actions, context=context):
            draws.append({a: (v[0] if isinstance(v, tuple) else v) for a, v in row.items()})
    return draws


def _credit(p_cred_draw: ProbDraw, selected: UnifiedActionId) -> float:
    """
    Evaluate the reward credited to a selection using an independent posterior draw.

    Parameters
    ----------
    p_cred_draw : ProbDraw
        An independent posterior draw (not the one used to make the selection).
    selected : UnifiedActionId
        The selected action: an ``ActionId`` for discrete actions or an ``(ActionId, quantity)``
        tuple for quantitative actions.

    Returns
    -------
    float
        The independent draw's reward for the selection.
    """
    if isinstance(selected, tuple):
        action_id, quantity = selected[0], np.asarray(selected[1])
        return float(p_cred_draw[action_id](quantity))
    return float(p_cred_draw[selected])


@validate_call(config=dict(arbitrary_types_allowed=True))
def tune_subsidy_factor(
    mab: BaseMab,
    context: np.ndarray | None = None,
    n_samples: PositiveInt = 1000,
    n_points: PositiveInt = 100,
    n_bootstrap: PositiveInt = 1000,
    confidence_level: confloat(gt=0, lt=1) = 0.9,
    random_seed: NonNegativeInt | None = None,
) -> SubsidyFactorTuningResult:
    """
    Discover a cost-control ``subsidy_factor`` from a fitted cost-control bandit.

    Sweeps ``subsidy_factor`` over ``[0, 1]``, estimates the mean reward of the resulting policy at
    each value by replaying the posterior through the strategy's ``select_action``, and returns the
    knee of that curve (the operating point past which reward damage accelerates). The knee is
    stabilised by bootstrapping the Monte-Carlo draws and taking the median knee.

    Parameters
    ----------
    mab : BaseMab
        A fitted cost-control bandit (``strategy`` must be a ``CostControlBandit``), e.g.
        ``SmabBernoulliCC`` or ``CmabBernoulliCC``. Discrete, quantitative and mixed action sets
        are supported. The input bandit is not modified.
    context : np.ndarray | None, default=None
        Context matrix of shape ``(n_rows, n_features)``. Required for contextual bandits and must
        be ``None`` otherwise. The mean reward is averaged over these rows, so they should be
        representative of the target population.
    n_samples : PositiveInt, default=1000
        Number of posterior draws used to estimate the reward curve. For a contextual bandit this is
        the number of posterior passes over ``context``; the total number of draws is
        ``n_samples * len(context)`` and the selection cost scales with ``draws * n_points``, so use
        a modest value (e.g. 20-100) with a representative context.
    n_points : PositiveInt, default=100
        Number of subsidy-factor grid points over ``[0, 1]``. Must be at least 3.
    n_bootstrap : PositiveInt, default=1000
        Number of bootstrap replicates used to stabilise the knee and form its confidence interval.
    confidence_level : float, default=0.9
        Confidence level (in ``(0, 1)``) for the knee's bootstrap interval.
    random_seed : NonNegativeInt | None, default=None
        Seed for the sampling and bootstrap generators. With a seed the result is reproducible.

    Returns
    -------
    SubsidyFactorTuningResult
        The discovered subsidy factor, its confidence interval, a bandit copy with the factor
        applied, and the full sweep as a DataFrame.

    Raises
    ------
    TypeError
        If ``mab`` is not a cost-control bandit.
    NotImplementedError
        If ``mab`` uses a multi-objective cost-control strategy (the knee needs reward
        scalarisation, which is not defined here).
    ValueError
        If ``n_points < 3``, or ``context`` is missing for a contextual bandit / provided for a
        non-contextual one.
    """
    if isinstance(mab.strategy, MultiObjectiveCostControlBandit):
        raise NotImplementedError(
            "Multi-objective cost control is not supported: the reward is a vector, so the knee "
            "needs a reward scalarisation choice. Use a single-objective CostControlBandit."
        )
    if not isinstance(mab.strategy, CostControlBandit):
        raise TypeError(
            "tune_subsidy_factor requires a cost-control bandit (strategy=CostControlBandit), "
            "e.g. SmabBernoulliCC or CmabBernoulliCC."
        )
    if n_points < 3:
        raise ValueError("n_points must be at least 3 to detect a knee.")
    contextual = isinstance(mab, BaseCmabBernoulli)
    if contextual and context is None:
        raise ValueError("context is required for a contextual (cMAB) bandit.")
    if not contextual and context is not None:
        raise ValueError("context must be None for a non-contextual (sMAB) bandit.")
    if contextual and context is not None and len(context) == 0:
        raise ValueError("context must not be empty for a contextual (cMAB) bandit.")

    # Analyse the pure strategy selection on an isolated copy: disable epsilon-greedy exploration
    # and use a dedicated rng so the input bandit (and its rng state) is left untouched.
    working = mab.model_copy(
        update={"epsilon": None, "default_action": None, "default_action_fraction": None}, deep=True
    )
    if random_seed is not None:
        working._rng = np.random.default_rng(random_seed)
    actions = working.actions

    # Two independent posterior sample sets: one drives the selection, the other credits it. The
    # independent credit makes the reward an unbiased estimate of the selection's expected reward.
    p_sel = _sample_prob_draws(working, context, n_samples, contextual)
    p_cred = _sample_prob_draws(working, context, n_samples, contextual)
    n_draws = min(len(p_sel), len(p_cred))

    subsidy_factors = np.linspace(0.0, 1.0, n_points)
    action_ids = list(actions.keys())
    credit = np.empty((n_draws, n_points))
    selection_counts = {action_id: np.zeros(n_points) for action_id in action_ids}
    for j, subsidy_factor in enumerate(subsidy_factors):
        strategy = working.strategy.with_subsidy_factor(float(subsidy_factor))
        for d in range(n_draws):
            selected = strategy.select_action(p_sel[d], actions)
            base_action = selected[0] if isinstance(selected, tuple) else selected
            selection_counts[base_action][j] += 1
            credit[d, j] = _credit(p_cred[d], selected)

    mean_reward = credit.mean(axis=0)

    # Bootstrap the draws and take the median knee for a stable, noise-robust pick.
    boot_rng = np.random.default_rng(random_seed)
    knees = np.array(
        [
            _detect_knee(subsidy_factors, credit[boot_rng.integers(0, n_draws, n_draws)].mean(axis=0))
            for _ in range(n_bootstrap)
        ]
    )
    subsidy_factor = float(np.median(knees))
    lower = (1 - confidence_level) / 2 * 100
    upper = (1 + confidence_level) / 2 * 100
    subsidy_factor_ci = (float(np.percentile(knees, lower)), float(np.percentile(knees, upper)))

    reference = mean_reward[0] if mean_reward[0] != 0 else np.nan
    frontier = pd.DataFrame(
        {
            "subsidy_factor": subsidy_factors,
            "mean_reward": mean_reward,
            "mean_reward_pct": mean_reward / reference * 100,
        }
    )
    for action_id in action_ids:
        frontier[f"p_select[{action_id}]"] = selection_counts[action_id] / n_draws

    tuned_mab = mab.model_copy(update={"strategy": mab.strategy.with_subsidy_factor(subsidy_factor)}, deep=True)
    return SubsidyFactorTuningResult(
        subsidy_factor=subsidy_factor,
        subsidy_factor_ci=subsidy_factor_ci,
        mab=tuned_mab,
        frontier=frontier,
    )
