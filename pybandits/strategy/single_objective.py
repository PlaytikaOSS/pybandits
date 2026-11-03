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

from random import random
from typing import Any, Callable, Dict, List, Optional, Self, Tuple, Union

import numpy as np
from loguru import logger
from pydantic import field_validator, validate_call

from pybandits.base import ActionId, Float01, UnifiedActionId
from pybandits.base_model import BaseModel, BaseModelDP
from pybandits.strategy.base import CostControlStrategy, SingleObjectiveStrategy
from pybandits.utils import OptimizationFailedError, maximize_by_quantity


class ClassicBandit(SingleObjectiveStrategy):
    """
    Classic Thompson Sampling strategy for multi-armed bandits.

    This strategy implements pure exploitation by always selecting the action
    with the highest sampled probability of reward. It considers all actions
    without any filtering or cost considerations.

    References
    ----------
    Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
    http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal and Goyal, 2014)
    https://arxiv.org/pdf/1209.3352.pdf
    """

    def get_prerequisites(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute prerequisites for classic bandit strategy.

        Classic bandits don't require any prerequisites as they consider
        all actions equally without additional filtering criteria.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary mapping action IDs to probability functions or values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions (unused in classic bandit).
        forbidden_regions : Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]], default=None
            Per-arm feasibility constraints (unused in classic bandit).

        Returns
        -------
        Dict[str, Any]
            Empty dictionary as no prerequisites are needed.
        """
        return {}

    def _verify_action(self, score: float) -> bool:
        """
        Verify if an action should be considered for selection.

        Classic bandits consider all actions regardless of their scores.

        Parameters
        ----------
        score : float
            The probability or score of the action (unused).

        Returns
        -------
        bool
            Always True - all actions are considered in classic bandits.
        """
        return True

    def _verify_and_select_from_quantitative_action(
        self,
        score_func: Callable[[np.ndarray], float],
        model: BaseModel,
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        rng: Optional[Any] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        Find optimal quantity for a quantitative action.

        Classic bandits maximize the score function to find the best quantity
        vector for quantitative actions.

        Parameters
        ----------
        score_func : Callable[[np.ndarray], float]
            Function that computes probability given a quantity vector.
        model : BaseModel
            The model associated with this quantitative action.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions that quantity must satisfy.
        rng : Optional[Any], default=None
            Random generator passed to the optimizer for reproducibility.

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector that maximizes the score function, or None if optimization fails.
        """
        try:
            return maximize_by_quantity(score_func, model.dimension, constraint_list, seed=rng)
        except OptimizationFailedError as e:
            logger.warning(f"Optimization failed: {e}")
            return None

    def _select_from_refined_actions(
        self,
        refined_p: Dict[UnifiedActionId, float],
        actions: Dict[ActionId, BaseModel],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """
        Select the action with the highest probability.

        This implements pure exploitation by choosing the action with the
        maximum sampled reward probability.

        Parameters
        ----------
        refined_p : Dict[UnifiedActionId, float]
            Dictionary of unified action IDs to their probability values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models (unused).
        constraint : Optional[Callable[[np.ndarray], bool]], default=None
            Optional constraint function (unused).

        Returns
        -------
        UnifiedActionId
            The action with the highest probability value.
        """
        if not refined_p:
            raise ValueError("Cannot select action from empty refined_p dictionary")
        best_unified_action = max(refined_p, key=refined_p.get)
        return best_unified_action


class DynamicPricingBandit(SingleObjectiveStrategy):
    """
    Dynamic pricing Thompson Sampling strategy for multi-armed bandits.

    This strategy selects the action that maximizes the expected revenue, defined as the
    product of the action ``price`` and its sampled probability of a positive reward
    (i.e. purchase probability): ``revenue = price * P(purchase)``. Each action is associated
    with a predefined ``price``: a scalar for discrete actions, or a callable mapping a quantity
    vector to a price for quantitative (continuous price) actions. The probability of purchase is
    provided by the Bayesian posterior, mirroring the demand estimate of the reference algorithm.

    References
    ----------
    Revenue objective ``price * P(purchase)`` is inspired by the demand-times-price formulation in:
    Nonparametric Pricing Analytics with Customer Covariates (Chen and Gallego, 2021)
    https://arxiv.org/abs/1805.01136
    """

    def get_prerequisites(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModelDP],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute prerequisites for the dynamic pricing strategy.

        Dynamic pricing requires no prerequisites: the revenue of each action is evaluated
        directly during selection.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary mapping action IDs to probability functions or values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions (unused in dynamic pricing).
        forbidden_regions : Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]], default=None
            Per-arm feasibility constraints (unused in dynamic pricing).

        Returns
        -------
        Dict[str, Any]
            Empty dictionary as no prerequisites are needed.
        """
        return {}

    def _verify_action(self, score: float) -> bool:
        """
        Verify if an action should be considered for selection.

        Dynamic pricing considers all actions; revenue is evaluated during selection.

        Parameters
        ----------
        score : float
            The probability or score of the action (unused).

        Returns
        -------
        bool
            Always True - all actions are considered.
        """
        return True

    @staticmethod
    def _revenue(model: BaseModelDP, quantity: Optional[np.ndarray], proba: float) -> float:
        """
        Compute expected revenue for an action.

        Parameters
        ----------
        model : BaseModelDP
            The action model providing the price (scalar or callable).
        quantity : Optional[np.ndarray]
            Quantity vector for quantitative actions; None for discrete actions.
        proba : float
            Purchase probability.

        Returns
        -------
        float
            ``price * proba``.
        """
        price = model.price(quantity) if callable(model.price) else model.price
        return price * proba

    def _verify_and_select_from_quantitative_action(
        self,
        score_func: Callable[[np.ndarray], float],
        model: BaseModelDP,
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        rng: Optional[Any] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        Find the revenue-maximizing quantity for a quantitative action.

        Maximizes ``price(quantity) * P(purchase | quantity)`` over the quantity space.

        Parameters
        ----------
        score_func : Callable[[np.ndarray], float]
            Function that computes the purchase probability given a quantity vector.
        model : BaseModelDP
            The model associated with this quantitative action (provides the ``price`` callable).
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions that quantity must satisfy.
        rng : Optional[Any], default=None
            Random generator passed to the optimizer for reproducibility.

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector that maximizes revenue, or None if optimization fails.
        """
        try:
            return maximize_by_quantity(
                lambda x: self._revenue(model, x, score_func(x)),
                model.dimension,
                constraint_list,
                seed=rng,
            )
        except OptimizationFailedError as e:
            logger.warning(f"Optimization failed: {e}")
            return None

    def _select_from_refined_actions(
        self,
        refined_p: Dict[UnifiedActionId, float],
        actions: Dict[ActionId, BaseModelDP],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """
        Select the action with the highest expected revenue.

        Revenue is ``price * P(purchase)``, where the price is read from the action model
        (a scalar for discrete actions, or ``price(quantity)`` for quantitative actions) and
        the probability is the refined sampled value.

        Parameters
        ----------
        refined_p : Dict[UnifiedActionId, float]
            Dictionary of unified action IDs to their probability values.
        actions : Dict[ActionId, BaseModelDP]
            Dictionary mapping action IDs to their models (for price information).
        constraint : Optional[Callable[[np.ndarray], bool]], default=None
            Optional constraint function (unused).

        Returns
        -------
        UnifiedActionId
            The action with the highest expected revenue.
        """
        if not refined_p:
            raise ValueError("Cannot select action from empty refined_p dictionary")

        def revenue(item: Tuple[UnifiedActionId, float]) -> float:
            action, proba = item
            model = actions[action[0]] if isinstance(action, tuple) else actions[action]
            quantity = np.array(action[1]) if isinstance(action, tuple) else None
            return self._revenue(model, quantity, proba)

        best_unified_action = max(refined_p.items(), key=revenue)[0]
        return best_unified_action


class BestActionIdentificationBandit(ClassicBandit):
    """
    Best-Action Identification (BAI) strategy for multi-armed bandits.

    This strategy balances between exploitation and exploration by probabilistically
    choosing between the best action and the second-best action. It's designed for
    scenarios where identifying the truly best action is important.

    Parameters
    ----------
    exploit_p : Optional[Float01], default=0.5
        Probability of selecting the best action versus the second-best action.
        - If exploit_p = 1: Always selects the best action (pure exploitation/greedy).
        - If exploit_p = 0: Always selects the second-best action.
        - If exploit_p = 0.5: Equal probability of selecting best or second-best.

    References
    ----------
    Simple Bayesian Algorithms for Best-Arm Identification (Russo, 2018)
    https://arxiv.org/pdf/1602.08448.pdf
    """

    exploit_p: Optional[Float01] = 0.5

    @field_validator("exploit_p", mode="before")
    @classmethod
    def normalize_exploit_p(cls, v):
        """
        Normalize the exploit_p field value to its default if None.

        Parameters
        ----------
        v : Any
            The exploit_p value to normalize.

        Returns
        -------
        Float01
            The original value if not None, otherwise 0.5.
        """
        return cls._normalize_field(v, "exploit_p")

    @validate_call
    def with_exploit_p(self, exploit_p: Optional[Float01]) -> Self:
        """
        Create a new instance with a different exploitation probability.

        Parameters
        ----------
        exploit_p : Optional[Float01], default=0.5
            Probability of selecting the best action versus the second-best action.
            - If exploit_p = 1: Always selects the best action (pure exploitation).
            - If exploit_p = 0: Always selects the second-best action.
            - If exploit_p = 0.5: Equal probability of selecting best or second-best.

        Returns
        -------
        mutated_best_action_identification : BestActionIdentificationBandit
            A new instance with the specified exploitation probability.
        """
        mutated_best_action_identification = self._with_argument("exploit_p", exploit_p)
        return mutated_best_action_identification

    def _select_from_refined_actions(
        self,
        refined_p: Dict[UnifiedActionId, float],
        actions: Dict[ActionId, BaseModel],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """
        Select action based on BAI strategy.

        Probabilistically chooses between the best action (with probability exploit_p)
        and the second-best action (with probability 1 - exploit_p).

        Parameters
        ----------
        refined_p : Dict[UnifiedActionId, float]
            Dictionary of unified action IDs to their probability values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models (unused).
        constraint : Optional[Callable[[np.ndarray], bool]], default=None
            Optional constraint function (unused).

        Returns
        -------
        UnifiedActionId
            Either the best or second-best action based on exploit_p probability.
        """
        # First get the best action
        best_unified_action = super()._select_from_refined_actions(refined_p, actions, constraint)

        # exploit with probability exploit_p and not exploit with probability 1-exploit_p
        take_second_max = self.exploit_p <= random() if self.exploit_p != 1 else False

        # select the action with the second-highest probability
        if take_second_max:
            refined_p.pop(best_unified_action)

            # Get the second best action
            if refined_p:
                return super()._select_from_refined_actions(refined_p, actions, constraint)

        return best_unified_action


class CostControlBandit(SingleObjectiveStrategy, CostControlStrategy):
    """
    Cost-controlled Thompson Sampling strategy for multi-armed bandits.

    This strategy extends classic bandits by considering action costs. It first
    identifies a feasible set of actions whose rewards are within a tolerance of
    the best reward, then selects the lowest-cost action from this set.

    The feasible action set is defined as those with expected rewards in the range
    [(1-subsidy_factor)*max_reward, max_reward], where max_reward is the highest
    sampled reward value.

    Parameters
    ----------
    subsidy_factor : Optional[Float01], default=0.5
        Tolerance factor defining the feasible action set.
        - If subsidy_factor = 1: Always selects minimum cost action.
        - If subsidy_factor = 0: Always selects highest reward action (classic bandit).
        - Values in between balance reward and cost considerations.

    References
    ----------
    Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
    https://arxiv.org/abs/1911.00638

    Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
    https://arxiv.org/abs/2011.01488
    """

    def get_prerequisites(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute the best available reward for defining the feasible action set.

        This method finds the maximum reward value across all actions, which is
        used to determine the reward threshold for feasible actions.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary mapping action IDs to probability functions or values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions for quantitative actions.
        forbidden_regions : Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]], default=None
            Per-arm feasibility constraints (``>= 0`` feasible). Passed through so the reward threshold is
            computed over the feasible quantity space, not the forbidden regions.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'best_value': the maximum reward value.
        """
        classic_bandit = ClassicBandit()
        # constraint_list holds the (single) global constraint from select_action; the per-arm forbidden
        # regions are passed separately so best_value reflects only the feasible quantity space.
        constraint = constraint_list[0] if constraint_list else None
        best_classic_unified_action = classic_bandit.select_action(
            p, actions, constraint, forbidden_regions=forbidden_regions
        )
        best_value = (
            p[best_classic_unified_action]
            if isinstance(best_classic_unified_action, str)
            else p[best_classic_unified_action[0]](best_classic_unified_action[1])
        )
        return {"best_value": best_value}

    def _select_from_refined_actions(
        self,
        refined_p: Dict[UnifiedActionId, float],
        actions: Dict[ActionId, BaseModel],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """
        Select the lowest-cost action from the feasible set.

        Actions are sorted primarily by cost (ascending) and secondarily by
        probability (descending) to break ties.

        Parameters
        ----------
        refined_p : Dict[UnifiedActionId, float]
            Dictionary of feasible actions and their probability values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models (for cost information).
        constraint : Optional[Callable[[np.ndarray], bool]], default=None
            Optional constraint function (unused).

        Returns
        -------
        UnifiedActionId
            The action with minimum cost among feasible actions.
        """

        # Apply cost control logic
        sortable_actions = []
        for action, proba in refined_p.items():
            cost = actions[action[0]].cost(action[1]) if isinstance(action, tuple) else actions[action].cost
            sortable_actions.append((cost, -proba, action))

        if not sortable_actions:
            return max(refined_p, key=refined_p.get)

        # select the action with the min cost (and the highest mean of probabilities in case of cost equality)
        _, _, best_unified_action = sorted(sortable_actions)[0]

        # return cheapest action from the set of feasible actions
        return best_unified_action

    def _verify_action(self, score: float, best_value: float) -> bool:
        """
        Check if an action's reward is within the feasible threshold.

        An action is feasible if its reward is at least (1-subsidy_factor) times
        the best available reward.

        Parameters
        ----------
        score : float
            The reward/probability of the action.
        best_value : float
            The maximum reward across all actions.

        Returns
        -------
        bool
            True if the action's reward is above the threshold, False otherwise.
        """
        return score >= best_value * (1 - self.subsidy_factor)

    def _verify_and_select_from_quantitative_action(
        self,
        score_func: Callable[[np.ndarray], float],
        model: BaseModel,
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        best_value: float,
        rng: Optional[Any] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        Find the minimum-cost quantity that meets the reward threshold.

        This method adds a reward threshold constraint and then minimizes cost
        subject to all constraints.

        Parameters
        ----------
        score_func : Callable[[np.ndarray], float]
            Function that computes reward given a quantity vector.
        model : BaseModel
            The model associated with this quantitative action.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of existing constraint functions.
        best_value : float
            The maximum reward across all actions.
        rng : Optional[Any], default=None
            Random generator passed to the optimizer for reproducibility.

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector that minimizes cost while meeting the
            reward threshold, or None if no feasible solution exists.
        """

        def cost_control_constraint(x: np.ndarray) -> bool:
            return score_func(x) - best_value * (1 - self.subsidy_factor)

        # Build a fresh list so the caller's constraint_list is not mutated across
        # successive quantitative actions (which would accumulate cost constraints).
        local_constraints = (constraint_list.copy() if constraint_list is not None else []) + [cost_control_constraint]
        try:
            return maximize_by_quantity(lambda x: -model.cost(x), model.dimension, local_constraints, seed=rng)
        except OptimizationFailedError:
            return None
