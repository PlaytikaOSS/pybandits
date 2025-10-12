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

from abc import ABC, abstractmethod
from random import random
from typing import Any, Callable, ClassVar, Dict, Generator, List, Optional, Type, TypeVar, Union

import numpy as np
from loguru import logger
from typing_extensions import Self

from pybandits.base import ActionId, Float01, PyBanditsBaseModel, UnifiedActionId
from pybandits.base_model import BaseModel
from pybandits.pydantic_version_compatibility import PrivateAttr, field_validator, validate_call
from pybandits.quantitative_model import QuantitativeModel
from pybandits.utils import OptimizationFailedError, maximize_by_quantity

StrategyType = TypeVar("StrategyType", bound="BaseStrategy")


class BaseStrategy(PyBanditsBaseModel, ABC):
    """
    Abstract base strategy for selecting actions in multi-armed bandits.

    This class defines the interface that all bandit strategies must implement.
    Strategies determine how to select actions based on their estimated rewards
    and other criteria.
    """

    @validate_call
    @abstractmethod
    def select_action(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        **kwargs,
    ) -> UnifiedActionId:
        """
        Select an action based on the strategy's selection criteria.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary mapping action IDs to either:
            - float: Fixed probability of positive reward
            - Callable: Function that computes probability given quantity vector
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        **kwargs
            Additional strategy-specific parameters.

        Returns
        -------
        UnifiedActionId
            The selected action ID, either a simple ActionId or a tuple of
            (ActionId, quantity_vector) for quantitative actions.
        """


class SingleObjectiveStrategy(BaseStrategy, ABC):
    """
    Abstract strategy for single-objective multi-armed bandits.

    This class handles bandits where each action has a single scalar reward.
    It provides a framework for refining actions based on constraints and
    selecting the best action according to strategy-specific criteria.

    """

    _dummy_quantitative_action: ClassVar[str] = "dummy_quantitative_action"

    @validate_call
    def select_action(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """
        Select an action for single-objective optimization.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary mapping action IDs to either:
            - float: Fixed probability of positive reward
            - Callable: Function that computes probability given quantity vector
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        constraint : Optional[Callable[[np.ndarray], bool]], default=None
            Optional constraint function that returns True if a quantity vector
            satisfies the constraints.

        Returns
        -------
        UnifiedActionId
            The selected action ID, either a simple ActionId or a tuple of
            (ActionId, quantity_vector) for quantitative actions.
        """
        constraint_list = [constraint] if constraint is not None else None
        refined_p = self.refine_p(p, actions, constraint_list)
        best_unified_action = self._select_from_refined_actions(refined_p, actions, constraint_list)
        return best_unified_action

    def refine_p(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
    ) -> Dict[UnifiedActionId, float]:
        """
        Refine action probabilities by evaluating quantitative actions and filtering.

        This method processes both standard and quantitative actions, evaluating
        quantitative functions at optimal points and filtering actions based on
        strategy-specific criteria.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary of actions and their probability functions or values.
        actions : Dict[ActionId, BaseModel]
            Dictionary of actions and their associated models.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions for quantitative actions.

        Returns
        -------
        refined_p: Dict[UnifiedActionId, float]
            Dictionary mapping unified action IDs to their refined probability values.
        """
        if not p or not actions:
            return {}
        prerequisites = self.get_prerequisites(p, actions, constraint_list)
        refined_p = {}
        for action, proba in p.items():
            model = actions[action]
            if callable(proba):  # Quantitative action
                quantity = self._verify_and_select_from_quantitative_action(
                    proba, model, constraint_list, **prerequisites
                )
                if quantity is not None:
                    proba_value = proba(quantity)
                    refined_p[(action, tuple(quantity))] = proba_value
            elif self._verify_action(proba, **prerequisites):  # Standard action
                refined_p[action] = proba
        return refined_p

    @abstractmethod
    def get_prerequisites(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
    ) -> Dict[str, Any]:
        """
        Compute prerequisites needed for strategy-specific action selection.

        This method allows strategies to pre-compute values needed for their
        selection logic, such as the best available reward for cost control.

        Parameters
        ----------
        p : Dict[ActionId, Union[float, Callable[[np.ndarray], float]]]
            Dictionary mapping action IDs to probability functions or values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions for quantitative actions.

        Returns
        -------
        Dict[str, Any]
            Dictionary of prerequisite values needed by the strategy.
        """

    @abstractmethod
    def _select_from_refined_actions(
        self,
        refined_p: Dict[UnifiedActionId, float],
        actions: Dict[ActionId, BaseModel],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """
        Apply strategy-specific logic to select from refined actions.

        Parameters
        ----------
        refined_p : Dict[UnifiedActionId, float]
            Dictionary of unified action IDs to their refined probability values.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.
        constraint : Optional[Callable[[np.ndarray], bool]], default=None
            Optional constraint function for additional filtering.

        Returns
        -------
        UnifiedActionId
            The selected unified action ID based on strategy criteria.
        """

    @abstractmethod
    def _verify_action(self, score: float, **kwargs) -> bool:
        """
        Determine if a standard action should be considered for selection.

        Parameters
        ----------
        score : float
            The probability or score associated with the action.
        **kwargs
            Additional strategy-specific parameters from prerequisites.

        Returns
        -------
        bool
            True if the action meets the strategy's criteria for consideration,
            False otherwise.
        """

    @abstractmethod
    def _verify_and_select_from_quantitative_action(
        self,
        score_func: Callable[[np.ndarray], float],
        model: BaseModel,
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        Find optimal quantity for a quantitative action if it meets criteria.

        Parameters
        ----------
        score_func : Callable[[np.ndarray], float]
            Function that computes probability/score given a quantity vector.
        model : BaseModel
            The model associated with this quantitative action.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions that quantity must satisfy.
        **kwargs
            Additional strategy-specific parameters from prerequisites.

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector if the action meets criteria,
            None if it should not be considered.
        """

    def verify_and_select_from_quantitative_action(
        self,
        score_func: Callable[[np.ndarray], float],
        model: BaseModel,
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
    ) -> Optional[np.ndarray]:
        """
        Public interface for verifying and selecting from quantitative actions.

        This method wraps the private implementation to provide a clean public API
        for finding optimal quantities for quantitative actions.

        Parameters
        ----------
        score_func : Callable[[np.ndarray], float]
            Function that computes probability/score given a quantity vector.
        model : BaseModel
            The model associated with this quantitative action.
        constraint_list : Optional[List[Callable[[np.ndarray], bool]]]
            List of constraint functions that quantity must satisfy.

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector if found, None otherwise.
        """
        p = {self._dummy_quantitative_action: score_func}
        actions = {self._dummy_quantitative_action: model}
        prerequisites = self.get_prerequisites(p, actions, constraint_list)
        return self._verify_and_select_from_quantitative_action(score_func, model, constraint_list, **prerequisites)


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

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector that maximizes the score function, or None if optimization fails.
        """
        try:
            return maximize_by_quantity(score_func, model.dimension, constraint_list)
        except OptimizationFailedError:
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


class CostControlStrategy(PyBanditsBaseModel):
    """
    Mixin class for cost-aware action selection strategies.

    This class provides functionality for strategies that consider action costs
    in addition to rewards. It defines a feasible action set based on a tolerance
    threshold and selects the lowest-cost action from this set.

    Parameters
    ----------
    subsidy_factor : Optional[Float01], default=0.5
        Tolerance factor defining the feasible action set as those with rewards
        in the range [(1-subsidy_factor)*max_reward, max_reward].
        - If subsidy_factor = 1: Selects minimum cost action (ignores rewards).
        - If subsidy_factor = 0: Selects highest reward action (ignores costs).
        - If subsidy_factor = 0.5: Balances between reward and cost.

    References
    ----------
    Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
    https://arxiv.org/abs/1911.00638

    Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
    https://arxiv.org/abs/2011.01488
    """

    subsidy_factor: Optional[Float01] = 0.5

    @field_validator("subsidy_factor", mode="before")
    @classmethod
    def normalize_subsidy_factor(cls, v):
        """
        Normalize the subsidy_factor field value to its default if None.

        Parameters
        ----------
        v : Any
            The subsidy_factor value to normalize.

        Returns
        -------
        Float01
            The original value if not None, otherwise 0.5.
        """
        return cls._normalize_field(v, "subsidy_factor")

    @validate_call
    def with_subsidy_factor(self, subsidy_factor: Optional[Float01]) -> Self:
        """
        Create a new instance with a different subsidy factor.

        Parameters
        ----------
        subsidy_factor : Optional[Float01], default=0.5
            Tolerance factor defining the feasible action set.
            - If subsidy_factor = 1: Selects minimum cost action (ignores rewards).
            - If subsidy_factor = 0: Selects highest reward action (ignores costs).
            - Values in between balance reward and cost considerations.

        Returns
        -------
        mutated_cost_control_bandit
            A new instance with the specified subsidy factor.
        """
        mutated_cost_control_bandit = self._with_argument("subsidy_factor", subsidy_factor)
        return mutated_cost_control_bandit


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

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'best_value': the maximum reward value.
        """
        classic_bandit = ClassicBandit()
        best_classic_unified_action = classic_bandit.select_action(p, actions, constraint_list)
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

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector that minimizes cost while meeting the
            reward threshold, or None if no feasible solution exists.
        """

        def cost_control_constraint(x: np.ndarray) -> bool:
            return score_func(x) >= best_value * (1 - self.subsidy_factor)

        if constraint_list is not None:
            constraint_list.append(cost_control_constraint)
        else:
            constraint_list = [cost_control_constraint]
        try:
            return maximize_by_quantity(lambda x: -model.cost(x), model.dimension, constraint_list)
        except OptimizationFailedError:
            return None


class MultiObjectiveStrategy(BaseStrategy, ABC):
    """
    Abstract strategy for multi-objective multi-armed bandits.

    This class handles bandits where each action has multiple reward objectives.
    It selects actions from the Pareto front - the set of non-dominated actions
    where no other action is better in all objectives.
    """

    # Class variable to define how to select the best action for each objective
    objective_selector_class: ClassVar[Type[SingleObjectiveStrategy]]
    _objective_selector: SingleObjectiveStrategy = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._objective_selector = self.objective_selector_class(**data)

    @validate_call
    def select_action(
        self,
        p: Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        actions: Dict[ActionId, BaseModel],
    ) -> UnifiedActionId:
        """
        Select an action from the Pareto front.

        This method finds all Pareto-optimal actions and randomly selects one,
        giving equal probability to each non-dominated action.

        Parameters
        ----------
        p : Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]]
            Dictionary mapping action IDs to either:
            - List[float]: Fixed reward vector for multiple objectives
            - Callable: Function that computes reward vector given quantity
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.

        Returns
        -------
        UnifiedActionId
            A randomly selected action from the Pareto front.
        """
        pareto_front = self._get_pareto_front(p=p, actions=actions)
        return np.random.choice(pareto_front)

    def _get_feasible_solutions(
        self,
        p: Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        actions: Dict[ActionId, BaseModel],
    ) -> Dict[UnifiedActionId, List[float]]:
        """
        Get feasible solutions for each objective.

        Applies logic independently to each objective, finding actions that meet the selection that objective.

        Parameters
        ----------
        p : Dict[ActionId, List[float]]
            Dictionary mapping action IDs to reward vectors.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.

        Returns
        -------
        Dict[UnifiedActionId, List[float]]
            Feasible actions considering logic for each objective.
        """
        action_id = list(p.keys())[0]
        if isinstance(action_id, tuple):
            action_id = action_id[0]
        n_objectives = len(actions[action_id].models)
        feasible_solutions = {}
        # Separate discrete and quantitative actions
        discrete_actions = {aid: prob_or_func for aid, prob_or_func in p.items() if not callable(prob_or_func)}
        quantitative_actions = {aid: prob_or_func for aid, prob_or_func in p.items() if callable(prob_or_func)}

        # For discrete actions, add directly (they already have full reward vectors)
        feasible_solutions.update(discrete_actions)

        # For quantitative actions, refine per objective
        if quantitative_actions:
            for i in range(n_objectives):
                # Fix closure bug: create a new function that captures i by value
                def make_objective_extractor(obj_idx):
                    return lambda x: x[obj_idx]

                objective_p = {action_id: make_objective_extractor(i) for action_id in quantitative_actions.keys()}
                objective_actions = {
                    action_id: actions[action_id].models[i] for action_id in quantitative_actions.keys()
                }

                refined = self._objective_selector.refine_p(objective_p, objective_actions, None)

                # Build multi-objective vectors from per-objective results
                for unified_action_id in refined.keys():
                    if unified_action_id not in feasible_solutions:
                        feasible_solutions[unified_action_id] = quantitative_actions[unified_action_id[0]](
                            unified_action_id[1]
                        )

        return feasible_solutions

    def _get_exact_pareto_front(
        self, p: Dict[UnifiedActionId, List[float]], actions: Dict[ActionId, BaseModel]
    ) -> List[UnifiedActionId]:
        """
        Compute the exact Pareto front for discrete action sets.

        An action is Pareto-optimal if no other action dominates it (i.e., is
        better or equal in all objectives and strictly better in at least one).

        Parameters
        ----------
        p : Dict[UnifiedActionId, List[float]]
            Dictionary mapping unified action IDs to their reward vectors.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.

        Returns
        -------
        List[UnifiedActionId]
            List of Pareto-optimal action IDs.
        """
        feasible_solutions = self._get_feasible_solutions(p, actions)
        # store non dominated actions
        pareto_front = []

        for this_action in feasible_solutions.keys():
            is_pareto = True  # we assume that action is Pareto Optimal until proven otherwise
            other_actions = [a for a in feasible_solutions.keys() if a != this_action]

            for other_action in other_actions:
                # check if this_action is not dominated by other_action based on
                # multiple objectives reward prob vectors
                is_dominated = not (
                    # an action cannot be dominated by an identical one
                    (feasible_solutions[this_action] == feasible_solutions[other_action])
                    # otherwise, apply the classical definition
                    or any(
                        feasible_solutions[this_action][i] > feasible_solutions[other_action][i]
                        for i in range(len(feasible_solutions[this_action]))
                    )
                )

                if is_dominated:
                    # this_action dominated by at least one other_action,
                    # this_action is not pareto optimal
                    is_pareto = False
                    break

            if is_pareto:
                # this_action is pareto optimal
                pareto_front.append(this_action)

        return pareto_front

    def _get_approximate_pareto_front(
        self,
        p: Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        actions: Dict[ActionId, BaseModel],
        n_divisions: int = 10,
    ) -> List[UnifiedActionId]:
        """
        Approximate the Pareto front for continuous/quantitative actions.

        Uses the Normal Constraint method with Das-Dennis weight generation to
        systematically sample the Pareto front for quantitative actions.

        Parameters
        ----------
        p : Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]]
            Dictionary mapping action IDs to reward vectors or functions.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.
        n_divisions : int, default=10
            Number of divisions for weight vector generation. Higher values
            provide better approximation but increase computation.

        Returns
        -------
        List[UnifiedActionId]
            List of approximately Pareto-optimal actions.
        """
        if not p:
            return []

        approximate_p = {}
        n_objectives = len(actions[list(p.keys())[0]].models)

        for action_id, prob_or_func in p.items():
            if callable(prob_or_func):
                # Quantitative action - find Pareto optimal input points
                pareto_input_points = self._find_pareto_front_normal_constraint(
                    prob_or_func, actions[action_id].dimension, n_objectives, n_divisions, actions[action_id]
                )
                approximate_p.update(
                    {(action_id, tuple(input_point)): prob_or_func(input_point) for input_point in pareto_input_points}
                )
            else:
                # Standard action with fixed reward vector
                approximate_p[action_id] = prob_or_func

        return self._get_exact_pareto_front(approximate_p, actions)

    @validate_call
    def _find_pareto_front_normal_constraint(
        self,
        func: Callable[[np.ndarray], List[float]],
        input_dim: int,
        n_objectives: int,
        n_divisions: int,
        model: BaseModel,
    ) -> List[np.ndarray]:
        """
        Find Pareto front using Normal Constraint method with Das-Dennis weight generation for a single function.

        This method systematically explores the Pareto front by solving constrained
        optimization problems with different weight vectors.

        Parameters
        ----------
        func : Callable[[np.ndarray], List[float]]
            Function mapping quantity vectors to reward vectors.
        input_dim : int
            Dimension of the input quantity vector.
        n_objectives : int
            Number of reward objectives.
        n_divisions : int
            Number of divisions for weight generation (controls approximation quality).
        model : BaseModel
            The model for this quantitative action.

        Returns
        -------
        List[np.ndarray]
            List of Pareto-optimal quantity vectors.

        References
        ----------
        The normalized normal constraint method for generating the Pareto frontier (Messac et al., 2003)
        https://ieeexplore.ieee.org/document/938649
        """
        # Step 1: Find anchor points using optimization for each objective
        anchor_points = [
            self._objective_selector.verify_and_select_from_quantitative_action(
                lambda x: func(x)[i], model.models[i], None
            )
            for i in range(n_objectives)
        ]
        anchor_rewards = [func(anchor_point) for anchor_point in anchor_points]

        anchor_matrix = np.array(anchor_rewards)  # n_objectives x n_objectives
        anchor_points = np.array(anchor_points)  # n_objectives x input_dim

        # Step 2: Generate Das-Dennis weight vectors
        weight_vectors = self._das_dennis_weights(n_objectives, n_divisions)

        # Step 3: For each weight vector, solve NC subproblem
        nc_solutions = set(tuple(anchor_point) for anchor_point in anchor_points)
        utopia = np.max(anchor_matrix, axis=0)  # Ideal point

        for weight in weight_vectors:
            solution = self._solve_nc_subproblem(func, anchor_matrix, anchor_points, utopia, weight, model)
            if solution is not None:
                nc_solutions.add(tuple(solution))

        return list(nc_solutions)

    @staticmethod
    def _das_dennis_weights(n_objectives: int, n_divisions: int) -> np.ndarray:
        """
        Generate Das-Dennis weight vectors for systematic Pareto front sampling.

        Creates uniformly distributed weight vectors on the unit simplex using
        the Das-Dennis method, which provides systematic coverage of the
        objective space.

        Parameters
        ----------
        n_objectives : int
            Number of objectives/dimensions.
        n_divisions : int
            Number of divisions per dimension. Total weights generated is
            approximately (n_divisions + n_objectives - 1)! / (n_divisions! * (n_objectives - 1)!).

        Returns
        -------
        np.ndarray
            Array of shape (n_weights, n_objectives) containing weight vectors.
        """

        def generate_recursive(
            n_obj: int, n_div: int, current_weight: List[int], depth: int = 0
        ) -> Generator[np.ndarray, None, None]:
            """
            Recursively generate weight combinations for Das-Dennis method.

            Parameters
            ----------
            n_obj : int
                Number of objectives.
            n_div : int
                Remaining divisions to allocate.
            current_weight : List[int]
                Current partial weight vector being built.
            depth : int
                Current recursion depth (objective index).

            Yields
            ------
            np.ndarray
                Normalized weight vectors summing to 1.
            """
            if depth == n_obj - 1:
                current_weight.append(n_div)
                yield np.array(current_weight) / n_divisions
                current_weight.pop()
                return

            for i in range(n_div + 1):
                current_weight.append(i)
                yield from generate_recursive(n_obj, n_div - i, current_weight, depth + 1)
                current_weight.pop()

        weights = list(generate_recursive(n_objectives, n_divisions, []))
        return np.array(weights)

    def _solve_nc_subproblem(
        self,
        func: Callable,
        anchor_matrix: np.ndarray,
        utopia: np.ndarray,
        weight: np.ndarray,
        model: BaseModel,
        epsilon: float = 1e-10,
    ) -> Optional[np.ndarray]:
        """
        Solve a single Normal Constraint optimization subproblem.

        Maximizes a weighted objective while constraining other objectives to lie
        on the "reference point side" of hyperplanes through anchor points.

        Parameters
        ----------
        func : Callable
            The multi-objective function to optimize.
        anchor_matrix : np.ndarray
            Matrix of anchor points (extreme points for each objective).
        utopia : np.ndarray
            The utopia point (ideal but typically unachievable point).
        weight : np.ndarray
            Weight vector determining the reference point and primary objective.
        model : BaseModel
            The model for constraint evaluation.
        epsilon : float, default=1e-10
            Numerical tolerance for constraint satisfaction.

        Returns
        -------
        Optional[np.ndarray]
            Optimal solution if found and feasible, None otherwise.
        """
        n_objectives = len(weight)
        primary_obj = np.argmax(weight)

        # Step #1: Create the utopia-based coordinate system
        # Transform the problem so utopia is at origin
        transformed_anchors = anchor_matrix - utopia  # Anchors relative to utopia

        # Step #2: Find reference point using weight vector from utopia
        # This is where the weight ray from utopia intersects the anchor hyperplane
        reference_point_transformed = self._find_utopia_reference_point(transformed_anchors, weight, epsilon)
        reference_point = reference_point_transformed + utopia  # Back to original coordinates

        # Step #3: Create Normal Constraint boundaries using utopia geometry
        constraint_normals = []
        constraint_intercepts = []

        for i in range(n_objectives):
            if i != primary_obj:
                # Normal vector points from anchor_i towards utopia
                # This creates the "feasible cone" emanating from utopia
                normal_direction = reference_point - anchor_matrix[i]

                # The constraint hyperplane passes through anchor_i with this normal
                if np.linalg.norm(normal_direction) > epsilon:
                    normal = normal_direction / np.linalg.norm(normal_direction)
                    intercept = np.dot(normal, anchor_matrix[i])

                    constraint_normals.append(normal)
                    constraint_intercepts.append(intercept)

        def reference_based_constraints(x: np.ndarray) -> bool:
            """
            Check if a point satisfies Normal Constraint boundaries.

            Verifies that the function value at x lies on the correct side of all
            constraint hyperplanes defined by the anchor points and reference point.

            Parameters
            ----------
            x : np.ndarray
                Input point to evaluate.

            Returns
            -------
            bool
                True if all constraints are satisfied, False otherwise.
            """
            rewards = np.array(func(x))

            for normal, intercept in zip(constraint_normals, constraint_intercepts):
                # Constraint: normal · f(x) >= intercept
                # Geometric meaning: f(x) is on the reference point side of the boundary
                constraint_value = np.dot(normal, rewards) - intercept

                if constraint_value < -epsilon:  # Tolerance for numerical errors
                    return False
            return True

        def objective_function(x: np.ndarray) -> float:
            """
            Extract the primary objective value for maximization.

            Parameters
            ----------
            x : np.ndarray
                Input point to evaluate.

            Returns
            -------
            float
                Value of the primary objective at x.
            """
            return func(x)[primary_obj]

        # Solve the constrained optimization
        try:
            solution = self._objective_selector.verify_and_select_from_quantitative_action(
                objective_function, model.models[primary_obj], reference_based_constraints
            )

            if reference_based_constraints(solution):
                return solution
            else:
                return None

        except Exception as e:
            logger.error(f"NC subproblem failed: {e}")
            return None

    @classmethod
    def _find_utopia_reference_point(
        cls, transformed_anchors: np.ndarray, weight: np.ndarray, epsilon: float
    ) -> np.ndarray:
        """
        Find the reference point for Normal Constraint method.

        Computes where a ray from the utopia point in the direction of the weight
        vector intersects the hyperplane defined by the anchor points.

        Parameters
        ----------
        transformed_anchors : np.ndarray
            Anchor points transformed relative to utopia point.
        weight : np.ndarray
            Direction vector from utopia point.
        epsilon : float
            Numerical tolerance for degeneracy detection.

        Returns
        -------
        np.ndarray
            The reference point in the transformed coordinate system.
        """

        # ray-hyperplane intersection
        anchor_center = np.mean(transformed_anchors, axis=0)
        anchor_vectors = transformed_anchors - anchor_center

        try:
            # Find hyperplane normal
            U, _, _ = np.linalg.svd(anchor_vectors.T, full_matrices=True)
            hyperplane_normal = U[:, -1]

            # Ray intersection
            numerator = np.dot(hyperplane_normal, anchor_center)
            denominator = np.dot(hyperplane_normal, weight)

            if abs(denominator) > epsilon:
                t = numerator / denominator
                intersection = t * weight
                return intersection
            else:
                return np.dot(weight, transformed_anchors)

        except np.linalg.LinAlgError:
            return np.dot(weight, transformed_anchors)

    def _get_pareto_front(
        self,
        p: Dict[ActionId, Union[List[float], List[Callable[[np.ndarray], float]]]],
        actions: Dict[ActionId, BaseModel],
    ) -> List[UnifiedActionId]:
        """
        Compute the Pareto front, using exact or approximate methods as appropriate.

        Automatically selects between exact computation (for discrete actions) and
        approximation (when quantitative actions are present).

        Parameters
        ----------
        p : Dict[ActionId, Union[List[float], List[Callable[[np.ndarray], float]]]]
            Dictionary mapping action IDs to reward vectors or functions.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.

        Returns
        -------
        List[UnifiedActionId]
            List of Pareto-optimal actions.
        """
        includes_quantitative_actions = any(isinstance(actions[a], QuantitativeModel) for a in p.keys())
        return (
            self._get_approximate_pareto_front(p, actions)
            if includes_quantitative_actions
            else self._get_exact_pareto_front(p, actions)
        )


class MultiObjectiveBandit(MultiObjectiveStrategy):
    """
    Multi-objective Thompson Sampling strategy for multi-armed bandits.

    This strategy handles vector-valued rewards where each action produces multiple
    reward outcomes. Actions are selected from the Pareto front - the set of
    non-dominated actions where no other action is superior in all objectives.

    The strategy uses Thompson Sampling for exploration by sampling from posterior
    distributions and then selecting uniformly from the resulting Pareto front.



    References
    ----------
    Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
    https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem
    """

    # Use ClassicBandit's selection strategy for finding extreme points
    objective_selector_class: ClassVar[Type[SingleObjectiveStrategy]] = ClassicBandit


class MultiObjectiveCostControlBandit(MultiObjectiveStrategy, CostControlStrategy):
    """
    Multi-objective strategy with cost control for multi-armed bandits.

    Combines multi-objective optimization with cost awareness. For each objective,
    identifies actions within a tolerance of the best reward, then considers only
    the lowest-cost actions from these feasible sets when computing the Pareto front.

    This strategy is useful when actions have both multiple reward objectives and
    associated costs, requiring a balance between Pareto-optimality and cost efficiency.



    """

    # Use CostControlBandit's selection strategy for finding extreme points
    objective_selector_class: ClassVar[Type[SingleObjectiveStrategy]] = CostControlBandit
