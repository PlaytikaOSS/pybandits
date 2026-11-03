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
from typing import Any, Callable, ClassVar, Dict, List, Optional, Self, TypeVar, Union

import numpy as np
from pydantic import field_validator, validate_call

from pybandits.base import ActionId, Float01, PyBanditsBaseModel, UnifiedActionId
from pybandits.base_model import BaseModel

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
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
        rng: Optional[Any] = None,
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
        forbidden_regions : Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]], default=None
            Per-arm feasibility constraints (``>= 0`` feasible) restricting a quantitative arm's quantity space.
            Merged with ``constraint`` for the relevant arm during quantity optimization.
        rng : Optional[Any], default=None
            Random generator passed to the quantity optimizer for reproducibility.

        Returns
        -------
        UnifiedActionId
            The selected action ID, either a simple ActionId or a tuple of
            (ActionId, quantity_vector) for quantitative actions.
        """
        constraint_list = [constraint] if constraint is not None else None
        refined_p = self.refine_p(p, actions, constraint_list, forbidden_regions, rng=rng)
        best_unified_action = self._select_from_refined_actions(refined_p, actions, constraint)
        return best_unified_action

    def refine_p(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
        rng: Optional[Any] = None,
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
            List of (global) constraint functions for quantitative actions.
        forbidden_regions : Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]], default=None
            Per-arm feasibility constraints (``>= 0`` feasible). For each quantitative arm, its forbidden-region
            constraints are appended to ``constraint_list`` so the optimizer avoids the forbidden quantity space.
            An arm whose quantity space is fully forbidden fails optimization and is dropped from the result.
        rng : Optional[Any], default=None
            Random generator passed to the quantity optimizer for reproducibility.

        Returns
        -------
        refined_p: Dict[UnifiedActionId, float]
            Dictionary mapping unified action IDs to their refined probability values.
        """
        if not p or not actions:
            return {}
        prerequisites = self.get_prerequisites(p, actions, constraint_list, forbidden_regions)
        refined_p = {}
        for action, proba in p.items():
            model = actions[action]
            if callable(proba):  # Quantitative action
                # Merge global constraints with this arm's forbidden-region constraints (fresh list per arm).
                arm_constraints = list(constraint_list) if constraint_list else []
                if forbidden_regions and action in forbidden_regions:
                    arm_constraints.extend(forbidden_regions[action])
                quantity = self._verify_and_select_from_quantitative_action(
                    proba, model, arm_constraints or None, rng=rng, **prerequisites
                )
                if quantity is not None:
                    proba_value = proba(quantity)
                    refined_p[(action, tuple(quantity))] = proba_value
            elif self._verify_action(proba, **prerequisites):  # Standard action
                refined_p[action] = proba
        if not refined_p:
            raise ValueError("No actions met the criteria for selection. Please check the constraints and the actions.")
        return refined_p

    @abstractmethod
    def get_prerequisites(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
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
        forbidden_regions : Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]], default=None
            Per-arm feasibility constraints (``>= 0`` feasible) so prerequisites are computed over the
            feasible quantity space only.

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
        rng: Optional[Any] = None,
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
        rng : Optional[Any], default=None
            Random generator passed to the quantity optimizer for reproducibility.

        Returns
        -------
        Optional[np.ndarray]
            Optimal quantity vector if found, None otherwise.
        """
        p = {self._dummy_quantitative_action: score_func}
        actions = {self._dummy_quantitative_action: model}
        prerequisites = self.get_prerequisites(p, actions, constraint_list)
        return self._verify_and_select_from_quantitative_action(
            score_func, model, constraint_list, rng=rng, **prerequisites
        )


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
        # _with_argument uses model_copy(update=...), which bypasses field validators;
        # apply the same normalization the validator runs so None -> 0.5 is never skipped.
        subsidy_factor = self._normalize_field(subsidy_factor, "subsidy_factor")
        mutated_cost_control_bandit = self._with_argument("subsidy_factor", subsidy_factor)
        return mutated_cost_control_bandit
