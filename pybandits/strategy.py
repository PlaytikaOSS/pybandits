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
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np
from scipy.stats import ttest_ind_from_stats
from typing_extensions import Self

from pybandits.base import ActionId, Float01, Probability, PyBanditsBaseModel, UnifiedActionId
from pybandits.base_model import BaseModel
from pybandits.model import Beta, BetaMOCC
from pybandits.pydantic_version_compatibility import field_validator, validate_call

StrategyType = TypeVar("StrategyType", bound="Strategy")


class Strategy(PyBanditsBaseModel, ABC):
    """
    Strategy to select actions in multi-armed bandits.
    """

    @abstractmethod
    def select_action(
        self, p: Dict[UnifiedActionId, Probability], actions: Optional[Dict[ActionId, BaseModel]]
    ) -> UnifiedActionId:
        """
        Select the action.
        """

    @classmethod
    @validate_call
    def numerize_field(cls, v, field_name: str):
        return v if v is not None else cls.model_fields[field_name].default

    @classmethod
    @validate_call
    def get_expected_value_from_state(cls, state: Dict[str, Any], field_name: str) -> float:
        return cls.numerize_field(state["strategy"].get(field_name), field_name)


class ClassicBandit(Strategy):
    """
    Classic multi-armed bandits strategy.

    References
    ----------
    Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
    http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal and Goyal, 2014)
    https://arxiv.org/pdf/1209.3352.pdf
    """

    @validate_call
    def select_action(
        self,
        p: Dict[UnifiedActionId, float],
        actions: Optional[Dict[UnifiedActionId, BaseModel]] = None,
    ) -> UnifiedActionId:
        """
        Select the action with the highest probability of getting a positive reward.

        Parameters
        ----------
        p : Dict[UnifiedActionId, Probability]
            The dictionary of actions and their sampled probability of getting a positive reward.
        actions : Optional[Dict[UnifiedActionId, BaseModel]]
            The dictionary of actions and their associated model.

        Returns
        -------
        selected_action: UnifiedActionId
            The selected action.
        """
        return max(p, key=p.get)


class BestActionIdentificationBandit(Strategy):
    """
    Best-Action Identification (BAI) strategy for multi-armed bandits.

    References
    ----------
    Simple Bayesian Algorithms for Best-Arm Identification (Russo, 2018)
    https://arxiv.org/pdf/1602.08448.pdf

    Parameters
    ----------
    exploit_p: Optional[Float01], 0.5 if not specified
        Tuning parameter taking value in [0, 1] which specifies the probability of selecting the best or an alternative
        action.
        If exploit_p is 1, the bandit always selects the action with the highest probability of
        getting a positive reward. That is, it behaves as a Greedy strategy.
        If exploit_p is 0, the bandit always select the action with 2nd highest probability of getting a positive
        reward.
    """

    exploit_p: Optional[Float01] = 0.5

    @field_validator("exploit_p", mode="before")
    @classmethod
    def numerize_exploit_p(cls, v):
        return cls.numerize_field(v, "exploit_p")

    @validate_call
    def with_exploit_p(self, exploit_p: Optional[Float01]) -> Self:
        """
        Instantiate a mutated cost control bandit strategy with an altered subsidy factor.

        Parameters
        ----------
        exploit_p: Optional[Float01], 0.5 if not specified
            Tuning parameter taking value in [0, 1] which specifies the probability of selecting the best or an alternative
            action.
            If exploit_p is 1, the bandit always selects the action with the highest probability of
            getting a positive reward. That is, it behaves as a Greedy strategy.
            If exploit_p is 0, the bandit always select the action with 2nd highest probability of getting a positive
            reward.

        Returns
        -------
        mutated_best_action_identification : BestActionIdentificationBandit
            The mutated best action identification strategy.
        """
        mutated_best_action_identification = self._with_argument("exploit_p", exploit_p)
        return mutated_best_action_identification

    @validate_call
    def select_action(
        self,
        p: Dict[UnifiedActionId, float],
        actions: Optional[Dict[UnifiedActionId, BaseModel]] = None,
    ) -> UnifiedActionId:
        """
        Select with probability self.exploit_p the best action (i.e. the action with the highest probability of getting
        a positive reward), and with probability 1-self.exploit_p it returns the second best action (i.e. the action
        with the second highest probability of getting a positive reward).

        Parameters
        ----------
        p : Dict[UnifiedActionId, Probability]
            The dictionary of actions and their sampled probability of getting a positive reward.
        actions : Optional[Dict[UnifiedActionId, BaseModel]]
            The dictionary of actions and their associated model.

        Returns
        -------
        selected_action: UnifiedActionId
            The selected action.
        """
        p = p.copy()

        # select the action with the highest probability
        selected_action = max(p, key=p.get)

        # exploit with probability exploit_p and not exploit with probability 1-exploit_p
        take_second_max = self.exploit_p <= random() if self.exploit_p != 1 else False

        # select the action with the second-highest probability
        if take_second_max:
            _ = p.pop(selected_action)
            selected_action = max(p, key=p.get)

        return selected_action

    # TODO: WIP this is valid only for SmabBernoulli
    def compare_best_actions(self, actions: Dict[UnifiedActionId, Beta]) -> float:
        """
        Compare the 2 best actions, hence the 2 actions with the highest expected means of getting a positive reward.

        Parameters
        ----------
        actions: Dict[UnifiedActionId, Beta]

        Returns
        ----------
        pvalue: float
            p-value result of the statistical test.
        """
        sorted_actions_mean = sorted([(counter.mean, a) for a, counter in actions.items()], reverse=True)

        _, first_best_action = sorted_actions_mean[0]
        _, second_best_action = sorted_actions_mean[1]

        _, pvalue = ttest_ind_from_stats(
            actions[first_best_action].mean,
            actions[first_best_action].std,
            actions[first_best_action].count,
            actions[second_best_action].mean,
            actions[second_best_action].std,
            actions[second_best_action].count,
            alternative="greater",
        )
        return pvalue


class CostControlStrategy(Strategy, ABC):
    """
    Cost Control (CC) strategy for multi-armed bandits.

    Bandits are extended to include a control of the action cost. Each action is associated with a predefined "cost".
    """

    @classmethod
    @validate_call
    def _average(cls, p_of_action: Union[Probability, List[Probability]]):
        return np.mean(p_of_action)

    @classmethod
    @validate_call
    def _evaluate_and_select(
        cls,
        p: Union[Dict[UnifiedActionId, Probability], Dict[UnifiedActionId, List[Probability]]],
        actions: Dict[UnifiedActionId, BaseModel],
        feasible_actions: List[UnifiedActionId],
    ) -> UnifiedActionId:
        """
        Evaluate the feasible actions and select the one with the minimum cost.

        Parameters
        ----------
        p: Union[Dict[UnifiedActionId, Probability], Dict[UnifiedActionId, List[Probability]]]
            The dictionary of actions and their sampled probability of getting a positive reward.
        actions: Dict[UnifiedActionId, BaseModel]
            The dictionary of actions and their associated model.
        feasible_actions: List[UnifiedActionId]
            The list of feasible actions.

        Returns
        -------
        selected_action: UnifiedActionId
            The selected action.
        """
        # feasible actions enriched with their characteristics (cost, np.mean(probabilities), action_id)
        # the negative probability ensures that if we order the actions based on their minimum quantities the one with
        # higher probability will be selected
        sortable_actions = [
            (
                actions[a[0]].cost(*a[1]) if cls._is_quantitative_action(a) else actions[a].cost,
                -cls._average(p[a]),
                str(a),
            )
            for a in feasible_actions
        ]

        # select the action with the min cost (and the highest mean of probabilities in case of cost equality)
        _, _, selected_action = sorted(sortable_actions)[0]

        # return cheapest action from the set of feasible actions
        return selected_action

    @staticmethod
    def _is_quantitative_action(action: UnifiedActionId) -> bool:
        """
        Check whether action represents a standard action or a quantitive one.

        Parameters
        ----------
        action : UnifiedActionId
            The action identifier to validate.

        Returns
        -------
        bool
            True for quantitive action, False for standard action
        """
        return isinstance(action, tuple)


class CostControlBandit(CostControlStrategy):
    """
    Cost Control (CC) strategy for multi-armed bandits.

    Bandits are extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards are above a pre-defined lower bound.
    Among these actions, the one with the lowest associated cost is recommended. The expected reward interval for
    feasible actions is defined as [(1-subsidy_factor)*max_p, max_p], where max_p is the highest expected reward sampled
    value.

    References
    ----------
    Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
    https://arxiv.org/abs/1911.00638

    Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
    https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    subsidy_factor: Optional[Float01], 0.5 if not specified
        Number in [0, 1] to define smallest tolerated probability reward, hence the set of feasible actions.
        If subsidy_factor is 1, the bandits always selects the action with the minimum cost.
        If subsidy_factor is 0, the bandits always selects the action with highest probability of getting a positive
        reward (it behaves as a classic Bernoulli bandit).
    """

    subsidy_factor: Optional[Float01] = 0.5

    @field_validator("subsidy_factor", mode="before")
    @classmethod
    def numerize_subsidy_factor(cls, v):
        return cls.numerize_field(v, "subsidy_factor")

    @validate_call
    def with_subsidy_factor(self, subsidy_factor: Optional[Float01]) -> Self:
        """
        Instantiate a mutated cost control bandit strategy with an altered subsidy factor.

        Parameters
        ----------
        subsidy_factor : Optional[Float01], 0.5 if not specified
            Number in [0, 1] to define smallest tolerated probability reward, hence the set of feasible actions.
            If subsidy_factor is 1, the bandits always selects the action with the minimum cost.
            If subsidy_factor is 0, the bandits always selects the action with highest probability of getting a positive
            reward (it behaves as a classic Bernoulli bandit).

        Returns
        -------
        mutated_cost_control_bandit : CostControlBandit
            The mutated cost control bandit strategy.
        """
        mutated_cost_control_bandit = self._with_argument("subsidy_factor", subsidy_factor)
        return mutated_cost_control_bandit

    @validate_call
    def select_action(
        self, p: Dict[UnifiedActionId, Probability], actions: Dict[UnifiedActionId, BaseModel]
    ) -> UnifiedActionId:
        """
        Select the action with the minimum cost among the set of feasible actions (the actions whose expected rewards
        are above a certain lower bound defined as [(1-subsidy_factor)*max_p, max_p], where max_p is the highest
        expected reward sampled value.

        Parameters
        ----------
        p: Dict[UnifiedActionId, Probability]
            The dictionary or actions and their sampled probability of getting a positive reward.
        actions: Dict[UnifiedActionId, BetaCC]
            The dictionary or actions and their cost.

        Returns
        -------
        selected_action: UnifiedActionId
            The selected action.
        """
        # get the highest expected reward sampled value
        max_p = max(p.values())

        # define the set of feasible actions
        feasible_actions = [a for a in p.keys() if p[a] >= (1 - self.subsidy_factor) * max_p]

        selected_action = self._evaluate_and_select(p, actions, feasible_actions)
        return selected_action


class MultiObjectiveStrategy(Strategy, ABC):
    """
    Multi Objective Strategy to select actions in multi-armed bandits.
    """

    @classmethod
    @validate_call
    def get_pareto_front(cls, p: Dict[UnifiedActionId, List[Probability]]) -> List[UnifiedActionId]:
        """
        Create Pareto optimal set of actions (Pareto front) A* identified as actions that are not dominated by
        any action out of the set A*.

        Parameters:
        -----------
        p: Dict[UnifiedActionId, Probability]
            The dictionary or actions and their sampled probability of getting a positive reward for each objective.

        Return
        ------
        pareto_front: set
            The list of Pareto optimal actions
        """
        # store non dominated actions
        pareto_front = []

        for this_action in p.keys():
            is_pareto = True  # we assume that action is Pareto Optimal until proven otherwise
            other_actions = [a for a in p.keys() if a != this_action]

            for other_action in other_actions:
                # check if this_action is not dominated by other_action based on
                # multiple objectives reward prob vectors
                is_dominated = not (
                    # an action cannot be dominated by an identical one
                    (p[this_action] == p[other_action])
                    # otherwise, apply the classical definition
                    or any(p[this_action][i] > p[other_action][i] for i in range(len(p[this_action])))
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


class MultiObjectiveBandit(MultiObjectiveStrategy):
    """
    Multi-Objective (MO) strategy for multi-armed bandits.

    The reward pertaining to an action is a multidimensional vector instead of a scalar value. In this setting,
    different actions are compared according to Pareto order between their expected reward vectors, and those actions
    whose expected rewards are not inferior to that of any other actions are called Pareto optimal actions, all of which
    constitute the Pareto front.

    References
    ----------
    Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
    https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem
    """

    @validate_call
    def select_action(self, p: Dict[UnifiedActionId, List[Probability]], **kwargs) -> UnifiedActionId:
        """
        Select an action at random from the Pareto optimal set of action. The Pareto optimal action set (Pareto front)
        A* is the set of actions not dominated by any other actions not in A*. Dominance relation is established based
        on the objective reward probabilities vectors.

        Parameters
        ----------
        p: Dict[ActionId, List[Probability]]
             The dictionary of actions and their sampled probability of getting a positive reward for each objective.

        Returns
        -------
        selected_action: ActionId
            The selected action.
        """
        return np.random.choice(self.get_pareto_front(p=p))


class MultiObjectiveCostControlBandit(MultiObjectiveStrategy, CostControlStrategy):
    """
    Multi-Objective (MO) with Cost Control (CC) strategy for multi-armed bandits.

    This strategy allows the reward to be a multidimensional vector and include a control of the action cost. It merges
    the Multi-Objective and Cost Control strategies.
    """

    @validate_call
    def select_action(
        self, p: Dict[UnifiedActionId, List[Probability]], actions: Dict[UnifiedActionId, BetaMOCC]
    ) -> UnifiedActionId:
        """
        Select the action with the minimum cost among the Pareto optimal set of action. The Pareto optimal
        action set (Pareto front) A* is the set of actions not dominated by any other actions not in A*. Dominance
        relation is established based on the objective reward probabilities vectors.

        Parameters
        ----------
        p: Dict[UnifiedActionId, List[Probability]]
             The dictionary of actions and their sampled probability of getting a positive reward for each objective.

        Returns
        -------
        selected_action: UnifiedActionId
            The selected action.
        """
        pareto_set = self.get_pareto_front(p=p)

        selected_action = self._evaluate_and_select(p, actions, pareto_set)
        return selected_action
