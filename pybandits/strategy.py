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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import PrivateAttr, field_validator, validate_call
from scipy.stats import ttest_ind_from_stats
from typing_extensions import Self

from pybandits.base import ActionId, BinaryReward, Float01, Probability, PyBanditsBaseModel
from pybandits.model import Beta, Model


class Strategy(PyBanditsBaseModel, ABC):
    """
    Strategy to select actions in multi-armed bandits.
    """

    def _with_argument(self, argument_name: str, argument_value: Any) -> Self:
        """
        Instantiate a mutated strategy with an altered argument_value for argument_name.

        Parameters
        ----------
        argument_name: str
            The name of the argument.
        argument_value: Any
            The value of the argument.

        Returns
        -------
        mutated_strategy: Strategy
            The mutated strategy.
        """
        mutated_strategy = self.model_copy(update={argument_name: argument_value}, deep=True)
        return mutated_strategy

    @abstractmethod
    def select_action(self, p: Dict[ActionId, Probability], actions: Optional[Dict[ActionId, Model]]) -> ActionId:
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

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Reference: Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal and Goyal, 2014)
               https://arxiv.org/pdf/1209.3352.pdf
    """

    @validate_call
    def select_action(
        self,
        p: Dict[ActionId, float],
        actions: Optional[Dict[ActionId, Model]] = None,
    ) -> ActionId:
        """
        Select the action with the highest probability of getting a positive reward.

        Parameters
        ----------
        p : Dict[ActionId, Probability]
            The dictionary of actions and their sampled probability of getting a positive reward.
        actions : Optional[Dict[ActionId, Model]]
            The dictionary of actions and their associated model.

        Returns
        -------
        selected_action: ActionId
            The selected action.
        """
        return max(p, key=p.get)


class BestActionIdentificationBandit(Strategy):
    """
    Best-Action Identification (BAI) strategy for multi-armed bandits.

    Reference: Simple Bayesian Algorithms for Best-Arm Identification (Russo, 2018)
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
        p: Dict[ActionId, float],
        actions: Optional[Dict[ActionId, Model]] = None,
    ) -> ActionId:
        """
        Select with probability self.exploit_p the best action (i.e. the action with the highest probability of getting
        a positive reward), and with probability 1-self.exploit_p it returns the second best action (i.e. the action
        with the second highest probability of getting a positive reward).

        Parameters
        ----------
        p : Dict[ActionId, Probability]
            The dictionary of actions and their sampled probability of getting a positive reward.
        actions : Optional[Dict[ActionId, Model]]
            The dictionary of actions and their associated model.

        Returns
        -------
        selected_action: ActionId
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
    def compare_best_actions(self, actions: Dict[ActionId, Beta]) -> float:
        """
        Compare the 2 best actions, hence the 2 actions with the highest expected means of getting a positive reward.

        Parameters
        ----------
        actions: Dict[ActionId, Beta]

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

    Parameters
    ----------
    subsidy_factor: Optional[Union[Float01, Beta, List[Beta]]], 0.5 if not specified
        Number in [0, 1] to define smallest tolerated probability reward, hence the set of feasible actions.
        If subsidy_factor is 1, the bandits always selects the action with the minimum cost.
        If subsidy_factor is 0, the bandits always selects the action with highest probability of getting a positive
            reward (it behaves as a classic Bernoulli bandit).
    loss_factor : Float01, defaults to 0
        Number in [0, 1] that controls the tradeoff between the cost and the expected reward.
        The tradeoff is characterized via the convex combination of the normalized cost
        and the negated normalized expected reward.
        If loss_factor is 0 or 1, the criterion for choosing the optimal action from the feasible set is based only
        on the cost or expected reward, respectively.
    """

    subsidy_factor: Optional[Union[Float01, Beta, List[Beta]]] = 0.5
    loss_factor: Float01 = 0.0
    _buffer = []

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

    def reset(self):
        if isinstance(self.subsidy_factor, Beta) or isinstance(self.subsidy_factor, list):
            self._buffer = []

    def _realize_subsidy_factor(self, subsidy_factor: Union[Float01, Beta]) -> Float01:
        """
        Realize the subsidy factor.
        If the subsidy factor is a model, sample from it. Otherwise, return the subsidy factor.

        Returns
        -------
        subsidy_factor: Float01
            The realized subsidy factor.
        """
        if isinstance(subsidy_factor, Beta):
            subsidy_factor = subsidy_factor.sample_proba()
        elif isinstance(subsidy_factor, float):
            subsidy_factor = subsidy_factor
        else:
            raise ValueError(f"Invalid subsidy factor type: {type(subsidy_factor)}")
        return subsidy_factor

    @validate_call
    def _evaluate_feasible_actions(
        self, p: Dict[ActionId, Probability], actions: Dict[ActionId, Model], subsidy_factor: Union[Float01, Beta]
    ) -> Tuple[Dict[ActionId, Probability], ActionId]:
        """
        Select the action with the minimum cost among the set of feasible actions.
        Feasible actions are the ones whose expected rewards are above (1-subsidy_factor)*max_p.

        Parameters
        ----------
        p : Dict[ActionId, Probability]
            The dictionary or actions and their sampled probability of getting a positive reward.
        actions : Dict[ActionId, Model]
            The dictionary of actions and their characteristics.
        subsidy_factor : Union[Float01, Beta]
            The subsidy factor.


        Returns
        -------
        feasible_actions_loss : Dict[ActionId, Probability]
            The dictionary of feasible actions and their loss.
        max_p_action : ActionId
            The highest expected reward action identifier.
        """
        realized_subsidy_factor = self._realize_subsidy_factor(subsidy_factor)
        # get the highest expected reward sampled value
        max_p = max(p.values())
        # find action whose value is max_p and has the lowest cost
        max_p_actions = [a for a in p.keys() if p[a] == max_p]
        max_p_action = min(max_p_actions, key=lambda action_id: (actions[action_id].cost, action_id))
        max_p_action_cost = actions[max_p_action].cost
        if max_p_action_cost == 0 or max_p == 0:
            feasible_actions_loss = {max_p_action: 0.0}
        else:
            feasible_actions_loss = {
                a: self.loss_factor * (1 - p[a] / max_p) + (1 - self.loss_factor) * actions[a].cost / max_p_action_cost
                for a in p.keys()
                if p[a] >= (1 - realized_subsidy_factor) * max_p and actions[a].cost <= max_p_action_cost
            }

        return feasible_actions_loss, max_p_action

    @validate_call
    def update(self, rewards: Union[List[BinaryReward], List[List[BinaryReward]]]):
        """
        Update the subsidy factor based on the rewards and the buffered relative costs.

        Parameters
        ----------
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]]
            The binary reward for each sample.
            If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                rewards = [1, 0, 1, 1, 1, ...]
            If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]

        """
        if type(self.subsidy_factor) is Beta:
            self._update_subsidy_factor(subsidy_factor=self.subsidy_factor, rewards=rewards, buffer=self._buffer)
        elif isinstance(self.subsidy_factor, list) and all(
            isinstance(subsidy_factor, Beta) for subsidy_factor in self.subsidy_factor
        ):
            for subsidy_factor, single_objective_rewards, single_objective_buffer in zip(
                self.subsidy_factor, zip(*rewards), zip(*self._buffer)
            ):
                self._update_subsidy_factor(
                    subsidy_factor=subsidy_factor, rewards=single_objective_rewards, buffer=single_objective_buffer
                )

    def _update_subsidy_factor(self, subsidy_factor: Beta, rewards: List[BinaryReward], buffer: List[Probability]):
        """
        Update the subsidy factor based on the rewards and the relative cost.

        Parameters
        ----------
        subsidy_factor : Beta
            The subsidy factor model.
        rewards : List[BinaryReward]
            The binary reward for each sample.
        buffer : List[Probability]
            The buffer of relative costs.

        """

        rewards = [
            (1 - r) * self.loss_factor + relative_cost > 0.5
            for relative_cost, r in zip(buffer, rewards)
            if relative_cost is not None
        ]
        subsidy_factor.update(rewards=rewards)


class CostControlBandit(CostControlStrategy):
    """
    Cost Control (CC) strategy for multi-armed bandits.

    Bandits are extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards are above a pre-defined lower bound.
    Among these actions, the one with the lowest associated cost is recommended. The expected reward interval for
    feasible actions is defined as [(1-subsidy_factor)*max_p, max_p], where max_p is the highest expected reward sampled
    value.

    Reference: Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
               https://arxiv.org/abs/1911.00638

               Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
               https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    subsidy_factor: Optional[Union[Float01, Beta]], 0.5 if not specified
        Number in [0, 1] to define smallest tolerated probability reward, hence the set of feasible actions.
        If subsidy_factor is 1, the bandits always selects the action with the minimum cost.
        If subsidy_factor is 0, the bandits always selects the action with highest probability of getting a positive
            reward (it behaves as a classic Bernoulli bandit).
    loss_factor : Float01, defaults to 0
        Number in [0, 1] that controls the tradeoff between the cost and the expected reward.
        The tradeoff is characterized via the convex combination of the normalized cost
        and the negated normalized expected reward.
        If loss_factor is 0 or 1, the criterion for choosing the optimal action from the feasible set is based only
        on the cost or expected reward, respectively.

    """

    subsidy_factor: Optional[Union[Float01, Beta]] = 0.5

    @validate_call
    def select_action(self, p: Dict[ActionId, Probability], actions: Dict[ActionId, Model]) -> ActionId:
        """
        Select the action with the minimum cost among the set of feasible actions (the actions whose expected rewards
        are above a certain lower bound. defined as [(1-subsidy_factor)*max_p, max_p], where max_p is the highest
        expected reward sampled value.

        Parameters
        ----------
        p: Dict[ActionId, Probability]
            The dictionary or actions and their sampled probability of getting a positive reward.
        actions: Dict[ActionId, BetaCC]
            The dictionary or actions and their cost.

        Returns
        -------
        selected_action: ActionId
            The selected action.
        """
        feasible_actions_loss, max_p_action = self._evaluate_feasible_actions(
            p=p, actions=actions, subsidy_factor=self.subsidy_factor
        )
        selected_action = min(
            feasible_actions_loss,
            key=lambda action_id: (
                feasible_actions_loss[action_id],
                -p[action_id] if self.loss_factor > 0.5 else actions[action_id].cost,
                actions[action_id].cost if self.loss_factor > 0.5 else -p[action_id],
                action_id,
            ),
        )
        max_p_action_cost = actions[max_p_action].cost
        self._buffer.append(
            (1 - self.loss_factor) * actions[selected_action].cost / max_p_action_cost
            if max_p_action_cost > 0
            else None
        )
        return selected_action


class MultiObjectiveStrategy(Strategy, ABC):
    """
    Multi Objective Strategy to select actions in multi-armed bandits.
    """

    @classmethod
    @validate_call
    def _get_pareto_front(cls, p: Dict[ActionId, List[Probability]]) -> List[ActionId]:
        """
        Create Pareto optimal set of actions (Pareto front) A* identified as actions that are not dominated by
        any action out of the set A*.

        Parameters:
        -----------
        p: Dict[ActionId, Probability]
            The dictionary or actions and their sampled probability of getting a positive reward for each objective.

        Return
        ------
        pareto_front: List[ActionId]
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

    @validate_call
    def select_action(self, p: Dict[ActionId, List[Probability]], **kwargs) -> ActionId:
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
        return np.random.choice(self._get_pareto_front(self._alter_p(p, **kwargs)))

    @abstractmethod
    def _alter_p(self, p: Dict[ActionId, List[Probability]], **kwargs) -> Dict[ActionId, List[Probability]]:
        """
        Alter the probability of getting a positive reward for each objective.

        Parameters
        ----------
        p : Dict[ActionId, List[Probability]]
            The dictionary of actions and their sampled probability of getting a positive reward for each objective.

        Returns
        -------
        altered_p : Dict[ActionId, List[Probability]]
            The altered probability of getting a positive reward for each objective.
        """
        pass


class MultiObjectiveBandit(MultiObjectiveStrategy):
    """
    Multi-Objective (MO) strategy for multi-armed bandits.

    The reward pertaining to an action is a multidimensional vector instead of a scalar value. In this setting,
    different actions are compared according to Pareto order between their expected reward vectors, and those actions
    whose expected rewards are not inferior to that of any other actions are called Pareto optimal actions, all of which
    constitute the Pareto front.

    Reference: Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
               https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem
    """

    def _alter_p(self, p: Dict[ActionId, List[Probability]], **kwargs) -> Dict[ActionId, List[Probability]]:
        """
        Return the probability of getting a positive reward for each objective.

        Parameters
        ----------
        p : Dict[ActionId, List[Probability]]
            The dictionary of actions and their sampled probability of getting a positive reward for each objective.

        Returns
        -------
        altered_p : Dict[ActionId, List[Probability]]
            The dictionary of actions and their sampled probability of getting a positive reward for each objective.
        """
        altered_p = p
        return altered_p


class MultiObjectiveCostControlBandit(MultiObjectiveStrategy, CostControlStrategy):
    """
    Multi-Objective (MO) with Cost Control (CC) strategy for multi-armed bandits.

    This strategy allows the reward to be a multidimensional vector and include a control of the action cost. It merges
    the Multi-Objective and Cost Control strategies.
    """

    subsidy_factor: Optional[Union[Float01, List[Beta]]] = 0.5
    _feasible_actions_losses = PrivateAttr([])

    @validate_call
    def select_action(self, p: Dict[ActionId, List[Probability]], actions: Dict[ActionId, Model]) -> ActionId:
        """
        Select an action at random from the Pareto optimal set of action. The Pareto optimal action set (Pareto front)
        A* is the set of actions not dominated by any other actions not in A*. Dominance relation is established based
        on the objective reward probabilities vectors.

        Parameters
        ----------
        p : Dict[ActionId, List[Probability]]
             The dictionary of actions and their sampled probability of getting a positive reward for each objective.
        actions : Dict[ActionId, Model]
            The dictionary of actions and their associated model.

        Returns
        -------
        selected_action: ActionId
            The selected action.
        """
        self._buffer.append([])

        selected_action = super().select_action(p=p, actions=actions)

        for feasible_actions_loss, max_p_action in self._feasible_actions_losses:
            max_p_action_cost = actions[max_p_action].cost
            if selected_action in feasible_actions_loss and max_p_action_cost > 0:
                max_p_action_cost = actions[max_p_action].cost
                self._buffer[-1].append((1 - self.loss_factor) * actions[selected_action].cost / max_p_action_cost)
            else:
                self._buffer[-1].append(None)
        return selected_action

    def _alter_p(
        self, p: Dict[ActionId, List[Probability]], actions: Dict[ActionId, Model]
    ) -> Dict[ActionId, List[Probability]]:
        """
        Alter the probability of getting a positive reward for each objective. The probability is replaced with the
        convex combination of the relative cost and the negated normalized expected reward.

        Parameters
        ----------
        p : Dict[ActionId, List[Probability]]
            The dictionary of actions and their sampled probability of getting a positive reward for each objective.
        actions : Dict[ActionId, Model]
            The dictionary of actions and their associated model.

        Returns
        -------
        altered_p : Dict[ActionId, List[Probability]]
            The altered probability of getting a positive reward for each objective
        """
        self._feasible_actions_losses = []
        action_ids = p.keys()
        altered_p = {action_id: [] for action_id in action_ids}
        for objective_id, single_objective_rewards in enumerate(zip(*p.values())):
            single_objective_p = dict(zip(action_ids, single_objective_rewards))
            feasible_actions_loss, max_p_action = self._evaluate_feasible_actions(
                p=single_objective_p,
                actions=actions,
                subsidy_factor=self.subsidy_factor[objective_id]
                if isinstance(self.subsidy_factor, list)
                else self.subsidy_factor,
            )
            {
                action_id: altered_p[action_id].append(feasible_actions_loss.get(action_id, 1.0))
                for action_id in action_ids
            }

            self._feasible_actions_losses.append((feasible_actions_loss, max_p_action))
        return altered_p
