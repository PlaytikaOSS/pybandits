# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
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

from abc import ABC
from typing import List, Optional, Set, Union

from pybandits.actions_manager import (
    SmabActionsManager,
    SmabActionsManagerCC,
    SmabActionsManagerMO,
    SmabActionsManagerMOCC,
    SmabActionsManagerSO,
)
from pybandits.base import (
    ActionId,
    BinaryReward,
    SmabPredictions,
)
from pybandits.mab import BaseMab
from pybandits.model import BaseBeta
from pybandits.pydantic_version_compatibility import PositiveInt, validate_call
from pybandits.quantitative_model import BaseSmabZoomingModel
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
)


class BaseSmabBernoulli(BaseMab, ABC):
    """
    Base model for a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Parameters
    ----------
    actions: Dict[ActionId, Union[BaseBeta, BaseSmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    """

    actions_manager: SmabActionsManager[Union[BaseBeta, BaseSmabZoomingModel]]

    @validate_call
    def predict(
        self,
        n_samples: PositiveInt = 1,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ) -> SmabPredictions:
        """
        Predict actions.

        Parameters
        ----------
        n_samples : PositiveInt, default=1
            Number of samples to predict.
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions: List[UnifiedActionId]
            The actions selected by the multi-armed bandit model.
        probs: Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, MOProbability]]]
            The probabilities of getting a positive reward for each action.
        """

        probs = self._get_action_probabilities(forbidden_actions=forbidden_actions, n_samples=n_samples)
        selected_actions = [self._select_epsilon_greedy_action(p=prob, actions=self.actions) for prob in probs]

        return selected_actions, probs

    @validate_call
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
    ):
        """
        Update the stochastic Bernoulli bandit given the list of selected actions and their corresponding binary
        rewards.

        Parameters
        ----------
        actions : List[ActionId] of shape (n_samples,), e.g. ['a1', 'a2', 'a3', 'a4', 'a5']
            The selected action for each sample.
        rewards : List[Union[BinaryReward, List[BinaryReward]]] of shape (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        super().update(
            actions=actions,
            rewards=rewards,
            quantities=quantities,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
        )


class SmabBernoulli(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling.

    References
    ----------
    Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
    http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions_manager: SmabActionsManagerSO
        The manager for actions and their associated models.
    strategy: ClassicBandit
        The strategy used to select actions.
    """

    actions_manager: SmabActionsManagerSO
    strategy: ClassicBandit


class SmabBernoulliBAI(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    References
    ----------
    Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
    http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions_manager: SmabActionsManagerSO
        The manager for actions and their associated models.
    strategy: BestActionIdentificationBandit
        The strategy used to select actions.
    """

    actions_manager: SmabActionsManagerSO
    strategy: BestActionIdentificationBandit


class SmabBernoulliCC(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control strategy.

    The sMAB is extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards is above a pre-defined lower bound. Among
    these actions, the one with the lowest associated cost is recommended. The expected reward interval for feasible
    actions is defined as [(1-subsidy_factor) * max_p, max_p], where max_p is the highest expected reward sampled value.

    References
    ----------
    Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
    https://arxiv.org/abs/1911.00638

    Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
    https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    actions_manager: SmabActionsManagerCC
        The manager for actions and their associated models.
    strategy: CostControlBandit
        The strategy used to select actions.
    """

    actions_manager: SmabActionsManagerCC
    strategy: CostControlBandit


class SmabBernoulliMO(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Multi-Objectives strategy.

    The reward pertaining to an action is a multidimensional vector instead of a scalar value. In this setting,
    different actions are compared according to Pareto order between their expected reward vectors, and those actions
    whose expected rewards are not inferior to that of any other actions are called Pareto optimal actions, all of which
    constitute the Pareto front.

    References
    ----------
    Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
    https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem

    Parameters
    ----------
    actions_manager: SmabActionsManagerMO
        The manager for actions and their associated models.
    strategy: MultiObjectiveBandit
        The strategy used to select actions.
    """

    actions_manager: SmabActionsManagerMO
    strategy: MultiObjectiveBandit


class SmabBernoulliMOCC(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling implementation for Multi-Objective (MO) with Cost
    Control (CC) strategy.

    This Bandit allows the reward to be a multidimensional vector and include a control of the action cost. It merges
    the Multi-Objective and Cost Control strategies.

    Parameters
    ----------
    actions_manager: SmabActionsManagerMOCC
        The manager for actions and their associated models.
    strategy: MultiObjectiveCostControlBandit
        The strategy used to select actions.
    """

    actions_manager: SmabActionsManagerMOCC
    strategy: MultiObjectiveCostControlBandit
