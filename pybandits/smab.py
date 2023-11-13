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

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import NonNegativeFloat, PositiveInt, validate_arguments, validator

from pybandits.base import (
    ActionId,
    BaseMab,
    BinaryReward,
    Float01,
    Probability,
    Strategy,
)
from pybandits.model import BaseBeta, BaseBetaMO, Beta, BetaCC, BetaMO, BetaMOCC
from pybandits.strategy import (
    BestActionIdentification,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
)


class BaseSmabBernoulli(BaseMab):
    """
    Base model for a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Parameters
    ----------
    actions: Dict[ActionId, BaseBeta]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BaseBeta]

    @validate_arguments
    def predict(
        self,
        n_samples: PositiveInt = 1,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ) -> Tuple[List[ActionId], List[Dict[ActionId, Probability]]]:
        """
        Predict actions.

        Parameters
        ----------
        n_samples : int > 0, default=1
            Number of samples to predict.
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions: List[ActionId] of shape (n_samples,)
            The actions selected by the multi-armed bandit model.
        probs: List[Dict[ActionId, Probability]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action.
        """
        valid_actions = self._get_valid_actions(forbidden_actions)

        selected_actions: List[ActionId] = []
        probs: List[Dict[ActionId, Probability]] = []

        for _ in range(n_samples):
            p = {action: model.sample_proba() for action, model in self.actions.items() if action in valid_actions}
            selected_actions.append(self.strategy.select_action(p=p, actions=self.actions))
            probs.append(p)

        return selected_actions, probs

    @validate_arguments
    def update(
        self,
        actions: List[ActionId],
        rewards: List[Union[BinaryReward, List[BinaryReward]]],
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
        """
        self._check_update_params(actions=actions, rewards=rewards)

        rewards_dict = defaultdict(list)

        for a, r in zip(actions, rewards):
            rewards_dict[a].append(r)

        for a in set(actions):
            self.actions[a].update(rewards=rewards_dict[a])


class SmabBernoulli(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[ActionId, Beta]
        The list of possible actions, and their associated Model.
    strategy: ClassicBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Beta]
    strategy: ClassicBandit

    def __init__(self, actions: Dict[ActionId, Beta]):
        super().__init__(actions=actions, strategy=ClassicBandit())

    @classmethod
    def from_state(cls, state: dict) -> "SmabBernoulli":
        return cls(actions=state["actions"])

    @validate_arguments
    def update(self, actions: List[ActionId], rewards: List[BinaryReward]):
        super().update(actions=actions, rewards=rewards)


class SmabBernoulliBAI(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[ActionId, Beta]
        The list of possible actions, and their associated Model.
    strategy: BestActionIdentification
        The strategy used to select actions.
    """

    actions: Dict[ActionId, Beta]
    strategy: BestActionIdentification

    def __init__(self, actions: Dict[ActionId, Beta], exploit_p: Optional[Float01] = None):
        strategy = BestActionIdentification() if exploit_p is None else BestActionIdentification(exploit_p=exploit_p)
        super().__init__(actions=actions, strategy=strategy)

    @classmethod
    def from_state(cls, state: dict) -> "SmabBernoulliBAI":
        return cls(actions=state["actions"], exploit_p=state["strategy"].get("exploit_p", None))

    @validate_arguments
    def update(self, actions: List[ActionId], rewards: List[BinaryReward]):
        super().update(actions=actions, rewards=rewards)


class SmabBernoulliCC(BaseSmabBernoulli):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control strategy.

    The sMAB is extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards is above a pre-defined lower bound. Among
    these actions, the one with the lowest associated cost is recommended. The expected reward interval for feasible
    actions is defined as [(1-subsidy_factor) * max_p, max_p], where max_p is the highest expected reward sampled value.

    Reference: Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
               https://arxiv.org/abs/1911.00638

               Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
               https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    actions: Dict[ActionId, BetaCC]
        The list of possible actions, and their associated Model.
    strategy: CostControlBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BetaCC]
    strategy: CostControlBandit

    def __init__(self, actions: Dict[ActionId, BetaCC], subsidy_factor: Optional[Float01] = None):
        strategy = CostControlBandit() if subsidy_factor is None else CostControlBandit(subsidy_factor=subsidy_factor)
        super().__init__(actions=actions, strategy=strategy)

    @classmethod
    def from_state(cls, state: dict) -> "SmabBernoulliCC":
        return cls(actions=state["actions"], subsidy_factor=state["strategy"].get("subsidy_factor", None))

    @validate_arguments
    def update(self, actions: List[ActionId], rewards: List[BinaryReward]):
        super().update(actions=actions, rewards=rewards)


class BaseSmabBernoulliMO(BaseSmabBernoulli):
    """
    Base model for a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling implementation, and Multi-Objectives
    strategy.

    Parameters
    ----------
    actions: Dict[ActionId, BetaMO]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BaseBetaMO]
    strategy: Strategy

    @validator("actions", pre=False)
    @classmethod
    def all_actions_have_same_number_of_objectives(cls, actions: Dict[ActionId, BaseBetaMO]):
        n_objs_per_action = [len(beta.counters) for beta in actions.values()]
        if len(set(n_objs_per_action)) != 1:
            raise ValueError("All actions should have the same number of objectives")
        return actions

    @validate_arguments
    def update(self, actions: List[ActionId], rewards: List[List[BinaryReward]]):
        super().update(actions=actions, rewards=rewards)


class SmabBernoulliMO(BaseSmabBernoulliMO):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Multi-Objectives strategy.

    The reward pertaining to an action is a multidimensional vector instead of a scalar value. In this setting,
    different actions are compared according to Pareto order between their expected reward vectors, and those actions
    whose expected rewards are not inferior to that of any other actions are called Pareto optimal actions, all of which
    constitute the Pareto front.

    Reference: Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
               https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem

    Parameters
    ----------
    actions: Dict[ActionId, BetaMO]
        The list of possible actions, and their associated Model.
    strategy: MultiObjectiveBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BetaMO]
    strategy: MultiObjectiveBandit

    def __init__(self, actions: Dict[ActionId, Beta]):
        super().__init__(actions=actions, strategy=MultiObjectiveBandit())

    @classmethod
    def from_state(cls, state: dict) -> "SmabBernoulliMO":
        return cls(actions=state["actions"])


class SmabBernoulliMOCC(BaseSmabBernoulliMO):
    """
    Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling implementation for Multi-Objective (MO) with Cost
    Control (CC) strategy.

    This Bandit allows the reward to be a multidimensional vector and include a control of the action cost. It merges
    the Multi-Objective and Cost Control strategies.

    Parameters
    ----------
    actions: Dict[ActionId, BetaMOCC]
        The list of possible actions, and their associated Model.
    strategy: MultiObjectiveCostControlBandit
        The strategy used to select actions.
    """

    actions: Dict[ActionId, BetaMOCC]
    strategy: MultiObjectiveCostControlBandit

    def __init__(self, actions: Dict[ActionId, Beta]):
        super().__init__(actions=actions, strategy=MultiObjectiveCostControlBandit())

    @classmethod
    def from_state(cls, state: dict) -> "SmabBernoulliMOCC":
        return cls(actions=state["actions"])


@validate_arguments
def create_smab_bernoulli_cold_start(action_ids: Set[ActionId]) -> SmabBernoulli:
    """
    Utility function to create a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, with default
    parameters.

    Parameters
    ----------
    action_ids: Set[ActionId]
        The list of possible actions.

    Returns
    -------
    smab: SmabBernoulli
        Stochastic Multi-Armed Bandit with strategy = ClassicBandit
    """
    actions = {}
    for a in set(action_ids):
        actions[a] = Beta()
    return SmabBernoulli(actions=actions)


@validate_arguments
def create_smab_bernoulli_bai_cold_start(
    action_ids: Set[ActionId], exploit_p: Optional[Float01] = None
) -> SmabBernoulliBAI:
    """
    Utility function to create a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action
    Identification strategy, with default parameters.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    action_ids: Set[ActionId]
        The list of possible actions.
    exploit_p: Float_0_1 (default=0.5)
        Number in [0, 1] which specifies the amount of exploitation.
        If exploit_p is 1, the bandits always selects the action with highest probability of getting a positive reward,
            (it behaves as a Greedy strategy).
        If exploit_p is 0, the bandits always select the action with 2nd highest probability of getting a positive
            reward.

    Returns
    -------
    smab: SmabBernoulliBAI
        Stochastic Multi-Armed Bandit with strategy = BestActionIdentification
    """
    actions = {}
    for a in set(action_ids):
        actions[a] = Beta()
    return SmabBernoulliBAI(actions=actions, exploit_p=exploit_p)


@validate_arguments
def create_smab_bernoulli_cc_cold_start(
    action_ids_cost: Dict[ActionId, NonNegativeFloat],
    subsidy_factor: Optional[Float01] = None,
) -> SmabBernoulliCC:
    """
    Utility function to create a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control
    strategy, with default parameters.

    The sMAB is extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards is above a pre-defined lower bound. Among
    these actions, the one with the lowest associated cost is recommended. The expected reward interval for feasible
    actions is defined as [(1-subsidy_factor) * max_p, max_p], where max_p is the highest expected reward sampled value.

    Reference: Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
               https://arxiv.org/abs/1911.00638

               Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
               https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    action_ids_cost: Dict[ActionId, NonNegativeFloat]
        The list of possible actions, and their cost.
    subsidy_factor: Optional[Float_0_1], default=0.5
        Number in [0, 1] to define smallest tolerated probability reward, hence the set of feasible actions.
        If subsidy_factor is 1, the bandits always selects the action with the minimum cost.
        If subsidy_factor is 0, the bandits always selects the action with highest probability of getting a positive
            reward (it behaves as a classic Bernoulli bandit).

    Returns
    -------
    smab: SmabBernoulliCC
        Stochastic Multi-Armed Bandit with strategy = CostControlBandit
    """
    actions = {}
    for a, cost in action_ids_cost.items():
        actions[a] = BetaCC(cost=cost)
    return SmabBernoulliCC(actions=actions, subsidy_factor=subsidy_factor)


@validate_arguments
def create_smab_bernoulli_mo_cold_start(action_ids: Set[ActionId], n_objectives: PositiveInt) -> SmabBernoulliMO:
    """
    Utility function to create a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling, and Multi-Objectives
    strategy, with default parameters.

    The reward pertaining to an action is a multidimensional vector instead of a scalar value. In this setting,
    different actions are compared according to Pareto order between their expected reward vectors, and those actions
    whose expected rewards are not inferior to that of any other actions are called Pareto optimal actions, all of which
    constitute the Pareto front.

    Reference: Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
               https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem

    Parameters
    ----------
    action_ids: Set[ActionId]
        The list of possible actions.
    n_objectives: PositiveInt
        The number of objectives to optimize. The bandit assumes the same number of objectives for all actions.

    Returns
    -------
    smab: SmabBernoulliMO
        Stochastic Multi-Armed Bandit with strategy = MultiObjectiveBandit
    """
    actions = {}
    for a in set(action_ids):
        actions[a] = BetaMO(counters=n_objectives * [Beta()])
    return SmabBernoulliMO(actions=actions)


@validate_arguments
def create_smab_bernoulli_mo_cc_cold_start(
    action_ids_cost: Dict[ActionId, NonNegativeFloat], n_objectives: PositiveInt
) -> SmabBernoulliMOCC:
    """
    Utility function to create a Stochastic Bernoulli Multi-Armed Bandit with Thompson Sampling implementation for
    Multi-Objective (MO) with Cost Control (CC) strategy, with default parameters.

    This Bandit allows the reward to be a multidimensional vector and include a control of the action cost. It merges
    the Multi-Objective and Cost Control strategies.

    Parameters
    ----------
    action_ids_cost: Dict[ActionId, NonNegativeFloat]
        The list of possible actions, and their cost.
    n_objectives: PositiveInt
        The number of objectives to optimize. The bandit assumes the same number of objectives for all actions.

    Returns
    -------
    smab: SmabBernoulliMO
        Stochastic Multi-Armed Bandit with strategy = MultiObjectiveBandit
    """
    actions = {}
    for a, cost in action_ids_cost.items():
        actions[a] = BetaMOCC(counters=n_objectives * [Beta()], cost=cost)
    return SmabBernoulliMOCC(actions=actions)
