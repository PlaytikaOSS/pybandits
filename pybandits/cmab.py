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

from typing import Dict, List, Optional, Set, Tuple, Union

from numpy import array
from numpy.random import choice
from numpy.typing import ArrayLike
from pydantic import NonNegativeFloat, PositiveInt, field_validator, validate_call

from pybandits.base import ActionId, BaseMab, BinaryReward, Float01, Probability
from pybandits.model import (
    BaseBayesianLogisticRegression,
    BayesianLogisticRegression,
    BayesianLogisticRegressionCC,
    UpdateMethods,
    create_bayesian_logistic_regression_cc_cold_start,
    create_bayesian_logistic_regression_cold_start,
)
from pybandits.strategy import (
    BestActionIdentification,
    ClassicBandit,
    CostControlBandit,
)


class BaseCmabBernoulli(BaseMab):
    """
    Base model for a Contextual Multi-Armed Bandit for Bernoulli bandits with Thompson Sampling.

    Parameters
    ----------
    actions: Dict[ActionId, BaseBayesianLogisticRegression]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    predict_with_proba: bool
        If True predict with sampled probabilities, else predict with weighted sums.
    predict_actions_randomly: bool
        If True predict actions randomly (where each action has equal probability to be selected), else predict with the
        bandit strategy.
    """

    actions: Dict[ActionId, BaseBayesianLogisticRegression]
    predict_with_proba: bool
    predict_actions_randomly: bool

    @field_validator("actions", mode="after")
    @classmethod
    def check_bayesian_logistic_regression_models(cls, v):
        action_models = list(v.values())
        first_action = action_models[0]
        first_action_type = type(first_action)
        for action in action_models[1:]:
            if not isinstance(action, first_action_type):
                raise AttributeError("All actions should follow the same type.")
            if not len(action.betas) == len(first_action.betas):
                raise AttributeError("All actions should have the same number of betas.")
            if not action.update_method == first_action.update_method:
                raise AttributeError("All actions should have the same update method.")
            if not action.update_kwargs == first_action.update_kwargs:
                raise AttributeError("All actions should have the same update kwargs.")
        return v

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        context: ArrayLike,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ) -> Tuple[List[ActionId], List[Dict[ActionId, Probability]]]:
        """
        Predict actions.

        Parameters
        ----------
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
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

        # cast inputs to numpy arrays to facilitate their manipulation
        context = array(context)

        if len(context) < 1:
            raise AttributeError("Context must have at least one row")

        if self.predict_actions_randomly:
            # check that context has the expected number of columns
            if context.shape[1] != len(list(self.actions.values())[0].betas):
                raise AttributeError("Context must have {n_betas} columns")

            selected_actions = choice(list(valid_actions), size=len(context)).tolist()  # predict actions randomly
            probs = len(context) * [{k: 0.5 for k in valid_actions}]  # all probs are set to 0.5
            weighted_sums = len(context) * [{k: 0 for k in valid_actions}]  # all weighted sum are set to 1
        else:
            # p is a dict of the sampled probability "prob" and weighted_sum "ws", e.g.
            #
            # p = {'a1': ([0.5, 0.2, 0.3], [200, 100, 130]), 'a2': ([0.4, 0.5, 0.6], [180, 200, 230]), ...}
            #               |               |                           |               |
            #              prob             ws                          prob            ws
            p = {
                action: model.sample_proba(context=context)  # sample probabilities for the entire context matrix
                for action, model in self.actions.items()
                if action in valid_actions
            }

            prob = {a: x[0] for a, x in p.items()}  # e.g. prob = {'a1': [0.5, 0.4, ...], 'a2': [0.4, 0.3, ...], ...}
            ws = {a: x[1] for a, x in p.items()}  # e.g. ws = {'a1': [200, 100, ...], 'a2': [100, 50, ...], ...}

            # select either "prob" or "ws" to use as input argument in select_actions()
            p_to_select_action = prob if self.predict_with_proba else ws

            # predict actions, probs, weighted_sums
            selected_actions = [
                self._select_epsilon_greedy_action(
                    p={a: p_to_select_action[a][i] for a in p_to_select_action}, actions=self.actions
                )
                for i in range(len(context))
            ]
            probs = [{a: prob[a][i] for a in prob} for i in range(len(context))]
            weighted_sums = [{a: ws[a][i] for a in ws} for i in range(len(context))]

        return selected_actions, probs, weighted_sums

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self, context: ArrayLike, actions: List[ActionId], rewards: List[Union[BinaryReward, List[BinaryReward]]]
    ):
        """
        Update the contextual Bernoulli bandit given the list of selected actions and their corresponding binary
        rewards.

        Parameters
        ----------
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
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
        if len(context) != len(rewards):
            raise AttributeError(f"Shape mismatch: actions and rewards should have the same length {len(actions)}.")

        # cast inputs to numpy arrays to facilitate their manipulation
        context, actions, rewards = array(context), array(actions), array(rewards)

        for a in set(actions):
            # get context and rewards of the samples associated to action a
            context_of_a = context[actions == a]
            rewards_of_a = rewards[actions == a].tolist()

            # update model associated to action a
            self.actions[a].update(context=context_of_a, rewards=rewards_of_a)

        # always set predict_actions_randomly after update
        self.predict_actions_randomly = False


class CmabBernoulli(BaseCmabBernoulli):
    """
    Contextual  Bernoulli Multi-Armed Bandit with Thompson Sampling.

    Reference: Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal and Goyal, 2014)
               https://arxiv.org/pdf/1209.3352.pdf

    Parameters
    ----------
    actions: Dict[ActionId, BayesianLogisticRegression]
        The list of possible actions, and their associated Model.
    strategy: ClassicBandit
        The strategy used to select actions.
    predict_with_proba: bool
        If True predict with sampled probabilities, else predict with weighted sums
    predict_actions_randomly: bool
        If True predict actions randomly (where each action has equal probability to be selected), else predict with the
        bandit strategy.
    """

    actions: Dict[ActionId, BayesianLogisticRegression]
    strategy: ClassicBandit
    predict_with_proba: bool = False
    predict_actions_randomly: bool = False

    def __init__(
        self,
        actions: Dict[ActionId, BaseBayesianLogisticRegression],
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
    ):
        super().__init__(actions=actions, strategy=ClassicBandit(), epsilon=epsilon, default_action=default_action)

    @classmethod
    def from_state(cls, state: dict) -> "CmabBernoulli":
        return cls(actions=state["actions"])

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(self, context: ArrayLike, actions: List[ActionId], rewards: List[BinaryReward]):
        super().update(context=context, actions=actions, rewards=rewards)


class CmabBernoulliBAI(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[ActionId, BayesianLogisticRegression]
        The list of possible actions, and their associated Model.
    strategy: BestActionIdentification
        The strategy used to select actions.
    predict_with_proba: bool
        If True predict with sampled probabilities, else predict with weighted sums
    predict_actions_randomly: bool
        If True predict actions randomly (where each action has equal probability to be selected), else predict with the
        bandit strategy.
    """

    actions: Dict[ActionId, BayesianLogisticRegression]
    strategy: BestActionIdentification
    predict_with_proba: bool = False
    predict_actions_randomly: bool = False

    def __init__(
        self,
        actions: Dict[ActionId, BayesianLogisticRegression],
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        exploit_p: Optional[Float01] = None,
    ):
        strategy = BestActionIdentification() if exploit_p is None else BestActionIdentification(exploit_p=exploit_p)
        super().__init__(actions=actions, strategy=strategy, epsilon=epsilon, default_action=default_action)

    @classmethod
    def from_state(cls, state: dict) -> "CmabBernoulliBAI":
        return cls(actions=state["actions"], exploit_p=state["strategy"].get("exploit_p", None))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(self, context: ArrayLike, actions: List[ActionId], rewards: List[BinaryReward]):
        super().update(context=context, actions=actions, rewards=rewards)


# TODO: add tests
class CmabBernoulliCC(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control strategy.

    The Cmab is extended to include a control of the action cost. Each action is associated with a predefined "cost".
    At prediction time, the model considers the actions whose expected rewards is above a pre-defined lower bound. Among
    these actions, the one with the lowest associated cost is recommended. The expected reward interval for feasible
    actions is defined as [(1-subsidy_factor) * max_p, max_p], where max_p is the highest expected reward sampled value.

    Reference: Thompson Sampling for Contextual Bandit Problems with Auxiliary Safety Constraints (Daulton et al., 2019)
               https://arxiv.org/abs/1911.00638

               Multi-Armed Bandits with Cost Subsidy (Sinha et al., 2021)
               https://arxiv.org/abs/2011.01488

    Parameters
    ----------
    actions: Dict[ActionId, BayesianLogisticRegressionCC]
        The list of possible actions, and their associated Model.
    strategy: CostControlBandit
        The strategy used to select actions.
    predict_with_proba: bool
        If True predict with sampled probabilities, else predict with weighted sums
    predict_actions_randomly: bool
        If True predict actions randomly (where each action has equal probability to be selected), else predict with the
        bandit strategy.
    """

    actions: Dict[ActionId, BayesianLogisticRegressionCC]
    strategy: CostControlBandit
    predict_with_proba: bool = True
    predict_actions_randomly: bool = False

    def __init__(
        self,
        actions: Dict[ActionId, BayesianLogisticRegressionCC],
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        subsidy_factor: Optional[Float01] = None,
    ):
        strategy = CostControlBandit() if subsidy_factor is None else CostControlBandit(subsidy_factor=subsidy_factor)
        super().__init__(actions=actions, strategy=strategy, epsilon=epsilon, default_action=default_action)

    @classmethod
    def from_state(cls, state: dict) -> "CmabBernoulliCC":
        return cls(actions=state["actions"], subsidy_factor=state["strategy"].get("subsidy_factor", None))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(self, context: ArrayLike, actions: List[ActionId], rewards: List[BinaryReward]):
        super().update(context=context, actions=actions, rewards=rewards)


@validate_call
def create_cmab_bernoulli_cold_start(
    action_ids: Set[ActionId],
    n_features: PositiveInt,
    epsilon: Optional[Float01] = None,
    default_action: Optional[ActionId] = None,
    update_method: UpdateMethods = "MCMC",
    update_kwargs: Optional[dict] = None,
) -> CmabBernoulli:
    """
    Utility function to create a Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, with default
    parameters. Until the very first update the model will predict actions randomly, where each action has equal
    probability to be selected.

    Parameters
    ----------
    action_ids: Set[ActionId]
        The list of possible actions.
    n_features: PositiveInt
        The number of features expected after in the context matrix. This is also the number of betas of the
        Bayesian Logistic Regression model.
    epsilon: Optional[Float01]
        epsilon for epsilon-greedy approach. If None, epsilon-greedy is not used.
    default_action: Optional[ActionId]
        The default action to select with a probability of epsilon when using the epsilon-greedy approach.
        If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
    update_method: UpdateMethods, defaults to MCMC
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method of each of the action models.

    Returns
    -------
    cmab: CmabBernoulli
        Contextual Multi-Armed Bandit with strategy = ClassicBandit
    """
    actions = {}
    for action_id in set(action_ids):
        actions[action_id] = create_bayesian_logistic_regression_cold_start(
            n_betas=n_features, update_method=update_method, update_kwargs=update_kwargs
        )
    mab = CmabBernoulli(actions=actions, epsilon=epsilon, default_action=default_action)
    mab.predict_actions_randomly = True
    return mab


@validate_call
def create_cmab_bernoulli_bai_cold_start(
    action_ids: Set[ActionId],
    n_features: PositiveInt,
    exploit_p: Optional[Float01] = None,
    epsilon: Optional[Float01] = None,
    default_action: Optional[ActionId] = None,
    update_method: UpdateMethods = "MCMC",
    update_kwargs: Optional[dict] = None,
) -> CmabBernoulliBAI:
    """
    Utility function to create a Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action
    Identification strategy, with default parameters. Until the very first update the model will predict actions
    randomly, where each action has equal probability to be selected.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    action_ids: Set[ActionId]
        The list of possible actions.
    n_features: PositiveInt
        The number of features expected after in the context matrix. This is also the number of betas of the
        Bayesian Logistic Regression model.
    exploit_p: Float_0_1 (default=0.5)
        Number in [0, 1] which specifies the amount of exploitation.
        If exploit_p is 1, the bandits always selects the action with highest probability of getting a positive reward,
            (it behaves as a Greedy strategy).
        If exploit_p is 0, the bandits always select the action with 2nd highest probability of getting a positive
            reward.
    epsilon: Optional[Float01]
        epsilon for epsilon-greedy approach. If None, epsilon-greedy is not used.
    default_action: Optional[ActionId]
        The default action to select with a probability of epsilon when using the epsilon-greedy approach.
        If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
    update_method: UpdateMethods, defaults to MCMC
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method of each of the action models.

    Returns
    -------
    cmab: CmabBernoulliBAI
        Contextual Multi-Armed Bandit with strategy = BestActionIdentification
    """
    actions = {}
    for a in set(action_ids):
        actions[a] = create_bayesian_logistic_regression_cold_start(
            n_betas=n_features,
            update_method=update_method,
            update_kwargs=update_kwargs,
        )
    mab = CmabBernoulliBAI(actions=actions, exploit_p=exploit_p, epsilon=epsilon, default_action=default_action)
    mab.predict_actions_randomly = True
    return mab


@validate_call
def create_cmab_bernoulli_cc_cold_start(
    action_ids_cost: Dict[ActionId, NonNegativeFloat],
    n_features: PositiveInt,
    subsidy_factor: Optional[Float01] = None,
    epsilon: Optional[Float01] = None,
    default_action: Optional[ActionId] = None,
    update_method: UpdateMethods = "MCMC",
    update_kwargs: Optional[dict] = None,
) -> CmabBernoulliCC:
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
    n_features: PositiveInt
        The number of features expected after in the context matrix. This is also the number of betas of the
        Bayesian Logistic Regression model.
    subsidy_factor: Optional[Float_0_1], default=0.5
        Number in [0, 1] to define smallest tolerated probability reward, hence the set of feasible actions.
        If subsidy_factor is 1, the bandits always selects the action with the minimum cost.
        If subsidy_factor is 0, the bandits always selects the action with highest probability of getting a positive
            reward (it behaves as a classic Bernoulli bandit).
    epsilon: Optional[Float01]
        epsilon for epsilon-greedy approach. If None, epsilon-greedy is not used.
    default_action: Optional[ActionId]
        The default action to select with a probability of epsilon when using the epsilon-greedy approach.
        If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
    update_method: UpdateMethods, defaults to MCMC
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method.

    Returns
    -------
    cmab: CmabBernoulliCC
        Contextual Multi-Armed Bandit with strategy = CostControl
    """
    actions = {}
    for a, cost in action_ids_cost.items():
        actions[a] = create_bayesian_logistic_regression_cc_cold_start(
            n_betas=n_features, cost=cost, update_method=update_method, update_kwargs=update_kwargs
        )
    mab = CmabBernoulliCC(
        actions=actions, subsidy_factor=subsidy_factor, epsilon=epsilon, default_action=default_action
    )
    mab.predict_actions_randomly = True
    return mab
