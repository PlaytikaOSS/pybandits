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

from typing import Dict, List, Optional, Set, Union

from numpy import array
from numpy.random import choice
from numpy.typing import ArrayLike

from pybandits.base import ActionId, BinaryReward, CmabPredictions
from pybandits.mab import BaseMab
from pybandits.model import BayesianLogisticRegression, BayesianLogisticRegressionCC
from pybandits.pydantic_version_compatibility import field_validator, validate_call
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
)


class BaseCmabBernoulli(BaseMab):
    """
    Base model for a Contextual Multi-Armed Bandit for Bernoulli bandits with Thompson Sampling.

    Parameters
    ----------
    actions: Dict[ActionId, BayesianLogisticRegression]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    predict_with_proba: bool
        If True predict with sampled probabilities, else predict with weighted sums.
    predict_actions_randomly: bool
        If True predict actions randomly (where each action has equal probability to be selected), else predict with the
        bandit strategy.
    """

    actions: Dict[ActionId, BayesianLogisticRegression]
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
    ) -> CmabPredictions:
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
        ws : List[Dict[ActionId, float]]
            The weighted sum of logistic regression logits.
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
        self._validate_update_params(actions=actions, rewards=rewards)
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


class CmabBernoulliBAI(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    Reference: Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
               http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions: Dict[ActionId, BayesianLogisticRegression]
        The list of possible actions, and their associated Model.
    strategy: BestActionIdentificationBandit
        The strategy used to select actions.
    predict_with_proba: bool
        If True predict with sampled probabilities, else predict with weighted sums
    predict_actions_randomly: bool
        If True predict actions randomly (where each action has equal probability to be selected), else predict with the
        bandit strategy.
    """

    actions: Dict[ActionId, BayesianLogisticRegression]
    strategy: BestActionIdentificationBandit
    predict_with_proba: bool = False
    predict_actions_randomly: bool = False


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
