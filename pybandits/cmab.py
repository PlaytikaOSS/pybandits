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
from typing import Dict, List, Optional, Set, Union

from numpy import array
from numpy.typing import ArrayLike

from pybandits.actions_manager import CmabActionsManager, CmabActionsManagerCC, CmabActionsManagerSO
from pybandits.base import ActionId, BinaryReward, CmabPredictions, PositiveProbability, Serializable
from pybandits.mab import BaseMab
from pybandits.model import BaseBayesianNeuralNetwork, BnnLayerParams, BnnParams, StudentTArray
from pybandits.pydantic_version_compatibility import validate_call
from pybandits.quantitative_model import BaseCmabZoomingModel
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
)


class BaseCmabBernoulli(BaseMab, ABC):
    """
    Base model for a Contextual Multi-Armed Bandit for Bernoulli bandits with Thompson Sampling.

    Parameters
    ----------
    actions : Dict[ActionId, Union[BaseBayesianLogisticRegression, BaseCmabZoomingModel]]
        The list of possible actions, and their associated Model.
    strategy : Strategy
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManager[Union[BaseBayesianNeuralNetwork, BaseCmabZoomingModel]]
    _predict_with_proba: bool

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
        probs: Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, MOProbability]]]
            The probabilities of getting a positive reward for each action.
        ws : Union[List[Dict[UnifiedActionId, float]], List[Dict[UnifiedActionId, List[float]]]]
            The weighted sum of logistic regression logits.
        """

        # cast inputs to numpy arrays to facilitate their manipulation
        context = array(context)

        if len(context) < 1:
            raise AttributeError("Context must have at least one row")

        # p is a dict of the sampled probability "prob" and weighted_sum "ws", e.g.
        #
        # p = {'a1': ([0.5, 0.2, 0.3], [200, 100, 130]), 'a2': ([0.4, 0.5, 0.6], [180, 200, 230]), ...}
        #               |               |                           |               |
        #              prob             ws                          prob            ws
        probs_weights = self._get_action_probabilities(forbidden_actions=forbidden_actions, context=context)

        probs = [
            {a: x[0] for a, x in prob_weight.items()} for prob_weight in probs_weights
        ]  # e.g. prob = {'a1': [0.5, 0.4, ...], 'a2': [0.4, 0.3, ...], ...}
        weighted_sums = [
            {a: x[1] for a, x in prob_weight.items()} for prob_weight in probs_weights
        ]  # e.g. ws = {'a1': [200, 100, ...], 'a2': [100, 50, ...], ...}

        # select either "prob" or "ws" to use as input argument in select_actions()
        p_to_select_action = probs if self._predict_with_proba else weighted_sums

        # predict actions, probs, weighted_sums
        selected_actions = [self._select_epsilon_greedy_action(p=p, actions=self.actions) for p in p_to_select_action]

        return selected_actions, probs, weighted_sums

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        context: ArrayLike,
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        context_memory: Optional[ArrayLike] = None,
    ):
        """
        Update the contextual Bernoulli bandit given the list of selected actions and their corresponding binary
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
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        context_memory : Optional[ArrayLike] of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        super().update(
            actions=actions,
            rewards=rewards,
            quantities=quantities,
            context=context,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
            context_memory=context_memory,
        )

    @classmethod
    def update_old_state(
        cls, state: Dict[str, Serializable], delta: Optional[PositiveProbability]
    ) -> Dict[str, Serializable]:
        """
        Update the model state to the current version.
        Besides the updates in the MAB class, it also loads legacy Bayesian Logistic Regression model parmeters into the new Bayesian Neural Network model.

        Parameters
        ----------
        state : Dict[str, Serializable]
            The internal state of a model (actions, strategy, etc.) of the same type.
            The state is expected to be in the old format of PyBandits below the current supported version.
        delta : Optional[PositiveProbability]
            The delta value to be set in the actions_manager. If None, it will not be set.
            This is relevant only for adaptive window models.

        Returns
        -------
        state : Dict[str, Serializable]
            The updated state of the model.
            The state is in the current format of PyBandits, with actions_manager and delta added if needed.
        """
        state = super().update_old_state(state, delta)

        if "predict_with_proba" in state:
            state.pop("predict_with_proba")

        if "predict_actions_randomly" in state:
            state.pop("predict_actions_randomly")

        # the state is in the old format of PyBandits < 3.0.0.
        for action_id, action_state in state["actions_manager"]["actions"].items():
            # Load legacy Bayesian Logistic Regression model parmeters into the new Bayesian Neural Network model.
            if ("alpha" in action_state) and ("betas" in action_state):
                bias = StudentTArray.cold_start(
                    mu=[action_state["alpha"]["mu"]],
                    sigma=[action_state["alpha"]["sigma"]],
                    nu=[action_state["alpha"]["nu"]],
                    shape=1,
                )
                mu_list = []
                sigma_list = []
                nu_list = []
                for beta in action_state["betas"]:
                    mu_list.append([beta["mu"]])
                    sigma_list.append([beta["sigma"]])
                    nu_list.append([beta["nu"]])

                weight = StudentTArray(mu=mu_list, sigma=sigma_list, nu=nu_list)
                layer_params = BnnLayerParams(weight=weight, bias=bias)

                # add model_params_init - in case we need to reset the model
                bias_init = StudentTArray.cold_start(shape=1)
                weight_init = StudentTArray.cold_start(shape=(len(mu_list), 1))
                layer_params_init = BnnLayerParams(weight=weight_init, bias=bias_init)

                model_params = BnnParams(
                    bnn_layer_params=[layer_params], bnn_layer_params_init=[layer_params_init]
                )._apply_version_adjusted_method("model_dump", "dict")
                action_state["model_params"] = model_params

                action_state.pop("alpha")
                action_state.pop("betas")

        return state


class CmabBernoulli(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling.

    References
    ----------
    Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal and Goyal, 2014)
    https://arxiv.org/pdf/1209.3352.pdf

    Parameters
    ----------
    actions_manager: CmabActionsManagerSO
        The manager for actions and their associated models.
    strategy: ClassicBandit
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManagerSO
    strategy: ClassicBandit
    _predict_with_proba: bool = False


class CmabBernoulliBAI(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Best Action Identification strategy.

    References
    ----------
    Analysis of Thompson Sampling for the Multi-armed Bandit Problem (Agrawal and Goyal, 2012)
    http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf

    Parameters
    ----------
    actions_manager: CmabActionsManagerSO
        The manager for actions and their associated models.
    strategy: BestActionIdentificationBandit
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManagerSO
    strategy: BestActionIdentificationBandit
    _predict_with_proba: bool = False


class CmabBernoulliCC(BaseCmabBernoulli):
    """
    Contextual Bernoulli Multi-Armed Bandit with Thompson Sampling, and Cost Control strategy.

    The Cmab is extended to include a control of the action cost. Each action is associated with a predefined "cost".
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
    actions_manager: CmabActionsManagerCC
        The manager for actions and their associated models.
    strategy: CostControlBandit
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManagerCC
    strategy: CostControlBandit
    _predict_with_proba: bool = True
