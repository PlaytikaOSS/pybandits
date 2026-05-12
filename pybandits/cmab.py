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

import numpy as np
from pydantic import validate_call

from pybandits.actions_manager import (
    CmabActionsManager,
    CmabActionsManagerCC,
    CmabActionsManagerMO,
    CmabActionsManagerMOCC,
    CmabActionsManagerSO,
    CmabModelType,
)
from pybandits.base import (
    ActionId,
    BinaryReward,
    CmabPredictions,
    MOProbabilityWeight,
    ProbabilityWeight,
    Serializable,
)
from pybandits.mab import BaseMab
from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BaseBayesianNeuralNetworkMO,
    FeaturesConfig,
)
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
    MultiObjectiveStrategy,
)


class BaseCmabBernoulli(BaseMab, ABC):
    """
    Base model for a Contextual Multi-Armed Bandit for Bernoulli bandits with Thompson Sampling.

    Parameters
    ----------
    actions : Dict[ActionId, Union[BaseBayesianNeuralNetwork, BaseQuantitativeBayesianNeuralNetwork]]
        The list of possible actions, and their associated Model.
    strategy : Strategy
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManager[CmabModelType]
    _predict_with_proba: bool

    @property
    def input_dim(self) -> int:
        """Returns the input feature dimension (number of context features)."""
        return next(iter(self.actions.values())).input_dim

    @staticmethod
    def _extract_element_from_probability_weight(
        index: int, prob_weight: Union[ProbabilityWeight, MOProbabilityWeight]
    ) -> Union[float, List[float]]:
        """
        Extract the element from the probability weight.
        """
        if isinstance(prob_weight, tuple):  # ProbabilityWeight
            return prob_weight[index]
        elif isinstance(prob_weight, list) and all(
            isinstance(value, tuple) for value in prob_weight
        ):  # MOProbabilityWeight
            return [value[index] for value in prob_weight]
        else:
            raise TypeError(f"Unsupported probability weight type: {type(prob_weight)}")

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        context: np.ndarray,
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

        if len(context) < 1:
            raise AttributeError("Context must have at least one row")

        # p is a dict of the sampled probability "prob" and weighted_sum "ws", e.g.
        #
        # p = {'a1': ([0.5, 0.2, 0.3], [200, 100, 130]), 'a2': ([0.4, 0.5, 0.6], [180, 200, 230]), ...}
        #               |               |                           |               |
        #              prob             ws                          prob            ws
        probs_weights = self._get_action_probabilities(forbidden_actions=forbidden_actions, context=context)

        probs = [
            {a: self._extract_element_from_probability_weight(0, x) for a, x in prob_weight.items()}
            for prob_weight in probs_weights
        ]  # e.g. prob = {'a1': [0.5, 0.4, ...], 'a2': [0.4, 0.3, ...], ...}
        weighted_sums = [
            {a: self._extract_element_from_probability_weight(1, x) for a, x in prob_weight.items()}
            for prob_weight in probs_weights
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
        context: np.ndarray,
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        context_memory: Optional[np.ndarray] = None,
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
    def update_old_state(cls, state: Dict[str, Serializable]) -> Dict[str, Serializable]:
        """
        Update the model state to the current version.
        Besides the updates in the MAB class, it also adapts internal Bayesian Neural Network models.

        Parameters
        ----------
        state : Dict[str, Serializable]
            The internal state of a model (actions, strategy, etc.) of the same type.
            The state is expected to be in the old format of PyBandits below the current supported version.

        Returns
        -------
        state : Dict[str, Serializable]
            The updated state of the model.
            The state is in the current format of PyBandits, with actions_manager and delta added if needed.
        """
        state = super().update_old_state(state)

        # Migrate update_kwargs from old PyMC format to new NumPyro format
        for action_id, action_state in state["actions_manager"]["actions"].items():
            if "feature_config" not in action_state:  # v5.0.0 compatability
                layer_params = action_state["model_params"]["bnn_layer_params"]
                if not layer_params:
                    raise ValueError("Cannot infer feature_config: bnn_layer_params is empty.")
                # weight.mu is List[List[float]]; outer length == input_dim == n_features for numerical-only models
                n_features = len(layer_params[0][BaseBayesianNeuralNetwork.weight_var_name]["mu"])
                fc = FeaturesConfig(n_features=n_features, categorical_features_configs=[])
                action_state["feature_config"] = fc.model_dump()

            if "update_kwargs" in action_state and action_state["update_kwargs"] is not None:  # v6.0.0 compatability
                kwargs = action_state["update_kwargs"]

                # Migrate VI kwargs: "fit" dict → flat keys
                if "fit" in kwargs:
                    fit = kwargs.pop("fit")
                    if "n" in fit:
                        kwargs["num_steps"] = fit.pop("n")
                    if "method" in fit:
                        kwargs["method"] = fit.pop("method")

                # Migrate MCMC kwargs: "trace" dict → flat keys + "nuts" sub-dict
                if "trace" in kwargs:
                    trace = kwargs.pop("trace")
                    if "tune" in trace:
                        kwargs["num_warmup"] = trace.pop("tune")
                    if "draws" in trace:
                        kwargs["num_samples"] = trace.pop("draws")
                    if "chains" in trace:
                        kwargs["num_chains"] = trace.pop("chains")
                    if "progressbar" in trace:
                        kwargs["progress_bar"] = trace.pop("progressbar")
                    nuts = {}
                    if "target_accept" in trace:
                        nuts["target_accept_prob"] = trace.pop("target_accept")
                    if nuts:
                        kwargs["nuts"] = nuts
                    # Remove PyMC-only keys
                    for pymc_key in ("init", "cores", "return_inferencedata"):
                        trace.pop(pymc_key, None)

                # Migrate optimizer kwargs: learning_rate → step_size
                if "optimizer_kwargs" in kwargs:
                    opt_kwargs = kwargs["optimizer_kwargs"]
                    if "learning_rate" in opt_kwargs:
                        opt_kwargs["step_size"] = opt_kwargs.pop("learning_rate")

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


class BaseCmabBernoulliMO(BaseCmabBernoulli, ABC):
    """
    Base model for a Contextual Multi-Armed Bandit with Thompson Sampling and Multi-Objective strategy.

    Parameters
    ----------
    actions_manager: CmabActionsManager[BaseBayesianNeuralNetworkMO]
        The manager for actions and their associated models.
    strategy : MultiObjectiveStrategy
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManager[BaseBayesianNeuralNetworkMO]
    strategy: MultiObjectiveStrategy


class CmabBernoulliMO(BaseCmabBernoulliMO):
    """
    Contextual Multi-Armed Bandit with Thompson Sampling and Multi-Objective strategy.

    The reward for an action is a multidimensional vector. Actions are compared using Pareto order between their expected reward vectors.
    Pareto optimal actions are those not strictly dominated by any other action.

    Reference
    ---------
    Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
    https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem

    Parameters
    ----------
    actions_manager: CmabActionsManagerMO
        The manager for actions and their associated models.
    strategy : MultiObjectiveBandit
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManagerMO
    strategy: MultiObjectiveBandit
    _predict_with_proba: bool = False


class CmabBernoulliMOCC(BaseCmabBernoulliMO):
    """
    Contextual Multi-Armed Bandit with Thompson Sampling for Multi-Objective (MO) and Cost Control (CC) strategy.

    This bandit allows the reward to be a multidimensional vector and includes control of the action cost, merging
    Multi-Objective and Cost Control strategies.

    Parameters
    ----------
    actions_manager: CmabActionsManagerMOCC
        The manager for actions and their associated models.
    strategy : MultiObjectiveCostControlBandit
        The strategy used to select actions.
    """

    actions_manager: CmabActionsManagerMOCC
    strategy: MultiObjectiveCostControlBandit
    _predict_with_proba: bool = True
