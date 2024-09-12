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

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, get_args

import numpy as np
from pydantic import field_validator, model_validator, validate_call

from pybandits.base import (
    ACTION_IDS_PREFIX,
    ActionId,
    ActionRewardLikelihood,
    BinaryReward,
    Float01,
    Predictions,
    PyBanditsBaseModel,
)
from pybandits.model import Model
from pybandits.strategy import Strategy
from pybandits.utils import extract_argument_names_from_function


class BaseMab(PyBanditsBaseModel, ABC):
    """
    Multi-armed bandit superclass.

    Parameters
    ----------
    actions : Dict[ActionId, Model]
        The list of possible actions, and their associated Model.
    strategy : Strategy
        The strategy used to select actions.
    epsilon : Optional[Float01], 0 if not specified.
        The probability of selecting a random action.
    default_action : Optional[ActionId], None if not specified.
        The default action to select with a probability of epsilon when using the epsilon-greedy approach.
        If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
    strategy_kwargs : Dict[str, Any]
        Relevant only if strategy was not provided. This argument contains the parameters for the strategy,
        which in turn will be used to instantiate the strategy.
    """

    actions: Dict[ActionId, Model]
    strategy: Strategy
    epsilon: Optional[Float01] = None
    default_action: Optional[ActionId] = None

    def __init__(
        self,
        actions: Dict[ActionId, Model],
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        **strategy_kwargs,
    ):
        if "strategy" in strategy_kwargs:
            strategy = strategy_kwargs["strategy"]
            if len(strategy_kwargs) > 1:
                raise ValueError("strategy should be the only keyword argument.")
        else:
            strategy_class = self.model_fields["strategy"].annotation
            strategy = strategy_class(**strategy_kwargs)

        super().__init__(actions=actions, strategy=strategy, epsilon=epsilon, default_action=default_action)

    ############################################ Instance Input Validators #############################################

    @field_validator("actions", mode="before")
    @classmethod
    def at_least_one_action_is_defined(cls, v):
        # validate number of actions
        if len(v) == 0:
            raise AttributeError("At least one action should be defined.")
        elif len(v) == 1:
            warnings.warn("Only a single action was supplied. This MAB will be deterministic.")
        # validate that all actions are of the same configuration
        action_models = list(v.values())
        first_action = action_models[0]
        first_action_type = type(first_action)
        if any(not isinstance(action, first_action_type) for action in action_models[1:]):
            raise AttributeError("All actions should follow the same type.")
        return v

    @model_validator(mode="after")
    def check_default_action(self):
        if not self.epsilon and self.default_action:
            raise AttributeError("A default action should only be defined when epsilon is defined.")
        if self.default_action and self.default_action not in self.actions:
            raise AttributeError("The default action must be valid action defined in the actions set.")
        return self

    @model_validator(mode="after")
    def validate_default_action(self):
        if not self.epsilon and self.default_action:
            raise AttributeError("A default action should only be defined when epsilon is defined.")
        if self.default_action and self.default_action not in self.actions:
            raise AttributeError("The default action should be defined in the actions.")
        return self

    ############################################# Method Input Validators ##############################################

    def _get_valid_actions(self, forbidden_actions: Optional[Set[ActionId]]) -> Set[ActionId]:
        """
        Given a set of forbidden action IDs, return a set of valid action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[Set[ActionId]]
            The set of forbidden action IDs.

        Returns
        -------
        valid_actions: Set[ActionId]
            The list of valid (i.e. not forbidden) action IDs.
        """
        if forbidden_actions is None:
            forbidden_actions = set()

        if not all(a in self.actions.keys() for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = set(self.actions.keys()) - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError("All actions are forbidden. You must allow at least 1 action.")
        if self.default_action and self.default_action not in valid_actions:
            raise ValueError("The default action is forbidden.")

        return valid_actions

    def _validate_update_params(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
    ):
        """
        Verify that the given list of action IDs is a subset of the currently defined actions and that
         the rewards type matches the strategy type.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: List[Union[BinaryReward, List[BinaryReward]]]
            The reward for each sample.
        """
        invalid = set(actions) - set(self.actions.keys())
        if invalid:
            raise AttributeError(f"The following invalid action(s) were specified: {invalid}.")
        if len(actions) != len(rewards):
            raise AttributeError(f"Shape mismatch: actions and rewards should have the same length {len(actions)}.")

    ####################################################################################################################

    @validate_call
    def update(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], *args, **kwargs
    ):
        """
        Update the multi-armed bandit model.

        actions: List[ActionId]
            The selected action for each sample.
        rewards: List[Union[BinaryReward, List[BinaryReward]]]
            The reward for each sample.
        """
        self._validate_update_params(actions=actions, rewards=rewards)
        self._update(actions=actions, rewards=rewards, *args, **kwargs)
        if hasattr(self.strategy, "update"):
            self.strategy.update(rewards=rewards)

    @abstractmethod
    def _update(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], *args, **kwargs
    ):
        """
        Update the multi-armed bandit model.

        actions : List[ActionId]
            The selected action for each sample.
        rewards : List[Union[BinaryReward, List[BinaryReward]]]
            The reward for each sample.
        """
        pass

    @validate_call
    def predict(self, forbidden_actions: Optional[Set[ActionId]] = None, **kwargs) -> Predictions:
        """
        Predict actions.

        Parameters
        ----------
        forbidden_actions : Optional[Set[ActionId]], default=None
            Set of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions : List[ActionId] of shape (n_samples,)
            The actions selected by the multi-armed bandit model.
        probs : List[Dict[ActionId, Probability]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action
        ws : List[Dict[ActionId, float]], only relevant for some of the MABs
            The weighted sum of logistic regression logits.
        """
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()
        valid_actions = self._get_valid_actions(forbidden_actions)
        return self._predict(valid_actions=valid_actions, **kwargs)

    @abstractmethod
    def _predict(self, valid_actions: Set[ActionId], **kwargs) -> Predictions:
        """
        Predict actions.

        Parameters
        ----------
        valid_actions : Set[ActionId]
            The set of valid actions.

        Returns
        -------
        actions: List[ActionId] of shape (n_samples,)
            The actions selected by the multi-armed bandit model.
        probs: List[Dict[ActionId, Probability]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action
        ws : List[Dict[ActionId, float]], only relevant for some of the MABs
            The weighted sum of logistic regression logits.
        """
        pass

    def get_state(self) -> (str, dict):
        """
        Access the complete model internal state, enough to create an exact copy of the same model from it.
        Returns
        -------
        model_class_name: str
            The name of the class of the model.
        model_state: dict
            The internal state of the model (actions, scores, etc.).
        """
        model_name = self.__class__.__name__
        state: dict = self.model_dump()
        return model_name, state

    @validate_call
    def _select_epsilon_greedy_action(
        self,
        p: ActionRewardLikelihood,
        actions: Optional[Dict[ActionId, Model]] = None,
    ) -> ActionId:
        """
        Wraps self.strategy.select_action function with epsilon-greedy strategy,
        such that with probability epsilon a default_action is selected,
        and with probability 1-epsilon the select_action function is triggered to choose action.
        If no default_action is provided, a random action is selected.

        Reference: Reinforcement Learning: An Introduction, Ch. 2 (Sutton and Burto, 2018)
               https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf&ved=2ahUKEwjMy8WV9N2HAxVe0gIHHVjjG5sQFnoECEMQAQ&usg=AOvVaw3bKK-Y_1kf6XQVwR-UYrBY

        Parameters
        ----------
        p: Union[Dict[ActionId, float], Dict[ActionId, Probability], Dict[ActionId, List[Probability]]]
            The dictionary or actions and their sampled probability of getting a positive reward.
            For MO strategy, the sampled probability is a list with elements corresponding to the objectives.
        actions: Optional[Dict[ActionId, Model]]
            The dictionary of actions and their associated Model.

        Returns
        -------
        selected_action: ActionId
            The selected action.

        Raises
        ------
        KeyError
            If self.default_action is not present as a key in the probabilities dictionary.
        """

        if self.epsilon:
            if self.default_action and self.default_action not in p.keys():
                raise KeyError(f"Default action {self.default_action} not in actions.")
            if np.random.binomial(1, self.epsilon):
                selected_action = self.default_action or np.random.choice(list(p.keys()))
            else:
                selected_action = self.strategy.select_action(p=p, actions=actions)
        else:
            selected_action = self.strategy.select_action(p=p, actions=actions)
        return selected_action

    @classmethod
    def from_state(cls, state: dict) -> "BaseMab":
        """
        Create a new instance of the class from a given model state.
        The state can be obtained by applying get_state() to a model.

        Parameters
        ----------
        state: dict
            The internal state of a model (actions, strategy, etc.) of the same type.

        Returns
        -------
        model: BaseMab
            The new model instance.

        """
        model_attributes = extract_argument_names_from_function(cls.__init__, True)
        strategy_attributes = list(state["strategy"].keys())
        attributes_mapping = {k: state[k] for k in model_attributes if k not in strategy_attributes and k in state}
        attributes_mapping.update({k: state["strategy"][k] for k in strategy_attributes})
        return cls(**attributes_mapping)

    @classmethod
    def cold_start(
        cls,
        action_ids: Optional[Set[ActionId]] = None,
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        **kwargs,
    ) -> "BaseMab":
        """
        Factory method to create a Multi-Armed Bandit with Thompson Sampling, with default
        parameters.

        Parameters
        ----------
        action_ids: Optional[Set[ActionId]]
            The list of possible actions.
        epsilon: Optional[Float01]
            epsilon for epsilon-greedy approach. If None, epsilon-greedy is not used.
        default_action: Optional[ActionId]
            The default action to select with a probability of epsilon when using the epsilon-greedy approach.
            If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
        kwargs: Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        mab: BaseMab
            Multi-Armed Bandit
        """
        action_specific_kwargs, kwargs = cls._extract_action_specific_kwargs(**kwargs)

        # Extract inner_action_ids
        inner_action_ids = action_ids or set(action_specific_kwargs.keys())
        if not inner_action_ids:
            raise ValueError(
                "inner_action_ids should be provided either directly or via keyword argument in the form of "
                "action_id_{model argument name} = {action_id: value}."
            )

        # Assign model for each action
        action_model_cold_start, action_general_kwargs = cls._extract_action_model_class_and_attributes(**kwargs)
        actions = {}
        for a in inner_action_ids:
            actions[a] = action_model_cold_start(**action_general_kwargs, **action_specific_kwargs.get(a, {}))

        # Instantiate the MAB
        strategy_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in action_general_kwargs.keys()}
        strategy_class = cls.model_fields["strategy"].annotation
        strategy = strategy_class(**strategy_kwargs)
        mab = cls(actions=actions, strategy=strategy, epsilon=epsilon, default_action=default_action)
        # For contextual multi-armed bandit, until the very first update the model will predict actions randomly,
        # where each action has equal probability to be selected.
        if hasattr(mab, "predict_actions_randomly"):
            mab.predict_actions_randomly = True
        return mab

    @staticmethod
    def _extract_action_specific_kwargs(**kwargs) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """
        Utility function to extract kwargs that are specific for each action when constructing the action model.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        action_specific_kwargs : Dict[str, Dict]
            Dictionary of actions and the parameters of their associated model.
        kwargs : Dict[str, Any]
            Dictionary of parameters and their values, without the action_specific_kwargs.
        """
        action_specific_kwargs = defaultdict(dict)
        for keyword in list(kwargs):
            argument = kwargs[keyword]
            if keyword.startswith(ACTION_IDS_PREFIX) and type(argument) is dict:
                kwargs.pop(keyword)
                inner_keyword = keyword.split(ACTION_IDS_PREFIX)[1]
                for action_id, value in argument.items():
                    action_specific_kwargs[action_id][inner_keyword] = value
        return dict(action_specific_kwargs), kwargs

    @classmethod
    def _extract_action_model_class_and_attributes(cls, **kwargs) -> Tuple[Callable, Dict[str, Dict]]:
        """
        Utility function to extract kwargs that are specific for each action when constructing the action model.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        action_model_cold_start : Callable
            Function handle for factoring the required action model.
        action_general_kwargs : Dict[str, any]
            Dictionary of parameters and their values for the action model.
        """
        action_model_class = get_args(cls.model_fields["actions"].annotation)[1]
        if hasattr(action_model_class, "cold_start"):
            action_model_cold_start_init = action_model_cold_start = action_model_class.cold_start
        else:
            action_model_cold_start_init = action_model_class.__init__
            action_model_cold_start = action_model_class

        action_model_attributes = extract_argument_names_from_function(action_model_cold_start_init, True)

        action_general_kwargs = {k: kwargs[k] for k in action_model_attributes if k in kwargs.keys()}
        return action_model_cold_start, action_general_kwargs
