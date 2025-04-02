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
import json
from abc import ABC, abstractmethod
from inspect import isclass
from typing import Any, Dict, List, Optional, Set, Union, get_origin

import numpy as np

from pybandits.actions_manager import ActionsManager
from pybandits.base import (
    ActionId,
    ActionRewardLikelihood,
    BinaryReward,
    Float01,
    PositiveProbability,
    Predictions,
    PyBanditsBaseModel,
)
from pybandits.model import BaseModel, Model
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    model_validator,
    pydantic_version,
    validate_call,
)
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

    actions_manager: ActionsManager
    strategy: Strategy
    epsilon: Optional[Float01] = None
    default_action: Optional[ActionId] = None

    def __init__(
        self,
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        **kwargs,
    ):
        class_attributes = {
            attribute_name: self._get_instantiated_class_attribute(attribute_name, kwargs)
            for attribute_name in self._get_class_type_attributes()
        }
        if kwargs:
            raise ValueError(f"Unknown arguments: {kwargs.keys()}")
        super().__init__(**class_attributes, epsilon=epsilon, default_action=default_action)

    @classmethod
    def _get_instantiated_class_attribute(cls, attribute_name: str, kwargs: Dict[str, Any]) -> PyBanditsBaseModel:
        if attribute_name in kwargs:
            attribute = kwargs[attribute_name]
        else:
            attribute_class = cls._get_attribute_type(attribute_name)
            required_sub_attributes = extract_argument_names_from_function(attribute_class.__init__, True)
            if not required_sub_attributes:  # case of no native __init__ method, just pydantic generic __init__
                required_sub_attributes = list(attribute_class.model_fields.keys())
                sub_attributes = {k: kwargs.pop(k) for k in required_sub_attributes if k in kwargs}
            else:
                sub_attributes = {k: kwargs.pop(k) for k in required_sub_attributes if k in kwargs}
                if "kwargs" in required_sub_attributes:
                    sub_attributes["kwargs"] = kwargs

            attribute = attribute_class(**sub_attributes)
        kwargs.pop(attribute_name, None)
        return attribute

    ############################################ Instance Input Validators #############################################

    if pydantic_version == PYDANTIC_VERSION_1:

        @model_validator(mode="before")
        @classmethod
        def check_default_action(cls, values):
            epsilon = cls._get_value_with_default("epsilon", values)
            default_action = cls._get_value_with_default("default_action", values)
            if not epsilon and default_action:
                raise AttributeError("A default action should only be defined when epsilon is defined.")
            if default_action and default_action not in values["actions_manager"].actions:
                raise AttributeError("The default action must be valid action defined in the actions set.")
            return values

    elif pydantic_version == PYDANTIC_VERSION_2:

        @model_validator(mode="after")
        def check_default_action(self):
            if not self.epsilon and self.default_action:
                raise AttributeError("A default action should only be defined when epsilon is defined.")
            if self.default_action and self.default_action not in self.actions:
                raise AttributeError("The default action must be valid action defined in the actions set.")
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

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
        action_ids = set(self.actions.keys())
        if not all(a in action_ids for a in forbidden_actions):
            raise ValueError("forbidden_actions contains invalid action IDs.")
        valid_actions = action_ids - forbidden_actions
        if len(valid_actions) == 0:
            raise ValueError("All actions are forbidden. You must allow at least 1 action.")
        if self.default_action and self.default_action not in valid_actions:
            raise ValueError("The default action is forbidden.")

        return valid_actions

    ####################################################################################################################

    def model_post_init(self, __context: Any) -> None:
        if self.actions_manager.delta is not None and (not self.epsilon or self.default_action is not None):
            raise ValueError("Adaptive window requires epsilon greedy super strategy with not default action.")

    @property
    def actions(self) -> Dict[ActionId, Model]:
        return self.actions_manager.actions

    @validate_call
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        **kwargs,
    ):
        """
        Update the multi-armed bandit model.

        Parameters
        ----------
        actions: List[ActionId]
            The selected action for each sample.
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]] of shape (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        self.actions_manager.update(
            actions=actions, rewards=rewards, actions_memory=actions_memory, rewards_memory=rewards_memory, **kwargs
        )

    @abstractmethod
    @validate_call
    def predict(self, forbidden_actions: Optional[Set[ActionId]] = None) -> Predictions:
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
        actions: List[ActionId] of shape (n_samples,)
            The actions selected by the multi-armed bandit model.
        probs: List[Dict[ActionId, Probability]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action
        ws : List[Dict[ActionId, float]], only relevant for some of the MABs
            The weighted sum of logistic regression logits..
        """

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
        json_state = self._apply_version_adjusted_method("model_dump_json", "json")
        state = json.loads(json_state)
        return model_name, state

    @validate_call
    def _select_epsilon_greedy_action(
        self,
        p: ActionRewardLikelihood,
        actions: Optional[Dict[ActionId, BaseModel]] = None,
    ) -> ActionId:
        """
        Wraps self.strategy.select_action function with epsilon-greedy strategy,
        such that with probability epsilon a default_action is selected,
        and with probability 1-epsilon the select_action function is triggered to choose action.
        If no default_action is provided, a random action is selected.

        References
        ----------
        Reinforcement Learning: An Introduction, Ch. 2 (Sutton and Burto, 2018)
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
        class_attributes = {
            attribute_name: list(state[attribute_name].keys()) for attribute_name in cls._get_class_type_attributes()
        }
        flattened_class_attributes = [item for sublist in class_attributes.values() for item in sublist]
        class_attributes_mapping = {
            k: state[k] for k in model_attributes if k not in flattened_class_attributes and k in state
        }
        class_attributes_mapping.update(
            {
                k: state[attribute_name][k]
                for attribute_name, sub_class_attributes in class_attributes.items()
                for k in sub_class_attributes
            }
        )
        return cls(**class_attributes_mapping)

    @classmethod
    def from_old_state(
        cls,
        state: dict,
        delta: Optional[PositiveProbability] = None,
    ) -> "BaseMab":
        """
        Create a new instance of the class from a given model state.
        The state can be obtained by applying get_state() to a model.

        Parameters
        ----------
        state: dict
            The internal state of a model (actions, strategy, etc.) of the same type.
            The state is expected to be in the old format of PyBandits < 2.0.0.

        Returns
        -------
        model: BaseMab
            The new model instance.

        """
        if "actions" not in state or "actions_manager" in state:
            raise ValueError("The state is expected to be in the old format of PyBandits < 2.0.0.")
        state["actions_manager"] = {}
        state["actions_manager"]["actions"] = state.pop("actions")
        state["actions_manager"]["delta"] = delta

        return cls.from_state(state)

    @classmethod
    def _get_class_type_attributes(cls) -> List[str]:
        return [
            attribute_name
            for attribute_name in cls.model_fields.keys()
            if isclass(class_ := cls._get_attribute_type(attribute_name))
            and issubclass(
                class_,
                PyBanditsBaseModel,
            )
        ]

    @classmethod
    def _get_attribute_type(cls, attribute_name: str) -> PyBanditsBaseModel:
        attribute_type = cls._get_field_type(attribute_name)
        return get_origin(attribute_type) or attribute_type

    @classmethod
    def cold_start(
        cls,
        epsilon: Optional[Float01] = None,
        default_action: Optional[ActionId] = None,
        **kwargs,
    ) -> "BaseMab":
        """
        Factory method to create a Multi-Armed Bandit with Thompson Sampling, with default
        parameters.

        Parameters
        ----------
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

        # Instantiate the MAB
        mab = cls(epsilon=epsilon, default_action=default_action, **kwargs)

        # For contextual multi-armed bandit, until the very first update the model will predict actions randomly,
        # where each action has equal probability to be selected.
        if hasattr(mab, "predict_actions_randomly"):
            mab.predict_actions_randomly = True
        return mab
