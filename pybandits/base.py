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
from typing import Any, Dict, List, NewType, Optional, Set, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    Extra,
    NonNegativeInt,
    confloat,
    conint,
    constr,
    root_validator,
    validate_arguments,
    validator,
)

ActionId = NewType("ActionId", constr(min_length=1))
Float01 = NewType("Float_0_1", confloat(ge=0, le=1))
Probability = NewType("Probability", Float01)
Predictions = NewType("Predictions", Tuple[List[ActionId], List[Dict[ActionId, Probability]]])
BinaryReward = NewType("BinaryReward", conint(ge=0, le=1))


class PyBanditsBaseModel(BaseModel):
    """
    BaseModel of the PyBandits library.
    """

    class Config:
        extra = Extra.forbid


class Model(PyBanditsBaseModel, ABC):
    """
    Class to model the prior distributions.
    """

    @abstractmethod
    def sample_proba(self) -> Probability:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: List[Any]):
        """
        Update the model parameters.
        """


class Strategy(PyBanditsBaseModel, ABC):
    """
    Strategy to select actions in multi-armed bandits.
    """

    @abstractmethod
    def select_action(self, p: Dict[ActionId, Probability], actions: Optional[Dict[ActionId, Model]]) -> ActionId:
        """
        Select the action.
        """


class BaseMab(PyBanditsBaseModel, ABC):
    """
    Multi-armed bandit superclass.

    Parameters
    ----------
    actions: Dict[ActionId, Model]
        The list of possible actions, and their associated Model.
    strategy: Strategy
        The strategy used to select actions.
    epsilon: Optional[Float01], defaults to None
        The probability of selecting a random action.
    """

    actions: Dict[ActionId, Model]
    strategy: Strategy
    epsilon: Optional[Float01]
    default_action: Optional[ActionId]

    @validator("actions", pre=True)
    @classmethod
    def at_least_2_actions_are_defined(cls, v):
        if len(v) < 2:
            raise AttributeError("At least 2 actions should be defined.")
        return v

    @root_validator
    def check_default_action(cls, values):
        if not values["epsilon"] and values["default_action"]:
            raise AttributeError("A default action should only be defined when epsilon is defined.")
        if values["default_action"] and values["default_action"] not in values["actions"]:
            raise AttributeError("The default action should be defined in the actions.")
        return values

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

    def _check_update_params(
        self,
        actions: List[ActionId],
        rewards: List[Union[NonNegativeInt, List[NonNegativeInt]]],
    ):
        """
        Verify that the given list of action IDs is a subset of the currently defined actions.

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

    @abstractmethod
    @validate_arguments
    def update(
        self,
        actions: List[ActionId],
        rewards: List[Union[BinaryReward, List[BinaryReward]]],
        *args,
        **kwargs,
    ):
        """
        Update the stochastic multi-armed bandit model.

        actions: List[ActionId]
            The selected action for each sample.
        rewards: List[Union[BinaryReward, List[BinaryReward]]]
            The reward for each sample.
        """

    @abstractmethod
    @validate_arguments
    def predict(self, forbidden_actions: Optional[Set[ActionId]] = None):
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
        probs: List[Dict[ActionId, float]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action.
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
        state: dict = self.dict()
        return model_name, state

    @validate_arguments
    def _select_epsilon_greedy_action(
        self,
        p: Union[Dict[ActionId, float], Dict[ActionId, Probability], Dict[ActionId, List[Probability]]],
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
                selected_action = self.default_action if self.default_action else np.random.choice(list(p.keys()))
            else:
                selected_action = self.strategy.select_action(p=p, actions=actions)
        else:
            selected_action = self.strategy.select_action(p=p, actions=actions)
        return selected_action
