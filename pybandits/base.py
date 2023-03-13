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
from pydantic import BaseModel, Extra, NonNegativeInt, confloat, conint, constr, validate_arguments, validator
from typing import Any, Dict, List, NewType, Optional, Set, Tuple, Union

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
    """

    actions: Dict[ActionId, Model]
    strategy: Strategy

    @validator("actions", pre=True)
    @classmethod
    def at_least_2_actions_are_defined(cls, v):
        if len(v) < 2:
            raise AttributeError("At least 2 actions should be defined.")
        return v

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
            raise AttributeError(f"Actions and rewards should have the same length {len(actions)}.")

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
