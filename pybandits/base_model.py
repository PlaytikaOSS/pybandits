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

from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np

from pybandits.base import (
    BinaryReward,
    Float01,
    MOProbability,
    Probability,
    ProbabilityWeight,
    PyBanditsBaseModel,
    QuantitativeMOProbability,
    QuantitativeProbability,
    QuantitativeProbabilityWeight,
)
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    conlist,
    pydantic_version,
    validate_call,
)


class BaseModel(PyBanditsBaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions.
    """

    @abstractmethod
    def sample_proba(
        self, **kwargs
    ) -> Union[
        List[Probability],
        List[MOProbability],
        List[ProbabilityWeight],
        List[QuantitativeProbability],
        List[QuantitativeMOProbability],
        List[QuantitativeProbabilityWeight],
    ]:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: Union[List[BinaryReward], List[List[BinaryReward]]], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]],
            if nested list, len() should follow shape of (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        """

    @abstractmethod
    def reset(self):
        """
        Reset the model.
        """


class BaseModelSO(BaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions for single objective.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """

    n_successes: PositiveInt = 1
    n_failures: PositiveInt = 1

    @abstractmethod
    def sample_proba(
        self, **kwargs
    ) -> Union[
        List[Probability], List[ProbabilityWeight], List[QuantitativeProbability], List[QuantitativeProbabilityWeight]
    ]:
        """
        Sample the probability of getting a positive reward.
        """

    @validate_call(config=dict(arbitrary_types_allowed=True))  # config allows to account for context argument type
    def update(self, rewards: List[BinaryReward], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : List[BinaryReward],
            The binary reward for each sample.
        """
        self._update(rewards=rewards, **kwargs)
        self.n_successes += sum(rewards)
        self.n_failures += len(rewards) - sum(rewards)

    @abstractmethod
    def _update(self, rewards: List[BinaryReward], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards: List[BinaryReward]
            A list of binary rewards.
        """

    def reset(self):
        """
        Reset the model.
        """
        self.n_successes = 1
        self.n_failures = 1
        self._reset()

    @abstractmethod
    def _reset(self):
        """
        Reset the model.
        """

    @property
    def count(self) -> NonNegativeInt:
        """
        The total amount of successes and failures collected.
        """
        return self.n_successes + self.n_failures

    @property
    def mean(self) -> Float01:
        """
        The success rate i.e. n_successes / (n_successes + n_failures).
        """
        return self.n_successes / self.count


class BaseModelMO(BaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions for multi-objective.

    Parameters
    ----------
    models : List[BaseModelSO]
        The list of models for each objective.
    """

    if pydantic_version == PYDANTIC_VERSION_1:
        models: conlist(BaseModelSO, min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        models: conlist(BaseModelSO, min_length=1)
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    def sample_proba(self, **kwargs) -> Union[List[MOProbability], List[QuantitativeMOProbability]]:
        """
        Sample the probability of getting a positive reward.
        """
        return [list(p) for p in zip(*[model.sample_proba(**kwargs) for model in self.models])]

    @validate_call(config=dict(arbitrary_types_allowed=True))  # config allows to account for context argument type
    def update(self, rewards: List[List[BinaryReward]], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : List[List[BinaryReward]],
            if nested list, len() should follow shape of (n_samples, n_objectives)
            The binary rewards for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        """
        if any(len(x) != len(self.models) for x in rewards):
            raise AttributeError("The shape of rewards is incorrect")

        for i, model in enumerate(self.models):
            model.update([r[i] for r in rewards], **kwargs)

    def reset(self):
        """
        Reset the model.
        """
        for model in self.models:
            model.reset()


class BaseModelCC(PyBanditsBaseModel, ABC):
    """
    Class to model action cost.

    Parameters
    ----------
    cost: Union[NonNegativeFloat, Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]]
        Cost associated to the Beta distribution.
    """

    cost: Union[NonNegativeFloat, Callable[[Union[float, np.ndarray]], NonNegativeFloat]]
