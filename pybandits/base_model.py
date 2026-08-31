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
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from numpy.random import Generator
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    conlist,
    validate_call,
)

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


class BaseModel(PyBanditsBaseModel, ABC):
    """
    Class to model the prior distributions of standard actions and quantitative actions.
    """

    @abstractmethod
    def sample_proba(
        self, rng: Generator, **kwargs
    ) -> (
        list[Probability]
        | list[MOProbability]
        | list[ProbabilityWeight]
        | list[QuantitativeProbability]
        | list[QuantitativeMOProbability]
        | list[QuantitativeProbabilityWeight]
    ):
        """
        Sample the probability of getting a positive reward.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central numpy random generator provided by the MAB.
        """

    @abstractmethod
    def update(self, rewards: list[BinaryReward] | list[list[BinaryReward]], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : list[BinaryReward] | list[list[BinaryReward]],
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

    # Beta(1, 1) prior pseudo-count: the anchor that raw counts start from and that decay shrinks toward.
    _prior_pseudo_count: ClassVar[int] = 1

    n_successes: PositiveInt = _prior_pseudo_count
    n_failures: PositiveInt = _prior_pseudo_count

    # --- Transfer learning keys (own contributions for this class) ---
    _transfer_learned_keys: ClassVar[tuple[str, ...]] = ("n_successes", "n_failures")
    _transfer_extendable_keys: ClassVar[tuple[str, ...]] = ()
    _transfer_structural_keys: ClassVar[tuple[str, ...]] = ()

    # --- Transfer learning keys (accumulated up the MRO, used by transfer.py) ---
    # For BaseModelSO itself these equal the own contributions above.
    # For subclasses they are auto-computed by __init_subclass__.
    transfer_learned_keys: ClassVar[tuple[str, ...]] = ("n_successes", "n_failures")
    """Accumulated learned-state keys from all classes in the MRO.  Used by transfer.py to decide which keys to copy from source to target."""
    transfer_extendable_keys: ClassVar[tuple[str, ...]] = ()
    """Accumulated extendable keys from all classes in the MRO.  Changes to these emit warnings (but not errors) during transfer."""
    transfer_structural_keys: ClassVar[tuple[str, ...]] = ()
    """Accumulated structural keys from all classes in the MRO.  Mismatches in these raise ValueError during transfer."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for pub, priv in (
            ("transfer_learned_keys", "_transfer_learned_keys"),
            ("transfer_extendable_keys", "_transfer_extendable_keys"),
            ("transfer_structural_keys", "_transfer_structural_keys"),
        ):
            accumulated: tuple[str, ...] = ()
            for base in reversed(cls.__mro__):
                if priv in base.__dict__:
                    accumulated += base.__dict__[priv]
            setattr(cls, pub, accumulated)

    @abstractmethod
    def sample_proba(
        self, rng: Generator, **kwargs
    ) -> (
        list[Probability]
        | list[ProbabilityWeight]
        | list[QuantitativeProbability]
        | list[QuantitativeProbabilityWeight]
    ):
        """
        Sample the probability of getting a positive reward.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central numpy random generator provided by the MAB.
        """

    @validate_call(config=dict(arbitrary_types_allowed=True))  # config allows to account for context argument type
    def update(self, rewards: list[BinaryReward], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : list[BinaryReward],
            The binary reward for each sample.
        """
        self._update(rewards=rewards, **kwargs)
        self.record_rewards(rewards)

    def record_rewards(self, rewards: list[BinaryReward]) -> None:
        """Tally binary rewards into the success/failure counters.

        The canonical success/failure bookkeeping, factored out of ``update`` so callers that train
        through a different path (e.g. the joint cMAB SVI engine, which bypasses per-model ``update``)
        can keep the counters in sync without re-implementing the arithmetic.

        Parameters
        ----------
        rewards : list[BinaryReward]
            The binary reward for each sample.
        """
        successes = sum(rewards)
        self.n_successes += successes
        self.n_failures += len(rewards) - successes

    @abstractmethod
    def _update(self, rewards: list[BinaryReward], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards: list[BinaryReward]
            A list of binary rewards.
        """

    def reset(self):
        """
        Reset the model.
        """
        self.n_successes = self._prior_pseudo_count
        self.n_failures = self._prior_pseudo_count
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
    models : list[BaseModelSO]
        The list of models for each objective.
    """

    models: conlist(BaseModelSO, min_length=1)

    def sample_proba(self, rng: Generator, **kwargs) -> list[MOProbability] | list[QuantitativeMOProbability]:
        """
        Sample the probability of getting a positive reward.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central numpy random generator provided by the MAB.
        """
        return [list(p) for p in zip(*[model.sample_proba(rng=rng, **kwargs) for model in self.models])]

    @validate_call(config=dict(arbitrary_types_allowed=True))  # config allows to account for context argument type
    def update(self, rewards: list[list[BinaryReward]], **kwargs):
        """
        Update the model parameters.

        Parameters
        ----------
        rewards : list[list[BinaryReward]],
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
    cost: NonNegativeFloat | Callable[[float | NonNegativeFloat], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """

    cost: NonNegativeFloat | Callable[[float | np.ndarray], NonNegativeFloat]


class BaseModelDP(PyBanditsBaseModel, ABC):
    """
    Class to model action price.

    Parameters
    ----------
    price: NonNegativeFloat | Callable[[float | np.ndarray], NonNegativeFloat]
        Price associated to the action.
    """

    price: NonNegativeFloat | Callable[[float | np.ndarray], NonNegativeFloat]
