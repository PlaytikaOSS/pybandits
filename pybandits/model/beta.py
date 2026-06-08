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
from abc import ABC
from typing import List

import numpy as np
from numpy import sqrt
from pydantic import (
    PositiveInt,
    conlist,
    validate_call,
)
from typing_extensions import Self

from pybandits.base import (
    BinaryReward,
    Probability,
)
from pybandits.model.base import Model, ModelCC, ModelDP, ModelMO


class BaseBeta(Model, ABC):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """

    @property
    def std(self) -> float:
        """
        The corrected standard deviation (Bessel's correction) of the binary distribution of successes and failures.
        """
        return sqrt((self.n_successes * self.n_failures) / (self.count * (self.count - 1)))

    @validate_call
    def _update(self, rewards: List[BinaryReward]):
        """
        Update n_successes and n_failures.

        Parameters
        ----------
        rewards: List[BinaryReward]
            A list of binary rewards.
        """
        pass

    def _reset(self):
        pass

    def sample_proba(self, n_samples: PositiveInt, rng: np.random.Generator) -> List[Probability]:
        """
        Sample the probability of getting a positive reward.

        Parameters
        ----------
        n_samples : PositiveInt
            Number of samples to draw.
        rng : np.random.Generator
            Numpy random generator. Vectorized sampling via ``rng.beta``.

        Returns
        -------
        prob: Probability
            Probability of getting a positive reward.
        """
        return list(rng.beta(self.n_successes, self.n_failures, size=n_samples))


class Beta(BaseBeta):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """


class BetaCC(BaseBeta, ModelCC):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with cost control.

    Parameters
    ----------
    n_successes : PositiveInt = 1
        Counter of the number of successes.
    n_failures : PositiveInt = 1
        Counter of the number of failures.
    cost : NonNegativeFloat
        Cost associated to the Beta distribution.
    """


class BetaDP(BaseBeta, ModelDP):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with dynamic pricing.

    Parameters
    ----------
    n_successes : PositiveInt = 1
        Counter of the number of successes.
    n_failures : PositiveInt = 1
        Counter of the number of failures.
    price : NonNegativeFloat
        Price associated to the Beta distribution.
    """


class BaseBetaMO(ModelMO, ABC):
    """
    Base Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    models: List[Beta] of length (n_objectives,)
        List of Beta distributions.
    """

    models: conlist(Beta, min_length=1)

    @classmethod
    @validate_call
    def cold_start(cls, n_objectives: PositiveInt, **kwargs) -> Self:
        """
        Utility function to create a BetaMO or child model with cost control,
        with default parameters.

        Parameters
        ----------
        n_objectives : PositiveInt
            Number of objectives (models) to create.
        kwargs: Dict[str, Any]
            Additional arguments for the BaseBetaMO child model.

        Returns
        -------
        beta_mo: BetaMO
            The multi-objective Beta model.
        """
        models = [Beta() for _ in range(n_objectives)]
        beta_mo = cls(models=models, **kwargs)
        return beta_mo


class BetaMO(BaseBetaMO):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    models: List[Beta] of length (n_objectives,)
        List of Beta distributions.
    """


class BetaMOCC(BaseBetaMO, ModelCC):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives and cost control.

    Parameters
    ----------
    models: List[Beta] of shape (n_objectives,)
        List of Beta distributions.
    cost: NonNegativeFloat
        Cost associated to the Beta distribution.
    """
