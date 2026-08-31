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
from typing import ClassVar, List, Optional, Self, Tuple

import numpy as np
from numpy import sqrt
from pydantic import (
    PositiveFloat,
    PositiveInt,
    conlist,
    model_validator,
    validate_call,
)

from pybandits.base import (
    BinaryReward,
    PositiveFloat01,
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
    decay_factor: Optional[PositiveFloat01] = None
        Per-update forgetting factor in (0, 1] inherited from Model. When set, sampling is
        driven by the effective (decayed) counts below instead of the raw n_successes/n_failures.
    decayed_n_successes: Optional[PositiveFloat] = None
        Effective number of successes after decay, used for sampling when decay_factor is set.
        Seeded from n_successes on first use; None when decay is disabled.
    decayed_n_failures: Optional[PositiveFloat] = None
        Effective number of failures after decay, used for sampling when decay_factor is set.
        Seeded from n_failures on first use; None when decay is disabled.
    """

    decayed_n_successes: Optional[PositiveFloat] = None
    decayed_n_failures: Optional[PositiveFloat] = None

    # The effective decayed counts are learned state and must be transferred alongside the raw counts.
    _transfer_learned_keys: ClassVar[Tuple[str, ...]] = ("decayed_n_successes", "decayed_n_failures")

    @model_validator(mode="after")
    def _init_decayed_counts(self) -> Self:
        """Seed the effective decayed counts from the raw counts when decay is enabled."""
        if self.decay_factor is not None:
            if self.decayed_n_successes is None:
                self.decayed_n_successes = float(self.n_successes)
            if self.decayed_n_failures is None:
                self.decayed_n_failures = float(self.n_failures)
        return self

    @property
    def std(self) -> float:
        """
        The corrected standard deviation (Bessel's correction) of the binary distribution of successes and failures.
        """
        if self.decay_factor is not None:
            n_s, n_f = self.decayed_n_successes, self.decayed_n_failures
        else:
            n_s, n_f = float(self.n_successes), float(self.n_failures)
        total = n_s + n_f
        return sqrt((n_s * n_f) / (total * (total - 1)))

    @validate_call
    def _update(self, rewards: List[BinaryReward]):
        """
        Update the effective decayed counts (the raw n_successes/n_failures are updated by BaseModelSO).

        When decay_factor is set, historical evidence is discounted towards the Beta(1, 1) prior
        before the new rewards are added: ``n <- 1 + decay_factor * (n - 1) + new``. This keeps the
        effective counts at or above the prior, so the Beta posterior stays proper.

        Parameters
        ----------
        rewards: List[BinaryReward]
            A list of binary rewards.
        """
        if self.decay_factor is not None:
            n_successes = sum(rewards)
            n_failures = len(rewards) - n_successes
            prior = self._prior_pseudo_count
            self.decayed_n_successes = prior + self.decay_factor * (self.decayed_n_successes - prior) + n_successes
            self.decayed_n_failures = prior + self.decay_factor * (self.decayed_n_failures - prior) + n_failures

    def _reset(self):
        if self.decay_factor is not None:
            self.decayed_n_successes = float(self._prior_pseudo_count)
            self.decayed_n_failures = float(self._prior_pseudo_count)

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
        if self.decay_factor is not None:
            n_successes, n_failures = self.decayed_n_successes, self.decayed_n_failures
        else:
            n_successes, n_failures = self.n_successes, self.n_failures
        return list(rng.beta(n_successes, n_failures, size=n_samples))


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
    def cold_start(cls, n_objectives: PositiveInt, decay_factor: Optional[PositiveFloat01] = None, **kwargs) -> Self:
        """
        Utility function to create a BetaMO or child model with cost control,
        with default parameters.

        Parameters
        ----------
        n_objectives : PositiveInt
            Number of objectives (models) to create.
        decay_factor : Optional[PositiveFloat01]
            Per-update forgetting factor forwarded to each per-objective Beta model.
        kwargs: Dict[str, Any]
            Additional arguments for the BaseBetaMO child model.

        Returns
        -------
        beta_mo: BetaMO
            The multi-objective Beta model.
        """
        models = [Beta(decay_factor=decay_factor) for _ in range(n_objectives)]
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
