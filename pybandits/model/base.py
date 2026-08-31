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

import numpy as np
from pydantic import (
    NonNegativeFloat,
    conlist,
)

from pybandits.base import (
    MOProbability,
    PositiveFloat01,
    Probability,
    ProbabilityWeight,
)
from pybandits.base_model import BaseModelCC, BaseModelDP, BaseModelMO, BaseModelSO


class Model(BaseModelSO, ABC):
    """
    Class to model the prior distributions for single objective.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    decay_factor: PositiveFloat01 | None = None
        Per-update forgetting factor in (0, 1]. When set, historical evidence is discounted on each
        update so the model adapts faster to non-stationary environments. None (default) and 1.0
        preserve the standard, non-decaying behavior.
    """

    decay_factor: PositiveFloat01 | None = None

    @abstractmethod
    def sample_proba(
        self, rng: np.random.Generator, **kwargs
    ) -> list[Probability] | list[MOProbability] | list[ProbabilityWeight]:
        """
        Sample the probability of getting a positive reward.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central numpy random generator provided by the MAB.
        """


class ModelCC(BaseModelCC, ABC):
    """
    Class to model action cost.

    Parameters
    ----------
    cost: NonNegativeFloat
        Cost associated to the action.
    """

    cost: NonNegativeFloat


class ModelDP(BaseModelDP, ABC):
    """
    Class to model action price.

    Parameters
    ----------
    price: NonNegativeFloat
        Price associated to the action.
    """

    price: NonNegativeFloat


class ModelMO(BaseModelMO, ABC):
    """
    Class to model the prior distributions for multi-objective.

    Parameters
    ----------
    models : list[Model]
        The list of models for each objective.
    """

    models: conlist(Model, min_length=1)
