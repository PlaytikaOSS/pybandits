import random
from typing import Dict, get_args

import numpy as np

from pybandits.base import PyBanditsBaseModel
from pybandits.model import UpdateMethods
from pybandits.pydantic_version_compatibility import PositiveInt

literal_update_methods = get_args(UpdateMethods)


def sample_with_replacement(source: list, length: PositiveInt):
    return [random.choice(source) for _ in range(length)]


class FakeApproximation(PyBanditsBaseModel):
    n_draws: PositiveInt = 10
    n_features: PositiveInt

    def sample(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        return {
            "alpha": np.random.random(size=(1, self.n_draws)),
            "betas": np.random.random(size=(self.n_features, self.n_draws)),
        }
