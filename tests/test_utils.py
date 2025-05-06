import json
import pickle
import random
from tempfile import NamedTemporaryFile
from typing import Dict, get_args

import numpy as np

from pybandits.base import PyBanditsBaseModel
from pybandits.model import BaseBayesianNeuralNetwork, UpdateMethods
from pybandits.pydantic_version_compatibility import Optional, PositiveInt

literal_update_methods = get_args(UpdateMethods)


def sample_with_replacement(source: list, length: PositiveInt):
    return [random.choice(source) for _ in range(length)]


def to_temporary_pickle(model: PyBanditsBaseModel):
    with NamedTemporaryFile("wb") as file:
        pickle.dump(model, file)


class FakeApproximation(PyBanditsBaseModel):
    n_draws: PositiveInt = 10
    n_features: PositiveInt
    hidden_dim_list: Optional[list] = None

    def sample(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        sample_dict = {}
        if self.hidden_dim_list is None:
            self.hidden_dim_list = []
        dim_list = [self.n_features] + self.hidden_dim_list + [1]
        for i in range(len(dim_list) - 1):
            (weight_layer_params_name, bias_layer_params_name) = BaseBayesianNeuralNetwork.get_layer_params_name(i)
            sample_dict[weight_layer_params_name] = np.random.random(size=(self.n_draws, dim_list[i], dim_list[i + 1]))
            sample_dict[bias_layer_params_name] = np.random.random(size=(self.n_draws, dim_list[i + 1]))

        return sample_dict


def is_serializable(something) -> bool:
    try:
        json.dumps(something, default=dict)
        return True
    except Exception:
        return False
