import json
import pickle
import random
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple, get_args

import numpy as np
from bokeh.core.serialization import Serializable

from pybandits.base import PyBanditsBaseModel
from pybandits.model import BaseBayesianNeuralNetwork, UpdateMethods
from pybandits.pydantic_version_compatibility import Optional, PositiveInt, PrivateAttr

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
    _hist: Optional[np.ndarray] = PrivateAttr(default=None)

    @property
    def hist(self) -> Optional[np.ndarray]:
        return self._hist

    def sample(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        sample_dict = {}
        if self.hidden_dim_list is None:
            self.hidden_dim_list = []
        dim_list = [self.n_features] + self.hidden_dim_list + [1]
        for i in range(len(dim_list) - 1):
            weight_layer_params_name, bias_layer_params_name = BaseBayesianNeuralNetwork.get_layer_params_name(i)
            sample_dict[weight_layer_params_name] = np.random.random(size=(self.n_draws, dim_list[i], dim_list[i + 1]))
            sample_dict[bias_layer_params_name] = np.random.random(size=(self.n_draws, dim_list[i + 1]))

        return sample_dict


def pop_from_state(state: str, key: str) -> Tuple[Serializable, str]:
    """
    Pop a key from a JSON string state.

    Parameters
    ----------
    state: str
        The JSON string state.
    key: str
        The key to pop.

    Returns
    -------
    value : Any
        The value of the popped key.
    str

    """

    state_dict = json.loads(state)
    value = state_dict.pop(key, None)
    state = json.dumps(state_dict)
    return value, state


def push_to_state(state: str, key: str, value: Any):
    """
    Push a key-value pair to a JSON string state.

    Parameters
    ----------
    state: str
        The JSON string state.
    key: str
        The key to push.
    value: Any
        The value to push.

    Returns
    -------
    new_state: str
        The updated JSON string state.
    """

    state_dict = json.loads(state)
    state_dict[key] = value
    return json.dumps(state_dict)
