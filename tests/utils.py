import json
import pickle
import random
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple, get_args

import numpy as np
from bokeh.core.serialization import Serializable

from pybandits.base import PyBanditsBaseModel, UnifiedActionId
from pybandits.base_model import BaseModel
from pybandits.model import BaseBayesianNeuralNetwork, UpdateMethods
from pybandits.pydantic_version_compatibility import (
    Optional,
    PositiveInt,
    PrivateAttr,
)
from pybandits.quantitative_model import QuantitativeModel

literal_update_methods = get_args(UpdateMethods)


class _EvalArray:
    """Helper class that mimics the behavior of PyMC tensor-like objects with eval() method."""

    def __init__(self, array: np.ndarray):
        self.array = array

    def eval(self) -> np.ndarray:
        """
        Evaluate and return the array.

        Returns
        -------
        np.ndarray
            The underlying numpy array.
        """
        return self.array


def sample_with_replacement(source: list, length: PositiveInt):
    return [random.choice(source) for _ in range(length)]


def to_temporary_pickle(model: PyBanditsBaseModel):
    with NamedTemporaryFile("wb") as file:
        pickle.dump(model, file)


def to_unified_action_id(action_id: str, model: BaseModel) -> UnifiedActionId:
    if isinstance(model, QuantitativeModel):
        return (action_id, (np.random.random(),))
    else:
        return action_id


def mock_update(self, *args, **kwargs):
    pass


class FakeApproximation(PyBanditsBaseModel):
    n_draws: PositiveInt = 10
    n_features: PositiveInt
    hidden_dim_list: Optional[list] = None
    _hist: Optional[np.ndarray] = PrivateAttr(default=None)
    _mean: Optional[_EvalArray] = PrivateAttr(default=None)
    _std: Optional[_EvalArray] = PrivateAttr(default=None)
    _ordering: Optional[Dict[str, Tuple[Any, slice, Any, Any]]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize mean, std, and ordering after model initialization.

        Parameters
        ----------
        __context : Any
            The context passed from Pydantic initialization.
        """
        if self.hidden_dim_list is None:
            self.hidden_dim_list = []
        dim_list = [self.n_features] + self.hidden_dim_list + [1]

        # Calculate total size of flattened parameter array
        total_size = 0
        param_sizes = {}
        for i in range(len(dim_list) - 1):
            weight_layer_params_name, bias_layer_params_name = BaseBayesianNeuralNetwork.get_layer_params_name(i)
            weight_size = dim_list[i] * dim_list[i + 1]
            bias_size = dim_list[i + 1]
            param_sizes[weight_layer_params_name] = (total_size, weight_size)
            total_size += weight_size
            param_sizes[bias_layer_params_name] = (total_size, bias_size)
            total_size += bias_size

        # Create mean and std arrays
        mean_array = np.random.random(size=total_size)
        std_array = np.random.random(size=total_size)
        self._mean = _EvalArray(mean_array)
        self._std = _EvalArray(std_array)

        # Create ordering dictionary with slices
        ordering = {}
        for param_name, (start_idx, size) in param_sizes.items():
            end_idx = start_idx + size
            param_slice = slice(start_idx, end_idx)
            ordering[param_name] = (None, param_slice, None, None)

        self._ordering = ordering

    @property
    def hist(self) -> Optional[np.ndarray]:
        """Return the history array."""
        return self._hist

    @property
    def mean(self) -> _EvalArray:
        """
        Return the mean tensor-like object with eval() method.

        Returns
        -------
        _EvalArray
            The mean array wrapper.
        """
        return self._mean

    @property
    def std(self) -> _EvalArray:
        """
        Return the std tensor-like object with eval() method.

        Returns
        -------
        _EvalArray
            The std array wrapper.
        """
        return self._std

    @property
    def ordering(self) -> Dict[str, Tuple[Any, slice, Any, Any]]:
        """
        Return the ordering dictionary mapping parameter names to slices.

        Returns
        -------
        Dict[str, Tuple[Any, slice, Any, Any]]
            Dictionary mapping parameter names to tuples containing slices for indexing.
        """
        return self._ordering

    def sample(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Sample from the approximation."""
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
