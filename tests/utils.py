import json
import pickle
import random
from tempfile import NamedTemporaryFile
from typing import Any, List, Tuple, get_args

import numpy as np
from bokeh.core.serialization import Serializable

from pybandits.base import PyBanditsBaseModel, UnifiedActionId
from pybandits.base_model import BaseModel
from pybandits.model import BaseBayesianNeuralNetwork, BaseBayesianNeuralNetworkMO, BnnLayerParams, UpdateMethods
from pybandits.pydantic_version_compatibility import (
    PositiveInt,
)
from pybandits.quantitative_model import BaseQuantitativeBayesianNeuralNetwork, QuantitativeModel

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
    """Mock BNN update that validates inputs then sets random parameter values."""

    # For quantitative BNN models, delegate to the inner bnn. Skip context validation
    # because the raw context lacks the quantity columns expected by the inner bnn.
    if isinstance(self, BaseQuantitativeBayesianNeuralNetwork):
        mock_update(self.bnn, *args, **kwargs)
        return

    updated_layer_params_list = []
    for layer_params in self.model_params.bnn_layer_params:
        w_shape = layer_params.weight.shape
        b_shape = layer_params.bias.shape
        w_mu = np.random.random(w_shape)
        w_sigma = np.abs(np.random.random(w_shape)) + 1e-6
        b_mu = np.random.random(b_shape)
        b_sigma = np.abs(np.random.random(b_shape)) + 1e-6
        updated_weight = layer_params.weight.with_dist_parameters(mu=w_mu.tolist(), sigma=w_sigma.tolist())
        updated_bias = layer_params.bias.with_dist_parameters(mu=b_mu.tolist(), sigma=b_sigma.tolist())
        updated_layer_params_list.append(BnnLayerParams(weight=updated_weight, bias=updated_bias))
    self.model_params.bnn_layer_params = updated_layer_params_list


def apply_mock_update(actions: List[Any]) -> None:
    """Apply mock_update to every BNN model contained in a list of CMAB actions.

    Handles all three action-model variants:

    - ``BaseBayesianNeuralNetwork``: updated directly.
    - ``BaseBayesianNeuralNetworkMO``: each per-objective sub-model is updated.
    - ``BaseQuantitativeBayesianNeuralNetwork``: the wrapped ``bnn`` is updated.

    Parameters
    ----------
    actions : List[Any]
        The action-model objects extracted from a CMAB instance
        (e.g. ``list(cmab.actions.values())``).
    """
    for action in actions:
        if isinstance(action, BaseBayesianNeuralNetworkMO):
            for sub_model in action.models:
                mock_update(sub_model)
        elif isinstance(action, BaseBayesianNeuralNetwork):
            mock_update(action)
        elif isinstance(action, BaseQuantitativeBayesianNeuralNetwork):
            mock_update(action.bnn)


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
