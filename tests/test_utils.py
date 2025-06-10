import pickle
import random
from tempfile import NamedTemporaryFile
from typing import Dict, List, get_args

import numpy as np

from pybandits.base import ProbabilityWeight, PyBanditsBaseModel
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


class FakePrediction(PyBanditsBaseModel):
    n_samples: PositiveInt

    def sample_prior_predictive(self, *args, **kwargs) -> Dict[str, Dict[str, object]]:
        return {
            "prior": {
                BaseBayesianNeuralNetwork._prob_var_name: type(
                    "FakeArray", (), {"values": np.random.random(size=(self.n_samples))}
                ),
                BaseBayesianNeuralNetwork._logit_var_name: type(
                    "FakeArray", (), {"values": np.random.random(size=(self.n_samples))}
                ),
            }
        }


def fake_bnn_sample_proba(self, context: np.ndarray, *args, **kwargs) -> List[ProbabilityWeight]:
    n_samples = len(context)
    fake_prediction = FakePrediction(n_samples=n_samples)
    predictions = fake_prediction.sample_prior_predictive()["prior"]
    mock_probs = predictions[BaseBayesianNeuralNetwork._prob_var_name].values
    mock_weighted_sums = predictions[BaseBayesianNeuralNetwork._logit_var_name].values
    return list(zip(mock_probs, mock_weighted_sums))
