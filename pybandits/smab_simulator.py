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

import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pybandits.base import ActionId, BinaryReward, Probability, UnifiedActionId
from pybandits.pydantic_version_compatibility import Field, model_validator
from pybandits.quantitative_model import QuantitativeModel
from pybandits.simulator import Simulator
from pybandits.smab import BaseSmabBernoulli
from pybandits.utils import extract_argument_names_from_function

#                                        quantity
ParametricActionProbability = Callable[[np.ndarray], Probability]
SmabProbabilityValue = Union[Probability, ParametricActionProbability]
SmabActionProbabilityGroundTruth = Dict[ActionId, SmabProbabilityValue]


class SmabSimulator(Simulator):
    """
    Simulate environment for stochastic multi-armed bandits.

    This class performs simulation of stochastic Multi-Armed Bandits (sMAB). Data are processed in batches of size n>=1.
    Per each batch of simulated samples, the mab selects one action and collects the corresponding simulated reward for
    each sample. Then, prior parameters are updated based on returned rewards from recommended actions.

    Parameters
    ----------
    mab : BaseSmabBernoulli
        sMAB model.
    """

    probs_reward: Optional[Union[SmabActionProbabilityGroundTruth, Dict[str, SmabActionProbabilityGroundTruth]]] = None
    mab: BaseSmabBernoulli = Field(validation_alias="smab")
    _base_columns: List[str] = ["batch", "action", "reward"]

    @classmethod
    def _validate_probs_reward_values(cls, probability: SmabProbabilityValue, is_quantitative_action: bool):
        if not is_quantitative_action:
            if not isinstance(probability, float):
                raise ValueError("The probability must be a float.")
            if not 0 <= probability <= 1:
                raise ValueError("The probability must be in the interval [0, 1].")
        else:
            if not callable(probability):
                raise ValueError("The probability must be a callable function.")
            if len(extract_argument_names_from_function(probability)) != 1:
                raise ValueError("The probability function must have only one argument.")

    @model_validator(mode="before")
    @classmethod
    def replace_null_and_validate_probs_reward(cls, values):
        probs_reward = cls._get_value_with_default("probs_reward", values)
        if probs_reward is None:
            probs_reward = {
                action: cls._generate_prob_reward(model.dimension)
                if isinstance(model, QuantitativeModel)
                else np.random.random()
                for action, model in values["mab"].actions.items()
            }
            values["probs_reward"] = probs_reward
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_probs_reward_columns(cls, values):
        if "probs_reward" in values and values["probs_reward"] is not None:
            cls._validate_probs_reward_dict(values["probs_reward"], values["mab"].actions)
        return values

    def _initialize_results(self):
        """
        Initialize the results DataFrame. The results DataFrame is used to store the raw simulation results.
        """
        self._results = pd.DataFrame(
            columns=["batch", "action", "reward", "quantities", "selected_prob_reward", "max_prob_reward"]
        )

    def _draw_rewards(
        self, actions: List[UnifiedActionId], metadata: Dict[str, List], update_kwargs: Dict[str, np.ndarray]
    ) -> List[BinaryReward]:
        """
        Draw rewards for the selected actions according to probs_reward.

        Parameters
        ----------
        actions : List[UnifiedActionId]
            The actions selected by the multi-armed bandit model.
        metadata : Dict[str, List]
            The metadata for the selected actions. Not used in this implementation.

        Returns
        -------
        reward : List[BinaryReward]
            A list of binary rewards.
        """
        rewards = [int(random.random() < self._extract_ground_truth(a)) for a in actions]
        return rewards

    def _extract_ground_truth(self, action: UnifiedActionId) -> Probability:
        """
        Extract the ground truth probability for the action.

        Parameters
        ----------
        action : UnifiedActionId
            The action for which the ground truth probability is extracted.

        Returns
        -------
        Probability
            The ground truth probability for the action.
        """
        return (
            self.probs_reward[action[0]](np.array(action[1]))
            if isinstance(action, tuple) and action[1] is not None
            else self.probs_reward[action[0]]
            if isinstance(action, tuple)
            else self.probs_reward[action]
        )

    def _get_batch_step_kwargs_and_metadata(
        self, batch_index
    ) -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, List]]:
        """
        Extract context required for the sMAB's update and predict functionality,
        as well as metadata for sample group.

        Parameters
        ----------
        batch_index : int
            The index of the batch.

        Returns
        -------
        predict_kwargs : Dict[str, int]
            Dictionary containing the number of samples for sMAB prediction.
        update_kwargs : Dict[str, np.ndarray]
            Dictionary containing nothing.
        metadata : Dict[str, List]
            Dictionary containing nothing.
        """
        predict_kwargs = {"n_samples": self.batch_size}
        update_kwargs = {}
        metadata = {}
        return predict_kwargs, update_kwargs, metadata

    def _finalize_step(self, batch_results: pd.DataFrame, update_kwargs: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Finalize the step by adding additional information to the batch results.

        Parameters
        ----------
        batch_results : pd.DataFrame
            Raw batch results
        update_kwargs : Dict[str, np.ndarray]
            Placeholder for interface compatability

        Returns
        -------
        batch_results : pd.DataFrame
            Same raw batch results
        """
        action_id = batch_results.loc[:, "action"]
        quantity = batch_results.loc[:, "quantities"]
        selected_prob_reward = [self._extract_ground_truth((a, q)) for a, q in zip(action_id, quantity)]
        batch_results.loc[:, "selected_prob_reward"] = selected_prob_reward
        max_prob_reward = [
            max(
                self._maximize_prob_reward((lambda q: self.probs_reward[a](q)), m.dimension)
                if isinstance(m, QuantitativeModel)
                else self.probs_reward[a]
                for a, m in self.mab.actions.items()
            )
        ] * len(batch_results)
        batch_results.loc[:, "max_prob_reward"] = max_prob_reward
        return batch_results
