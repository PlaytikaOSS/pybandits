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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pybandits.base import ActionId, BinaryReward, Probability, UnifiedActionId
from pybandits.cmab import BaseCmabBernoulli
from pybandits.pydantic_version_compatibility import Field, model_validator
from pybandits.quantitative_model import QuantitativeModel
from pybandits.simulator import (
    DoubleParametricActionProbability,
    ParametricActionProbability,
    Simulator,
)
from pybandits.utils import extract_argument_names_from_function

CmabProbabilityValue = Union[ParametricActionProbability, DoubleParametricActionProbability]
CmabActionProbabilityGroundTruth = Dict[ActionId, CmabProbabilityValue]


class CmabSimulator(Simulator):
    """
    Simulate environment for contextual multi-armed bandit models.

    This class simulates information required by the contextual bandit. Generated data are processed by the bandit with
    batches of   size n>=1. For each batch of samples, actions are recommended by the bandit and corresponding simulated
    rewards collected. Bandit policy parameters are then updated based on returned rewards from recommended actions.

    Parameters
    ----------
    mab : BaseCmabBernoulli
        Contextual multi-armed bandit model
    context : np.ndarray of shape (n_samples, n_feature)
        Context matrix of samples features.
    group : Optional[List] with length=n_samples
        Group to which each sample belongs. Samples which belongs to the same group have features that come from the
        same distribution and they have the same probability to receive a positive/negative feedback from each action.
        If not supplied, all samples are assigned to the group.
    """

    probs_reward: Optional[Union[CmabActionProbabilityGroundTruth, Dict[str, CmabActionProbabilityGroundTruth]]] = None
    mab: BaseCmabBernoulli = Field(validation_alias="cmab")
    context: np.ndarray
    group: Optional[List] = None
    _base_columns: List[str] = ["batch", "action", "reward", "group"]

    @classmethod
    def _validate_probs_reward_values(cls, probability: CmabProbabilityValue, is_quantitative_action: bool):
        if not callable(probability):
            raise ValueError("The probability must be a callable function.")
        if not is_quantitative_action:
            if len(extract_argument_names_from_function(probability)) != 1:
                raise ValueError("The probability function must have only one argument.")
        else:
            if len(extract_argument_names_from_function(probability)) != 2:
                raise ValueError("The probability function must have only two argument.")

    @model_validator(mode="before")
    @classmethod
    def replace_nulls_and_validate_sizes_and_dtypes(cls, values):
        context = values["context"]
        batch_size = cls._get_value_with_default("batch_size", values)
        n_updates = cls._get_value_with_default("n_updates", values)
        group = cls._get_value_with_default("group", values)

        if len(context) != batch_size * n_updates:
            raise ValueError("Context length must equal to batch_size x n_updates.")
        if group is None:
            group = len(context) * ["0"]
            values["group"] = group
        else:
            if len(context) != len(group):
                raise ValueError("Mismatch between context length and group length")
            values["group"] = [str(g) for g in group]
        probs_reward = cls._get_value_with_default("probs_reward", values)
        if probs_reward is None:
            probs_reward = {
                g: {
                    action: cls._generate_prob_reward(values["context"].shape[1], model.dimension)
                    if isinstance(model, QuantitativeModel)
                    else cls._generate_prob_reward(values["context"].shape[1])
                    for action, model in values["mab"].actions.items()
                }
                for g in set(group)
            }
            values["probs_reward"] = probs_reward
        else:
            if probs_reward.shape[0] != len(set(group)):
                raise ValueError("number of probs_reward rows must match the number of groups.")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_probs_reward_columns(cls, values):
        if "probs_reward" in values and values["probs_reward"] is not None:
            groups = set(values["group"])
            if set(values["probs_reward"].keys()) != groups:
                raise ValueError("probs_reward keys must match groups.")
            for v in values["probs_reward"].values():
                cls._validate_probs_reward_dict(v, values["mab"].actions)
        return values

    def _initialize_results(self):
        """
        Initialize the results DataFrame. The results DataFrame is used to store the raw simulation results.
        """
        self._results = pd.DataFrame(
            columns=["action", "reward", "quantities", "group", "selected_prob_reward", "max_prob_reward"],
        )

    def _draw_rewards(
        self, actions: List[UnifiedActionId], metadata: Dict[str, List], update_kwargs: Dict[str, np.ndarray]
    ) -> List[BinaryReward]:
        """
        Draw rewards for the selected actions based on metadata according to probs_reward

        Parameters
        ----------
        actions : List[UnifiedActionId]
            The actions selected by the multi-armed bandit model.
        metadata : Dict[str, List]
            The metadata for the selected actions; should contain the batch groups association.

        Returns
        -------
        reward : List[BinaryReward]
            A list of binary rewards.
        """
        rewards = [
            int(random.random() < self._extract_ground_truth(a, g, c))
            for g, a, c in zip(metadata["group"], actions, update_kwargs["context"])
        ]
        return rewards

    def _extract_ground_truth(self, action: UnifiedActionId, group: str, context: np.ndarray) -> Probability:
        """
        Extract the ground truth probability for the action.

        Parameters
        ----------
        action : UnifiedActionId
            The action for which the ground truth probability is extracted.
        group : str
            The group to which the action was applied.
        context : np.ndarray
            The context for the action.

        Returns
        -------
        Probability
            The ground truth probability for the action.
        """
        return (
            self.probs_reward[group][action[0]](context, np.array(action[1]))
            if isinstance(action, tuple) and action[1] is not None
            else self.probs_reward[group][action[0]](context)
            if isinstance(action, tuple)
            else self.probs_reward[group][action](context)
        )

    def _get_batch_step_kwargs_and_metadata(
        self, batch_index
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List]]:
        """
        Extract context required for the cMAB's update and predict functionality,
        as well as metadata for sample group.

        Parameters
        ----------
        batch_index : int
            The index of the batch.

        Returns
        -------
        predict_kwargs : Dict[str, np.ndarray]
            Dictionary containing the context for the batch.
        update_kwargs : Dict[str, np.ndarray]
            Dictionary containing the context for the batch.
        metadata : Dict[str, List]
            Dictionary containing the group information for the batch.
        """
        idx_batch_min = batch_index * self.batch_size
        idx_batch_max = (batch_index + 1) * self.batch_size
        predict_and_update_kwargs = {"context": self.context[idx_batch_min:idx_batch_max]}
        metadata = {"group": self.group[idx_batch_min:idx_batch_max]}
        return predict_and_update_kwargs, predict_and_update_kwargs, metadata

    def _finalize_step(self, batch_results: pd.DataFrame, update_kwargs: Dict[str, np.ndarray]):
        """
        Finalize the step by adding additional information to the batch results.

        Parameters
        ----------
        batch_results : pd.DataFrame
            Raw batch results
        update_kwargs : Dict[str, np.ndarray]
            Context for the batch

        Returns
        -------
        batch_results : pd.DataFrame
            Batch results with added reward probability for selected action and most rewarding action
        """
        group_id = batch_results.loc[:, "group"]
        action_id = batch_results.loc[:, "action"]
        quantity = batch_results.loc[:, "quantities"]
        selected_prob_reward = [
            self._extract_ground_truth((a, q), g, c)
            for a, q, g, c in zip(action_id, quantity, group_id, update_kwargs["context"])
        ]
        batch_results.loc[:, "selected_prob_reward"] = selected_prob_reward
        max_prob_reward = [
            max(
                self._maximize_prob_reward((lambda q: self.probs_reward[g][a](c, q)), m.dimension)
                if isinstance(m, QuantitativeModel)
                else self.probs_reward[g][a](c)
                for a, m in self.mab.actions.items()
            )
            for g, c in zip(group_id, update_kwargs["context"])
        ]
        batch_results.loc[:, "max_prob_reward"] = max_prob_reward
        return batch_results
