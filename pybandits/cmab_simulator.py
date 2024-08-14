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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pybandits.base import ActionId, BinaryReward
from pybandits.cmab import BaseCmabBernoulli
from pybandits.pydantic_version_compatibility import Field, model_validator
from pybandits.simulator import Simulator


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

    mab: BaseCmabBernoulli = Field(validation_alias="cmab")
    context: np.ndarray
    group: Optional[List] = None
    _base_columns: List[str] = ["batch", "action", "reward", "group"]

    @model_validator(mode="before")
    @classmethod
    def replace_nulls_and_validate_sizes(cls, values):
        context = values["context"]
        batch_size = cls._get_value_with_default("batch_size", values)
        n_updates = cls._get_value_with_default("n_updates", values)
        group = cls._get_value_with_default("group", values)

        if len(context) != batch_size * n_updates:
            raise ValueError("Context length must equal to batch_size x n_updates.")
        if group is None:
            group = len(context) * [0]
            values["group"] = group
        else:
            if len(context) != len(group):
                raise ValueError("Mismatch between context length and group length")
        mab_action_ids = list(values["mab"].actions.keys())
        index = list(set(group))
        probs_reward = cls._get_value_with_default("probs_reward", values)
        if probs_reward is None:
            probs_reward = pd.DataFrame(0.5, index=index, columns=mab_action_ids)
            values["probs_reward"] = probs_reward
        else:
            if probs_reward.shape[0] != len(index):
                raise ValueError("number of probs_reward rows must match the number of groups.")
        return values

    def _initialize_results(self):
        """
        Initialize the results DataFrame. The results DataFrame is used to store the raw simulation results.
        """
        self._results = pd.DataFrame(
            columns=["action", "reward", "group", "selected_prob_reward", "max_prob_reward"],
        )

    def _draw_rewards(self, actions: List[ActionId], metadata: Dict[str, List]) -> List[BinaryReward]:
        """
        Draw rewards for the selected actions based on metadata according to probs_reward

        Parameters
        ----------
        actions : List[ActionId]
            The actions selected by the multi-armed bandit model.
        metadata : Dict[str, List]
            The metadata for the selected actions; should contain the batch groups association.

        Returns
        -------
        reward : List[BinaryReward]
            A list of binary rewards.
        """
        rewards = [int(random.random() < self.probs_reward.loc[g, a]) for g, a in zip(metadata["group"], actions)]
        return rewards

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

    def _finalize_step(self, batch_results: pd.DataFrame):
        """
        Finalize the step by adding additional information to the batch results.

        Parameters
        ----------
        batch_results : pd.DataFrame
            raw batch results

        Returns
        -------
        batch_results : pd.DataFrame
            batch results with added reward probability for selected a1nd most rewarding action
        """
        group_id = batch_results.loc[:, "group"]
        action_id = batch_results.loc[:, "action"]
        selected_prob_reward = [self.probs_reward.loc[g, a] for g, a in zip(group_id, action_id)]
        batch_results.loc[:, "selected_prob_reward"] = selected_prob_reward
        max_prob_reward = self.probs_reward.loc[group_id].max(axis=1)
        batch_results.loc[:, "max_prob_reward"] = max_prob_reward.tolist()
        return batch_results

    def _finalize_results(self):
        """
        Finalize the simulation process. Used to add regret and cumulative regret

        Returns
        -------
        None
        """
        self._results["regret"] = self._results["max_prob_reward"] - self._results["selected_prob_reward"]
        self._results["cum_regret"] = self._results["regret"].cumsum()
