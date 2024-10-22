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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pybandits.base import ActionId, BinaryReward
from pybandits.pydantic_version_compatibility import Field, model_validator
from pybandits.simulator import Simulator
from pybandits.smab import BaseSmabBernoulli


class SmabSimulator(Simulator):
    """
    Simulate environment for stochastic multi-armed bandits.

    This class performs simulation of stochastic Multi-Armed Bandits (sMAB). Data are processed in batches of size n>=1.
    Per each batch of simulated samples, the mab selects one a1nd collects the corresponding simulated reward for
    each sample. Then, prior parameters are updated based on returned rewards from recommended actions.

    Parameters
    ----------
    mab : BaseSmabBernoulli
        sMAB model.
    """

    mab: BaseSmabBernoulli = Field(validation_alias="smab")
    _base_columns: List[str] = ["batch", "action", "reward"]

    @model_validator(mode="before")
    @classmethod
    def replace_null_and_validate_probs_reward(cls, values):
        mab_action_ids = list(values["mab"].actions.keys())
        probs_reward = cls._get_value_with_default("probs_reward", values)
        if probs_reward is None:
            probs_reward = pd.DataFrame(0.5, index=[0], columns=mab_action_ids)
            values["probs_reward"] = probs_reward
        else:
            if len(probs_reward) != 1:
                raise ValueError("probs_reward must have exactly one row.")
        return values

    def _initialize_results(self):
        """
        Initialize the results DataFrame. The results DataFrame is used to store the raw simulation results.
        """
        self._results = pd.DataFrame(columns=["batch", "action", "reward"])

    def _draw_rewards(self, actions: List[ActionId], metadata: Dict[str, List]) -> List[BinaryReward]:
        """
        Draw rewards for the selected actions according to probs_reward.

        Parameters
        ----------
        actions : List[ActionId]
            The actions selected by the multi-armed bandit model.
        metadata : Dict[str, List]
            The metadata for the selected actions. Not used in this implementation.

        Returns
        -------
        reward : List[BinaryReward]
            A list of binary rewards.
        """
        rewards = [int(random.random() < self.probs_reward.loc[0, a]) for a in actions]
        return rewards

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

    def _finalize_step(self, batch_results: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize the step by adding additional information to the batch results.

        Parameters
        ----------
        batch_results : pd.DataFrame
            raw batch results

        Returns
        -------
        batch_results : pd.DataFrame
            same raw batch results
        """
        return batch_results

    def _finalize_results(self):
        """
        Finalize the simulation process. It can be used to add additional information to the results.

        Returns
        -------
        None
        """
        pass
