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
from json import dump

import numpy as np
import pandas as pd

from pybandits.core.cmab import Cmab, check_context_matrix


class SimulationCmab:
    """
    Simulate environment for contextual multi-armed bandit models.

    This class simulates information required by the contextual bandit. Generated data are processed by the bandit with
    batches of   size n>=1. For each batch of samples, actions are recommended by the bandit and corresponding simulated
    rewards collected. Bandit policy parameters are then updated based on returned rewards from recommended actions.

    Parameters
    ----------
    cmab : pybandits.core.cmab.Cmab
        Contextual multi-armed bandit model
    X : array_like of shape (n_samples, n_feature)
        Context matrix of samples features.
    batch_size: int, default=100
        The number of samples per batch.
    n_updates : int, default=10
        The number of updates in the simulation.
    group : list int with length=n_samples
        Group to which each sample belongs. Samples which belongs to the same group have features that come from the
        same distribution and they have the same probability to receive a positive/negative feedback from each action.
    prob_rewards : pd.DataFrame of shape (n_groups, n_actions)
        Matrix of positive reward probability for each group-action combination. If None all probs are set to 0.5.
    save : bool, default=False
        Boolean flag to save the results.
    path : string, default=''
        Path where results are saved if save=True
    verbose :  bool, default=False
        Enable verbose output. If True produce detailed logging information about the simulation.
    random_seed : int, default=None
        Seed for random state. If specified, the model outputs deterministic results.
    """

    def __init__(
        self,
        cmab,
        X,
        batch_size=100,
        n_updates=10,
        group=None,
        prob_rewards=None,
        save=False,
        path="",
        random_seed=None,
        verbose=False,
    ):
        # init cmab
        if type(cmab) is not Cmab:
            raise TypeError("cmab must be of type pybandits.core.cmab.Cmab")
        self._cmab = cmab

        # init batch-size
        if type(batch_size) is not int and batch_size <= 0:
            raise ValueError("batch_size must be an integer > 0")
        self._batch_size = batch_size

        # init n_updates
        if type(n_updates) is not int and n_updates <= 0:
            raise ValueError("n_updates must be an integer > 0")
        self._n_updates = n_updates

        # init X
        self._X = check_context_matrix(X, cmab._n_features)
        if len(self._X) != batch_size * n_updates:
            raise ValueError(
                "Mismatch between n_samples samples in the context matrix with batch_size and n_updates. "
                "len(X) must be equal to batch_size x n_updates."
            )

        # init group
        if group is None:
            group = len(X) * [0]  # if the input argument group is not specified, set all samples group to #0
        if len(group) != len(X):
            raise ValueError("The length of X must equal to the length of group")
        self._n_groups = len(set(group))

        # create matrix of probability rewards if None (by default all probs = 0.5)
        if prob_rewards is None:
            prob_rewards = pd.DataFrame(0.5, index=set(group), columns=cmab._actions_ids)
        if prob_rewards.shape[0] != self._n_groups and self._prob_rewards.shape[1] != len(self._cmab._actions_ids):
            raise ValueError(
                "matrix of probability rewards should have shape ({}, {}), while detected shape is {}".format(
                    len(group), len(cmab._actions_ids), prob_rewards.shape
                )
            )

        # init prob_rewards, save, path, verbose
        if type(path) is not str:
            raise TypeError("path must be a string")
        if type(save) is not bool:
            raise TypeError("save must be boolean (True/False)")
        if type(random_seed) is not int and random_seed is not None:
            raise TypeError("random_seed must be an integer")
        if type(verbose) is not bool:
            raise TypeError("verbose must be boolean (True/False)")
        self._prob_rewards = prob_rewards
        self._save = save
        self._path = path
        self._verbose = verbose

        # create rewards per each sample given the matrix of probability rewards
        random.seed(random_seed)
        self._rewards = [
            [1 if random.random() < prob_rewards.iloc[group[i], j] else 0 for j in range(len(cmab._actions_ids))]
            for i in range(batch_size * n_updates)
        ]
        self._rewards = pd.DataFrame(self._rewards, columns=cmab._actions_ids)

        # created DataFrame for simulation results
        self.results = pd.DataFrame(
            np.nan,
            columns=["action", "reward", "group", "selected_prob_reward", "max_prob_reward"],
            index=range(batch_size * n_updates),
        )
        self.results["group"] = pd.Series(group)

        if self._verbose:
            print("Setup simulation  completed.")
            df = pd.DataFrame(
                [
                    np.sum(self._rewards.loc[self.results["group"] == i]) / sum(self.results["group"] == i)
                    for i in range(self._n_groups)
                ]
            )
            df.index.name = "group"
            self._prob_rewards.index.name = "group"
            print("Simulated input probability rewards:\n", df, "\n")

    def run(self):
        """
        Start simulation process. It consists in the following steps:

            - for i=0 to n_updates
                - Extract batch[i] of samples from X
                - Model recommends the best actions as the action with the highest reward probability to each simulated
                  sample in batch[i] and collect corresponding simulated rewards
                - Model priors are updated using information from recommended actions and returned rewards
        """

        for i in range(self._n_updates):
            if self._verbose:
                print("Iteration #{}".format(i + 1))

            # extract simulated data for the current batch and scale the features
            idx_batch_min = i * self._batch_size
            idx_batch_max = (i + 1) * self._batch_size - 1
            X_batch = self._X.loc[idx_batch_min:idx_batch_max]

            # predict
            if self._verbose:
                print("Start predict batch {} ...".format(i + 1))

            actions, _ = self._cmab.fast_predict(X_batch)

            # Get reward
            rewards = [self._rewards.loc[j + idx_batch_min, actions[j]] for j in range(self._batch_size)]

            # update cmab
            if self._verbose:
                print("Start update batch {} ...".format(i + 1), "\n")
            self._cmab.update(X=X_batch, actions=actions, rewards=rewards)

            # write in simulation results
            self.results.loc[idx_batch_min:idx_batch_max, "action"] = pd.Series(
                actions, index=range(idx_batch_min, idx_batch_max + 1)
            )
            self.results.loc[idx_batch_min:idx_batch_max, "reward"] = pd.Series(
                rewards, index=range(idx_batch_min, idx_batch_max + 1)
            )

            # write for regret analysis:
            # 1. extract group information
            # 2. reward prob for selected action and
            # 3. reward probability from most rewarding action
            group_id = self.results.loc[idx_batch_min:idx_batch_max, "group"].tolist()
            selected_prob_reward = [
                self._prob_rewards.iloc[group_id[k], self._cmab._actions_ids.index(actions[k])]
                for k in range(len(actions))
            ]
            self.results.loc[idx_batch_min:idx_batch_max, "selected_prob_reward"] = pd.Series(
                selected_prob_reward, index=range(idx_batch_min, idx_batch_max + 1)
            )
            max_prob_reward = [self._prob_rewards.iloc[group_id[k],].max() for k in range(len(actions))]
            self.results.loc[idx_batch_min:idx_batch_max, "max_prob_reward"] = pd.Series(
                max_prob_reward, index=range(idx_batch_min, idx_batch_max + 1)
            )

            # save partial results
            if self._save:
                self.results.to_csv(self._path + "simulation_results.csv")

        # compute expected cumulative regrets
        self.results["regret"] = self.results["max_prob_reward"] - self.results["selected_prob_reward"]
        self.results["cum_regret"] = self.results["regret"].cumsum()

        if self._verbose:
            self._print_results()

        # store results
        if self._save:
            if self._verbose:
                print("Saving results...")
            self._save_results()

    def get_count_selected_actions(self):
        """
        Get the proportion of recommended actions per group at the end of the process.

        Returns
        -------
        df : pandas DataFrame
            Matrix of the proportion of recommended actions per group.
        """
        return {
            "group " + str(i): (self.results["action"].loc[self.results["group"] == i].value_counts()).to_dict()
            for i in range(self._n_groups)
        }

    def get_proportion_positive_reward(self):
        """
        Get the proportion of positive rewards per group/action at the end of the process.

        Returns
        -------
        df : pandas DataFrame
            Matrix of the proportion of positive rewards per group/action.
        """
        return {
            "group "
            + str(i): (
                self.results["action"].loc[(self.results["group"] == i) & (self.results["reward"] == 1)].value_counts()
                / self.results["action"].loc[(self.results["group"] == i)].value_counts()
            ).to_dict()
            for i in range(self._n_groups)
        }

    def get_cumulative_proportions(self, path=""):
        """
        Plot results of the simulation. It will create two plots per each group which display:
            - The cumulated proportion of action
            - The cumulated proportion of rewards

        Parameters
        ----------
        path: str, default=''
            Path in which plot figures are saved.
        """
        d = {}
        for i in range(self._n_groups):
            actions = pd.get_dummies(self.results["action"].loc[self.results["group"] == i]).reset_index(drop=True)
            actions_plot = actions.cumsum().div(actions.index.values + 1, axis=0)

            rewards = pd.get_dummies(self.results["action"].loc[self.results["group"] == i])
            rewards.loc[(self.results["group"] == i) & self.results["reward"] == 0] = 0
            rewards.reset_index(inplace=True, drop=True)
            rewards_plot = rewards.cumsum().div(actions.cumsum())

            d["group " + str(i)] = {"action": actions_plot, "reward": rewards_plot}
        return d

    def _print_results(self):
        """Private function to print results."""
        print("Simulation results (first 10 observations):\n", self.results.head(10), "\n")
        print("Count of actions selected by the bandit: \n", self.get_count_selected_actions(), "\n")
        print("Observed proportion of positive rewards for each action:\n", self.get_proportion_positive_reward(), "\n")

    def _save_results(self):
        """Private function to save results."""
        self.results.to_csv(self._path + "simulation_results.csv")

        f = open("count_selected_actions.json", "w")
        dump(self.get_count_selected_actions(), f)
        f.close()

        f = open("proportions_of_positive_rewards.json", "w")
        dump(self.get_proportion_positive_reward(), f)
        f.close()
