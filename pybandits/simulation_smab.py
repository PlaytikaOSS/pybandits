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

import numpy as np
import pandas as pd
import random

from pybandits.core.smab import Smab


class SimulationSmab:
    """
    Simulate environment for stochastic multi-armed bandits.

    This class performs simulation of stochastic Multi-Armed Bandits (sMAB). Data are processed in batches of size n>=1.
    Per each batch of simulated samples, the sMAB selects one action and collects the corresponding simulated reward for
    each sample. Then, prior parameters are updated based on returned rewards from recommended actions.

    Parameters
    ----------
    smab : pybandits.core.smab.Smab
        Stochastic multi-armed bandit model.
    n_updates : int, default=10
        The number of updates (i.e. batches of samples) in the simulation.
    batch_size: int, default=100
        The number of samples per batch.
    probs_reward : dict, default=None
        The reward probability for the different actions. If None probabilities are set to 0.5.
        The keys of the dict must match the smab actions_ids, and the values are float in the interval [0, 1].
        e.g. probs_reward={'action A': 0.6, 'action B': 0.8, 'action C': 1.}
    save : bool, default=False
        Boolean flag to save the results.
    path : string, default=''
        Path where results are saved if save=True
    random_seed : int, default=None
        Seed for random state. If specified, the model outputs deterministic results.
    verbose :  bool, default=False
        Enable verbose output. If True, detailed logging information about the simulation are provided.
    """

    def __init__(
        self,
        smab,
        n_updates=10,
        batch_size=100,
        probs_reward=None,
        save=False,
        path="",
        random_seed=None,
        verbose=False,
    ):
        if type(smab) is not Smab:
            raise TypeError("smab must be of type pybandits.core.smab.Smab")
        if type(n_updates) is not int and n_updates <= 0:
            raise ValueError("n_updates must be an integer > 0")
        if type(batch_size) is not int and batch_size <= 0:
            raise ValueError("batch_size must be an integer > 0")
        if type(save) is not bool:
            raise TypeError("save must be boolean (True/False)")
        if type(path) is not str:
            raise TypeError("path must be a string")
        if type(random_seed) is not int and random_seed is not None:
            raise TypeError("random_seed must be an integer")
        if type(verbose) is not bool:
            raise TypeError("verbose must be boolean (True/False)")

        if probs_reward is None:
            probs_reward = {k: v for (k, v) in zip(smab._actions_ids, len(smab._actions_ids) * [0.5])}
        if (
            type(probs_reward) is not dict
            or not all(isinstance(x, str) for x in probs_reward.keys())
            or not all(isinstance(x, float) for x in probs_reward.values())
        ):
            raise TypeError("probs_reward must be a dict with string as keys and float as values.")
        if set(probs_reward.keys()) != set(smab._actions_ids):
            raise ValueError("probs_reward dict keys must match smab actions_ids.")
        if all(v > 1 for v in probs_reward.values()) or all(v < 0 for v in probs_reward.values()):
            raise ValueError("probs_reward values must be in the interval [0, 1].")

        self._smab = smab
        self._n_updates = n_updates
        self._batch_size = batch_size
        self._probs_reward = probs_reward
        self._save = save
        self._path = path
        self._random_seed = random_seed
        self._verbose = verbose

        # created DataFrame for simulation results
        self.results = pd.DataFrame(np.nan, columns=["action", "reward"], index=range(batch_size * n_updates))

    def run(self):
        """
        Start simulation process. It consists in the following steps:
            for i=0 to n_updates
                Consider batch[i] of observation
                sMAB selects the best action as the action with the highest reward probability to each sample in
                    batch[i].
                Rewards are returned for each recommended action
                Prior parameters are updated based on recommended actions and returned rewards
        """
        for i in range(self._n_updates):
            # select actions for batch #i
            actions, _ = self._smab.predict(n_samples=self._batch_size)

            # find min and max indexes for batch #i
            idx_batch_min = i * self._batch_size
            idx_batch_max = (i + 1) * self._batch_size - 1

            # write the selected actions for batch #i in the results matrix
            self.results.loc[idx_batch_min:idx_batch_max, "action"] = actions

            for a in self._smab._actions_ids:
                # simulate the rewards
                random.seed(self._random_seed)
                rewards = [1 if random.random() < self._probs_reward[a] else 0 for i in range(actions.count(a))]

                # find indexes of the action 'a' in the array 'actions'
                idx = [i for i in range(len(actions)) if actions[i] == a]

                # write rewards for batch #i and action 'a' in the result matrix
                self.results.loc[[idx_batch_min + i for i in idx], "reward"] = rewards

                # update the stochastic multi-armed bandit model
                self._smab.update(action_id=a, n_successes=rewards.count(1), n_failures=rewards.count(0))

        # print results
        if self._verbose:
            self._print_results()

        # store results
        if self._save:
            if self._verbose:
                print("Saving results at {}".format(self._path))
            self._save_results()

    def get_count_selected_actions(self):
        """
        Get the count of actions selected by the bandit at the end of the process.

        Returns
        -------
        dict
            Dictionary with keys=action_ids and values=count of recommended actions.
        """
        return dict(self.results.action.value_counts())

    def get_proportion_positive_reward(self):
        """
        Get the observed proportion of positive rewards for each action at the end of the simulation process.

        Returns
        -------
        dict
            Dictionary with keys=action_ids and values=proportion of positive rewards for each action.
        """
        d = {}
        for a in self._smab._actions_ids:
            x = self.results[self.results.action == a]
            d[a] = sum(x.reward) / len(x)
        return d

    def get_cumulative_proportions(self):
        """
        Get (i) the cumulative action proportions and (ii) the cumulative reward proportions per action.

        Returns
        -------
        dict
            Dictionary with keys=(actions, reward) and
            values=(cumulative action proportions, cumulative reward proportions per action)
        """
        actions = pd.get_dummies(self.results["action"]).reset_index(drop=True)
        actions_plot = actions.cumsum().div(actions.index.values + 1, axis=0)

        rewards = pd.get_dummies(self.results["action"])
        rewards.loc[self.results["reward"] == 0] = 0
        rewards.reset_index(inplace=True, drop=True)
        rewards_plot = rewards.cumsum().div(actions.cumsum())

        return {"action": actions_plot, "reward": rewards_plot}

    def _print_results(self):
        """Private function to print results."""
        print("Simulation results (first 10 observations):\n", self.results.head(10), "\n")
        print("Count of actions selected by the bandit: \n", self.get_count_selected_actions(), "\n")
        print("Observed proportion of positive rewards for each action:\n", self.get_proportion_positive_reward(), "\n")

    def _save_results(self):
        """Private function to save results."""
        self.results.to_csv("simulation_results.csv", index=False)
        with open(self._path + "summary.txt", "w") as f:
            f.write(str(self.get_count_selected_actions()) + "\n" + str(self.get_proportion_positive_reward()))
