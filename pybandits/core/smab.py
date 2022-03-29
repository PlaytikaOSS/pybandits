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

import copy
import random


class Smab:
    """
    Stochastic Multi-Armed Bandit for Bernoulli bandits with Thompson Sampling.

    Parameters
    ----------
    action_ids: List[str]
        List of possible actions
    success_priors: Dict[str, int]
        Dictionary containing the prior number of positive feedback (successes) for each action. keys are
        action IDs and values are successes counts. If None, each action's prior is set to 1 by default.
        Success counts must be integers and > 0.
    failure_priors: Dict[str, int]
        Dictionary containing the prior number of negative feedback (failures) for each action. keys are
        action IDs and values are failures counts. If None, each action's prior is set to 1 by default.
        Failure counts must be integers and > 0.
    random_seed: int
        Seed for random state. If specified, the model outputs deterministic results.
    """
    def __init__(self, action_ids, success_priors=None, failure_priors=None, random_seed=None):
        """ Initialization. """
        if random_seed is not None:
            random.seed(random_seed)
        self._actions_ids = action_ids
        self._success_counters = dict((t, 1) for t in self._actions_ids) if success_priors is None else \
            copy.deepcopy(success_priors)
        """ dict: success counters initialized to 1 for each action"""
        self._failure_counters = dict((t, 1) for t in self._actions_ids) if failure_priors is None else \
            copy.deepcopy(failure_priors)
        """dict: failure counters initialized to 1 for each action"""

        # Input sanity checking
        if ((success_priors is None and failure_priors is not None) or (success_priors is not None and
                                                                        failure_priors is None)):
            raise ValueError('Either both or neither success_prior and failure_prior should be specified.')
        if set(self._success_counters.keys()) != set(action_ids) or \
                set(self._failure_counters.keys()) != set(action_ids):
            raise ValueError('Treatment IDs, successes keys and failure keys must be identical.')
        if len(action_ids) < 1:
            raise ValueError('There must be at least one action.')

        # Input type checking
        if type(self._actions_ids) is not list:
            raise TypeError('action_ids must be a list of strings.')
        if type(self._success_counters) is not dict or type(self._failure_counters) is not dict:
            raise TypeError('success_priors and failure_priors must be dictionaries.')
        for t in self._actions_ids:
            if self._success_counters[t] <= 0 or self._failure_counters[t] <= 0:
                raise ValueError('Success/failure counters must be > 0')
            if type(self._success_counters[t]) is not int or type(self._failure_counters[t]) is not int:
                raise TypeError('Success/failure counters must be integers.')
            if type(t) is not str:
                raise TypeError('Treatments must be strings.')

    def predict(self, n_samples=1, forbidden_actions=None):
        """
        Predict the best actions by randomly drawing samples from a beta distribution for each possible action.
        The action with the highest value is recommended to the user as the 'best action' considering current
        information. The Beta distributions' alpha and beta parameters for each action are its associated counts
        of success and failure, respectively.

        Parameters
        ----------
        n_samples: int
            Number of samples to predict (default 1).
        forbidden_actions: List[str]
            List of forbidden actions. If specified, the model will discard the forbidden_actions and it will only
            consider the remaining allowed_actions. By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        best_actions: list
            The best actions according to the model, i.e. the actions whose distribution gave the greater sample.
        probs: list
            The probabilities to get a positive reward for each action.
        """
        if forbidden_actions is None:
            forbidden_actions = []

        # Input type checking
        if type(n_samples) is not int:
            raise TypeError('n_samples must be an integer.')
        if n_samples < 1:
            raise ValueError('n_samples must be greater than 0.')
        if type(forbidden_actions) is not list:
            raise TypeError('forbidden_actions must be a list of strings.')
        if not all(a in self._actions_ids for a in forbidden_actions):
            raise ValueError('forbidden_actions contains invalid action_ids.')
        if len(forbidden_actions) != len(set(forbidden_actions)):
            raise ValueError('forbidden_actions cannot contains duplicates.')
        if set(forbidden_actions) == set(self._actions_ids):
            raise ValueError('All actions are forbidden. You must allow at least 1 action.')

        best_actions, probs = [], []
        for _ in range(n_samples):
            p = {t: random.betavariate(self._success_counters[t], self._failure_counters[t]) for t in self._actions_ids
                 if t not in set(forbidden_actions)}
            probs.append(p)
            best_actions.append(max(p, key=p.get))  # action with the highest probability

        return best_actions, probs

    def update(self, action_id, n_successes, n_failures):
        """
        This method updates the SMAB with feedbacks for a given action. The action's associated success (resp.
        failure) counter is incremented by the number of successes (resp. failures) received.

        Parameters
        ----------
        action_id: str
            The ID of the action to update.
        n_successes: int
            The number of successes received for action_id.
        n_failures: int
            The number of failures received for action_id.
        """
        if type(action_id) is not str or type(n_successes) is not int or type(n_failures) is not int:
            raise TypeError('action_id must be a string and n_successes/failures must be integers.')
        if (n_successes < 0) or (n_failures < 0):
            raise ValueError('The number of successes/failures must be >= 0')
        if action_id not in self._actions_ids:
            raise ValueError('Treatment', action_id, 'does not exist.')

        self._success_counters[action_id] += n_successes
        self._failure_counters[action_id] += n_failures

    def batch_update(self, batch):
        """
        This method updates the SMAB for several action IDs at once, iterating over the batch.

        Parameters
        ----------
            batch: List[dict]
                List of dicts in the form [{'action_id': <str>, 'n_successes': <int>, 'n_failures':<int>}]
        """
        assert(type(batch) == list), "batch type must be a list"
        for elem in batch:
            assert(type(elem) == dict), "batch must contain dicts only"
            assert("action_id" in elem and "n_successes" in elem and "n_failures" in elem),\
                "Batch must be in the form [{'action_id': <str>, 'n_successes': <int>, 'n_failures':<int>}]"
            self.update(action_id=elem['action_id'], n_successes=elem['n_successes'], n_failures=elem['n_failures'])
