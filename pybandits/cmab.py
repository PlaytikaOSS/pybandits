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
from functools import partial
from multiprocessing import Pool
from pymc3 import Bernoulli, Data, Deterministic, Model, StudentT, fast_sample_posterior_predictive, sample, set_data
from pymc3.math import sigmoid
from scipy.stats import t
from theano.tensor import dot


def check_context_matrix(context, n_features):
    """
    Check context matrix

    Parameters
    ----------
    X: array_like of shape (n_samples, n_features)
        Matrix with contextual features.
    """
    try:
        context = pd.DataFrame(context).reset_index(drop=True)
    except ValueError as e:
        print("{}\nCannot convert input arguments to pandas DataFrame".format(e))
        raise
    if context.shape[1] != n_features:
        raise ValueError("context must have {} columns.".format(n_features))
    return context


class Cmab:
    """
    Contextual Multi-Armed Bandit with binary rewards. It is based on Thompson Sampling with bayesian logistic
    regression. It assumes a prior distribution over the parameters and it computes the posterior reward
    distributions applying Bayes'theorem via Markov Chain Monte Carlo simulation (MCMC).

    Parameters
    ----------
    n_features: int
        The number of contextual features.
    actions_ids : list of strings with length = n_actions
        List of actions names.
    params_sample: dict
        Sampling parameters for pm.sample function from pymc3.
    n_jobs: int
        The number of jobs to run in parallel. If n_jobs > 1, both the update() and predict() functions will be run with
        parallelization via the multiprocessing package.
    mu_alpha: dict
        Mu (location) parameters for alpha prior Student's t distribution. By default all mu=0.
        The keys of the dict must be the actions_ids, and the values are floats.
        e.g. mu={'action1': 0., 'action2': 0.} with n_actions=2.
    mu_betas: dict
        Mu (location) parameters for betas prior Student's t distributions. By default all mu=0.
        The keys of the dict must be the actions_ids, and the values are lists of floats with length=n_features.
        e.g. mu={'action1': [0., 0., 0.], 'action2': [0., 0., 0.]} with n_actions=2 and n_features=3.
    sigma_alpha: dict
        Sigma (scale) parameters for alpha prior Student's t distribution. By default all sigma=10.
        The keys of the dict must be the actions_ids, and the values are floats.
        e.g. sigma={'action1': 10., 'action2': 10.} with n_actions=2.
    sigma_betas: dict
        Sigma (scale) parameters for betas prior Student's t distributions. By default all sigma=10.
        The keys of the dict must be the actions_ids, and the values are lists of floats with length=n_features.
        e.g. sigma={'action1': [10., 10., 10.], 'action2': [10., 10., 10.]} with n_actions=2 and n_features=3.
    nu_alpha: dict
        Nu (normality) parameters for alpha prior Student's t distribution. By default all nu=5.
        The keys of the dict must be the actions_ids, and the values are floats.
        e.g. nu={'action1': 5., 'action2': 5.} with n_actions=2.
    nu_betas: dict
        Nu (normality) parameters for betas prior Student's t distributions. By default all nu=5.
        The keys of the dict must be the actions_ids, and the values are lists of floats with length=n_features.
        e.g. nu={'action1': [5., 5., 5.], 'action2': [5., 5., 5.]} with n_actions=2 and n_features=3.
    random_seed: int
        Seed for random state. If specified, the model outputs deterministic results.
    """

    def __init__(
        self,
        n_features,
        actions_ids,
        params_sample=None,
        n_jobs=1,
        mu_alpha=None,
        mu_betas=None,
        sigma_alpha=None,
        sigma_betas=None,
        nu_alpha=None,
        nu_betas=None,
        random_seed=None,
    ):
        """Initialization."""

        # set default params_sample if not specified
        if params_sample is None:
            params_sample = {
                "tune": 500,
                "draws": 1000,
                "chains": 2,
                "init": "adapt_diag",
                "cores": 1,
                "target_accept": 0.95,
                "progressbar": False,
                "return_inferencedata": False,
            }
        # check input
        if type(n_features) is not int or type(n_jobs) is not int:
            raise TypeError("n_features, n_jobs must be integers.")
        if n_features <= 0 or n_jobs <= 0:
            raise ValueError("n_features, n_jobs must be > 0")
        if type(params_sample) is not dict:
            raise TypeError("params_sample must be a dictionary.")
        if n_jobs > 1 and params_sample["cores"] > 1:
            raise ValueError("n_jobs and cores cannot be both > 1.")
        if type(actions_ids) is not list or not all(isinstance(x, str) for x in actions_ids):
            raise TypeError("actions_ids must be a list of strings.")
        if len(actions_ids) < 1 or len(actions_ids) != len(set(actions_ids)):
            raise ValueError("actions_ids must be a non empty list without duplicates.")
        if type(random_seed) is not int and random_seed is not None:
            raise TypeError("random_seed must be an integer")

        self._models = {}  # dictionary of pymc3 models per each action: keys=actions_ids, values=pymc3.Model()
        self._actions_ids = actions_ids
        self._n_features = n_features
        self._params_sample = params_sample
        self._params_sample["random_seed"] = random_seed
        self._n_jobs = n_jobs
        self._mu_alpha = self._init_alpha(mu_alpha, 0.0)
        self._mu_betas = self._init_betas(mu_betas, 0.0)
        self._sigma_alpha = self._init_alpha(sigma_alpha, 10.0)
        self._sigma_betas = self._init_betas(sigma_betas, 10.0)
        self._nu_alpha = self._init_alpha(nu_alpha, 5.0)
        self._nu_betas = self._init_betas(nu_betas, 5.0)
        self._random_seed = random_seed
        self._traces = {a: None for a in actions_ids}

    def _init_alpha(self, prior, default):
        # set default prior
        if prior is None:
            prior = {a: default for a in self._actions_ids}
        if (
            type(prior) is not dict
            or not all(isinstance(x, str) for x in prior.keys())
            or not all(isinstance(x, float) for x in prior.values())
        ):
            raise TypeError("prior must be a dict with string as keys and float as values. prior =", prior)
        if set(prior.keys()) != set(self._actions_ids):
            raise ValueError("prior dict keys must be equal to the actions_ids. prior =", prior)
        return prior

    def _init_betas(self, prior, default):
        # set default prior
        if prior is None:
            prior = {k: v for (k, v) in zip(self._actions_ids, len(self._actions_ids) * [self._n_features * [default]])}
        if (
            type(prior) is not dict
            or not all(isinstance(x, str) for x in prior.keys())
            or not all(isinstance(x, list) for x in prior.values())
        ):
            raise TypeError("prior must be a dict with string as keys and lists as values. prior =", prior)
        if not all([item for list in [[isinstance(x, float) for x in v] for v in prior.values()] for item in list]):
            raise TypeError("prior dict values must be lists of float. prior =", prior)
        if set(prior.keys()) != set(self._actions_ids):
            raise ValueError("prior dict keys must be equal to the actions_ids. prior =", prior)
        if not np.all(np.array([len(x) for x in list(prior.values())]) == self._n_features):
            raise ValueError("prior values must be lists of length = n_features. prior =", prior)
        return prior

    def _update_trace(self, alpha, betas, X, rewards):
        """
        Compute the likelihood based on a logistic regression and update the trace.

        Parameters
        ----------
        alpha: TensorVariable
            intercept
        betas: TensorVariable
            coefficients
        X: pandas DataFrame of shape (n_samples, n_features)
            Matrix with contextual features.
        rewards: pandas Series of shape (n_samples,)
            Array of boolean rewards (0 or 1) per each sample.

        Returns
        -------
        trace : pymc3.MultiTrace
            New trace for hyper-parameters
        """
        X = Data("X", X)
        rewards = Data("rewards", rewards)

        # Likelihood (sampling distribution) of observations
        linear_combination = Deterministic("linear_combination", alpha + dot(betas, X.T))
        p = Deterministic("p", sigmoid(linear_combination))

        # Bernoulli random vector with probability of success given by sigmoid function and actual data as observed
        _ = Bernoulli("likelihood", p=p, observed=rewards)

        trace = sample(**self._params_sample)
        return trace

    def _worker_update(self, X, actions, rewards, a):
        """
        Update core function to run in parallel. Given (X, actions, rewards), it updates the model of each action a.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.
        actions : array_like of shape (n_samples,)
            Array of recommended actions per each sample.
        rewards: array_like of shape (n_samples,)
            Array of boolean rewards (0 or 1) per each sample.
        a: string
            Action to consider for the update

        Returns
        -------
        a: string
            Action to consider for the update
        trace : pymc3.MultiTrace
            Trace for the model of the action a
        """
        with Model() as model:
            # update intercept (alpha) and coefficients (betas)
            # if model was never updated priors_parameters = default arguments
            # else priors_parameters are calculated from traces of the previous update
            alpha = StudentT("alpha", mu=self._mu_alpha[a], sigma=self._sigma_alpha[a], nu=self._nu_alpha[a])
            betas = [
                StudentT(
                    "beta" + str(j), mu=self._mu_betas[a][j], sigma=self._sigma_betas[a][j], nu=self._nu_betas[a][j]
                )
                for j in range(len(X.columns))
            ]

            # update traces
            trace = self._update_trace(alpha, betas, X[actions == a], rewards[actions == a])

            return a, trace, model

    def update(self, X, actions, rewards):
        """
        Update internal state of the models. Compute posterior distributions using new data set (actions, rewards and
        context). If n_jobs > 1, the models of each actions will be updated in parallel.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.
        actions : array_like of shape (n_samples,)
            Array of recommended actions per each sample.
        rewards: array_like of shape (n_samples,)
            Array of boolean rewards (0 or 1) per each sample.
        """
        # check input
        try:
            X = pd.DataFrame(X).reset_index(drop=True)
            actions = pd.Series(actions).reset_index(drop=True)
            rewards = pd.Series(rewards).reset_index(drop=True)
        except ValueError as e:
            print("{}\nCannot convert input arguments to pandas DataFrame/Series".format(e))
            raise
        if not (len(X) == len(actions) == len(rewards)):
            raise ValueError("X, actions and rewards must have the same number of rows.")
        if X.shape[1] != self._n_features:
            raise ValueError("X must have {} columns.".format(self._n_features))
        if not set(actions).issubset(set(self._actions_ids)):
            raise ValueError("Invalid actions. Only actions in {} are allowed.".format(self._actions_ids))
        if not set(rewards).issubset({0, 1}):
            raise ValueError("Invalid rewards. Only rewards in {0, 1} are allowed")

        # if n_jobs = 1 then update the model sequentially, else update model in parallel with multiprocessing
        if self._n_jobs == 1:
            new_models = [self._worker_update(X, actions, rewards, a) for a in set(actions)]
        else:
            # create a pool object
            p = Pool(processes=self._n_jobs)

            # update pymc3 models
            new_models = p.map(partial(self._worker_update, X, actions, rewards), set(actions))

        for a in set(actions):
            # store traces of each actions
            self._traces[a], self._models[a] = next(
                ((trace, model) for (action, trace, model) in new_models if action == a), None
            )

            # compute mean and std of the coefficients distributions
            self._mu_alpha[a] = np.mean(self._traces[a]["alpha"])
            self._mu_betas[a] = [np.mean(self._traces[a]["beta" + str(j)]) for j in range(len(X.columns))]
            self._sigma_alpha[a] = np.std(self._traces[a]["alpha"], ddof=1)
            self._sigma_betas[a] = [np.std(self._traces[a]["beta" + str(j)], ddof=1) for j in range(len(X.columns))]

    def _worker_predict(self, X, a):
        """
        Predict core function to run in parallel. For each sample in X, it computes the probability to get a positive
        reward if action a is recommended.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.
        a: string
            Action to consider for the prediction

        Returns
        -------
        list of len (n_samples)
            List of probabilities (per each sample) to get a positive rewards given action a.
        """
        with self._models[a]:
            # update context information
            set_data({"X": X})

            # use the updated values and compute posterior predictive samples
            pps = fast_sample_posterior_predictive(
                trace=self._traces[a], random_seed=self._random_seed, var_names=["linear_combination"], samples=1
            )

            # compute the linear combination for each sample
            return pps["linear_combination"][0]

    def _predict_with_sampling(self, X):
        """
        Predict via sampling. It can be run only after the first update.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.

        Returns
        -------
        best_actions: list of len (n_samples)
            Best action per each sample, i.e. action with the highest probability to get a positive reward.
        probs: array_like of shape (n_actions, n_samples)
            Probability to get a positive reward per each action-sample
        """
        # create a pool object
        p = Pool(processes=self._n_jobs)

        # map list to target function
        z = p.map(partial(self._worker_predict, X), self._actions_ids)

        # find index of the best actions
        idx_max_prob = np.argmax(z, axis=0)

        # compute the best action per each sample
        best_actions = [self._actions_ids[i] for i in idx_max_prob]

        # compute the probability to get a positive reward as sigmoid function
        probs = 1.0 / (1.0 + np.exp(-np.array(z)))

        return best_actions, probs

    def _predict_random_actions(self, X):
        """
        Random recommendation of actions

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.

        Returns
        -------
        best_actions: list of len (n_samples)
            Best action per each sample, i.e. action with the highest probability to get a positive reward.
        probs: array_like of shape (n_actions, n_samples)
            Probability to get a positive reward per each action-sample
        """
        rng = np.random.default_rng(self._random_seed)
        return rng.choice(self._actions_ids, size=len(X)), np.full((len(self._actions_ids), len(X)), 0.5)

    def predict(self, X):
        """
        Generate posterior predictive probability from a model given the trace and the context X. It returns the action
        with the highest probability. If n_jobs > 1, the prediction for each model action will be run in parallel.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.

        Returns
        -------
        best_actions: list of len (n_samples)
            Best action per each sample, i.e. action with the highest probability to get a positive reward.
        probs: array_like of shape (n_actions, n_samples)
            Reward probability for each action-sample pair
        """
        # check input
        _X = check_context_matrix(X, self._n_features)

        # recommended actions at random if the model was never updated
        if not any(self._traces.values()):
            return self._predict_random_actions(_X)

        # if X has only 1 row add a dummy row with zeros
        # (due to known pymc3 bug in fast_sample_posterior_predictive)
        if len(X) == 1:
            _X.loc[len(_X)] = 0
            best_actions, probs = self._predict_with_sampling(_X)
            return best_actions[:-1], [p[:-1] for p in probs]  # discard dummy row
        return self._predict_with_sampling(_X)

    def fast_predict(self, X):
        """
        Compute the posterior reward probability for all actions sampling coefficients from student-t distribution
        as a real time faster alternative to fast_sample_posterior_predictive. It returns the action with the highest
        probability.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            Matrix with contextual features.

        Returns
        -------
        best_actions: list of len (n_samples)
            Best action per each sample, i.e. action with the highest probability to get a positive reward.
        probs: array_like of shape (n_actions, n_samples)
            Reward probability for each action-sample pair
        """
        # check input
        _X = check_context_matrix(X, self._n_features)

        # recommended actions at random if the model was never updated
        if not any(self._traces.values()):
            return self._predict_random_actions(_X)

        # add column of ones for the dot product with the intercept
        _X = np.c_[np.ones((_X.shape[0], 1)), _X]

        # compute probabilities via logistic regression
        probs = []
        weighted_sum = []

        for a in self._actions_ids:
            # sample coefficients only once per each sample from student-t distributions
            alpha = t.rvs(
                df=self._nu_alpha[a],
                loc=self._mu_alpha[a],
                scale=self._sigma_alpha[a],
                size=len(_X),
                random_state=self._random_seed,
            )
            betas = np.array(
                [
                    t.rvs(
                        df=self._nu_betas[a][b],
                        loc=self._mu_betas[a][b],
                        scale=self._sigma_betas[a][b],
                        size=len(_X),
                        random_state=self._random_seed,
                    )
                    for b in range(self._n_features)
                ]
            )

            # create coefficients matrix
            coeff = np.insert(arr=betas, obj=0, values=alpha, axis=0)

            # extract the weighted sum between X and coefficients
            ws = np.multiply(_X, coeff.T).sum(axis=1)
            weighted_sum.append(ws)

            # compute the probability with the sigmoid function
            probs.append(1.0 / (1.0 + np.exp(-ws)))

        # the max is computed on the weighted sum instead of sigmoid transformation because of 0 and 1 boundary values
        # whatever z above or below a given threshold.
        idx_max_prob = np.argmax(weighted_sum, axis=0)
        best_actions = [self._actions_ids[i] for i in idx_max_prob]

        return best_actions, probs
