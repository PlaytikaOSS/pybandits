# MIT License
#
# Copyright (c) 2023 Playtika Ltd.
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
import warnings
from abc import ABC, abstractmethod
from random import betavariate
from typing import Any, List, Literal, Optional, Tuple, Union, Dict

import numpy as np
import pymc.math as pmath
from numpy import array, c_, insert, mean, multiply, ones, sqrt, std
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, fit, sample
from pymc import Model as PymcModel
from pymc import StudentT as PymcStudentT
import pymc as pm
from pytensor.tensor import TensorVariable, dot
from scipy.stats import t

from pybandits.base import BinaryReward, Probability, PyBanditsBaseModel
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    Field,
    NonNegativeFloat,
    PositiveInt,
    confloat,
    model_validator,
    pydantic_version,
    validate_call,
)

UpdateMethods = Literal["MCMC", "VI"]


class Model(PyBanditsBaseModel, ABC):
    """
    Class to model the prior distributions.
    """

    @abstractmethod
    def sample_proba(self) -> Probability:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: List[Any]):
        """
        Update the model parameters.
        """


class BaseBeta(Model):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """

    n_successes: PositiveInt = 1
    n_failures: PositiveInt = 1

    @model_validator(mode="before")
    @classmethod
    def both_or_neither_counters_are_defined(cls, values):
        if hasattr(values, "n_successes") != hasattr(values, "n_failures"):
            raise ValueError("Either both or neither n_successes and n_failures should be specified.")
        return values

    @property
    def count(self) -> int:
        """
        The total amount of successes and failures collected.
        """
        return self.n_successes + self.n_failures

    @property
    def mean(self) -> float:
        """
        The success rate i.e. n_successes / (n_successes + n_failures).
        """
        return self.n_successes / self.count

    @property
    def std(self) -> float:
        """
        The corrected standard deviation (Bessel's correction) of the binary distribution of successes and failures.
        """
        return sqrt((self.n_successes * self.n_failures) / (self.count * (self.count - 1)))

    @validate_call
    def update(self, rewards: List[BinaryReward]):
        """
        Update n_successes and and n_failures.

        Parameters
        ----------
        rewards: List[BinaryReward]
            A list of binary rewards.
        """
        self.n_successes += sum(rewards)
        self.n_failures += len(rewards) - sum(rewards)

    def sample_proba(self) -> Probability:
        """
        Sample the probability of getting a positive reward.

        Returns
        -------
        prob: Probability
            Probability of getting a positive reward.
        """
        return betavariate(self.n_successes, self.n_failures)  # type: ignore


class Beta(BaseBeta):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.
    """


class BetaCC(BaseBeta):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with cost control.

    Parameters
    ----------
    cost: NonNegativeFloat
        Cost associated to the Beta distribution.
    """

    cost: NonNegativeFloat


class BetaMO(Model):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    counters: List[Beta] of shape (n_objectives,)
        List of Beta distributions.
    """

    counters: List[Beta]

    @validate_call
    def sample_proba(self) -> List[Probability]:
        """
        Sample the probability of getting a positive reward.

        Returns
        -------
        prob: List[Probability]
            Probabilities of getting a positive reward for each objective.
        """
        return [x.sample_proba() for x in self.counters]

    @validate_call
    def update(self, rewards: List[List[BinaryReward]]):
        """
        Update the Beta model using the provided rewards.

        Parameters
        ----------
        rewards: List[List[BinaryReward]]
            A list of rewards, where each reward is in turn a list containing the reward of the Beta model
            associated to each objective.
            For example, `[[1, 1], [1, 0], [1, 1], [1, 0], [1, 1]]`.
        """
        if any(len(x) != len(self.counters) for x in rewards):
            raise AttributeError("The shape of rewards is incorrect")

        for i, counter in enumerate(self.counters):
            counter.update([r[i] for r in rewards])

    @classmethod
    def cold_start(cls, n_objectives: PositiveInt, **kwargs) -> "BetaMO":
        """
        Utility function to create a Bayesian Logistic Regression model  or child model with cost control,
        with default parameters.

        It is modeled as:

            y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

        where the alpha and betas coefficients are Student's t-distributions.

        Parameters
        ----------
        n_betas : PositiveInt
            The number of betas of the Bayesian Logistic Regression model. This is also the number of features expected
            after in the context matrix.
        kwargs: Dict[str, Any]
            Additional arguments for the Bayesian Logistic Regression child model.

        Returns
        -------
        blr: BayesianLogisticRegrssion
            The Bayesian Logistic Regression model.
        """
        counters = n_objectives * [Beta()]
        blr = cls(counters=counters, **kwargs)
        return blr


class BetaMOCC(BetaMO):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives and cost control.

    Parameters
    ----------
    counters: List[BetaCC] of shape (n_objectives,)
        List of Beta distributions.
    cost: NonNegativeFloat
        Cost associated to the Beta distribution.
    """

    cost: NonNegativeFloat


class StudentT(PyBanditsBaseModel):
    """
    Student's t-distribution.

    Parameters
    ----------
    mu: float
        Mean of the Student's t-distribution.
    sigma: float
        Standard deviation of the Student's t-distribution.
    nu: float
        Degrees of freedom.
    """

    mu: confloat(allow_inf_nan=False) = 0.0
    sigma: confloat(allow_inf_nan=False) = 10.0
    nu: confloat(allow_inf_nan=False) = 5.0


class BayesianLogisticRegression(Model):
    """
    Base Bayesian Logistic Regression model.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    alpha : StudentT
        Student's t-distribution of the alpha coefficient.
    betas : StudentT
        Student's t-distributions of the betas coefficients.
    update_method : UpdateMethods, defaults to "MCMC"
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method.
    """

    alpha: StudentT
    if pydantic_version == PYDANTIC_VERSION_1:
        betas: List[StudentT] = Field(..., min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        betas: List[StudentT] = Field(..., min_length=1)
    else:
        raise ValueError("Invalid version.")
    update_method: UpdateMethods = "MCMC"
    update_kwargs: Optional[dict] = None
    _default_update_kwargs = dict(draws=1000, progressbar=False, return_inferencedata=False)
    _default_mcmc_kwargs = dict(
        tune=500,
        draws=1000,
        chains=2,
        init="adapt_diag",
        cores=1,
        target_accept=0.95,
        progressbar=False,
        return_inferencedata=False,
    )
    _default_variational_inference_kwargs = dict(method="advi")

    if pydantic_version == PYDANTIC_VERSION_1:

        @model_validator(mode="before")
        @classmethod
        def arrange_update_kwargs(cls, values):
            update_kwargs = cls._get_value_with_default("update_kwargs", values)
            update_method = cls._get_value_with_default("update_method", values)
            if update_kwargs is None:
                update_kwargs = cls._default_update_kwargs
            if update_method == "VI":
                update_kwargs = {**cls._default_variational_inference_kwargs, **update_kwargs}
            elif update_method == "MCMC":
                update_kwargs = {**cls._default_mcmc_kwargs, **update_kwargs}
            else:
                raise ValueError("Invalid update method.")
            values["update_kwargs"] = update_kwargs
            values["update_method"] = update_method
            return values

    elif pydantic_version == PYDANTIC_VERSION_2:

        @model_validator(mode="after")
        def arrange_update_kwargs(self):
            if self.update_kwargs is None:
                self.update_kwargs = self._default_update_kwargs
            if self.update_method == "VI":
                self.update_kwargs = {**self._default_variational_inference_kwargs, **self.update_kwargs}
            elif self.update_method == "MCMC":
                self.update_kwargs = {**self._default_mcmc_kwargs, **self.update_kwargs}
            else:
                raise ValueError("Invalid update method.")
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @classmethod
    def _stable_sigmoid(cls, x: Union[np.ndarray, TensorVariable]) -> Union[np.ndarray, TensorVariable]:
        """
        Vectorized sigmoid function that avoids overflow and underflow.
        Compatible with both numpy and PyMC3 tensors.

        Parameters
        ----------
        x : Union[np.ndarray, TensorVariable]
            Input values.

        Returns
        -------
        prob : Union[np.ndarray, TensorVariable]
            Sigmoid function applied to the input values.
        """
        backend = np if isinstance(x, np.ndarray) else pmath
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prob = backend.where(x >= 0, 1 / (1 + backend.exp(-x)), backend.exp(x) / (1 + backend.exp(x)))
        return prob

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def check_context_matrix(self, context: ArrayLike):
        """
        Check and cast context matrix.

        Parameters
        ----------
        context : ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.

        Returns
        -------
        context : pandas DataFrame of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        try:
            n_cols_context = array(context).shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {len(self.betas)} columns: {e}.")
        if n_cols_context != len(self.betas):
            raise AttributeError(f"Shape mismatch: context must have {len(self.betas)} columns.")

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: ArrayLike) -> Tuple[Probability, float]:
        """
        Compute the probability of getting a positive reward from the sampled regression coefficients and the context.

        Parameters
        ----------
        context : ArrayLike
            Context matrix of shape (n_samples, n_features).

        Returns
        -------
        prob: ndarray of shape (n_samples)
            Probability of getting a positive reward.
        weighted_sum: ndarray of shape (n_samples)
            Weighted sums between contextual feature values and sampled coefficients.
        """

        # check input args
        self.check_context_matrix(context=context)

        # extend context with a column of 1 to handle the dot product with the intercept
        context_ext = c_[ones((len(context), 1)), context]

        # sample alpha and beta coefficient values from student-t distributions once for each sample
        alpha = t.rvs(df=self.alpha.nu, loc=self.alpha.mu, scale=self.alpha.sigma, size=len(context_ext))
        betas = array(
            [
                t.rvs(df=self.betas[i].nu, loc=self.betas[i].mu, scale=self.betas[i].sigma, size=len(context_ext))
                for i in range(len(self.betas))
            ]
        )

        # create coefficients matrix
        coeff = insert(arr=betas, obj=0, values=alpha, axis=0)

        # extract the weighted sum between the context and the coefficients
        weighted_sum = multiply(context_ext, coeff.T).sum(axis=1)

        # compute the probability with the sigmoid function
        prob = self._stable_sigmoid(weighted_sum)

        return prob, weighted_sum

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(self, context: ArrayLike, rewards: List[BinaryReward]):
        """
        Update the model parameters.

        Parameters
        ----------
        context : ArrayLike
            Context matrix of shape (n_samples, n_features).
        rewards: List[BinaryReward]
            A list of binary rewards.
        """

        # check input args
        self.check_context_matrix(context=context)
        if len(context) != len(rewards):
            AttributeError("Shape mismatch: context and rewards must have the same length.")

        with PymcModel() as _:
            # update intercept (alpha) and coefficients (betas)
            # if model was never updated priors_parameters = default arguments
            # else priors_parameters are calculated from traces of the previous update
            alpha = PymcStudentT("alpha", mu=self.alpha.mu, sigma=self.alpha.sigma, nu=self.alpha.nu)
            beta_mu = [b.mu for b in self.betas]
            beta_sigma = [b.sigma for b in self.betas]
            beta_nu = [b.nu for b in self.betas]
            betas = PymcStudentT("betas", mu=beta_mu, sigma=beta_sigma, nu=beta_nu, shape=len(self.betas))

            context = Data("context", context, mutable=False)
            rewards = Data("rewards", rewards, mutable=False)

            # Likelihood (sampling distribution) of observations
            weighted_sum = Deterministic("weighted_sum", alpha + dot(betas, context.T))
            p = Deterministic("p", self._stable_sigmoid(weighted_sum))

            # Bernoulli random vector with probability of success given by sigmoid function and actual data as observed
            _ = Bernoulli("likelihood", p=p, observed=rewards)

            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                update_kwargs = self.update_kwargs.copy()
                approx = fit(method=update_kwargs.pop("method"))
                trace = approx.sample(**update_kwargs)
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs)
            else:
                raise ValueError("Invalid update method.")

            # compute mean and std of the coefficients distributions
            self.alpha.mu = mean(trace["alpha"])
            self.alpha.sigma = std(trace["alpha"], ddof=1)
            betas_mu = mean(trace["betas"], axis=0)
            betas_std = std(trace["betas"], axis=0, ddof=1)
            self.betas = [
                StudentT(mu=mu, sigma=sigma, nu=beta.nu) for mu, sigma, beta in zip(betas_mu, betas_std, self.betas)
            ]

    @classmethod
    def cold_start(
            cls,
            n_features: PositiveInt,
            update_method: UpdateMethods = "MCMC",
            update_kwargs: Optional[dict] = None,
            **kwargs,
    ) -> "BayesianLogisticRegression":
        """
        Utility function to create a Bayesian Logistic Regression model  or child model with cost control,
        with default parameters.

        It is modeled as:

            y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

        where the alpha and betas coefficients are Student's t-distributions.

        Parameters
        ----------
        n_features : PositiveInt
            The number of betas of the Bayesian Logistic Regression model. This is also the number of features expected
            after in the context matrix.
        update_method : UpdateMethods, defaults to "MCMC"
            The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
            chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
            full list.
        update_kwargs : Optional[dict], uses default values if not specified
            Additional arguments to pass to the update method.
        kwargs: Dict[str, Any]
            Additional arguments for the Bayesian Logistic Regression child model.

        Returns
        -------
        blr: BayesianLogisticRegrssion
            The Bayesian Logistic Regression model.
        """
        return cls(
            alpha=StudentT(),
            betas=[StudentT() for _ in range(n_features)],
            update_method=update_method,
            update_kwargs=update_kwargs,
            **kwargs,
        )


class BayesianLogisticRegressionCC(BayesianLogisticRegression):
    """
    Bayesian Logistic Regression model with cost control.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    alpha: StudentT
        Student's t-distribution of the alpha coefficient.
    betas: StudentT
        Student's t-distributions of the betas coefficients.
    update_method : UpdateMethods, defaults to "MCMC"
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method.
    cost: NonNegativeFloat
        Cost associated to the Bayesian Logistic Regression model.
    """

    cost: NonNegativeFloat


class BayesianModel:
    def __init__(self, in_dim, hid_dim, mu=0, sigma=10.):
        self.trace = None
        self.approx = None

        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.init_params(mu, sigma)
        self.init_model()

    def init_params(self, mu, sigma, seed=42):
        # init params
        rnd = np.random.default_rng(seed)

        # Initialize random weights between each layer
        init_w1 = mu + sigma * rnd.standard_normal(size=(self.in_dim, self.hid_dim))
        init_b1 = mu + sigma * rnd.standard_normal(size=self.hid_dim)
        init_w2 = mu + sigma * rnd.standard_normal(size=self.hid_dim)
        init_b2 = mu + sigma * rnd.standard_normal(size=1)

        self.init_dict = {
            "w1": dict(initval=init_w1, shape=(self.in_dim, self.hid_dim)),
            "b1": dict(initval=init_b1, shape=(self.hid_dim)),
            "w2": dict(initval=init_w2, shape=(self.hid_dim)),
            "b2": dict(initval=init_b2, shape=(1))
        }

        self.posterior_params = {name: dict(mu=mu, sigma=sigma) for name in self.init_dict.keys()}

    def init_model(self):

        X = np.zeros((1, self.in_dim))  # dummy data
        y = np.zeros(1)  # dummy data

        params_dict = {}
        with pm.Model() as self.model:
            # Define data variables using minibatches
            ann_input = pm.MutableData("ann_input", X)
            ann_output = pm.MutableData("ann_output", y)

            for name in self.init_dict.keys():
                params_dict[name] = pm.Normal(name=name, **self.posterior_params[name], **self.init_dict[name])

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, params_dict["w1"]) + params_dict["b1"])
            # act_out = pm.Deterministic("act_out",pm.math.sigmoid(pm.math.dot(act_1, weights_out) + bias_out))
            act_out = pm.Deterministic("act_out", pm.math.dot(act_1, params_dict["w2"]) + params_dict["b2"])
            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                logit_p=act_out,
                observed=ann_output
            )

    def mcmc_sample(self, draws=1000, tune=500):
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune)

    def update_variational(self, x_train, y_train, num_iter=10000, learning_rate=0.01):
        with self.model:
            pm.set_data(new_data={"ann_input": x_train, "ann_output": y_train})
            self.approx = pm.fit(n=num_iter, method='advi', obj_optimizer=pm.adam(learning_rate=learning_rate))

        trace = bayesian_model.approx.sample(draws=500)

        for name in self.posterior_params.keys():
            self.posterior_params[name]["mu"] = np.mean(trace['posterior'][name].squeeze(), axis=0)
            self.posterior_params[name]["sigma"] = np.std(trace['posterior'][name].squeeze(), axis=0)

    def vi_sample(self, x=None, draws=100):
        if self.approx is not None:
            if x is None:
                # create a dummy x:
                x = np.zeros((1, self.in_dim))

            y = np.zeros(len(x))
            with self.model:
                pm.set_data(new_data={"ann_input": x, "ann_output": y})

            approx_sample = pm.sample_approx(approx=self.approx, draws=draws)
            return approx_sample

        else:
            print("No variational approximation available. Run the fit_variational method first.")
            return None

    def summary(self):
        if self.trace is not None:
            return pm.summary(self.trace)
        elif self.approx is not None:
            return self.approx.summary()
        else:
            print("No trace or variational approximation available. Run the sample or fit_variational method first.")

    def update_data(self, new_data):
        self.data = new_data
        self.model = pm.Model()  # Reset the model
        self.build_model()  # Rebuild the model with new data


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # %%
    def create_data(c_params, n_samples, n_features, n_bias_features):
        features_obs = np.random.uniform(0, 1, size=(n_samples, n_features))
        amount = np.random.uniform(0, 2, size=(n_samples))
        logit_bias = c_params[-1] + np.matmul(features_obs[:, 0:n_bias_features], c_params[0:n_bias_features])
        logit_quant = c_params[n_bias_features] * np.multiply(features_obs[:, n_bias_features], amount) - c_params[
            n_bias_features + 1] * np.multiply(features_obs[:, n_bias_features + 1], amount ** 2)
        probs_obs = sigmoid(logit_bias + logit_quant)
        y_obs = np.random.binomial(1, probs_obs)
        x_obs = np.hstack((features_obs, amount.reshape(-1, 1)))
        x = torch.from_numpy(x_obs).float()
        y = torch.from_numpy(y_obs).float()

        return x, y, probs_obs


    c_params = [-4, 2, 3, 3, 1, -3]
    n_bias_features = 3
    n_features = 5
    n_samples_train = 30000
    n_samples_val = 2000
    n_samples_test = 10000

    # train
    x_train, y_train, probs_obs_train = create_data(c_params, n_samples_train, n_features, n_bias_features)
    x_val, y_val, probs_obs_val = create_data(c_params, n_samples_train, n_features, n_bias_features)
    x_test, y_test, probs_obs_test = create_data(c_params, n_samples_test, n_features, n_bias_features)

    bayesian_model = BayesianModel(x_train.shape[1], 10)
    bayesian_model.update_variational(x_train, y_train, num_iter=100, learning_rate=0.01)

    # %%
    plt.figure()
    plt.plot(bayesian_model.approx.hist)
    plt.ylabel("ELBO")
    plt.xlabel("iteration 1 ");
    plt.show()

    bayesian_model.update_variational(x_train, y_train, num_iter=100, learning_rate=0.01)
    # %%
    plt.figure()
    plt.plot(bayesian_model.approx.hist)
    plt.ylabel("ELBO")
    plt.xlabel("iteration 2 ");
    plt.show()
