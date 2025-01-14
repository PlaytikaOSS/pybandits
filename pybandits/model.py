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

class Quantitative_Cmab(Model):
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
    actions_to_polynom_order_dict: Dict[str, PositiveInt]
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
        if len(self.actions_to_polynom_order_dict) > 0:
            num_action_params = sum(self.actions_to_polynom_order_dict.values())
        else:
            num_action_params = 0

        try:
            n_cols_context = array(context).shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {len(self.betas) - num_action_params} columns: {e}.")



        if (n_cols_context + num_action_params) != len(self.betas):
            raise AttributeError(f"Shape mismatch: context must have {len(self.betas) - num_action_params} columns.")

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

        # set number of parameters
        self.actions_to_polynom_order_dict


        alpha = t.rvs(df=self.alpha.nu, loc=self.alpha.mu, scale=self.alpha.sigma, size=len(context_ext))
        betas = array(
            [
                t.rvs(df=self.betas[i].nu, loc=self.betas[i].mu, scale=self.betas[i].sigma, size=len(context_ext))
                for i in range(len(self.betas))
            ]
        )

        # optimize actions polynomials
        import sympy as sp
        num_action_quant_actions = len(self.actions_to_polynom_order_dict)
        quant_values = sp.symbols(f'x_0:{num_action_quant_actions}')
        num_actions_params = sum(self.actions_to_polynom_order_dict.values())
        action_betas = betas[:,0:num_actions_params]

        polynomials = 0
        for i, (_, polynom_order) in enumerate(self.actions_to_polynom_order_dict.items()):
            for j in range(polynom_order):
                polynomials += action_betas[:,i + j] * quant_values[i]**(j + 1)

        # calculate the gradient of the polynomial
        actions_list = []
        for polynomial in polynomials:
            gradient = [sp.diff(polynomial, x) for x in quant_values]
            critical_points = sp.solve(gradient, quant_values)

            # find maximum value:
            if len(critical_points) == 0:
                max_critical_point = None
            else:
                max_value = -np.inf
                for critical_point in critical_points:
                    values_dict = {x: critical_point[i] for i,x in enumerate(quant_values)}
                    print(polynomial.subs(values_dict))
                    if polynomial.subs(values_dict) > max_value:
                        max_critical_point = critical_point

            actions_list.append(max_critical_point)




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
        n_params: PositiveInt,
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
        n_params : PositiveInt
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
            betas=[StudentT() for _ in range(n_params)],
            update_method=update_method,
            update_kwargs=update_kwargs,
            **kwargs,
        )

if __name__ == '__main__':
    n_samples = 10
    n_features = 5
    actions_to_polynom_order_dict = {'a': 2, 'b': 3}
    num_actions_params = sum(actions_to_polynom_order_dict.values())


    blr = Quantitative_Cmab.cold_start(n_params=n_features + num_actions_params,  actions_to_polynom_order_dict=actions_to_polynom_order_dict)

    # context is numpy array
    context = np.random.uniform(low=-100.0, high=100.0, size=(n_samples, n_features))
    assert type(context) is np.ndarray
    prob, weighted_sum = blr.sample_proba(context=context)
