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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pymc.math as pmath
import pytensor.tensor as pt
from numpy import array, c_, insert, mean, multiply, ones, sqrt, std
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, MutableData, fit, math, sample, sample_prior_predictive
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
    root_validator,
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


class StudentTArray(PyBanditsBaseModel):
    """
    A model representing an array of Student's t-distributions.
    Attributes
    ----------
    shape : Optional[List[PositiveInt]]
        The shape of the arrays for the parameters. If not provided, `params_dict` must be specified.
    params_dict : Optional[Dict[str, Union[List[float], List[List[float]]]]]
        A dictionary containing the parameters 'mu', 'sigma', and 'nu'. If not provided, `shape` must be specified.
    mu : confloat(allow_inf_nan=False)
        The mean of the Student's t-distribution. Default is 0.0.
    sigma : confloat(allow_inf_nan=False)
        The scale (standard deviation) of the Student's t-distribution. Default is 10.0.
    nu : confloat(allow_inf_nan=False)
        The degrees of freedom of the Student's t-distribution. Default is 5.0.

    """

    shape: Optional[List[PositiveInt]] = None
    params_dict: Optional[Dict[str, Union[List[float], List[List[float]]]]] = None
    mu: confloat(allow_inf_nan=False) = 0.0
    sigma: confloat(allow_inf_nan=False) = 1.0
    nu: confloat(allow_inf_nan=False) = 20.0

    @root_validator(pre=False, skip_on_failure=False)
    def initialize_arrays(cls, values):
        if values.get("params_dict") is None:
            if values.get("shape") is None:
                raise ValueError("either legal 'shape' or 'params_dict' must be specified")

            shape = values.get("shape")
            values["params_dict"] = {}
            values["params_dict"]["mu"] = (np.zeros(shape) + values.get("mu")).tolist()
            values["params_dict"]["sigma"] = np.full(shape, values.get("sigma")).tolist()
            values["params_dict"]["nu"] = np.full(shape, values.get("nu")).tolist()
        else:
            mu = values["params_dict"].get("mu")
            sigma = values["params_dict"].get("sigma")
            nu = values["params_dict"].get("nu")

            if not (mu and sigma and nu):
                raise ValueError("params_dict must contain mu, sigma, and nu")
            if not (np.array(mu).shape == np.array(sigma).shape == np.array(nu).shape):
                raise ValueError("mu, sigma, and nu must have the same sizes")

        return values

    def __eq__(self, other):
        for key, value in self.params_dict.items():
            if not np.array_equal(value, other.params_dict[key]):
                return False
        return True

    class Config:
        arbitrary_types_allowed = True


class LegacyBayesianLogisticRegression(Model):
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
    ) -> "LegacyBayesianLogisticRegression":
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


class BayesianNeuralNetwork(Model):
    """Bayesian Neural Network model for binary classification.
    This class implements a Bayesian Neural Network using PyMC for binary classification tasks.
    It supports both MCMC and Variational Inference methods for posterior inference.

    Parameters
    ----------
    posterior_params : List[Dict[str, StudentTArray]]
        List of dictionaries containing weight and bias parameters for each layer.
        Each dictionary should have 'w' and 'b' keys with StudentTArray values.
    update_method : str, optional
        Method used for posterior inference, either "MCMC" or "VI" (default is "MCMC")
    update_kwargs : dict, optional
        Dictionary of keyword arguments for the update method.
        For MCMC: Contains 'trace' settings
        For VI: Contains both 'trace' and 'fit' settings

    Attributes
    ----------
    expected_input : int
        Number of input features expected by the model
    posterior_params : List[Dict[str, StudentTArray]]
        Current posterior parameters of the network
    Methods
    -------
    cold_start(dim_list, update_method="MCMC", update_kwargs=None, **kwargs)
        Create a new BNN instance with specified architecture
    sample_proba(context)
        Sample probabilities for given context
    update(context, rewards)
        Update model parameters using observed context-reward pairs
    create_posterior_params(dim_list)
        Create initial posterior parameters for network layers
    check_context_matrix(context)
        Validate input context matrix dimensions
    create_model(x, y, is_samplewise)
        Create PyMC model for inference

    Notes
    -----
    The network uses tanh activation for hidden layers and sigmoid for output layer.
    Supported PyMC sampling methods are MCMC (default) and Variational Inference (VI).

    Examples
    --------
    >>> dim_list = [5, 3, 1]  # 5 inputs, 1 hidden layer with 3 nodes, 1 output
    >>> bnn = BayesianNeuralNetwork.cold_start(dim_list)
    >>> probs, _ = bnn.sample_proba([[1,2,3,4,5]])
    >>> bnn.update([[1,2,3,4,5]], [1])
    """

    update_method: str = "MCMC"
    update_kwargs: Optional[Union[Dict[str, Any], dict[str, Dict[str, Any]]]] = None

    posterior_params: List[Dict[str, StudentTArray]]
    is_fitted: Optional[bool] = False

    _default_mcmc_trace_kwargs = dict(
        tune=500,
        draws=1000,
        chains=2,
        init="adapt_diag",
        cores=1,
        target_accept=0.95,
        progressbar=False,
        return_inferencedata=False,
    )

    _default_variational_inference_fit_kwargs = dict(method="advi")
    _default_variational_inference_trace_kwargs = dict(draws=1000, progressbar=False, return_inferencedata=False)
    class Config:
        arbitrary_types_allowed = True

    if pydantic_version == PYDANTIC_VERSION_1:

        @model_validator(mode="before")
        @classmethod
        def arrange_update_kwargs(cls, values):
            update_kwargs = cls._get_value_with_default("update_kwargs", values)
            update_method = cls._get_value_with_default("update_method", values)

            if update_kwargs is None:
                update_kwargs = dict()

            if update_method == "VI":
                update_kwargs["trace"] = {
                    **cls._default_variational_inference_trace_kwargs,
                    **update_kwargs.get("trace", {}),
                }
                update_kwargs["fit"] = {**cls._default_variational_inference_fit_kwargs, **update_kwargs.get("fit", {})}

            elif update_method == "MCMC":
                update_kwargs["trace"] = {**cls._default_mcmc_trace_kwargs, **update_kwargs.get("trace", {})}
            else:
                raise ValueError("Invalid update method.")

            values["update_kwargs"] = update_kwargs
            values["update_method"] = update_method
            return values

    elif pydantic_version == PYDANTIC_VERSION_2:

        @model_validator(mode="after")
        def arrange_update_kwargs(self):
            if self.update_kwargs is None:
                self.update_kwargs = dict()

            if self.update_method == "VI":
                self.update_kwargs["trace"] = {
                    **self._default_variational_inference_trace_kwargs,
                    **self.update_kwargs.get("trace", {}),
                }
                self.update_kwargs["fit"] = {
                    **self._default_variational_inference_fit_kwargs,
                    **self.update_kwargs.get("fit", {}),
                }

            elif self.update_method == "MCMC":
                self.update_kwargs["trace"] = {**self._default_mcmc_trace_kwargs, **self.update_kwargs.get("trace", {})}
            else:
                raise ValueError("Invalid update method.")
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @classmethod
    def create_posterior_params(cls, dim_list: List[PositiveInt]) -> List[Dict[str, StudentTArray]]:
        """Create a list of posterior parameters for a neural network.

        This classmethod creates posterior parameters for each layer in a neural network
        based on the provided dimensions. Each layer's parameters include weights (w) and
        biases (b) as StudentT distributions.

        Parameters
        ----------
        dim_list : List[PositiveInt]
            List of integers representing the dimensions of each layer in the neural
            network. The output dimension will be automatically appended as 1.

        Returns
        -------
        List[Dict[str, StudentTArray]]
            List of dictionaries containing posterior parameters for each layer.
            Each dictionary has two keys:
                - 'w': StudentTArray for weights with shape [input_dim, output_dim]
                - 'b': StudentTArray for biases with shape [output_dim]

        """
        _dim_list = dim_list.copy()
        _dim_list.append(1)

        posterior_params = []
        for layer_ind in range(len(_dim_list) - 1):
            input_dim = _dim_list[layer_ind]
            output_dim = _dim_list[layer_ind + 1]
            w_param = StudentTArray(shape=[input_dim, output_dim])
            b_param = StudentTArray(shape=[output_dim])
            posterior_params.append(dict(w=w_param, b=b_param))

        return posterior_params

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
            n_cols_context = np.array(context).shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {self.expected_input} columns: {e}.")
        if n_cols_context != self.expected_input:
            raise AttributeError(f"Shape mismatch: context must have {self.expected_input} columns.")

    @property
    def expected_input(self) -> PositiveInt:
        """
        Returns the expected input dimension of the model.

        Returns
        -------
        int
            The number of input features expected by the model, derived from
            the shape of the weight matrix in the first layer's posterior parameters.
        """
        return self.posterior_params[0]["w"].shape[0]

    def create_model(self, x: ArrayLike, y: Union[List[BinaryReward], np.ndarray], is_sampelwise: bool) -> PymcModel:
        """
        Create a PyMC model for Bayesian Neural Network.

        This method builds a PyMC model with the network architecture specified in posterior_params.
        The model uses tanh activation for hidden layers and sigmoid for the output layer.

        Parameters
        ----------
        x : ArrayLike
            Input features of shape (n_samples, n_features)
        y : Union[List[BinaryReward], np.ndarray]
            Binary target values of shape (n_samples,)
        is_sampelwise : bool
            If True, process samples independently. If False, process all samples at once.

        Returns
        -------
        PymcModel
            PyMC model object with the specified neural network architecture

        Notes
        -----
        The model structure follows these steps:
        1. For each layer, create weight and bias variables from StudentT distributions
        2. Apply linear transformations and activations through the layers.
           When is_sampelwise is True, the linear transformation is applied on each row separately (so random variables are not shared)
           When is_sampelwise is False, the linear transformation is applied on the whole matrix at once, so random variables are shared.
        3. Apply sigmoid activation at the output
        4. Use Bernoulli likelihood for binary classification
        """

        with PymcModel() as _model:
            # Define data variables using minibatches
            bnn_output = MutableData("bnn_output", y)
            bnn_input = MutableData("bnn_input", x)
            n_samples = len(x)

            next_layer_input = bnn_input
            for layer_ind in range(len(self.posterior_params)):
                layer_params = self.posterior_params[layer_ind]
                w_shape = np.array(layer_params["w"].params_dict["mu"]).shape  # without it n_features = 1 doesn't work
                b_shape = np.array(layer_params["b"].params_dict["mu"]).shape

                if self.is_fitted:
                    w_init_val = None
                else:
                    w_init_val = np.random.randn(*w_shape)
                    b_init_val = np.random.randn(*b_shape)

                if is_sampelwise:
                    w = PymcStudentT(
                        f"w{layer_ind}", **layer_params["w"].params_dict, shape=(n_samples,) + w_shape)
                      # this create a tensor of shape (n_samples, n_features, n_output)
                    b = PymcStudentT(f"b{layer_ind}", **layer_params["b"].params_dict, shape=(n_samples,) + b_shape)
                    linear_transform = pt.as_tensor_variable(
                        pt.batched_dot(next_layer_input, w) + b, name=f"linear_transform{layer_ind}"
                    )

                else:
                    w = PymcStudentT(f"w{layer_ind}", **layer_params["w"].params_dict, shape=w_shape,initval=w_init_val)
                    b = PymcStudentT(f"b{layer_ind}", **layer_params["b"].params_dict, shape=b_shape ,initval=b_init_val)
                    linear_transform = Deterministic(f"linear_transform{layer_ind}", math.dot(next_layer_input, w) + b)

                if layer_ind < len(self.posterior_params) - 1:
                    next_layer_input = math.tanh(linear_transform)

            logit = Deterministic("logit", linear_transform.squeeze())
            prob = Deterministic("prob", math.sigmoid(logit))

            Bernoulli("out", p=prob, observed=bnn_output)
        return _model

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: ArrayLike) -> Tuple[Probability, float]:
        """
        Samples probabilities and weighted sums from the prior predictive distribution.
        Parameters
        ----------
        context : ArrayLike
            The context matrix for which the probabilities are to be sampled.
        Returns
        -------
        Tuple[Probability, float]
            A tuple containing the sampled probabilities and the weighted sum.
        """

        # check input args
        self.check_context_matrix(context=context)

        _context = np.array(context, ndmin=2)
        dummy_y = np.zeros(len(context), dtype=np.int64)
        _model = self.create_model(_context, dummy_y, is_sampelwise=True)

        with _model:
            trace = sample_prior_predictive(samples=1)

        prob = trace["prior"]["prob"].values.reshape(-1)
        weighted_sum = trace["prior"]["logit"].values.reshape(-1)

        return prob, weighted_sum

    def update(self, context: ArrayLike, rewards: List[BinaryReward]):
        """
        Update the posterior_params with new context and rewards.
        Parameters
        ----------
        context : ArrayLike
            The context matrix where each row represents a context vector.
        rewards : List[BinaryReward]
            A list of binary rewards corresponding to each context vector.
        Raises
        ------
        AttributeError
            If the length of the context matrix does not match the length of the rewards list.
        ValueError
            If the update method is not recognized.
        Notes
        -----
        This method updates the model's posterior parameters by sampling from the posterior distribution
        using either Variational Inference (VI) or Markov Chain Monte Carlo (MCMC) methods.
        """
        self.check_context_matrix(context=context)
        if len(context) != len(rewards):
            AttributeError("Shape mismatch: context and rewards must have the same length.")

        _context = np.array(context, ndmin=2)
        _model = self.create_model(x=_context, y=rewards, is_sampelwise=False)
        with _model:
            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                update_kwargs = self.update_kwargs.copy()
                approx = fit(**update_kwargs["fit"])
                trace = approx.sample(**update_kwargs["trace"])
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs["trace"])
            else:
                raise ValueError("Invalid update method.")

        for layer_ind in range(len(self.posterior_params)):
            name = f"w{layer_ind}"

            w_mu = np.mean(trace[name], axis=0)
            w_sigma = np.std(trace[name], axis=0)

            self.posterior_params[layer_ind]["w"].params_dict["mu"] = w_mu.tolist()
            self.posterior_params[layer_ind]["w"].params_dict["sigma"] = w_sigma.tolist()

            name = f"b{layer_ind}"
            b_mu = np.mean(trace[name], axis=0)
            b_sigma = np.std(trace[name], axis=0)

            self.posterior_params[layer_ind]["b"].params_dict["mu"] = b_mu.tolist()
            self.posterior_params[layer_ind]["b"].params_dict["sigma"] = b_sigma.tolist()

        self.is_fitted = True

    @classmethod
    def cold_start(
        cls,
        dim_list: List[PositiveInt],
        update_method: UpdateMethods = "MCMC",
        update_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> "BayesianNeuralNetwork":
        """Create a new BayesianNeuralNetwork instance with specified architecture with default student-t distribution.

        Parameters
        ----------
        dim_list : List[PositiveInt]
            List specifying the number of nodes in each layer.
            Example: [5, 3, 1] means 5 input nodes, 3 hidden nodes, 1 output node
        update_method : UpdateMethods, optional
            Method used for posterior inference, by default "MCMC"
        update_kwargs : Optional[dict], optional
            Additional arguments for the update method, by default None
        **kwargs : dict
            Additional keyword arguments to pass to BayesianNeuralNetwork constructor

        Returns
        -------
        BayesianNeuralNetwork
            A new instance of BayesianNeuralNetwork with the specified architecture
            and initialization parameters
        """

        posterior_params = cls.create_posterior_params(dim_list)
        return cls(
            posterior_params=posterior_params, update_method=update_method, update_kwargs=update_kwargs, **kwargs
        )

    def __eq__(self, other):
        for i, layer in enumerate(self.posterior_params):
            if not layer["w"] == other.posterior_params[i]["w"] or not layer["b"] == other.posterior_params[i]["b"]:
                return False
        return True


class BayesianLogisticRegression(BayesianNeuralNetwork):
    """A Bayesian logistic regression model where the weights (betas) and
    intercept (alpha) follow Student-T distributions.

    The model output is calculated as follows:
        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)
    where the alpha and betas coefficients are Student's t-distributions.

    The implementation is based on the Bayesian Neural Network model with a single output node.

    Parameters
    ----------
    alpha : StudentT
        The intercept parameter following a Student-T distribution
    betas : List[StudentT]
        List of weight parameters, each following a Student-T distribution.
        Length must be equal to number of features (minimum 1)
    Attributes
    ----------
    posterior_params : list
        List containing dictionary of posterior parameters for weights (w) and bias (b)
    Methods
    -------
    cold_start(n_features, update_method='MCMC', update_kwargs=None, **kwargs)
        Creates a new instance with default initialization
        Parameters:
            n_features : int
                Number of input features
            update_method : str, optional
                Method for updating parameters (default is 'MCMC')
            update_kwargs : dict, optional
                Additional arguments for the update method
            **kwargs
                Additional keyword arguments
        Returns:
            BayesianLogisticRegression
                New instance of the model
    """

    alpha: StudentT
    if pydantic_version == PYDANTIC_VERSION_1:
        betas: List[StudentT] = Field(..., min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        betas: List[StudentT] = Field(..., min_length=1)
    else:
        raise ValueError("Invalid version.")

    @model_validator(mode="before")
    def set_posterior_params(cls, values):
        input_dim = len(values["betas"])
        output_dim = 1

        w_param = StudentTArray(shape=[input_dim, output_dim])
        betas = values["betas"].copy()

        for i, beta in enumerate(betas):
            if type(beta) is dict:
                beta = StudentT(**beta)  # handle from_state
            w_param.params_dict["mu"][i][0] = beta.mu
            w_param.params_dict["sigma"][i][0] = beta.sigma
            w_param.params_dict["nu"][i][0] = beta.nu

        b_param = StudentTArray(shape=[output_dim])
        alpha = values["alpha"].copy()

        if type(alpha) is dict:
            alpha = StudentT(**alpha)

        b_param.params_dict["mu"][0] = alpha.mu
        b_param.params_dict["sigma"][0] = alpha.sigma
        b_param.params_dict["nu"][0] = alpha.nu

        values["posterior_params"] = [dict(w=w_param, b=b_param)]
        return values

    @classmethod
    def cold_start(
        cls,
        n_features,
        update_method: UpdateMethods = "MCMC",
        update_kwargs: Optional[Union[Dict[str, Any], dict[str, Dict[str, Any]]]] = None,
        **kwargs,
    ) -> "BayesianLogisticRegression":
        return cls(
            alpha=StudentT(),
            betas=[StudentT() for _ in range(n_features)],
            update_method=update_method,
            update_kwargs=update_kwargs,
            **kwargs,
        )


class BayesianNeuralNetworkCC(BayesianNeuralNetwork):
    """
    Bayesian Neural Network with Cost Constraint.

    This class extends the BayesianNeuralNetwork to include a cost constraint.

    Attributes
    ----------
    cost : NonNegativeFloat
        The cost associated with the neural network, which must be a non-negative float.
    """

    cost: NonNegativeFloat
