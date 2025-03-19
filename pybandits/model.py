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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, ClassVar
from typing_extensions import Self
from functools import cached_property

import numpy as np
import pymc.math as pmath
import pytensor.tensor as pt
from numpy import array, c_, insert, mean, multiply, ones, sqrt, std
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, MutableData, fit, math, sample, sample_prior_predictive, Approximation
from pymc import Model as PymcModel
from pymc import StudentT as PymcStudentT
from scipy.stats import t

from pybandits.base import BinaryReward, Probability, PyBanditsBaseModel
from pydantic.dataclasses import dataclass
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    Field,
    NonNegativeFloat,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
    confloat,
    model_validator,
    field_validator,
    pydantic_version,
    validate_call,
    PrivateAttr,
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
    mu: Union[List[float], List[List[float]]]
    sigma: Union[List[NonNegativeFloat], List[List[NonNegativeFloat]]]
    nu: Union[List[PositiveFloat], List[List[PositiveFloat]]]

    @model_validator(mode="after")
    @classmethod
    def validate_inputs(cls, values):
        if pydantic_version == PYDANTIC_VERSION_1:
            if (np.array(values.get("mu")).shape != np.array(values.get("sigma")).shape) or (np.array(values.get("mu")).shape != np.array(values.get("nu")).shape):
                raise ValueError("mu, sigma, and nu must have the same shape.")
        elif pydantic_version == PYDANTIC_VERSION_2:
            if (np.array(values.mu).shape != np.array(values.sigma).shape) or (np.array(values.mu).shape != np.array(values.nu).shape):
                raise ValueError("mu, sigma, and nu must have the same shape.")
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

        return values
    
    @classmethod
    def cold_start(cls, shape: Tuple[PositiveInt], mu: float = 0.0, sigma: float = 10.0, nu: float = 5.0) -> "StudentTArray":
        mu = np.full(shape, mu).tolist()
        sigma = np.full(shape, sigma).tolist()
        nu = np.full(shape, nu).tolist()
        return cls(mu=mu, sigma=sigma, nu=nu)
    
    @cached_property
    def shape(self) -> Tuple[PositiveInt]:
        return np.array(self.mu).shape
    
    @cached_property
    def params(self):
        return dict(mu=np.array(self.mu), sigma=np.array(self.sigma), nu=np.array(self.nu))
    
    class Config:
        keep_untouched = (cached_property,)

    def __eq__(self, other: Self)  -> bool:
        return self.mu == other.mu and self.sigma == other.sigma and self.nu == other.nu

@dataclass 
class BnnLayerParams:
    weight: StudentTArray
    bias: StudentTArray

class BaseBayesianNeuralNetwork(Model):
    """Bayesian Neural Network model for binary classification.

    This class implements a Bayesian Neural Network with an arbitrary number of fully connected layers 
    using PyMC for binary classification tasks. It supports both Markov Chain Monte Carlo (MCMC) 
    and Variational Inference (VI) methods for posterior inference.

    Parameters
    ----------
    posterior_params : List[BnnLayerParams]
        A list of `BnnLayerParams` objects, each containing the weight and bias parameters 
        for a layer in the neural network. These parameters are modeled as Student's t-distributions.
    update_method : str, optional
        The method used for posterior inference, either "MCMC" or "VI" (default is "MCMC").
    update_kwargs : Optional[Union[Dict[str, Any], dict[str, Dict[str, Any]]]], optional
        A dictionary of keyword arguments for the update method. For MCMC, it contains 'trace' settings. 
        For VI, it contains both 'trace' and 'fit' settings.

    Notes
    -----
    - The model uses tanh activation for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled 
      using a Bernoulli likelihood.
    """

    _logit_var_name: ClassVar[str] = "logit"
    _prob_var_name: ClassVar[str] = "prob"
 
    update_method: str = "MCMC"
    update_kwargs: Optional[Union[Dict[str, Any], dict[str, Dict[str, Any]]]] = None

    posterior_params: List[BnnLayerParams]

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

    _approx : Approximation = PrivateAttr(None) 

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
    def create_posterior_params(cls, num_features: PositiveInt, hidden_dim_list: List[PositiveInt], **dist_params_init) -> BnnLayerParams:
        """
        Creates posterior parameters for a Bayesian neural network (BNN) layer.
        This method initializes the posterior parameters for each layer of a BNN 
        using the specified number of features, hidden dimensions, and distribution 
        initialization parameters.
        Parameters
        ----------
        num_features : PositiveInt
            The number of input features for the BNN.
        hidden_dim_list : List[PositiveInt]
            A list of integers specifying the number of hidden units in each hidden layer.
            If None, no hidden layers are added.
        **dist_params_init : dict, optional
            Additional parameters for initializing the distribution of weights and biases.
        Returns
        -------
        List[BnnLayerParams]
            A list of `BnnLayerParams` objects, where each object contains the weight 
            and bias parameters for a layer in the BNN.
        """

        if hidden_dim_list is None:
            _dim_list = [num_features]
        else:
            _dim_list = [num_features] + hidden_dim_list
        
        _dim_list.append(1)

        posterior_params = []
        for layer_ind in range(len(_dim_list) - 1):
            input_dim = _dim_list[layer_ind]
            output_dim = _dim_list[layer_ind + 1]
            w_param = StudentTArray.cold_start(shape=(input_dim,output_dim), **dist_params_init)
            b_param = StudentTArray.cold_start(shape=output_dim, **dist_params_init)
            posterior_params.append(BnnLayerParams(weight=w_param, bias=b_param))

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
            raise AttributeError(f"Context must be an ArrayLike with {self.input_dim} columns: {e}.")
        if n_cols_context != self.input_dim:
            raise AttributeError(f"Shape mismatch: context must have {self.input_dim} columns.")

    @property
    def input_dim(self) -> PositiveInt:
        """
        Returns the expected input dimension of the model.

        Returns
        -------
        int
            The number of input features expected by the model, derived from
            the shape of the weight matrix in the first layer's posterior parameters.
        """
        return self.posterior_params[0].weight.shape[0]

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
                w_shape = layer_params.weight.shape  # without it n_features = 1 doesn't work
                b_shape = layer_params.bias.shape

                if is_sampelwise:
                    # in this case we create n_samples different weights and biases - one for each sample
                    w = PymcStudentT(
                        f"weight_{layer_ind}", **layer_params.weight.params, shape=(n_samples,) + w_shape)  # (n_samples, n_features, n_output)
                    b = PymcStudentT(f"bias_{layer_ind}", **layer_params.bias.params, shape=(n_samples,) + b_shape)
                    linear_transform = pt.as_tensor_variable(
                        pt.batched_dot(next_layer_input, w) + b, name=f"linear_transform{layer_ind}"
                    )

                else:
                    # in this case we create one weight and bias for all samples
                    w = PymcStudentT(f"weight_{layer_ind}", **layer_params.weight.params, shape=w_shape,initval="prior")
                    b = PymcStudentT(f"bias_{layer_ind}", **layer_params.bias.params, shape=b_shape ,initval="prior")
                    linear_transform = Deterministic(f"linear_transform{layer_ind}", math.dot(next_layer_input, w) + b)

                if layer_ind < len(self.posterior_params) - 1:
                    next_layer_input = math.tanh(linear_transform)

            logit = Deterministic(self._logit_var_name, linear_transform.squeeze())
            prob = Deterministic(self._prob_var_name, math.sigmoid(logit))

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

        prob = trace["prior"][self._prob_var_name].values.reshape(-1)
        weighted_sum = trace["prior"][self._logit_var_name].values.reshape(-1)

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
                self._approx = fit(**update_kwargs["fit"])
                trace = self._approx .sample(**update_kwargs["trace"])
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs["trace"])
            else:
                raise ValueError("Invalid update method.")

        for layer_ind in range(len(self.posterior_params)):
            name = f"weight_{layer_ind}"

            w_mu = np.mean(trace[name], axis=0)
            w_sigma = np.std(trace[name], axis=0)
            self.posterior_params[layer_ind].weight= StudentTArray(mu=w_mu.tolist(), sigma=w_sigma.tolist(), nu=self.posterior_params[layer_ind].weight.nu)

            name = f"bias_{layer_ind}"
            b_mu = np.mean(trace[name], axis=0)
            b_sigma = np.std(trace[name], axis=0)
            self.posterior_params[layer_ind].bias = StudentTArray(mu=b_mu.tolist(), sigma=b_sigma.tolist(), nu=self.posterior_params[layer_ind].bias.nu)

    @classmethod
    def cold_start(
        cls,
        num_features: PositiveInt,
        hidden_dim_list: Optional[List[PositiveInt]] = None,
        update_method: UpdateMethods = "MCMC",
        update_kwargs: Optional[dict] = None,
        dist_params_init: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Self:
        """
        Initialize a Bayesian Neural Network with a cold start.

        Parameters
        ----------
        num_features : PositiveInt
            Number of input features for the network.
        hidden_dim_list : Optional[List[PositiveInt]], optional
            List of dimensions for the hidden layers of the network. If None, no hidden layers are added.
        update_method : UpdateMethods, optional
            Method to update the network, either "MCMC" or "VI". Default is "MCMC".
        update_kwargs : Optional[dict], optional
            Additional keyword arguments for the update method. Default is None.
        dist_params_init : Optional[Dict[str, float]], optional
            Initial distribution parameters for the network weights and biases. Default is None.
        **kwargs
            Additional keyword arguments for the BayesianNeuralNetwork constructor.

        Returns
        -------
        Self
            An instance of the Bayesian Neural Network initialized with the specified parameters.
        """

        if dist_params_init is None:
            dist_params_init = {}
            
        posterior_params = cls.create_posterior_params(num_features=num_features, hidden_dim_list=hidden_dim_list, **dist_params_init)
        return cls(
            posterior_params=posterior_params, update_method=update_method, update_kwargs=update_kwargs, **kwargs
        )

    def __eq__(self, other: Self) -> bool:
        for self_layer, other_layer in zip(self.posterior_params, other.posterior_params):
            if not self_layer.weight == other_layer.weight  or not self_layer.bias == other_layer.bias:
                return False
        
        return True


class BayesianNeuralNetwork(BaseBayesianNeuralNetwork):
    """
    Bayesian Neural Network class.
    This class implements a Bayesian Neural Network by extending the 
    BaseBayesianNeuralNetwork. It provides functionality for probabilistic 
    modeling and inference using neural networks.
    """

class BayesianNeuralNetworkCC(BaseBayesianNeuralNetwork):
    """
    Bayesian Neural Network with Cost Constraint.

    This class extends the BayesianNeuralNetwork to include a cost constraint.

    Attributes
    ----------
    cost : NonNegativeFloat
        The cost associated with the neural network, which must be a non-negative float.
    """

    cost: NonNegativeFloat


class BaseBayesianLogisticRegression(BaseBayesianNeuralNetwork): 
    """
    A Bayesian Logistic Regression model that inherits from BaseBayesianNeuralNetwork.
    This model is a specialized version of a Bayesian Neural Network with a single layer,
    designed specifically for logistic regression tasks. The posterior parameters are
    validated to ensure that the model adheres to this single-layer constraint.
    """
    @field_validator('posterior_params') # will be enabled in pydantic 2
    def validate_posterior_params(cls, posterior_params):
        if len(posterior_params) != 1:
            raise ValueError("The BayesianLogisticRegression model should have only one layer.") 
        return posterior_params

class BayesianLogisticRegression(BaseBayesianLogisticRegression):
    """
    A Bayesian Logistic Regression model.
    """
    
class BayesianLogisticRegressionCC(BaseBayesianLogisticRegression):
    """
    A Bayesian Logistic Regression model with cost control.
    """

    cost: NonNegativeFloat
