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
from abc import ABC, abstractmethod
from copy import deepcopy
from random import betavariate
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pytensor.tensor as pt
from numpy import sqrt
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, fit, math, sample, sample_prior_predictive
from pymc import Model as PymcModel
from pymc import StudentT as PymcStudentT
from pytensor.tensor import specify_broadcastable
from typing_extensions import Self

from pybandits.base import BinaryReward, MOProbability, Probability, ProbabilityWeight, PyBanditsBaseModel
from pybandits.base_model import BaseModelCC, BaseModelMO, BaseModelSO
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
    pydantic_version,
    validate_call,
)

UpdateMethods = Literal["VI", "MCMC"]


class Model(BaseModelSO, ABC):
    """
    Class to model the prior distributions for single objective.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """

    @abstractmethod
    def sample_proba(self, **kwargs) -> Union[List[Probability], List[MOProbability], List[ProbabilityWeight]]:
        """
        Sample the probability of getting a positive reward.
        """


class ModelCC(BaseModelCC, ABC):
    """
    Class to model action cost.

    Parameters
    ----------
    cost: NonNegativeFloat
        Cost associated to the action.
    """

    cost: NonNegativeFloat


class ModelMO(BaseModelMO, ABC):
    """
    Class to model the prior distributions for multi-objective.

    Parameters
    ----------
    models : List[Model]
        The list of models for each objective.
    """

    models: List[Model]


class BaseBeta(Model, ABC):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """

    @property
    def std(self) -> float:
        """
        The corrected standard deviation (Bessel's correction) of the binary distribution of successes and failures.
        """
        return sqrt((self.n_successes * self.n_failures) / (self.count * (self.count - 1)))

    @validate_call
    def _update(self, rewards: List[BinaryReward]):
        """
        Update n_successes and n_failures.

        Parameters
        ----------
        rewards: List[BinaryReward]
            A list of binary rewards.
        """
        pass

    def _reset(self):
        pass

    def sample_proba(self, n_samples: PositiveInt) -> List[Probability]:
        """
        Sample the probability of getting a positive reward.

        Returns
        -------
        prob: Probability
            Probability of getting a positive reward.
        """
        return [betavariate(self.n_successes, self.n_failures) for _ in range(n_samples)]


class Beta(BaseBeta):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """


class BetaCC(BaseBeta, ModelCC):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with cost control.

    Parameters
    ----------
    n_successes : PositiveInt = 1
        Counter of the number of successes.
    n_failures : PositiveInt = 1
        Counter of the number of failures.
    cost : NonNegativeFloat
        Cost associated to the Beta distribution.
    """


class BaseBetaMO(ModelMO, ABC):
    """
    Base beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    models: List[Beta] of length (n_objectives,)
        List of Beta distributions.
    """

    models: List[Beta]

    @validate_call
    def sample_proba(self, n_samples: PositiveInt) -> List[MOProbability]:
        """
        Sample the probability of getting a positive reward.

        Parameters
        ----------
        n_samples : PositiveInt
            Number of samples to draw.

        Returns
        -------
        prob: List[MOProbability]
            Probabilities of getting a positive reward for each sample and objective.
        """
        return [list(p) for p in zip(*[model.sample_proba(n_samples=n_samples) for model in self.models])]

    @validate_call
    def _update(self, rewards: List[List[BinaryReward]]):
        """
        Update the Beta model using the provided rewards.

        Parameters
        ----------
        rewards: List[List[BinaryReward]]
            A list of rewards, where each reward is in turn a list containing the reward of the Beta model
            associated to each objective.
            For example, `[[1, 1], [1, 0], [1, 1], [1, 0], [1, 1]]`.
        """
        if any(len(x) != len(self.models) for x in rewards):
            raise AttributeError("The shape of rewards is incorrect")

        for i, model in enumerate(self.models):
            model.update([r[i] for r in rewards])

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
        beta_mo: BetaMO
            The multi-objective Beta model.
        """
        models = n_objectives * [Beta()]
        beta_mo = cls(models=models, **kwargs)
        return beta_mo


class BetaMO(BaseBetaMO):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    models: List[Beta] of length (n_objectives,)
        List of Beta distributions.
    """


class BetaMOCC(BaseBetaMO, ModelCC):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives and cost control.

    Parameters
    ----------
    models: List[BetaCC] of shape (n_objectives,)
        List of Beta distributions.
    cost: NonNegativeFloat
        Cost associated to the Beta distribution.
    """


class StudentTArray(PyBanditsBaseModel):
    """
    A class representing an array of Student's t-distributions with parameters `mu`, `sigma`, and `nu`.
    A specific element (e.g, a single parameter of a layer) distribution is defined by the the corresponding elements in the lists.
    The mean values are represented by `mu`, the scale (standard deviation) values by `sigma`, and the degrees of freedom by `nu`.


    Parameters
    ----------
    mu : Union[List[float], List[List[float]]]
        The mean values of the Student's t-distributions. Can be a 1D (for the layer bias term) or 2D list (for the layer weight term).
    sigma : Union[List[NonNegativeFloat], List[List[NonNegativeFloat]]]
        The scale (standard deviation) values of the Student's t-distributions. Must be non-negative.
        Can be a 1D or 2D list.
    nu : Union[List[PositiveFloat], List[List[PositiveFloat]]]
        The degrees of freedom of the Student's t-distributions. Must be positive.
        Can be a 1D or 2D list.
    """

    mu: Union[List[float], List[List[float]]]
    sigma: Union[List[NonNegativeFloat], List[List[NonNegativeFloat]]]
    nu: Union[List[PositiveFloat], List[List[PositiveFloat]]]

    @staticmethod
    def convert_list_to_array(input_list: Union[List[float], List[List[float]]]) -> bool:
        if len(input_list) == 0:
            is_valid_input = False

        elif not isinstance(input_list[0], list):
            is_valid_input = True

        else:
            first_length = len(input_list[0])
            is_valid_input = all(
                isinstance(inner_list, list) and len(inner_list) == first_length for inner_list in input_list
            )

        if is_valid_input:
            return np.array(input_list)
        else:
            raise ValueError("Input list must be a 1D or 2D list with the same length for all inner lists.")

    @model_validator(mode="after")
    @classmethod
    def validate_inputs(cls, values):
        if pydantic_version == PYDANTIC_VERSION_1:
            mu_arr = cls.convert_list_to_array(values.get("mu"))
            sigma_arr = cls.convert_list_to_array(values.get("sigma"))
            nu_arr = cls.convert_list_to_array(values.get("nu"))
        elif pydantic_version == PYDANTIC_VERSION_2:
            mu_arr = cls.convert_list_to_array(values.mu)
            sigma_arr = cls.convert_list_to_array(values.sigma)
            nu_arr = cls.convert_list_to_array(values.nu)
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

        if (mu_arr.shape != sigma_arr.shape) or (mu_arr.shape != nu_arr.shape):
            raise ValueError(
                f"mu, sigma, and nu must have the same shape, "
                f"but are {mu_arr.shape}, {sigma_arr.shape}, and {nu_arr.shape}, respectively."
            )

        if any(dim_len == 0 for dim_len in mu_arr.shape):
            raise ValueError("mu, sigma, and nu must have at least one element in every dimension.")

        return values

    @classmethod
    def cold_start(
        cls,
        shape: Union[PositiveInt, Tuple[PositiveInt, ...]],
        mu: float = 0.0,
        sigma: NonNegativeFloat = 10.0,
        nu: PositiveFloat = 5.0,
    ) -> "StudentTArray":
        if isinstance(shape, int):
            shape = (shape,)

        if any(dim_len == 0 for dim_len in shape):
            raise ValueError("shape of mu, sigma, and nu must have at least one element in every dimension.")

        mu = np.full(shape, mu).tolist()
        sigma = np.full(shape, sigma).tolist()
        nu = np.full(shape, nu).tolist()
        return cls(mu=mu, sigma=sigma, nu=nu)

    @property
    def shape(self) -> Tuple[PositiveInt, ...]:
        return np.array(self.mu).shape

    @property
    def params(self):
        return dict(mu=np.array(self.mu), sigma=np.array(self.sigma), nu=np.array(self.nu))


class BnnLayerParams(PyBanditsBaseModel):
    """
    Represents the parameters of a Bayesian neural network (BNN) layer.

    Parameters
    ----------
    weight : StudentTArray
        The weight parameter of the BNN layer, represented as a StudentTArray.
    bias : StudentTArray
        The bias parameter of the BNN layer, represented as a StudentTArray.
    """

    weight: StudentTArray
    bias: StudentTArray


class BnnParams(PyBanditsBaseModel):
    """
    Represents the parameters of a Bayesian Neural Network (BNN), including
    both the current layer parameters and the initial layer parameters.
    We keep the init parameters in case we need to reset the model.

    Parameters
    ----------
    bnn_layer_params : List[BnnLayerParams]
        A list of BNN layer parameters representing the current state of the model.
    bnn_layer_params_init : List[BnnLayerParams]
        A list of BNN layer parameters representing the initial state of the model.
    """

    bnn_layer_params: Optional[List[BnnLayerParams]]
    bnn_layer_params_init: List[BnnLayerParams] = Field(default_factory=list, init=False, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, values):
        if values.get("bnn_layer_params_init") is None:
            values["bnn_layer_params_init"] = deepcopy(values["bnn_layer_params"])

        return values


class BaseBayesianNeuralNetwork(Model, ABC):
    """Bayesian Neural Network model for binary classification.

    This class implements a Bayesian Neural Network with an arbitrary number of fully connected layers
    using PyMC for binary classification tasks. It supports both Markov Chain Monte Carlo (MCMC)
    and Variational Inference (VI) methods for posterior inference.

    References
    ----------
    Bayesian Learning for Neural Networks (Radford M. Neal, 1995)
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=db869fa192a3222ae4f2d766674a378e47013b1b

    Parameters
    ----------
    model_params : BnnParams
        The parameters of the Bayesian Neural Network, including weights and biases for each layer and their initial values for resetting
    update_method : str, optional
        The method used for posterior inference, either "MCMC" or "VI" (default is "MCMC").
    update_kwargs : Optional[dict], optional
        A dictionary of keyword arguments for the update method. For MCMC, it contains 'trace' settings.
        For VI, it contains both 'trace' and 'fit' settings.

    Notes
    -----
    - The model uses tanh activation for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled
      using a Bernoulli likelihood.
    """

    model_params: BnnParams

    _logit_var_name: ClassVar[str] = "logit"
    _weight_var_name: ClassVar[str] = "weight"
    _bias_var_name: ClassVar[str] = "bias"

    update_method: str = "MCMC"
    update_kwargs: Optional[dict] = None

    _default_mcmc_trace_kwargs: ClassVar[dict] = dict(
        tune=500,
        draws=1000,
        chains=2,
        init="adapt_diag",
        cores=1,
        target_accept=0.95,
        progressbar=False,
        return_inferencedata=False,
    )

    _default_variational_inference_fit_kwargs: ClassVar[dict] = dict(method="advi")
    _default_variational_inference_trace_kwargs: ClassVar[dict] = dict(
        draws=1000, progressbar=False, return_inferencedata=False
    )

    _approx_history: np.ndarray = PrivateAttr(None)

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

    @property
    def approx_history(self) -> Optional[np.ndarray]:
        return self._approx_history

    @classmethod
    def _stable_sigmoid(cls, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    @classmethod
    def get_layer_params_name(cls, layer_ind: PositiveInt) -> Tuple[str, str]:
        weight_layer_params_name = f"{cls._weight_var_name}_{layer_ind}"
        bias_layer_params_name = f"{cls._bias_var_name}_{layer_ind}"
        return weight_layer_params_name, bias_layer_params_name

    @classmethod
    def create_model_params(
        cls, n_features: PositiveInt, hidden_dim_list: List[PositiveInt], **dist_params_init
    ) -> BnnParams:
        """
        Creates model parameters for a Bayesian neural network (BNN) model according to dist_params_init
        This method initializes the distribution's parameters for each layer of a BNN
        using the specified number of features, hidden dimensions, and distribution
        initialization parameters.

        Parameters
        ----------
        n_features : PositiveInt
            The number of input features for the BNN.
        hidden_dim_list : List[PositiveInt]
            A list of integers specifying the number of hidden units in each hidden layer.
            If None, no hidden layers are added.
        **dist_params_init : dict, optional
            Additional parameters for initializing the distribution of weights and biases.
        Returns
        -------
        BnnParams
            An instance of BnnParams containing the initialized layer parameters.
        """

        if hidden_dim_list is None:
            _dim_list = [n_features]
        else:
            _dim_list = [n_features] + hidden_dim_list

        _dim_list.append(1)

        layer_params_init = []
        for layer_ind in range(len(_dim_list) - 1):
            input_dim = _dim_list[layer_ind]
            output_dim = _dim_list[layer_ind + 1]
            w_param = StudentTArray.cold_start(shape=(input_dim, output_dim), **dist_params_init)
            b_param = StudentTArray.cold_start(shape=output_dim, **dist_params_init)
            layer_params_init.append(BnnLayerParams(weight=w_param, bias=b_param))

        return BnnParams(bnn_layer_params=layer_params_init)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def check_context_matrix(self, context: np.ndarray):
        """
        Check and cast context matrix.

        Parameters
        ----------
        context : np.ndarray of shape (n_samples, n_features)
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

        if not np.issubdtype(context.dtype, np.number):
            raise ValueError("Context array must contain only numeric values.")

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
            the shape of the weight matrix in the first layer's parameters.
        """
        return self.model_params.bnn_layer_params[0].weight.shape[0]

    def create_model(
        self, x: ArrayLike, y: Optional[Union[List[BinaryReward], np.ndarray]] = None, is_predict: bool = False
    ) -> PymcModel:
        """
        Create a PyMC model for Bayesian Neural Network.

        This method builds a PyMC model with the network architecture specified in model_params.
        The model uses tanh activation for hidden layers and sigmoid for the output layer.

        Parameters
        ----------
        x : ArrayLike
            Input features of shape (n_samples, n_features)
        y : Union[List[BinaryReward], np.ndarray]
            Binary target values of shape (n_samples,)
        is_predict : bool
            If True, process samples independently. If False, process all samples at once.
            In the predict step, we would like to sample the model parameters independently for each sample.
            In the update step, this is not required.

        Returns
        -------
        PymcModel
            PyMC model object with the specified neural network architecture

        Notes
        -----
        The model structure follows these steps:
        1. For each layer, create weight and bias variables from StudentT distributions.
        2. Apply linear transformations and activations through the layers.
           When is_sampelwise is True, the linear transformation is applied on each row separately (so random variables are not shared).
           When is_sampelwise is False, the linear transformation is applied on the whole matrix at once, so random variables are shared.
        3. Apply sigmoid activation at the output
        4. Use Bernoulli likelihood for binary classification
        """
        if y is None:
            y = np.zeros(len(x), dtype=np.int64)

        n_samples = len(x)
        is_multi_sample_predict = is_predict and (n_samples != 1)
        with PymcModel() as _model:
            # Define data variables using minibatches
            bnn_output = Data("bnn_output", y, mutable=True)

            if is_multi_sample_predict:
                # arrange  for batched dot product
                x_expanded = np.expand_dims(x, axis=1)
                bnn_input = Data("bnn_input", x_expanded, mutable=True)
                for dim, size in enumerate(x_expanded.shape):
                    if size == 1:
                        bnn_input = specify_broadcastable(bnn_input, dim)
            else:
                bnn_input = Data("bnn_input", x, mutable=True)

            next_layer_input = bnn_input
            for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                w_shape = layer_params.weight.shape  # without it n_features = 1 doesn't work
                b_shape = layer_params.bias.shape
                weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)

                if is_multi_sample_predict:
                    # in this case we create n_samples different weights and biases - one for each sample
                    w = PymcStudentT(
                        name=weight_layer_params_name, shape=(n_samples,) + w_shape, **layer_params.weight.params
                    )
                    b = PymcStudentT(
                        name=bias_layer_params_name, shape=(n_samples, 1) + b_shape, **layer_params.bias.params
                    )

                    batched_dot = pt.vectorize(
                        lambda x_batch, w_batch: pt.dot(x_batch, w_batch), signature="(n,m),(m,p)->(n,p)"
                    )
                    linear_transform = pt.as_tensor_variable(
                        batched_dot(next_layer_input, w) + b, name=f"linear_transform{layer_ind}"
                    )

                else:
                    # in this case we create one weight and bias for all samples
                    w = PymcStudentT(
                        name=weight_layer_params_name, shape=w_shape, **layer_params.weight.params, initval="prior"
                    )
                    b = PymcStudentT(
                        name=bias_layer_params_name, shape=b_shape, **layer_params.bias.params, initval="prior"
                    )

                    linear_transform = math.dot(next_layer_input, w) + b

                if layer_ind < len(self.model_params.bnn_layer_params) - 1:
                    next_layer_input = math.tanh(linear_transform)
            for dim in range(1, linear_transform.ndim):
                linear_transform = specify_broadcastable(linear_transform, dim)

            logit = Deterministic(self._logit_var_name, linear_transform.squeeze())

            if not is_predict:
                Bernoulli("out", logit_p=logit, observed=bnn_output)

        return _model

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: np.ndarray) -> List[ProbabilityWeight]:
        """
        Samples probabilities and weighted sums from the prior predictive distribution.

        Parameters
        ----------
        context : ArrayLike
            The context matrix for which the probabilities are to be sampled.
        Returns
        -------
        List[ProbabilityWeight]
            Each element is a tuple containing the probability of a positive reward and
            the corresponding weighted sum between contextual feature quantities and sampled coefficients.
        """

        # check input args
        self.check_context_matrix(context=context)

        _context = np.atleast_2d(context)
        _model = self.create_model(x=_context, is_predict=True)

        with _model:
            trace = sample_prior_predictive(samples=1)

        weighted_sum = trace["prior"][self._logit_var_name].values.reshape(-1)
        prob = self._stable_sigmoid(weighted_sum)

        return list(zip(prob, weighted_sum))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _update(self, context: np.ndarray, rewards: List[BinaryReward]):
        """
        Update the model_params with new context and rewards.
        Parameters
        ----------
        context : np.ndarray
            The context matrix where each row represents a context vector.
        rewards : List[BinaryReward]
            A list of binary rewards corresponding to each context vector.

        Notes
        -----
        This method updates the model's parameters by sampling from the posterior distribution
        using either Variational Inference (VI) or Markov Chain Monte Carlo (MCMC) methods.
        """
        self.check_context_matrix(context=context)

        if len(context) != len(rewards):
            raise AttributeError("Shape mismatch: context and rewards must have the same length.")

        _context = np.atleast_2d(context)
        _model = self.create_model(x=_context, y=rewards, is_predict=False)
        with _model:
            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                update_kwargs = self.update_kwargs.copy()
                approx = fit(**update_kwargs["fit"])
                trace = approx.sample(**update_kwargs["trace"])
                self._approx_history = approx.hist
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs["trace"])
            else:
                raise ValueError("Invalid update method.")

        for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
            weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)

            w_mu = np.mean(trace[weight_layer_params_name], axis=0)
            w_sigma = np.std(trace[weight_layer_params_name], axis=0)
            layer_params.weight = StudentTArray(
                mu=w_mu.tolist(), sigma=w_sigma.tolist(), nu=self.model_params.bnn_layer_params[layer_ind].weight.nu
            )

            b_mu = np.mean(trace[bias_layer_params_name], axis=0)
            b_sigma = np.std(trace[bias_layer_params_name], axis=0)
            layer_params.bias = StudentTArray(
                mu=b_mu.tolist(), sigma=b_sigma.tolist(), nu=self.model_params.bnn_layer_params[layer_ind].bias.nu
            )

    @classmethod
    def cold_start(
        cls,
        n_features: PositiveInt,
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
        n_features : PositiveInt
            Number of input features for the network.
        hidden_dim_list : Optional[List[PositiveInt]], optional
            List of dimensions for the hidden layers of the network. If None, no hidden layers are added.
        update_method : UpdateMethods
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

        model_params = cls.create_model_params(
            n_features=n_features, hidden_dim_list=hidden_dim_list, **dist_params_init
        )
        return cls(model_params=model_params, update_method=update_method, update_kwargs=update_kwargs, **kwargs)

    def _reset(self):
        """
        Reset the model.
        """
        self.model_params.bnn_layer_params = deepcopy(self.model_params.bnn_layer_params_init)


class BayesianNeuralNetwork(BaseBayesianNeuralNetwork):
    """
    Bayesian Neural Network class.
    This class implements a Bayesian Neural Network by extending the
    BaseBayesianNeuralNetwork. It provides functionality for probabilistic
    modeling and inference using neural networks.
    """


class BayesianNeuralNetworkCC(BaseBayesianNeuralNetwork, ModelCC):
    """Bayesian Neural Network model for binary classification with cost constraint.

    This class implements a Bayesian Neural Network with an arbitrary number of fully connected layers
    using PyMC for binary classification tasks. It supports both Markov Chain Monte Carlo (MCMC)
    and Variational Inference (VI) methods for posterior inference.

    References
    ----------
    Bayesian Learning for Neural Networks (Radford M. Neal, 1995)
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=db869fa192a3222ae4f2d766674a378e47013b1b

    Parameters
    ----------
    model_params : BnnParams
        The parameters of the Bayesian Neural Network, including weights and biases for each layer and their initial values for resetting
    update_method : str, optional
        The method used for posterior inference, either "MCMC" or "VI" (default is "MCMC").
    update_kwargs : Optional[dict], optional
        A dictionary of keyword arguments for the update method. For MCMC, it contains 'trace' settings.
        For VI, it contains both 'trace' and 'fit' settings.
    cost : NonNegativeFloat
        Cost associated to the Bayesian Neural Network model.

    Notes
    -----
    - The model uses tanh activation for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled
      using a Bernoulli likelihood.
    """


class BayesianLogisticRegression(BayesianNeuralNetwork):
    """
    A Bayesian Logistic Regression model that inherits from BayesianNeuralNetwork.
    This model is a specialized version of a Bayesian Neural Network with a single layer,
    designed specifically for logistic regression tasks. The model parameters are
    validated to ensure that the model adheres to this single-layer constraint.
    """

    @field_validator("model_params")
    def validate_model_params(cls, model_params):
        if (len(model_params.bnn_layer_params_init) != 1) or (len(model_params.bnn_layer_params) != 1):
            raise ValueError("The Bayesian Logistic Regression model should have only one layer.")
        return model_params


class BayesianLogisticRegressionCC(BayesianLogisticRegression, ModelCC):
    """
    A Bayesian Logistic Regression model with cost control.
    """
