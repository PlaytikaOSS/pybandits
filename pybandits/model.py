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
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pymc
from numpy import sqrt
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, Minibatch, fit, math, sample
from pymc import Model as PymcModel
from pymc import StudentT as PymcStudentT
from scipy.special import erf
from scipy.stats import t
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
    conlist,
    field_validator,
    model_validator,
    pydantic_version,
    validate_call,
)

UpdateMethods = Literal["VI", "MCMC"]
ActivationFunctions = Literal["tanh", "relu", "sigmoid", "gelu"]


# Module-level activation functions for pickling compatibility
def _pymc_relu(x):
    """ReLU activation function for PyMC."""
    return math.maximum(0, x)


def _pymc_gelu(x):
    """GELU activation function for PyMC."""
    return 0.5 * x * (1 + math.erf(x / np.sqrt(2.0)))


def _numpy_relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function for NumPy."""
    return np.maximum(0, x)


def _numpy_gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function for NumPy."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2.0)))


def _stable_sigmoid(x):
    """Stable sigmoid activation function for NumPy."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


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

    if pydantic_version == PYDANTIC_VERSION_1:
        models: conlist(Model, min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        models: conlist(Model, min_length=1)
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")


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

    if pydantic_version == PYDANTIC_VERSION_1:
        models: conlist(Beta, min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        models: conlist(Beta, min_length=1)
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

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

    _mu_array: np.ndarray = PrivateAttr()
    _sigma_array: np.ndarray = PrivateAttr()
    _nu_array: np.ndarray = PrivateAttr()
    _params: Dict[str, np.ndarray] = PrivateAttr()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StudentTArray):
            return False
        return (
            np.all(self._mu_array == other._mu_array)
            and np.all(self._sigma_array == other._sigma_array)
            and np.all(self._nu_array == other._nu_array)
        )

    @staticmethod
    def maybe_convert_list_to_array(input_list: Union[List[float], List[List[float]]]) -> bool:
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

    @model_validator(mode="before")
    @classmethod
    def validate_input_shapes(cls, values):
        mu_input = values.get("mu")
        sigma_input = values.get("sigma")
        nu_input = values.get("nu")

        mu_arr = cls.maybe_convert_list_to_array(mu_input)
        sigma_arr = cls.maybe_convert_list_to_array(sigma_input)
        nu_arr = cls.maybe_convert_list_to_array(nu_input)

        if (mu_arr.shape != sigma_arr.shape) or (mu_arr.shape != nu_arr.shape):
            raise ValueError(
                f"mu, sigma, and nu must have the same shape, "
                f"but are {mu_arr.shape}, {sigma_arr.shape}, and {nu_arr.shape}, respectively."
            )

        if any(dim_len == 0 for dim_len in mu_arr.shape):
            raise ValueError("mu, sigma, and nu must have at least one element in every dimension.")

        for key, value in zip(["mu", "sigma", "nu"], [mu_input, sigma_input, nu_input]):
            if isinstance(value, np.ndarray):
                values[key] = value.tolist()
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

        mu = np.full(shape, mu)
        sigma = np.full(shape, sigma)
        nu = np.full(shape, nu)
        return cls(mu=mu, sigma=sigma, nu=nu)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize private numpy array attributes by converting lists to arrays once at initialization.

        Parameters
        ----------
        __context : Any
            Pydantic context (unused).
        """
        self._mu_array = np.array(self.mu)
        self._sigma_array = np.array(self.sigma)
        self._nu_array = np.array(self.nu)
        self._params = dict(mu=self._mu_array, sigma=self._sigma_array, nu=self._nu_array)

    @property
    def shape(self) -> Tuple[PositiveInt, ...]:
        """
        Get the shape of the mu array.

        Returns
        -------
        Tuple[PositiveInt, ...]
            The shape of the mu array.
        """
        return self._mu_array.shape

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters as a dictionary of numpy arrays.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'mu', 'sigma', and 'nu' as numpy arrays.
        """
        return self._params


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
    activation : str, optional
        The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
    use_residual_connections : bool, optional
        Whether to use residual connections in the network. Residual connections are only added when
        the layer output dimension is greater than or equal to the input dimension (default is False).

    Notes
    -----
    - The model uses the specified activation function for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled
      using a Bernoulli likelihood.
    - When use_residual_connections is True, residual connections are added to hidden layers where the output
      dimension is >= input dimension. For expanding dimensions, the residual is zero-padded.
    """

    model_params: BnnParams

    _logit_var_name: ClassVar[str] = "logit"
    _prob_var_name: ClassVar[str] = "prob"
    _weight_var_name: ClassVar[str] = "weight"
    _bias_var_name: ClassVar[str] = "bias"
    _vi_update_params: ClassVar[list] = ["optimizer_type", "optimizer_kwargs", "batch_size"]
    _supported_optimizers: ClassVar[list] = [
        "sgd",
        "momentum",
        "nesterov_momentum",
        "adagrad",
        "rmsprop",
        "adadelta",
        "adam",
        "adamax",
    ]
    _pymc_activations: ClassVar[dict] = {
        "tanh": math.tanh,
        "relu": _pymc_relu,
        "sigmoid": math.sigmoid,
        "gelu": _pymc_gelu,
    }
    _numpy_activations: ClassVar[dict] = {
        "tanh": np.tanh,
        "relu": _numpy_relu,
        "sigmoid": _stable_sigmoid,
        "gelu": _numpy_gelu,
    }

    update_method: str = "VI"
    update_kwargs: Optional[dict] = None
    activation: ActivationFunctions = "tanh"
    use_residual_connections: bool = False

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

    _approx_history: np.ndarray = PrivateAttr(None)
    _numpy_activation_fn: Callable = PrivateAttr(None)
    _pymc_activation_fn: Callable = PrivateAttr(None)

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
                update_kwargs["fit"] = {**cls._default_variational_inference_fit_kwargs, **update_kwargs.get("fit", {})}
                optimizer_type = update_kwargs.get("optimizer_type", None)

                if optimizer_type is not None:
                    if optimizer_type not in cls._supported_optimizers:
                        raise ValueError(
                            f"Invalid optimizer type: {optimizer_type}. Supported optimizers are: {cls._supported_optimizers}"
                        )

            elif update_method == "MCMC":
                for param in cls._vi_update_params:
                    if param in update_kwargs:
                        raise ValueError(
                            f"Invalid update MCMC parameter: {param}. {cls._vi_update_params} are VI parameters."
                        )

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
                self.update_kwargs["fit"] = {
                    **self._default_variational_inference_fit_kwargs,
                    **self.update_kwargs.get("fit", {}),
                }
                optimizer_type = self.update_kwargs.get("optimizer_type", None)
                if optimizer_type is not None:
                    if optimizer_type not in self._supported_optimizers:
                        raise ValueError(
                            f"Invalid optimizer type: {optimizer_type}. Supported optimizers are: {self._supported_optimizers}"
                        )

            elif self.update_method == "MCMC":
                for param in self._vi_update_params:
                    if param in self.update_kwargs:
                        raise ValueError(
                            f"Invalid update MCMC parameter: {param}. {self._vi_update_params} are VI parameters."
                        )

                self.update_kwargs["trace"] = {**self._default_mcmc_trace_kwargs, **self.update_kwargs.get("trace", {})}
            else:
                raise ValueError("Invalid update method.")
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v):
        if v not in cls._pymc_activations.keys():
            raise ValueError(
                f"Invalid activation function: {v}. Supported activations are: {list(cls._pymc_activations.keys())}"
            )
        return v

    @property
    def approx_history(self) -> Optional[np.ndarray]:
        return self._approx_history

    @property
    def optimizer(self) -> Callable:
        optimizer_type = self.update_kwargs.get("optimizer_type", None)
        if optimizer_type is not None:
            optimizer = getattr(pymc, optimizer_type)
            optimizer_kwargs = self.update_kwargs.get("optimizer_kwargs", {})
            _optimizer = optimizer(**optimizer_kwargs)
        else:
            _optimizer = None

        return _optimizer

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
        PositiveInt
            The number of input features expected by the model, derived from
            the shape of the weight matrix in the first layer's parameters.
        """
        return self.model_params.bnn_layer_params[0].weight.shape[0]

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize activation function PrivateAttr based on the activation setting.
        """
        # Initialize activation functions (always set to ensure they're available after model_copy)
        self._numpy_activation_fn = self._numpy_activations[self.activation]
        self._pymc_activation_fn = self._pymc_activations[self.activation]

    def create_update_model(
        self, x: ArrayLike, y: Union[List[BinaryReward], np.ndarray], batch_size: Optional[PositiveInt] = None
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

        Returns
        -------
        PymcModel
            PyMC model object with the specified neural network architecture

        Notes
        -----
        The model structure follows these steps:
        1. For each layer, create weight and bias variables from StudentT distributions.
        2. Apply linear transformations and activations through the layers.
        3. Apply sigmoid activation at the output
        4. Use Bernoulli likelihood for binary classification
        """
        y = np.array(y, dtype=np.int32)
        with PymcModel() as _model:
            # Define data variables
            if batch_size is None:
                bnn_output = Data("bnn_output", y)
                bnn_input = Data("bnn_input", x)
            else:
                bnn_input, bnn_output = Minibatch(x, y, batch_size=batch_size)

            next_layer_input = bnn_input

            for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                w_shape = layer_params.weight.shape  # without it n_features = 1 doesn't work
                b_shape = layer_params.bias.shape
                weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)
                input_dim = w_shape[0]
                output_dim = w_shape[1]

                # For training, use shared weights and biases
                w = PymcStudentT(
                    name=weight_layer_params_name, shape=w_shape, **layer_params.weight.params, initval="prior"
                )
                b = PymcStudentT(
                    name=bias_layer_params_name, shape=b_shape, **layer_params.bias.params, initval="prior"
                )

                linear_transform = math.dot(next_layer_input, w) + b

                if layer_ind < len(self.model_params.bnn_layer_params) - 1:
                    activated_output = self._pymc_activation_fn(linear_transform)

                    # Add residual connection if enabled and dimensions allow
                    if self.use_residual_connections and output_dim >= input_dim:
                        if output_dim == input_dim:
                            next_layer_input = activated_output + next_layer_input
                        else:
                            residual_padded = math.concatenate(
                                [next_layer_input, math.zeros((next_layer_input.shape[0], output_dim - input_dim))],
                                axis=1,
                            )
                            next_layer_input = activated_output + residual_padded
                    else:
                        next_layer_input = activated_output

            # Final output processing
            logit = Deterministic(self._logit_var_name, linear_transform.squeeze())

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
        n_samples = len(_context)
        # Sample from StudentT distributions for each layer
        next_layer_input = _context

        for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
            # Sample weights and biases from StudentT distributions
            w_params = layer_params.weight.params
            b_params = layer_params.bias.params
            input_dim = layer_params.weight.shape[0]
            output_dim = layer_params.weight.shape[1]

            # Sample weights and biases using scipy.stats
            w = t.rvs(
                w_params["nu"],
                loc=w_params["mu"],
                scale=w_params["sigma"],
                size=(n_samples, len(w_params["nu"]), len(w_params["nu"][0])),
            )
            b = t.rvs(
                b_params["nu"], loc=b_params["mu"], scale=b_params["sigma"], size=(n_samples, len(b_params["nu"]))
            )

            # Linear transformation
            linear_transform = np.einsum("...i,...ij->...j", next_layer_input, w) + b

            # Apply activation function for hidden layers, sigmoid for output
            if layer_ind < len(self.model_params.bnn_layer_params) - 1:
                activated_output = self._numpy_activation_fn(linear_transform)

                # Add residual connection if enabled and dimensions allow
                if self.use_residual_connections and output_dim >= input_dim:
                    if output_dim == input_dim:
                        next_layer_input = activated_output + next_layer_input
                    else:
                        residual_padded = np.pad(
                            next_layer_input, ((0, 0), (0, output_dim - input_dim)), mode="constant", constant_values=0
                        )
                        next_layer_input = activated_output + residual_padded
                else:
                    next_layer_input = activated_output
            else:
                # Output layer - apply sigmoid
                weighted_sum = linear_transform.squeeze(-1)
                prob = _stable_sigmoid(weighted_sum)

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

        batch_size = self.update_kwargs.get("batch_size", None)
        _context = np.atleast_2d(context)
        _model = self.create_update_model(x=_context, y=rewards, batch_size=batch_size)
        with _model:
            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                update_kwargs = self.update_kwargs.copy()

                if self.optimizer is not None:
                    approx = fit(obj_optimizer=self.optimizer, **update_kwargs["fit"])
                else:
                    approx = fit(**update_kwargs["fit"])

                self._approx_history = approx.hist
                approx_mean_eval = approx.mean.eval()
                approx_std_eval = approx.std.eval()
                approx_posterior_mapping = {
                    param: (approx_mean_eval[slice_], approx_std_eval[slice_])
                    for (param, (_, slice_, _, _)) in approx.ordering.items()
                }
                for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                    weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)
                    w_shape = layer_params.weight.shape
                    b_shape = layer_params.bias.shape
                    w_mu = approx_posterior_mapping[weight_layer_params_name][0].reshape(w_shape)
                    w_sigma = approx_posterior_mapping[weight_layer_params_name][1].reshape(w_shape)
                    b_mu = approx_posterior_mapping[bias_layer_params_name][0].reshape(b_shape)
                    b_sigma = approx_posterior_mapping[bias_layer_params_name][1].reshape(b_shape)
                    layer_params.weight = StudentTArray(
                        mu=w_mu, sigma=w_sigma, nu=self.model_params.bnn_layer_params[layer_ind].weight.nu
                    )
                    layer_params.bias = StudentTArray(
                        mu=b_mu, sigma=b_sigma, nu=self.model_params.bnn_layer_params[layer_ind].bias.nu
                    )
                    self.model_params.bnn_layer_params[layer_ind] = layer_params
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs["trace"])

                for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                    weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)

                    w_mu = np.mean(trace[weight_layer_params_name], axis=0)
                    w_sigma = np.std(trace[weight_layer_params_name], axis=0)
                    layer_params.weight = StudentTArray(
                        mu=w_mu.tolist(),
                        sigma=w_sigma.tolist(),
                        nu=self.model_params.bnn_layer_params[layer_ind].weight.nu,
                    )

                    b_mu = np.mean(trace[bias_layer_params_name], axis=0)
                    b_sigma = np.std(trace[bias_layer_params_name], axis=0)
                    layer_params.bias = StudentTArray(
                        mu=b_mu.tolist(),
                        sigma=b_sigma.tolist(),
                        nu=self.model_params.bnn_layer_params[layer_ind].bias.nu,
                    )
            else:
                raise ValueError("Invalid update method.")

    @classmethod
    def cold_start(
        cls,
        n_features: PositiveInt,
        hidden_dim_list: Optional[List[PositiveInt]] = None,
        update_method: UpdateMethods = "VI",
        update_kwargs: Optional[dict] = None,
        dist_params_init: Optional[Dict[str, float]] = None,
        activation: ActivationFunctions = "tanh",
        use_residual_connections: bool = False,
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
        activation : str
            The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
        use_residual_connections : bool
            Whether to use residual connections in the network (default is False).
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
        return cls(
            model_params=model_params,
            update_method=update_method,
            update_kwargs=update_kwargs,
            activation=activation,
            use_residual_connections=use_residual_connections,
            **kwargs,
        )

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


class BaseBayesianNeuralNetworkMO(ModelMO, ABC):
    """
    Base class for Bayesian Neural Network with multi-objective.

    Parameters
    ----------
    models : List[BayesianNeuralNetwork]
        The list of Bayesian Neural Network models for each objective.
    """

    if pydantic_version == PYDANTIC_VERSION_1:
        models: conlist(BayesianNeuralNetwork, min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        models: conlist(BayesianNeuralNetwork, min_length=1)
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    def model_post_init(self, __context: Any) -> None:
        """
        Validate that all models have the same number of features.
        """
        n_features = self.models[0].input_dim
        for model in self.models[1:]:
            if model.input_dim != n_features:
                raise ValueError(f"All models must have the same number of features: {model.input_dim} != {n_features}")

    @classmethod
    def cold_start(
        cls,
        n_objectives: PositiveInt,
        n_features: PositiveInt,
        hidden_dim_list: Optional[List[PositiveInt]] = None,
        update_method: UpdateMethods = "VI",
        update_kwargs: Optional[dict] = None,
        dist_params_init: Optional[Dict[str, float]] = None,
        activation: ActivationFunctions = "tanh",
        use_residual_connections: bool = False,
        **kwargs,
    ) -> Self:
        """
        Initialize a multi-objective Bayesian Neural Network with a cold start.

        Parameters
        ----------
        n_objectives : PositiveInt
            Number of objectives (models) to create.
        n_features : PositiveInt
            Number of input features for each network.
        hidden_dim_list : Optional[List[PositiveInt]], optional
            List of dimensions for the hidden layers of each network.
        update_method : UpdateMethods
            Method to update the networks.
        update_kwargs : Optional[dict], optional
            Additional keyword arguments for the update method.
        dist_params_init : Optional[Dict[str, float]], optional
            Initial distribution parameters for the network weights and biases.
        activation : str
            The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
        use_residual_connections : bool
            Whether to use residual connections in the network (default is False).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        BayesianNeuralNetworkMO
            A multi-objective BNN with the specified number of objectives.
        """

        models = [
            BayesianNeuralNetwork.cold_start(
                n_features=n_features,
                hidden_dim_list=hidden_dim_list,
                update_method=update_method,
                update_kwargs=update_kwargs,
                dist_params_init=dist_params_init,
                activation=activation,
                use_residual_connections=use_residual_connections,
            )
            for _ in range(n_objectives)
        ]
        return cls(models=models, **kwargs)


class BayesianNeuralNetworkMO(BaseBayesianNeuralNetworkMO):
    """
    Bayesian Neural Network model for multi-objective.

    Parameters
    ----------
    models : List[BayesianNeuralNetwork]
        The list of Bayesian Neural Network models for each objective.
    """


class BayesianNeuralNetworkMOCC(BaseBayesianNeuralNetworkMO, ModelMO, ModelCC):
    """
    Bayesian Neural Network model for multi-objective with cost control.

    Parameters
    ----------
    models : List[BayesianNeuralNetwork]
        The list of Bayesian Neural Network models for each objective.
    cost : NonNegativeFloat
        Cost associated to the Bayesian Neural Network model.
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
