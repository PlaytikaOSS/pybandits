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
from math import ceil
from random import betavariate
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pymc
from numpy import sqrt
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, Minibatch, fit, math, sample
from pymc import Model as PymcModel
from pymc import Normal as PymcNormal
from pymc import StudentT as PymcStudentT
from pymc.variational.callbacks import CheckParametersConvergence
from scipy.special import erf
from scipy.stats import norm, t
from typing_extensions import Self

from pybandits.base import BinaryReward, MOProbability, Probability, ProbabilityWeight, PyBanditsBaseModel
from pybandits.base_model import BaseModelCC, BaseModelMO, BaseModelSO
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
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


def _numpy_sigmoid(x):
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


class BaseLocationScaleArray(PyBanditsBaseModel, ABC):
    """
    Abstract base class for location-scale distribution arrays used in Bayesian Neural Networks.

    Parameters
    ----------
    mu : Union[List[float], List[List[float]]]
        The mean values of the distributions. Can be a 1D (for the layer bias term) or 2D list (for the layer weight term).
    sigma : Union[List[NonNegativeFloat], List[List[NonNegativeFloat]]]
        The scale (standard deviation) values of the distributions. Must be non-negative.
        Can be a 1D or 2D list.
    """

    mu: Union[List[float], List[List[float]]]
    sigma: Union[List[NonNegativeFloat], List[List[NonNegativeFloat]]]

    _mu_array: np.ndarray = PrivateAttr()
    _sigma_array: np.ndarray = PrivateAttr()
    _params: Dict[str, np.ndarray] = PrivateAttr()
    _pymc_class: ClassVar[type]
    _sampler: ClassVar[Callable]
    param_map: ClassVar[Dict[str, str]] = {"mu": "loc", "sigma": "scale"}

    def to_pymc_distribution(self, name: str, shape: Tuple[PositiveInt, ...]) -> Union[PymcStudentT, PymcNormal]:
        """
        Create a PyMC distribution from this prior distribution array.

        Parameters
        ----------
        name : str
            Name for the PyMC random variable.
        shape : Tuple[PositiveInt, ...]
            Shape of the distribution array.

        Returns
        -------
        Union[PymcStudentT, PymcNormal]
            A PyMC distribution instance.
        """
        if self._pymc_class is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define _pymc_class ClassVar to specify the PyMC distribution class."
            )
        return self._pymc_class(name=name, shape=shape, **self.params)

    def with_dist_parameters(self, **kwargs) -> "BaseLocationScaleArray":
        """
        Create a new instance with updated distribution parameters.

        Parameters
        ----------
        **kwargs
            Parameters to update (e.g., `mu`, `sigma`, `nu` for StudentTArray).
            If empty, returns self unchanged.

        Returns
        -------
        BaseLocationScaleArray
            A new instance with the updated parameters.
        """
        if not kwargs:
            return self

        # Convert to dict, update with new parameters, and validate to create new instance
        updated_dict = self.apply_version_adjusted_method("model_dump", "dict")
        updated_dict.update(kwargs)
        if pydantic_version == PYDANTIC_VERSION_1:
            return self.__class__(**updated_dict)
        elif pydantic_version == PYDANTIC_VERSION_2:
            return self.__class__.model_validate(updated_dict)
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    def _to_sampler_kwargs(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {self.param_map[k]: v for k, v in params.items()}

    def sample_rvs(self, size: Tuple[int, ...]) -> np.ndarray:
        """
        Sample random variates from this distribution.

        Parameters
        ----------
        size : Tuple[int, ...]
            Shape of the output array.

        Returns
        -------
        np.ndarray
            Array of sampled values.
        """
        return self._sampler(size=size, **self._to_sampler_kwargs(self._params))

    def sample_at_indices(self, indices: Union[List[NonNegativeInt], np.ndarray]) -> np.ndarray:
        """
        Sample one row-vector per entry in ``indices`` from a 2-D distribution matrix.

        For each ``i``, draws independently from the distribution at row ``indices[i]``.
        This is equivalent to ``sample_rvs(size=(len(indices), *row_shape))[np.arange(len(indices)), indices]``
        but allocates only ``O(len(indices) × ncols)`` memory instead of
        ``O(len(indices) × nrows × ncols)``.

        Parameters
        ----------
        indices : Union[List[NonNegativeInt], np.ndarray] of shape (n,) with dtype int
            Row indices to sample from.

        Returns
        -------
        np.ndarray of shape (n, ncols)
            Sampled instances.
        """
        sliced = {k: v[indices] for k, v in self._params.items()}
        return self._sampler(**self._to_sampler_kwargs(sliced))

    @staticmethod
    def maybe_convert_list_to_array(input_list: Union[List[float], List[List[float]]]) -> np.ndarray:
        """
        Convert a list or list of lists to a numpy array.

        Parameters
        ----------
        input_list : Union[List[float], List[List[float]]]
            Input list to convert.

        Returns
        -------
        np.ndarray
            Converted numpy array.

        Raises
        ------
        ValueError
            If the input list is not a valid 1D or 2D list.
        """
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
        """
        Validate that all array-like parameters have the same shape.

        Parameters
        ----------
        values : dict or BaseLocationScaleArray instance
            Dictionary of field values or an already-instantiated object.

        Returns
        -------
        dict or BaseLocationScaleArray instance
            Validated values dictionary or the object itself if already instantiated.

        Raises
        ------
        ValueError
            If array-like parameters have different shapes or empty dimensions.
        """
        # If values is already an instance of this class or a subclass, return it as-is
        if isinstance(values, cls):
            return values

        # If values is not a dict, it might be another type - return as-is for Pydantic to handle
        if not isinstance(values, dict):
            return values

        # Find all array-like values (lists or numpy arrays) that need shape validation
        array_like_keys = []
        array_like_values = []
        arrays = []

        for key, value in values.items():
            # Skip None values and non-array-like types
            if value is None:
                continue
            if isinstance(value, (list, np.ndarray)):
                array_like_keys.append(key)
                array_like_values.append(value)
                # Convert to array for shape comparison
                if isinstance(value, list):
                    arr = cls.maybe_convert_list_to_array(value)
                else:
                    arr = value
                arrays.append(arr)

        # If we have array-like values, validate they all have the same shape
        if arrays:
            reference_shape = arrays[0].shape

            # Check all arrays have the same shape
            for key, arr in zip(array_like_keys, arrays):
                if arr.shape != reference_shape:
                    raise ValueError(
                        f"All array-like parameters must have the same shape, but {key} has shape {arr.shape} "
                        f"while the reference shape is {reference_shape}."
                    )

            # Check no empty dimensions
            if any(dim_len == 0 for dim_len in reference_shape):
                param_names = ", ".join(array_like_keys)
                raise ValueError(
                    f"All array-like parameters ({param_names}) must have at least one element in every dimension."
                )

            # Convert numpy arrays back to lists for Pydantic
            for key, value in zip(array_like_keys, array_like_values):
                if isinstance(value, np.ndarray):
                    values[key] = value.tolist()

        return values

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
        self._params = dict(mu=self._mu_array, sigma=self._sigma_array)

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
            Dictionary containing 'mu' and 'sigma' as numpy arrays.
        """
        return self._params

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another distribution array.

        Parameters
        ----------
        other : Any
            Other object to compare with.

        Returns
        -------
        bool
            True if distributions are equal, False otherwise.
        """
        if not isinstance(other, BaseLocationScaleArray):
            return False
        return (
            np.all(self._mu_array == other._mu_array)
            and np.all(self._sigma_array == other._sigma_array)
            and type(self) is type(other)
        )

    @classmethod
    def cold_start(
        cls,
        shape: Union[PositiveInt, Tuple[PositiveInt, ...]],
        mu: float = 0.0,
        sigma: NonNegativeFloat = 10.0,
        use_layerwise_scaling: bool = False,
        **kwargs,
    ) -> "BaseLocationScaleArray":
        """
        Template method for cold start initialization.

        Common logic for shape normalization, validation, and parameter array creation
        is handled here. Subclasses override `_get_distribution_specific_params` to
        provide distribution-specific parameters.

        Parameters
        ----------
        shape : Union[PositiveInt, Tuple[PositiveInt, ...]]
            Dimensions of the distribution array.
        mu : float
            Mean of the distribution, by default 0.0.
        sigma : NonNegativeFloat
            Standard deviation of the distribution, by default 10.0.
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
            When applied, the sigma is scaled by the square root of the input dimension.
            This is useful to enable smoother convergence with Gaussian Process-like behavior.
        **kwargs
            Additional keyword arguments for distribution-specific parameters
            (e.g., `nu` for StudentTArray).

        Returns
        -------
        BaseLocationScaleArray
            An instance of the distribution array with the specified parameters.

        Raises
        ------
        ValueError
            If shape has empty dimensions.
        """
        if isinstance(shape, int):
            shape = (shape,)

        if any(dim_len == 0 for dim_len in shape):
            raise ValueError("shape must have at least one element in every dimension.")

        mu_array = np.full(shape, mu)
        sigma_array = np.full(shape, sigma / np.sqrt(shape[0]) if use_layerwise_scaling else sigma)

        # Get distribution-specific parameters from subclass
        dist_params = cls._get_distribution_specific_params(shape, **kwargs)

        return cls(mu=mu_array, sigma=sigma_array, **dist_params)

    @classmethod
    @abstractmethod
    def _get_distribution_specific_params(cls, shape: Tuple[PositiveInt, ...], **kwargs) -> Dict[str, np.ndarray]:
        """
        Get distribution-specific parameters for cold start initialization.

        Subclasses must implement this method to provide distribution-specific
        parameters (e.g., `nu` for StudentTArray).

        Note: Subclasses can omit `**kwargs` from their signature if they don't
        need to accept additional parameters. This provides stricter validation
        by raising TypeError for unexpected arguments.

        Parameters
        ----------
        shape : Tuple[PositiveInt, ...]
            Shape of the distribution array.
        **kwargs
            Additional keyword arguments containing distribution-specific parameters.
            May be omitted in subclass implementations if not needed.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping parameter names to numpy arrays of the specified shape.
        """
        pass


class StudentTArray(BaseLocationScaleArray):
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

    nu: Union[List[PositiveFloat], List[List[PositiveFloat]]]

    _nu_array: np.ndarray = PrivateAttr()
    _pymc_class: ClassVar[type] = PymcStudentT
    _sampler: ClassVar[Callable] = t.rvs
    param_map: ClassVar[Dict[str, str]] = {**BaseLocationScaleArray.param_map, "nu": "df"}

    @model_validator(mode="before")
    @classmethod
    def validate_input_shapes(cls, values):
        # The parent class method is now generic and handles all array-like parameters
        # including mu, sigma, and nu, so we can just call it directly
        return super().validate_input_shapes(values)

    def __eq__(self, other: Any) -> bool:
        """Check equality including nu parameter."""
        if not isinstance(other, StudentTArray):
            return False
        return super().__eq__(other) and np.all(self._nu_array == other._nu_array)

    @classmethod
    def _get_distribution_specific_params(
        cls, shape: Tuple[PositiveInt, ...], nu: PositiveFloat = 5.0
    ) -> Dict[str, np.ndarray]:
        """
        Get distribution-specific parameters for Student's t-distribution.

        Parameters
        ----------
        shape : Tuple[PositiveInt, ...]
            Shape of the distribution array.
        nu : PositiveFloat
            Degrees of freedom of the Student's t-distribution, by default 5.0.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'nu' parameter as a numpy array.
        """
        return {"nu": np.full(shape, nu)}

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize private numpy array attributes by converting lists to arrays once at initialization.

        Parameters
        ----------
        __context : Any
            Pydantic context (unused).
        """
        super().model_post_init(__context)
        self._nu_array = np.array(self.nu)
        self._params["nu"] = self._nu_array

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


class NormalArray(BaseLocationScaleArray):
    """
    A class representing an array of Normal distributions with parameters `mu` and `sigma`.
    A specific element (e.g, a single parameter of a layer) distribution is defined by the corresponding elements in the lists.
    The mean values are represented by `mu` and the standard deviation values by `sigma`.

    Normal distributions are simpler and faster than Student-t distributions, but less robust to outliers.
    They provide standard L2-like regularization.

    Parameters
    ----------
    mu : Union[List[float], List[List[float]]]
        The mean values of the Normal distributions. Can be a 1D (for the layer bias term) or 2D list (for the layer weight term).
    sigma : Union[List[NonNegativeFloat], List[List[NonNegativeFloat]]]
        The standard deviation values of the Normal distributions. Must be non-negative.
        Can be a 1D or 2D list.

    Examples
    --------
    >>> # Create NormalArray with default parameters
    >>> normal = NormalArray.cold_start(shape=(10, 5), mu=0.0, sigma=1.0)
    >>> # Use in BNN
    >>> bnn = BayesianNeuralNetwork.cold_start(
    ...     n_features=10,
    ...     dist_type="normal",
    ...     dist_params_init={"mu": 0, "sigma": 1}
    ... )
    """

    _pymc_class: ClassVar[type] = PymcNormal
    _sampler: ClassVar[Callable] = norm.rvs

    @classmethod
    def _get_distribution_specific_params(cls, shape: Tuple[PositiveInt, ...]) -> Dict[str, np.ndarray]:
        """
        Get distribution-specific parameters for Normal distribution.

        Normal distributions only require mu and sigma, which are handled
        by the base class, so this method returns an empty dictionary.

        Parameters
        ----------
        shape : Tuple[PositiveInt, ...]
            Shape of the distribution array.

        Returns
        -------
        Dict[str, np.ndarray]
            Empty dictionary (no additional parameters needed for Normal distribution).
        """
        return {}


class CategoricalFeatureConfig(PyBanditsBaseModel):
    """
    Configuration for a single categorical feature with Bayesian embedding.

    The caller is responsible for pre-encoding categorical values as integer indices
    in the range ``[0, cardinality)``.

    Parameters
    ----------
    column_index : NonNegativeInt
        Column position of this feature in the input numpy array.
    cardinality : PositiveInt
        Number of distinct integer category codes. The context array must contain
        pre-encoded integer indices in the range ``[0, cardinality)``.
    embedding_dim : PositiveInt
        Dimensionality of the embedding vector for this feature. Default is 8.
    """

    column_index: NonNegativeInt
    cardinality: PositiveInt
    embedding_dim: PositiveInt


class FeaturesConfig(PyBanditsBaseModel):
    """
    Specification of the structure of a numpy context array.

    Columns can appear in any order. Categorical features are identified by their
    explicit ``column_index``; all remaining columns are treated as numerical.

    Parameters
    ----------
    n_features : int
        Total number of columns in the input numpy array. Default 0.
    categorical_features_configs : List[CategoricalFeatureConfig]
        List of categorical feature configurations.
    """

    n_features: NonNegativeInt = 0
    categorical_features_configs: List[CategoricalFeatureConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _validate_categorical_columns(cls, values):
        if not isinstance(values, dict):
            return values
        cat_configs = values.get("categorical_features_configs", [])
        n_features = values.get("n_features", 0)
        col_indices = [cfg["column_index"] if isinstance(cfg, dict) else cfg.column_index for cfg in cat_configs]
        if len(col_indices) != len(set(col_indices)):
            raise ValueError("Duplicate column_index values found in categorical_features_configs.")
        for col_index in col_indices:
            if col_index >= n_features:
                raise ValueError(f"column_index {col_index} is out of range for n_features={n_features}.")
        return values

    @property
    def numerical_indices(self) -> List[int]:
        """Sorted list of column positions treated as numerical (not used by any categorical)."""
        cat_cols = {cfg.column_index for cfg in self.categorical_features_configs}
        return [i for i in range(self.n_features) if i not in cat_cols]

    @property
    def n_numerical(self) -> int:
        """Number of numerical columns (derived: n_features minus categorical count)."""
        return self.n_features - len(self.categorical_features_configs)

    @property
    def has_categorical(self) -> bool:
        """True if at least one categorical feature is configured."""
        return bool(self.categorical_features_configs)

    @property
    def total_output_dim(self) -> int:
        """
        Total dimensionality of the concatenated vector fed into the first BNN layer.

        = n_numerical + sum(cat.embedding_dim)
        """
        return self.n_numerical + sum(cfg.embedding_dim for cfg in self.categorical_features_configs)


class EmbeddingParams(PyBanditsBaseModel):
    """
    Stores Bayesian embedding matrices for all categorical features.

    Each embedding matrix has shape ``(cardinality, embedding_dim)`` and is stored as a
    ``BaseLocationScaleArray`` (StudentTArray or NormalArray) — the same representation
    used for layer weights in ``BnnLayerParams``.

    Parameters
    ----------
    embeddings : List[Union[StudentTArray, NormalArray]]
        Ordered list of embedding matrix distributions, matching the order of
        ``FeaturesConfig.categorical_features_configs``.
        Shape of each matrix: ``(cardinality, embedding_dim)``.
    embeddings_init : List[Union[StudentTArray, NormalArray]]
        Frozen copy of the initial embeddings for resetting. Set automatically.
    """

    embeddings: List[Union[StudentTArray, NormalArray]]
    embeddings_init: List[Union[StudentTArray, NormalArray]] = Field(default_factory=list, init=False, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _set_embeddings_init(cls, values):
        if isinstance(values, dict):
            if not values.get("embeddings_init"):
                values["embeddings_init"] = deepcopy(values.get("embeddings", []))
        return values

    @classmethod
    def cold_start(
        cls,
        feature_config: "FeaturesConfig",
        dist_class: type[BaseLocationScaleArray] = StudentTArray,
        **dist_params_init,
    ) -> "EmbeddingParams":
        """
        Create ``EmbeddingParams`` with prior distributions for all categorical features.

        Parameters
        ----------
        feature_config : FeaturesConfig
            Feature configuration containing categorical feature specs.
        dist_class : type
            The distribution class to use for embedding priors, by default ``StudentTArray``.
        **dist_params_init
            Distribution parameters passed to ``BaseLocationScaleArray.cold_start``
            (e.g. ``mu``, ``sigma``, ``nu`` for StudentT; ``mu``, ``sigma`` for Normal).

        Returns
        -------
        EmbeddingParams
            An ``EmbeddingParams`` instance with one embedding matrix per categorical feature,
            each of shape ``(cardinality, embedding_dim)``, initialised from ``dist_class`` cold-start priors.
        """
        embeddings = []
        for cat_cfg in feature_config.categorical_features_configs:
            shape = (cat_cfg.cardinality, cat_cfg.embedding_dim)
            embeddings.append(dist_class.cold_start(shape=shape, **dist_params_init))
        return cls(embeddings=embeddings)


class BnnLayerParams(PyBanditsBaseModel):
    """
    Represents the parameters of a Bayesian neural network (BNN) layer.

    Parameters
    ----------
    weight : Union[NormalArray, StudentTArray]
        The weight parameter of the BNN layer, represented as either a NormalArray or StudentTArray.
    bias : Union[StudentTArray, NormalArray]
        The bias parameter of the BNN layer, represented as either a StudentTArray or NormalArray.
    """

    weight: Union[NormalArray, StudentTArray]
    bias: Union[StudentTArray, NormalArray]


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
    embedding_params : Optional[EmbeddingParams]
        Bayesian embedding matrices for categorical features. ``None`` when no
        categorical features are configured.
    embedding_params_init : Optional[EmbeddingParams]
        Frozen copy of the initial embedding parameters for resetting. Set automatically.
    """

    bnn_layer_params: Optional[List[BnnLayerParams]]
    bnn_layer_params_init: List[BnnLayerParams] = Field(default_factory=list, init=False, frozen=True)
    embedding_params: Optional[EmbeddingParams] = None
    embedding_params_init: Optional[EmbeddingParams] = Field(default=None, init=False, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, values):
        if values.get("bnn_layer_params_init") is None:
            values["bnn_layer_params_init"] = deepcopy(values["bnn_layer_params"])
        if values.get("embedding_params_init") is None and values.get("embedding_params") is not None:
            values["embedding_params_init"] = deepcopy(values["embedding_params"])

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
        For VI, it contains 'fit' settings and additional parameters like 'epochs', 'optimizer_type',
        'optimizer_kwargs', 'batch_size', and 'early_stopping_kwargs'. The 'epochs' parameter specifies
        the number of iterations for VI (maps to 'n' in PyMC's fit function).
    activation : str, optional
        The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
    use_residual_connections : bool, optional
        Whether to use residual connections in the network. Residual connections are only added when
        the layer output dimension is greater than or equal to the input dimension (default is False).
    early_stopping_config : Optional[EarlyStoppingConfig], optional
        Configuration for early stopping during VI training. If None, no early stopping is used (default is None).
        Only applicable when update_method is "VI".

    Examples
    --------
    >>> # Create BNN with Student-t priors (default)
    >>> bnn = BayesianNeuralNetwork.cold_start(
    ...     n_features=2,
    ...     hidden_dim_list=[16, 16],
    ...     dist_type="studentt",
    ...     dist_params_init={"mu": 0, "sigma": 1, "nu": 5}
    ... )
    >>> # Create BNN with Normal priors
    >>> bnn = BayesianNeuralNetwork.cold_start(
    ...     n_features=2,
    ...     hidden_dim_list=[16, 16],
    ...     dist_type="normal",
    ...     dist_params_init={"mu": 0, "sigma": 1}
    ... )

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
    weight_var_name: ClassVar[str] = "weight"
    _bias_var_name: ClassVar[str] = "bias"
    _embedding_var_name: ClassVar[str] = "embedding"
    _vi_update_params: ClassVar[list] = [
        "optimizer_type",
        "optimizer_kwargs",
        "batch_size",
        "early_stopping_kwargs",
        "epochs",
    ]
    _distribution_mapping: ClassVar[Dict[str, type]] = {"normal": NormalArray, "studentt": StudentTArray}
    _embedding_dim_divisor: ClassVar[int] = 4
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
        "sigmoid": _numpy_sigmoid,
        "gelu": _numpy_gelu,
    }

    update_method: UpdateMethods = "VI"
    update_kwargs: Optional[dict] = None
    activation: ActivationFunctions = "tanh"
    use_residual_connections: bool = False
    feature_config: FeaturesConfig

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
    _obj_optimizer: Optional[Callable] = PrivateAttr(None)
    _early_stopping_callback: Optional[CheckParametersConvergence] = PrivateAttr(None)
    _update_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    _transfer_learned_keys: ClassVar[Tuple[str, ...]] = ("model_params",)
    _transfer_extendable_keys: ClassVar[Tuple[str, ...]] = ("update_method",)
    _transfer_structural_keys: ClassVar[Tuple[str, ...]] = ("activation", "use_residual_connections")

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v):
        if v not in cls._pymc_activations.keys():
            raise ValueError(
                f"Invalid activation function: {v}. Supported activations are: {list(cls._pymc_activations.keys())}"
            )
        return v

    @classmethod
    def get_embedding_var_name(cls, feat_index: int) -> str:
        """Return the PyMC variable name for a categorical embedding matrix."""
        return f"{cls._embedding_var_name}_{feat_index}"

    @property
    def approx_history(self) -> Optional[np.ndarray]:
        return self._approx_history

    def _prepare_context_arrays(self, context: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Split a numpy context array into numerical and categorical index arrays.

        Numerical columns are all columns not assigned to a categorical feature.
        Categorical columns are extracted by their explicit ``column_index``.

        Parameters
        ----------
        context : np.ndarray of shape (n_samples, input_dim)
            Input array.

        Returns
        -------
        numerical_arr : np.ndarray of shape (n_samples, n_numerical)
            Numerical features, same dtype as input. Empty ``(n_samples, 0)`` if none.
        cat_indices_dict : Dict[int, np.ndarray]
            Mapping from feature index (0-based) to int32 array of shape ``(n_samples,)``
            containing integer category codes.
        """
        n_samples = len(context)
        numeric_indices = self.feature_config.numerical_indices

        if numeric_indices:
            numerical_arr = context[:, numeric_indices]
        else:
            numerical_arr = np.empty((n_samples, 0), dtype=context.dtype)

        configs = self.feature_config.categorical_features_configs
        cat_indices_dict = {i: context[:, cfg.column_index].astype(np.int32) for i, cfg in enumerate(configs)}

        return numerical_arr, cat_indices_dict

    def _get_obj_optimizer(self) -> Optional[Callable]:
        if self.update_kwargs is None:
            return None
        optimizer_type = self.update_kwargs.get("optimizer_type", None)
        if optimizer_type is not None:
            if optimizer_type not in self._supported_optimizers:
                raise ValueError(
                    f"Invalid optimizer type: {optimizer_type}. Supported optimizers are: {self._supported_optimizers}"
                )
            optimizer = getattr(pymc, optimizer_type)
            optimizer_kwargs = self.update_kwargs.get("optimizer_kwargs", {})
            try:
                _optimizer = optimizer(**optimizer_kwargs)
            except (TypeError, ValueError, KeyError) as e:
                raise e.__class__(f"Invalid optimizer kwargs: {optimizer_kwargs}.\n{e}")
        else:
            _optimizer = None

        return _optimizer

    def _get_early_stopping_callback(self) -> Optional[CheckParametersConvergence]:
        if self.update_kwargs is None:
            return None
        early_stopping_kwargs = self.update_kwargs.get("early_stopping_kwargs", None)
        if early_stopping_kwargs is not None:
            try:
                early_stopping_callback = CheckParametersConvergence(**early_stopping_kwargs)
            except (TypeError, ValueError, KeyError) as e:
                raise e.__class__(f"Invalid early stopping kwargs: {early_stopping_kwargs}.\n{e}")
        else:
            early_stopping_callback = None
        return early_stopping_callback

    @classmethod
    def get_layer_params_name(cls, layer_ind: PositiveInt) -> Tuple[str, str]:
        weight_layer_params_name = f"{cls.weight_var_name}_{layer_ind}"
        bias_layer_params_name = f"{cls._bias_var_name}_{layer_ind}"
        return weight_layer_params_name, bias_layer_params_name

    @classmethod
    def create_model_params(
        cls,
        feature_config: FeaturesConfig,
        hidden_dim_list: Optional[List[PositiveInt]],
        use_layerwise_scaling: bool = False,
        dist_class: type[BaseLocationScaleArray] = StudentTArray,
        **dist_params_init,
    ) -> BnnParams:
        """
        Creates model parameters for a Bayesian neural network (BNN) model according to dist_params_init.
        This method initializes the distribution's parameters for each layer of a BNN
        using the specified number of features, hidden dimensions, and distribution
        initialization parameters.

        Parameters
        ----------
        feature_config : FeaturesConfig
            Full input layout description. First-layer input dimension is ``feature_config.total_output_dim``.
            ``EmbeddingParams`` are created when ``feature_config.categorical_features_configs`` is non-empty.
        hidden_dim_list : Optional[List[PositiveInt]]
            Number of hidden units per hidden layer. If None, no hidden layers are added.
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
        dist_class : type
            The distribution class to use for weights, biases, and embeddings, by default ``StudentTArray``.
        **dist_params_init : dict, optional
            Additional parameters for initializing the distribution of weights and biases.

        Returns
        -------
        BnnParams
            An instance of BnnParams containing the initialized layer parameters.
        """
        effective_n_features = feature_config.total_output_dim

        if hidden_dim_list is None:
            _dim_list = [effective_n_features]
        else:
            _dim_list = [effective_n_features] + hidden_dim_list

        _dim_list.append(1)

        layer_params_init = []
        for layer_ind in range(len(_dim_list) - 1):
            input_dim = _dim_list[layer_ind]
            output_dim = _dim_list[layer_ind + 1]
            w_param = dist_class.cold_start(
                shape=(input_dim, output_dim), use_layerwise_scaling=use_layerwise_scaling, **dist_params_init
            )
            b_param = dist_class.cold_start(shape=output_dim, **dist_params_init)
            layer_params_init.append(BnnLayerParams(weight=w_param, bias=b_param))

        if feature_config.categorical_features_configs:
            embedding_params = EmbeddingParams.cold_start(
                feature_config=feature_config,
                dist_class=dist_class,
                **dist_params_init,
            )
        else:
            embedding_params = None

        return BnnParams(
            bnn_layer_params=layer_params_init,
            embedding_params=embedding_params,
        )

    def check_context_matrix(self, context: np.ndarray):
        """
        Validate the context input.

        Context must be an array-like with numeric values and the correct number of columns.
        Categorical columns are validated to contain valid integer indices within their vocab range.

        Parameters
        ----------
        context : np.ndarray
            Matrix of contextual features of shape ``(n_samples, n_cols)``.
        """
        expected_cols = self.feature_config.n_features

        try:
            n_cols_context = context.shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {expected_cols} columns: {e}.")

        if not np.issubdtype(context.dtype, np.number):
            raise ValueError("Context array must contain only numeric values.")

        if n_cols_context != expected_cols:
            raise AttributeError(f"Shape mismatch: context must have {expected_cols} columns.")

        configs = self.feature_config.categorical_features_configs
        if configs:
            col_indices = [c.column_index for c in configs]
            cardinalities = np.array([c.cardinality for c in configs])
            cat_cols = context[:, col_indices].astype(int)
            in_range = (cat_cols >= 0) & (cat_cols < cardinalities[np.newaxis, :])
            if not np.all(in_range):
                bad_idx = int(np.argmax(~np.all(in_range, axis=0)))
                cfg = configs[bad_idx]
                raise ValueError(
                    f"Categorical feature at column {cfg.column_index} (index {bad_idx}) has indices out of range "
                    f"[0, {cfg.cardinality})."
                )

    @property
    def input_dim(self) -> PositiveInt:
        """
        Returns the number of raw context columns expected by the model.

        Returns
        -------
        PositiveInt
            Equal to ``feature_config.n_features``: the number of columns the
            context numpy array must have.  For categorical models this differs
            from the post-embedding dimension (``feature_config.total_output_dim``).
        """
        return self.feature_config.n_features

    def _arrange_update_kwargs(self):
        if self.update_kwargs is None:
            self.update_kwargs = dict()

        if self.update_method == "VI":
            # Validate that epochs and n are mutually exclusive
            has_epochs = "epochs" in self.update_kwargs
            has_n = "n" in self.update_kwargs.get("fit", {})
            if has_epochs and has_n:
                raise ValueError(
                    "Cannot specify both 'epochs' and 'n' in update_kwargs. "
                    "Use 'epochs' for epoch-based training or 'fit': {'n': ...} for iteration-based training."
                )

            self.update_kwargs["fit"] = {
                **self._default_variational_inference_fit_kwargs,
                **self.update_kwargs.get("fit", {}),
            }

        elif self.update_method == "MCMC":
            for param in self._vi_update_params:
                if param in self.update_kwargs:
                    raise ValueError(
                        f"Invalid update MCMC parameter: {param}. {self._vi_update_params} are VI parameters."
                    )

            self.update_kwargs["trace"] = {**self._default_mcmc_trace_kwargs, **self.update_kwargs.get("trace", {})}
        else:
            raise ValueError("Invalid update method.")

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize activation function PrivateAttr based on the activation setting.
        """
        # Initialize activation functions (always set to ensure they're available after model_copy)
        self._numpy_activation_fn = self._numpy_activations[self.activation]
        self._pymc_activation_fn = self._pymc_activations[self.activation]
        self._arrange_update_kwargs()
        self._obj_optimizer = self._get_obj_optimizer()
        self._early_stopping_callback = self._get_early_stopping_callback()
        self._update_kwargs = deepcopy(self.update_kwargs)
        if self._obj_optimizer is not None:
            self._update_kwargs["fit"]["obj_optimizer"] = self._obj_optimizer
        if self._early_stopping_callback is not None:
            self._update_kwargs["fit"]["callbacks"] = [self._early_stopping_callback]

    def _apply_bnn_layers(self, network_input: Any) -> Any:
        """
        Apply the BNN layer graph to the input.

        Returns the final logit (output layer pre-activation).
        """
        next_layer_input = network_input
        for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
            w_shape = layer_params.weight.shape
            b_shape = layer_params.bias.shape
            weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)
            input_dim = w_shape[0]
            output_dim = w_shape[1]

            w = layer_params.weight.to_pymc_distribution(name=weight_layer_params_name, shape=w_shape)
            b = layer_params.bias.to_pymc_distribution(name=bias_layer_params_name, shape=b_shape)

            linear_transform = math.dot(next_layer_input, w) + b

            if layer_ind < len(self.model_params.bnn_layer_params) - 1:
                activated_output = self._pymc_activation_fn(linear_transform)
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

        return linear_transform

    def _make_embedding_matrix_inputs(self, configs: List[CategoricalFeatureConfig], cat_index_tensors: List) -> List:
        """
        Build PyMC embedding variables and return their indexed outputs.

        For each categorical feature ``i``, creates a PyMC random variable for the
        ``(cardinality, embedding_dim)`` embedding matrix and indexes it with the
        corresponding integer-index tensor, yielding a ``(n_rows, embedding_dim)`` output.

        Parameters
        ----------
        configs : List[CategoricalFeatureConfig]
            Categorical feature specs (one per embedding).
        cat_index_tensors : list of PyMC-compatible tensors
            One integer-index tensor per categorical feature, in the same order as ``configs``.
            May be plain ``Data`` nodes or ``Minibatch`` slices.

        Returns
        -------
        list of PyMC tensors
            One ``(n_rows, embedding_dim)`` tensor per categorical feature.
        """
        parts = []
        for i, _ in enumerate(configs):
            emb_dist = self.model_params.embedding_params.embeddings[i]
            emb_matrix = emb_dist.to_pymc_distribution(name=self.get_embedding_var_name(i), shape=emb_dist.shape)
            parts.append(emb_matrix[cat_index_tensors[i]])
        return parts

    def _prepare_bnn_input_output(
        self,
        numerical_arr: np.ndarray,
        cat_indices_dict: Dict[int, np.ndarray],
        y: np.ndarray,
        configs: List[CategoricalFeatureConfig],
        batch_size: Optional[PositiveInt] = None,
    ) -> Tuple[List, Any, Optional[int]]:
        """
        Wrap raw arrays as PyMC tensors (``Data`` or ``Minibatch``), build embedding
        look-ups, and return everything the BNN graph needs.

        Parameters
        ----------
        numerical_arr : np.ndarray
            Numerical feature columns, shape ``(n_samples, n_numerical)``.
        cat_indices_dict : Dict[int, np.ndarray]
            Integer-index arrays for each categorical feature.
        y : np.ndarray
            Binary target values, shape ``(n_samples,)``.
        configs : list of CategoricalFeatureConfig
            Categorical feature specs.
        batch_size : Optional[PositiveInt]
            When set, arrays are wrapped in a single co-indexed ``Minibatch``.

        Returns
        -------
        input_parts : list of PyMC tensors
            Numerical and embedded-categorical inputs ready for concatenation.
        bnn_output : PyMC-compatible tensor
            Observed binary targets.
        n_total : Optional[int]
            Full dataset size (set only when using minibatch; ``None`` otherwise).
        """
        if batch_size is not None:
            n_total = len(y)
            mb_arrays = [numerical_arr] if numerical_arr.shape[1] > 0 else []
            mb_arrays += [cat_indices_dict[i] for i in range(len(configs))]
            mb_arrays.append(y)
            mb = Minibatch(*mb_arrays, batch_size=batch_size)

            offset = 0
            input_parts = []
            if numerical_arr.shape[1] > 0:
                input_parts.append(mb[offset])
                offset += 1
            cat_tensors = [mb[offset + i] for i in range(len(configs))]
            bnn_output = mb[offset + len(configs)]
        else:
            n_total = None
            bnn_output = Data("bnn_output", y)
            input_parts = [Data("bnn_numerical_input", numerical_arr)] if numerical_arr.shape[1] > 0 else []
            cat_tensors = [Data(f"cat_idx_{i}", cat_indices_dict[i]) for i in range(len(configs))]

        input_parts += self._make_embedding_matrix_inputs(configs, cat_tensors)
        return input_parts, bnn_output, n_total

    def create_update_model(
        self,
        x: ArrayLike,
        y: Union[List[BinaryReward], np.ndarray],
        batch_size: Optional[PositiveInt] = None,
    ) -> PymcModel:
        """
        Create a PyMC model for Bayesian Neural Network.

        ``x`` must be a numpy array with ``feature_config.n_features`` columns.
        Numerical columns are passed through as-is. Categorical columns (identified by their
        ``column_index``) are modelled with Bayesian embedding matrices inside the PyMC graph.

        Parameters
        ----------
        x : ArrayLike
            Input features of shape ``(n_samples, n_features)``.
        y : Union[List[BinaryReward], np.ndarray]
            Binary target values of shape ``(n_samples,)``.
        batch_size : Optional[PositiveInt]
            When set, Minibatch is used for numerical-only models.
            Minibatch with categorical embedding features is not supported.

        Returns
        -------
        PymcModel
            PyMC model object with the specified neural network architecture.
        """
        y = np.array(y, dtype=np.int32)
        numerical_arr, cat_indices_dict = self._prepare_context_arrays(x)
        configs = self.feature_config.categorical_features_configs

        with PymcModel() as _model:
            input_parts, bnn_output, n_total = self._prepare_bnn_input_output(
                numerical_arr, cat_indices_dict, y, configs, batch_size
            )

            network_input = input_parts[0] if len(input_parts) == 1 else math.concatenate(input_parts, axis=1)
            _logit = self._apply_bnn_layers(network_input)
            logit = Deterministic(self._logit_var_name, _logit.squeeze())
            bernoulli_kwargs = dict(logit_p=logit, observed=bnn_output)
            if n_total is not None:
                bernoulli_kwargs["total_size"] = n_total
            Bernoulli("out", **bernoulli_kwargs)

        return _model

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_weights(self, n_samples: PositiveInt) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample weights and biases for each sample and each layer.

        Parameters
        ----------
        n_samples : PositiveInt
            The number of samples (users) to draw weights for. Must be positive.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            A list of length num_layers, where each element is (weights, biases) for that layer.
            - weights shape: (n_samples, input_dim, output_dim)
            - biases shape: (n_samples, output_dim)
        """
        sampled_weights = []
        for layer_params in self.model_params.bnn_layer_params:
            input_dim = layer_params.weight.shape[0]
            output_dim = layer_params.weight.shape[1]

            w = layer_params.weight.sample_rvs(size=(n_samples, input_dim, output_dim))
            b = layer_params.bias.sample_rvs(size=(n_samples, output_dim))

            sampled_weights.append((w, b))

        return sampled_weights

    def sample_embeddings(self, context: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Sample embedding vectors for each categorical feature given the context.

        For each categorical feature, extracts the integer indices from the context
        and samples from the corresponding embedding distribution at those indices.

        Parameters
        ----------
        context : np.ndarray
            Context matrix, shape ``(n_samples, feature_config.n_features)``.
            Categorical columns contain integer indices into the embedding vocabulary.

        Returns
        -------
        Optional[List[np.ndarray]]
            One array per categorical feature, each of shape ``(n_samples, emb_dim)``.
            ``None`` when the model has no categorical features.
        """
        if not self.feature_config.has_categorical:
            return None
        _context = np.atleast_2d(context)
        _, cat_indices_dict = self._prepare_context_arrays(_context)
        return [
            self.model_params.embedding_params.embeddings[i]
            .sample_at_indices(cat_indices_dict[i])
            .reshape(len(_context), -1)
            for i in range(len(self.feature_config.categorical_features_configs))
        ]

    @staticmethod
    def extract_sample(
        sampled_weights: List[Tuple[np.ndarray, np.ndarray]],
        sampled_embeddings: Optional[List[np.ndarray]],
        sample_idx: NonNegativeInt,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[List[np.ndarray]]]:
        """
        Extract the weights, biases, and embeddings for a specific sample.

        Parameters
        ----------
        sampled_weights : List[Tuple[np.ndarray, np.ndarray]]
            List of (weights, biases) per layer.
            Each weights has shape (n_samples, input_dim, output_dim), biases (n_samples, output_dim).
        sampled_embeddings : Optional[List[np.ndarray]]
            Pre-sampled embedding vectors, one per categorical feature, each of shape
            ``(n_samples, emb_dim)``. ``None`` when no categorical features.
        sample_idx : NonNegativeInt
            The index of the sample to extract.

        Returns
        -------
        Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[List[np.ndarray]]]
            ``(weights_idx, embeddings_idx)`` sliced to a single sample (batch dim = 1).
        """
        weights_idx = [(w[sample_idx : sample_idx + 1], b[sample_idx : sample_idx + 1]) for w, b in sampled_weights]
        embeddings_idx = (
            [ev[sample_idx : sample_idx + 1] for ev in sampled_embeddings] if sampled_embeddings is not None else None
        )
        return weights_idx, embeddings_idx

    def _prepare_forward_input(self, context: np.ndarray, sampled_embeddings: Optional[List[np.ndarray]]) -> np.ndarray:
        """
        Replace categorical columns with pre-sampled embedding vectors.

        Parameters
        ----------
        context : np.ndarray
            Context matrix, shape ``(n_samples, feature_config.n_features)``.
        sampled_embeddings : Optional[List[np.ndarray]]
            One array per categorical feature, each of shape ``(n_samples, emb_dim)``.
            ``None`` when the model has no categorical features.

        Returns
        -------
        np.ndarray
            Network input with categorical columns replaced by embedding vectors,
            shape ``(n_samples, feature_config.total_output_dim)``.
        """
        if sampled_embeddings is None:
            return context
        _context = np.atleast_2d(context)
        numerical_arr, _ = self._prepare_context_arrays(_context)
        parts = [numerical_arr] if numerical_arr.shape[1] > 0 else []
        parts.extend(sampled_embeddings)
        return np.concatenate(parts, axis=1)

    def forward_pass(
        self,
        sampled_weights: List[Tuple[np.ndarray, np.ndarray]],
        context: np.ndarray,
        sampled_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[ProbabilityWeight]:
        """
        Apply the neural network forward pass using pre-sampled weights, biases, and embeddings.

        All stochastic parameters must be sampled externally (via ``sample_weights`` and
        ``sample_embeddings``) before calling this method.

        Parameters
        ----------
        sampled_weights : List[Tuple[np.ndarray, np.ndarray]]
            List of (weights, biases) per layer from ``sample_weights``.
            Each weights has shape (n_samples, input_dim, output_dim), biases (n_samples, output_dim).
        context : np.ndarray
            Context matrix, shape (n_samples, feature_config.n_features).
            Categorical columns contain integer indices into the embedding vocabulary.
        sampled_embeddings : Optional[List[np.ndarray]]
            Pre-sampled embedding vectors from ``sample_embeddings``, one array per
            categorical feature, each of shape ``(n_samples, emb_dim)``.
            ``None`` when the model has no categorical features.

        Returns
        -------
        List[ProbabilityWeight]
            Each element is (probability, weighted_sum) per sample.
        """
        next_layer_input = self._prepare_forward_input(context, sampled_embeddings)
        n_layers = len(self.model_params.bnn_layer_params)

        for layer_ind, (w, b) in enumerate(sampled_weights):
            layer_params = self.model_params.bnn_layer_params[layer_ind]
            input_dim = layer_params.weight.shape[0]
            output_dim = layer_params.weight.shape[1]

            # Linear transformation: same as sample_proba
            linear_transform = np.einsum("...i,...ij->...j", next_layer_input, w) + b

            # Apply activation function for hidden layers
            if layer_ind < n_layers - 1:
                # Apply activation function
                activated_output = self._numpy_activation_fn(linear_transform)
                # Add residual connection if enabled and dimensions allow
                if self.use_residual_connections and output_dim >= input_dim:
                    if output_dim == input_dim:
                        next_layer_input = activated_output + next_layer_input
                    else:
                        residual_padded = np.pad(
                            next_layer_input,
                            ((0, 0), (0, output_dim - input_dim)),
                            mode="constant",
                            constant_values=0,
                        )
                        next_layer_input = activated_output + residual_padded
                else:
                    next_layer_input = activated_output
            else:
                weighted_sum = linear_transform.squeeze(-1)
                prob = _numpy_sigmoid(weighted_sum)

        return list(zip(prob, weighted_sum))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: np.ndarray) -> List[ProbabilityWeight]:
        """
        Samples probabilities and logits from the prior predictive distribution.

        Parameters
        ----------
        context : np.ndarray
            The context matrix for which the probabilities are to be sampled.

        Returns
        -------
        List[ProbabilityWeight]
            Each element is a tuple containing the probability of a positive reward and
            the network logit.
        """
        self.check_context_matrix(context=context)

        _context = np.atleast_2d(context)
        n_samples = len(_context)

        sampled_weights = self.sample_weights(n_samples)
        sampled_embeddings = self.sample_embeddings(_context)

        return self.forward_pass(
            sampled_weights=sampled_weights, context=_context, sampled_embeddings=sampled_embeddings
        )

    def _create_updated_layer_params(
        self,
        w_mu: np.ndarray,
        w_sigma: np.ndarray,
        b_mu: np.ndarray,
        b_sigma: np.ndarray,
        original_weight: BaseLocationScaleArray,
        original_bias: BaseLocationScaleArray,
    ) -> BnnLayerParams:
        """
        Create updated layer parameters with new mu and sigma values, preserving distribution type.

        Parameters
        ----------
        w_mu : np.ndarray
            Updated weight mean values.
        w_sigma : np.ndarray
            Updated weight standard deviation values.
        b_mu : np.ndarray
            Updated bias mean values.
        b_sigma : np.ndarray
            Updated bias standard deviation values.
        original_weight : BaseLocationScaleArray
            Original weight distribution to preserve type and nu (if StudentTArray).
        original_bias : BaseLocationScaleArray
            Original bias distribution to preserve type and nu (if StudentTArray).

        Returns
        -------
        BnnLayerParams
            New layer parameters with updated values.
        """

        updated_weight = original_weight.with_dist_parameters(mu=w_mu.tolist(), sigma=w_sigma.tolist())
        updated_bias = original_bias.with_dist_parameters(mu=b_mu.tolist(), sigma=b_sigma.tolist())

        return BnnLayerParams(weight=updated_weight, bias=updated_bias)

    def _update_embedding_params_from_mapping(self, mu_sigma_mapping: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Update embedding matrices in-place from the VI approximate posterior.

        After variational inference converges, the approximate posterior is available as a flat mapping
        ``{var_name: (mu_flat, sigma_flat)}``.  For each categorical feature ``i``, this method looks up
        the corresponding variable (``emb_0``, ``emb_1``, …), reshapes the flat posterior arrays back to
        ``(cardinality, embedding_dim)``, and replaces the prior distribution parameters with the
        posterior mean and standard deviation via ``with_dist_parameters``.

        Parameters
        ----------
        mu_sigma_mapping : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Mapping from PyMC variable name to a ``(mu_flat, sigma_flat)`` pair of 1-D arrays
            as returned by the VI approximation posterior.
        """
        if self.model_params.embedding_params is None:
            return
        updated_embeddings = []
        for i, orig_emb in enumerate(self.model_params.embedding_params.embeddings):
            emb_var_name = self.get_embedding_var_name(i)
            emb_shape = orig_emb.shape
            emb_mu = mu_sigma_mapping[emb_var_name][0].reshape(emb_shape)
            emb_sigma = mu_sigma_mapping[emb_var_name][1].reshape(emb_shape)
            updated_embeddings.append(orig_emb.with_dist_parameters(mu=emb_mu.tolist(), sigma=emb_sigma.tolist()))
        self.model_params.embedding_params.embeddings = updated_embeddings

    def _update_embedding_params_from_trace(self, trace) -> None:
        """Update embedding matrices from an MCMC trace (mean/std over samples)."""
        if self.model_params.embedding_params is None:
            return
        updated_embeddings = []
        for i, orig_emb in enumerate(self.model_params.embedding_params.embeddings):
            emb_var_name = self.get_embedding_var_name(i)
            emb_values = trace[emb_var_name]
            emb_mu = np.mean(emb_values, axis=0)
            emb_sigma = np.std(emb_values, axis=0)
            updated_embeddings.append(orig_emb.with_dist_parameters(mu=emb_mu.tolist(), sigma=emb_sigma.tolist()))
        self.model_params.embedding_params.embeddings = updated_embeddings

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

        batch_size = self._update_kwargs.get("batch_size", None)
        _context = np.atleast_2d(context)
        _model = self.create_update_model(x=_context, y=rewards, batch_size=batch_size)
        with _model:
            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                fit_kwargs = self._update_kwargs["fit"].copy()
                # Handle epochs parameter - convert to 'n' based on effective batch size
                if "epochs" in self._update_kwargs:
                    num_samples = len(_context)
                    effective_batch_size = batch_size if batch_size is not None else num_samples
                    fit_kwargs["n"] = int(self._update_kwargs["epochs"] * num_samples / effective_batch_size)

                approx = fit(**fit_kwargs)

                self._approx_history = approx.hist
                approx_mean_eval = approx.mean.eval()
                approx_std_eval = approx.std.eval()
                approx_posterior_mapping = {
                    param: (approx_mean_eval[slice_], approx_std_eval[slice_])
                    for (param, (_, slice_, _, _)) in approx.ordering.items()
                }
                updated_layer_params_list = []
                for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                    weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)
                    w_shape = layer_params.weight.shape
                    b_shape = layer_params.bias.shape
                    w_mu = approx_posterior_mapping[weight_layer_params_name][0].reshape(w_shape)
                    w_sigma = approx_posterior_mapping[weight_layer_params_name][1].reshape(w_shape)
                    b_mu = approx_posterior_mapping[bias_layer_params_name][0].reshape(b_shape)
                    b_sigma = approx_posterior_mapping[bias_layer_params_name][1].reshape(b_shape)

                    updated_layer_params = self._create_updated_layer_params(
                        w_mu, w_sigma, b_mu, b_sigma, layer_params.weight, layer_params.bias
                    )
                    updated_layer_params_list.append(updated_layer_params)

                self._update_embedding_params_from_mapping(approx_posterior_mapping)

            elif self.update_method == "MCMC":
                trace = sample(**self.update_kwargs["trace"])

                updated_layer_params_list = []
                for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                    weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)

                    w_values = trace[weight_layer_params_name]
                    b_values = trace[bias_layer_params_name]

                    w_mu = np.mean(w_values, axis=0)
                    w_sigma = np.std(w_values, axis=0)
                    b_mu = np.mean(b_values, axis=0)
                    b_sigma = np.std(b_values, axis=0)

                    updated_layer_params = self._create_updated_layer_params(
                        w_mu, w_sigma, b_mu, b_sigma, layer_params.weight, layer_params.bias
                    )
                    updated_layer_params_list.append(updated_layer_params)

                self._update_embedding_params_from_trace(trace)

            else:
                raise ValueError("Invalid update method.")

            self.model_params.bnn_layer_params = updated_layer_params_list

    @classmethod
    def cold_start(
        cls,
        n_features: PositiveInt,
        hidden_dim_list: Optional[List[PositiveInt]] = None,
        update_method: UpdateMethods = "VI",
        update_kwargs: Optional[dict] = None,
        dist_type: Literal["normal", "studentt"] = "studentt",
        dist_params_init: Optional[Dict[str, float]] = None,
        activation: ActivationFunctions = "tanh",
        use_residual_connections: bool = False,
        use_layerwise_scaling: bool = False,
        categorical_features: Optional[Dict[NonNegativeInt, NonNegativeInt]] = None,
        **kwargs,
    ) -> Self:
        """
        Initialize a Bayesian Neural Network with a cold start.

        Parameters
        ----------
        n_features : PositiveInt
            Total number of columns in the context array, including any categorical columns.
        hidden_dim_list : Optional[List[PositiveInt]], optional
            List of dimensions for the hidden layers of the network. If None, no hidden layers are added.
        update_method : UpdateMethods
            Method to update the network, either "MCMC" or "VI". Default is "VI".
        update_kwargs : Optional[dict], optional
            Additional keyword arguments for the update method. Default is None.
        dist_type : Literal["normal", "studentt"]
            Type of distribution to use for priors. Default is "studentt".
        dist_params_init : Optional[Dict[str, float]], optional
            Initial distribution parameters for the network weights and biases. Default is None.
            For Student-t distributions: requires "mu", "sigma", and "nu" parameters.
            For Normal distributions: requires "mu" and "sigma" parameters (no "nu" needed).
        activation : str
            The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
        use_residual_connections : bool
            Whether to use residual connections in the network (default is False).
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
            When applied, the sigma is scaled by the square root of the input dimension.
            This is useful to enable smoother convergence with Gaussian Process-like behavior.
        categorical_features : Optional[Dict[int, int]], optional
            Categorical columns as ``{column_index: cardinality}``. Each categorical column is
            modelled with a Bayesian embedding matrix; ``embedding_dim`` is set automatically
            to ``ceil(cardinality / _embedding_dim_divisor)``. Columns absent from this dict are treated as numerical.
        **kwargs
            Additional keyword arguments for the BayesianNeuralNetwork constructor.

        Returns
        -------
        Self
            An instance of the Bayesian Neural Network initialized with the specified parameters.
        """
        cat_configs = [
            CategoricalFeatureConfig(
                column_index=col_idx,
                cardinality=cardinality,
                embedding_dim=ceil(cardinality / cls._embedding_dim_divisor),
            )
            for col_idx, cardinality in (categorical_features or {}).items()
        ]
        feature_config = FeaturesConfig(n_features=n_features, categorical_features_configs=cat_configs)

        dist_params_init = (dist_params_init or {}).copy()

        if dist_type not in cls._distribution_mapping:
            raise ValueError(
                f"Invalid dist_type: {dist_type}. Must be one of {list(cls._distribution_mapping.keys())}. "
                f"Example: dist_type='normal', dist_params_init={{'mu': 0, 'sigma': 1}}"
            )

        dist_class = cls._distribution_mapping[dist_type]
        model_params = cls.create_model_params(
            feature_config=feature_config,
            hidden_dim_list=hidden_dim_list,
            use_layerwise_scaling=use_layerwise_scaling,
            dist_class=dist_class,
            **dist_params_init,
        )
        return cls(
            model_params=model_params,
            update_method=update_method,
            update_kwargs=update_kwargs,
            activation=activation,
            use_residual_connections=use_residual_connections,
            feature_config=feature_config,
            **kwargs,
        )

    def _reset(self):
        """
        Reset the model to its initial parameters.
        """
        self.model_params.bnn_layer_params = deepcopy(self.model_params.bnn_layer_params_init)
        if self.model_params.embedding_params is not None:
            self.model_params.embedding_params.embeddings = deepcopy(self.model_params.embedding_params.embeddings_init)

    @classmethod
    def from_old_state(cls, state: Dict[str, Any]) -> Self:
        """
        Construct an instance from a pre-4.4 state dict that may not contain ``feature_config``.

        Use this when deserializing a ``BayesianNeuralNetwork`` that was saved before version 4.4
        (when ``feature_config`` was introduced).  If ``feature_config`` is absent, an
        all-numerical ``FeaturesConfig`` is inferred from the first layer's weight shape.

        Parameters
        ----------
        state : Dict[str, Any]
            Raw state dict as returned by ``model_dump()`` / ``dict()`` for a pre-4.4 model.

        Returns
        -------
        Self
            A fully initialised instance of the calling class.

        Examples
        --------
        >>> import json
        >>> raw = json.loads(old_model.get_state())   # pre-4.4 JSON
        >>> model = BayesianNeuralNetwork.from_old_state(raw)
        """
        if "feature_config" not in state:
            layer_params = state["model_params"]["bnn_layer_params"]
            if not layer_params:
                raise ValueError("Cannot infer feature_config: bnn_layer_params is empty.")
            # weight.mu is List[List[float]]; outer length == input_dim == n_features for numerical-only models
            n_features = len(layer_params[0][BaseBayesianNeuralNetwork.weight_var_name]["mu"])
            fc = FeaturesConfig(n_features=n_features, categorical_features_configs=[])
            state = {**state, "feature_config": fc.apply_version_adjusted_method("model_dump", "dict")}

        if pydantic_version == PYDANTIC_VERSION_2:
            return cls.model_validate(state)
        elif pydantic_version == PYDANTIC_VERSION_1:
            return cls.parse_obj(state)
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")


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

    @property
    def input_dim(self) -> PositiveInt:
        """
        Returns the expected input dimension of the model.

        Returns
        -------
        PositiveInt
            The number of input features expected by the model, derived from
            the shape of the weight matrix in the first layer's parameters of the first objective model.
        """
        return self.models[0].input_dim

    @classmethod
    def cold_start(
        cls,
        n_objectives: PositiveInt,
        n_features: PositiveInt,
        hidden_dim_list: Optional[List[PositiveInt]] = None,
        update_method: UpdateMethods = "VI",
        update_kwargs: Optional[dict] = None,
        dist_type: Literal["normal", "studentt"] = "studentt",
        dist_params: Optional[Dict[str, float]] = None,
        activation: ActivationFunctions = "tanh",
        use_residual_connections: bool = False,
        use_layerwise_scaling: bool = False,
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
        dist_type : Literal["normal", "studentt"]
            Type of distribution to use for priors. Default is "studentt".
        dist_params : Optional[Dict[str, float]], optional
            Initial distribution parameters for the network weights and biases.
        activation : str
            The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
        use_residual_connections : bool
            Whether to use residual connections in the network (default is False).
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
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
                dist_type=dist_type,
                dist_params_init=dist_params,
                activation=activation,
                use_residual_connections=use_residual_connections,
                use_layerwise_scaling=use_layerwise_scaling,
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
