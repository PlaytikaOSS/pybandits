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
from typing import Any, ClassVar, Dict, List, Optional, Self, Tuple, Union

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as npdist
from numpyro.distributions import Normal as NumpyroNormal
from numpyro.distributions import StudentT as NumpyroStudentT
from pydantic import (
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    model_validator,
    validate_call,
)

from pybandits.base import (
    PyBanditsBaseModel,
)


class BaseLocationScaleArray(PyBanditsBaseModel, ABC):
    """
    Abstract base class for location-scale distribution arrays used in Bayesian Neural Networks.

    Parameters
    ----------
    mu : Union[List[float], List[List[float]]]
        The mean values of the distributions. Can be a 1D (for the layer bias term) or 2D list (for the layer weight term).
    sigma : Union[List[PositiveFloat], List[List[PositiveFloat]]]
        The scale (standard deviation) values of the distributions. Must be strictly positive.
        Can be a 1D or 2D list.
    """

    mu: Union[List[float], List[List[float]]]
    sigma: Union[List[PositiveFloat], List[List[PositiveFloat]]]

    _mu_array: np.ndarray = PrivateAttr()
    _sigma_array: np.ndarray = PrivateAttr()
    _params: Dict[str, np.ndarray] = PrivateAttr()
    _numpyro_dist_class: ClassVar[type]
    _sampler: ClassVar[str]  # name of the numpy Generator method
    _sampler_kwargs: ClassVar[Dict[str, str]] = {}  # internal param name → numpy kwarg name
    param_map: ClassVar[Dict[str, str]] = {"mu": "loc", "sigma": "scale"}

    def to_numpyro_distribution(self) -> npdist.Distribution:
        """
        Create a NumPyro distribution from this prior distribution array.

        Maps internal parameter names (mu, sigma, nu) to NumPyro parameter names
        (loc, scale, df) using the subclass-defined param_map.

        Returns
        -------
        npdist.Distribution
            A NumPyro distribution instance.
        """
        numpyro_params = {self.param_map[k]: jnp.array(v) for k, v in self.params.items()}
        return self._numpyro_dist_class(**numpyro_params)

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
        updated_dict = self.model_dump()
        updated_dict.update(kwargs)
        return self.__class__.model_validate(updated_dict)

    def _draw(
        self, params: Dict[str, np.ndarray], rng: np.random.Generator, size: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Apply the loc-scale transform: ``mu + sigma * rng.<_sampler>(**_sampler_kwargs)``."""
        _size = size if size is not None else params["mu"].shape
        extra = {rng_key: params[param_key] for param_key, rng_key in self._sampler_kwargs.items()}
        return params["mu"] + params["sigma"] * getattr(rng, self._sampler)(size=_size, **extra)

    def sample_rvs(self, size: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """
        Sample random variates from this distribution.

        Parameters
        ----------
        size : Tuple[int, ...]
            Shape of the output array.
        rng : np.random.Generator
            Numpy random generator.

        Returns
        -------
        np.ndarray
            Array of sampled values.
        """
        return self._draw(self._params, rng, size=size)

    def sample_at_indices(
        self, indices: Union[List[NonNegativeInt], np.ndarray], rng: np.random.Generator
    ) -> np.ndarray:
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
        rng : np.random.Generator
            Numpy random generator.

        Returns
        -------
        np.ndarray of shape (n, ncols)
            Sampled instances.
        """
        sliced = {k: v[indices] for k, v in self._params.items()}
        return self._draw(sliced, rng)

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

    @classmethod
    @validate_call
    def cold_start(
        cls,
        shape: Union[PositiveInt, Tuple[PositiveInt, ...]],
        mu: float = 0.0,
        sigma: PositiveFloat = 10.0,
        use_layerwise_scaling: bool = False,
        **kwargs,
    ) -> Self:
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
        sigma : PositiveFloat
            Standard deviation of the distribution, by default 10.0. Must be strictly positive.
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
    sigma : Union[List[PositiveFloat], List[List[PositiveFloat]]]
        The scale (standard deviation) values of the Student's t-distributions. Must be strictly positive (> 0).
        Can be a 1D or 2D list.
    nu : Union[List[PositiveFloat], List[List[PositiveFloat]]]
        The degrees of freedom of the Student's t-distributions. Must be positive.
        Can be a 1D or 2D list.
    """

    nu: Union[List[PositiveFloat], List[List[PositiveFloat]]]

    _nu_array: np.ndarray = PrivateAttr()
    _numpyro_dist_class: ClassVar[type] = NumpyroStudentT
    _sampler: ClassVar[str] = "standard_t"
    _sampler_kwargs: ClassVar[Dict[str, str]] = {"nu": "df"}
    param_map: ClassVar[Dict[str, str]] = {**BaseLocationScaleArray.param_map, "nu": "df"}

    @model_validator(mode="before")
    @classmethod
    def validate_input_shapes(cls, values):
        # The parent class method is now generic and handles all array-like parameters
        # including mu, sigma, and nu, so we can just call it directly
        return super().validate_input_shapes(values)

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
    sigma : Union[List[PositiveFloat], List[List[PositiveFloat]]]
        The standard deviation values of the Normal distributions. Must be strictly positive (> 0).
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

    _numpyro_dist_class: ClassVar[type] = NumpyroNormal
    _sampler: ClassVar[str] = "standard_normal"
    _sampler_kwargs: ClassVar[Dict[str, str]] = {}

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
