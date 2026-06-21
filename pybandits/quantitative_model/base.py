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

import functools
import inspect
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, List, Optional, Tuple, Union

import numpy as np
from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    field_serializer,
    field_validator,
    validate_call,
)

from pybandits.base import (
    BinaryReward,
    PyBanditsBaseModel,
    QuantitativeProbability,
    QuantitativeProbabilityWeight,
)
from pybandits.base_model import BaseModelCC, BaseModelDP, BaseModelSO


class QuantitativeModel(BaseModelSO, ABC):
    """
    Base class for quantitative models.

    Each concrete model wraps an inner single-objective base model (e.g. a ``Beta`` per Zooming
    segment, or a ``BayesianNeuralNetwork``). By convention, ``cold_start`` accepts
    ``base_model_cold_start_kwargs``: a dict forwarded to that inner base model's constructor /
    cold start (e.g. ``decay_factor``), since per-update behavior is an attribute of the base model
    rather than the quantitative wrapper.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    """

    dimension: PositiveInt
    _transfer_structural_keys: ClassVar[Tuple[str, ...]] = ("dimension",)

    @abstractmethod
    def sample_proba(
        self, rng: np.random.Generator, **kwargs
    ) -> Union[List[QuantitativeProbability], List[QuantitativeProbabilityWeight]]:
        """
        Sample the model.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central numpy random generator provided by the MAB.

        Returns
        -------
        Union[List[QuantitativeProbability], List[QuantitativeProbabilityWeight]]
            A list of callables: either probability functions (quantity -> Probability)
            or (probability, weight) tuples. List length is equal to the number of samples.
        """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _update(
        self,
        quantities: Optional[List[Union[float, List[float]]]],
        rewards: List[BinaryReward],
        **kwargs,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: List[BinaryReward]
            The reward for each sample.
        """

        if quantities:
            self._quantitative_update(quantities=quantities, rewards=rewards, **kwargs)

    @abstractmethod
    def _quantitative_update(
        self,
        quantities: List[Union[float, List[float], None]],
        rewards: List[BinaryReward],
        **kwargs,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: List[BinaryReward]
            The reward for each sample.
        """


class CallableFieldSerde(PyBanditsBaseModel, ABC):
    """
    Mixin providing (de)serialization for a model field that holds a named callable.

    Pydantic cannot serialize a function directly, so this mixin serializes a named
    (non-lambda) function to its source code and reconstructs it on deserialization.
    Concrete models declare their own field-specific ``field_serializer`` /
    ``field_validator`` hooks that delegate to :meth:`_serialize_callable` /
    :meth:`_deserialize_callable` (see :class:`QuantitativeModelCC` for ``cost`` and
    :class:`QuantitativeModelDP` for ``price``).
    """

    @staticmethod
    def _serialize_function(func: Callable) -> str:
        """
        Serialize a function to its source code.

        Parameters
        ----------
        func : Callable
            Function to serialize as string.

        Returns
        -------
        str
            The serialized function code.
        """
        if inspect.isfunction(func) and not func.__name__ == "<lambda>":
            try:
                return inspect.getsource(func).strip()
            except OSError:  # Dynamically evaluated functions may not have source code available
                return globals()[func.__name__].__source__  # Fallback to the global scope if source is not available
        # Anonymous/builtin callables (lambdas, partials of lambdas, builtins) cannot be
        # round-tripped: str(func) yields a repr that _deserialize_function cannot eval.
        raise ValueError(
            f"Cannot serialize {func!r}: only named module-level functions (and functools.partial "
            "wrapping them) are supported; lambdas and other anonymous callables are not."
        )

    @staticmethod
    def _deserialize_function(code: str) -> Callable:
        """
        Deserialize a function from its source code.

        Parameters
        ----------
        code : str
            python code representing a function or a callable object.

        Returns
        -------
        Callable
            The deserialized function or callable object.
        """
        if code.startswith("def "):
            exec(code, globals())
            func_name = code.split("(")[0][4:].strip()
            globals()[func_name].__source__ = code.strip()  # Register the function in the global scope
        else:
            func_name = code.strip()
        return eval(func_name)

    @staticmethod
    def _serialize_callable(value) -> str:
        """Serialize a callable value to its string representation."""
        if isinstance(value, functools.partial):
            return f"functools.partial({CallableFieldSerde._serialize_function(value.func)}, {value.args}, {value.keywords})"
        elif callable(value):
            return CallableFieldSerde._serialize_function(value)
        else:
            raise ValueError(f"Unrecognized callable for serialization: {value}")

    @classmethod
    def _deserialize_callable(cls, value):
        """Deserialize a callable from its string representation if needed."""
        if isinstance(value, str):
            if value.startswith("functools.partial"):
                inner_func_split = "(".join(value.split("(")[1:]).split(",")
                # Extract function and arguments from pattern: functools.partial(func_name, args, kwargs)
                func_str = ",".join(inner_func_split[:-2]).strip()
                func = cls._deserialize_function(func_str)
                args_str = ")".join(",".join(inner_func_split[-2:]).split(")")[:-1]).strip()
                args_parts = eval(args_str) if args_str else ((), {})
                return functools.partial(func, *args_parts[0], **args_parts[1])
            else:
                return cls._deserialize_function(value)
        return value


class QuantitativeModelCC(CallableFieldSerde, BaseModelCC, ABC):
    """
    Class to model quantitative action cost.

    Parameters
    ----------
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """

    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]

    @field_serializer("cost")
    def encode_cost(self, value):
        return self._serialize_callable(value).encode("ascii")

    @field_validator("cost", mode="before")
    @classmethod
    def validate_cost(cls, value):
        """
        Deserialize cost from string representation if needed.
        """
        return cls._deserialize_callable(value)


class QuantitativeModelDP(CallableFieldSerde, BaseModelDP, ABC):
    """
    Class to model quantitative action price.

    Parameters
    ----------
    price: Callable[[Union[float, np.ndarray]], NonNegativeFloat]
        Price associated to the Beta distribution.
    """

    price: Callable[[Union[float, np.ndarray]], NonNegativeFloat]

    @field_serializer("price")
    def encode_price(self, value):
        return self._serialize_callable(value).encode("ascii")

    @field_validator("price", mode="before")
    @classmethod
    def validate_price(cls, value):
        """
        Deserialize price from string representation if needed.
        """
        return cls._deserialize_callable(value)
