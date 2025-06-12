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

from typing import Any, Dict, List, Mapping, NewType, Optional, Tuple, Union, _GenericAlias, get_args, get_origin

from typing_extensions import Self

from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    BaseModel,
    confloat,
    conint,
    constr,
    pydantic_version,
)
from pybandits.utils import classproperty

ActionId = NewType("ActionId", constr(min_length=1))
QuantitativeActionId = Tuple[ActionId, Tuple[float, ...]]
UnifiedActionId = Union[ActionId, QuantitativeActionId]
Float01 = NewType("Float_0_1", confloat(ge=0, le=1))
Probability = NewType("Probability", Float01)
PositiveProbability = NewType("PositiveProbability", confloat(gt=0, le=1))
ProbabilityWeight = Tuple[Probability, float]
MOProbability = List[Probability]
MOProbabilityWeight = List[ProbabilityWeight]
# QuantitativeProbability generalizes probability to include both action quantities and their associated probability
QuantitativeProbability = Tuple[Tuple[Tuple[Float01, ...], Probability], ...]
QuantitativeProbabilityWeight = Tuple[Tuple[Tuple[Float01, ...], ProbabilityWeight], ...]
QuantitativeMOProbability = Tuple[Tuple[Tuple[Float01, ...], List[Probability]], ...]
QuantitativeMOProbabilityWeight = Tuple[Tuple[Tuple[Float01, ...], List[ProbabilityWeight]], ...]
UnifiedProbability = Union[Probability, QuantitativeProbability]
UnifiedProbabilityWeight = Union[ProbabilityWeight, QuantitativeProbabilityWeight]
UnifiedMOProbability = Union[MOProbability, QuantitativeMOProbability]
UnifiedMOProbabilityWeight = Union[MOProbabilityWeight, QuantitativeMOProbabilityWeight]
# SmabPredictions is a tuple of two lists: the first list contains the selected action ids,
# and the second list contains their associated probabilities
SmabPredictions = NewType(
    "SmabPredictions",
    Tuple[
        List[UnifiedActionId],
        Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, MOProbability]]],
    ],
)
# CmabPredictions is a tuple of three lists: the first list contains the selected action ids,
# the second list contains their associated probabilities,
# and the third list contains their associated weighted sums
CmabPredictions = NewType(
    "CmabPredictions",
    Union[
        Tuple[List[UnifiedActionId], List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, float]]],
        Tuple[
            List[UnifiedActionId], List[Dict[UnifiedActionId, MOProbability]], List[Dict[UnifiedActionId, List[float]]]
        ],
    ],
)
Predictions = NewType("Predictions", Union[SmabPredictions, CmabPredictions])
BinaryReward = NewType("BinaryReward", conint(ge=0, le=1))
ActionRewardLikelihood = NewType(
    "ActionRewardLikelihood",
    Union[Dict[UnifiedActionId, float], Dict[UnifiedActionId, Probability], Dict[UnifiedActionId, List[Probability]]],
)
ACTION_IDS_PREFIX = "action_ids_"
ACTIONS = "actions"
QUANTITATIVE_ACTION_IDS_PREFIX = f"quantitative_{ACTION_IDS_PREFIX}"
SerializablePrimitive = Union[str, int, float, bool, None]
Serializable = Union[SerializablePrimitive, Dict[str, "Serializable"], List["Serializable"]]


class PyBanditsBaseModel(BaseModel):
    """
    BaseModel of the PyBandits library.
    """

    if pydantic_version == PYDANTIC_VERSION_1:

        class Config:
            extra = "forbid"

        def __init__(self, **data):
            super(PyBanditsBaseModel, self).__init__(**data)
            self.model_post_init(None)

        def model_post_init(self, __context: Any) -> None:
            pass

    elif pydantic_version == PYDANTIC_VERSION_2:
        model_config = {"extra": "forbid"}

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    def _validate_params_lengths(
        self,
        force_values: bool = False,
        **kwargs,
    ):
        """
        Verify that the given keyword arguments have the same length.
        """
        reference = None
        for val in kwargs.values():
            if val is not None:
                reference = len(val)
                break
        if reference is not None:
            for k, v in kwargs.items():
                if (v is None or len(v) != reference) if force_values else (v is not None and len(v) != reference):
                    raise AttributeError(f"Shape mismatch: {k} should have the same length as the other parameters.")

    def _apply_version_adjusted_method(self, v2_method_name: str, v1_method_name: str, **kwargs) -> Any:
        """
        Apply the method with the given name, adjusting for the pydantic version.

        Parameters
        ----------
        v2_method_name : str
            The method name for pydantic v2.
        v1_method_name : str
            The method name for pydantic v1.
        """
        if pydantic_version == PYDANTIC_VERSION_1:
            return getattr(self, v1_method_name)(**kwargs)
        elif pydantic_version == PYDANTIC_VERSION_2:
            return getattr(self, v2_method_name)(**kwargs)
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    def _with_argument(self, argument_name: str, argument_value: Any) -> Self:
        """
        Instantiate a mutated model with an altered argument_value for argument_name.

        Parameters
        ----------
        argument_name: str
            The name of the argument.
        argument_value: Any
            The value of the argument.

        Returns
        -------
        mutated_strategy: PyBanditsBaseModel
            The mutated model.
        """
        mutated_strategy = self._apply_version_adjusted_method(
            "model_copy", "copy", update={argument_name: argument_value}
        )
        return mutated_strategy

    @classmethod
    def _get_value_with_default(cls, key: str, values: Dict[str, Any]) -> Any:
        return values.get(key, cls.model_fields[key].default)

    @classmethod
    def _get_field_type(cls, key: str) -> Any:
        if pydantic_version == PYDANTIC_VERSION_1:
            annotation = cls.model_fields[key].type_
        elif pydantic_version == PYDANTIC_VERSION_2:
            annotation = cls.model_fields[key].annotation
            if isinstance(annotation, _GenericAlias) and get_origin(annotation) is dict:
                annotation = get_args(annotation)[1]  # refer to the type of the Dict values
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")
        if get_origin(annotation) is Union:
            annotation = get_args(annotation)
        return annotation

    if pydantic_version == PYDANTIC_VERSION_1:

        @classproperty
        def model_fields(cls) -> Dict[str, Any]:
            """
            Get the model fields.

            Returns
            -------
            List[str]
                The model fields.
            """
            return cls.__fields__

        def model_copy(self, *, update: Optional[Mapping[str, Any]] = None, deep: bool = False) -> Self:
            """
            Create a new instance of the model with the same quantities.

            Parameters
            ----------
            update : Mapping[str, Any], optional
                The quantities to update, by default None

            deep : bool, optional
                Whether to copy the quantities deeply, by default False

            Returns
            -------
            Self
                The new instance of the model.
            """
            return self.copy(update=update, deep=deep)

        @classmethod
        def model_validate_json(
            cls,
            json_data: Union[str, bytes, bytearray],
        ) -> Self:
            """
            Validate a PyBandits BaseModel model instance.

            Parameters
            ----------
            json_data : str
                JSON string of the object to validate.

            Raises
            ------
                ValidationError: If the object could not be validated.

            Returns
            -------
            Self
                The validated model instance.
            """
            return cls.parse_raw(json_data)
