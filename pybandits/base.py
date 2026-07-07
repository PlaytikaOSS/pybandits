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

from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
    Set,
    Tuple,
    Union,
    _GenericAlias,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    confloat,
    conint,
    constr,
)
from typing_extensions import Self

ActionId = NewType("ActionId", constr(min_length=1))
QuantitativeActionId = Tuple[ActionId, Tuple[float, ...]]
UnifiedActionId = Union[ActionId, QuantitativeActionId]
Float01 = NewType("Float_0_1", confloat(ge=0, le=1))
PositiveFloat01 = NewType("PositiveFloat01", confloat(gt=0, le=1))
Probability = NewType("Probability", Float01)
PositiveProbability = NewType("PositiveProbability", confloat(gt=0, le=1))
ProbabilityWeight = Tuple[Probability, float]
MOProbability = List[Probability]
MOProbabilityWeight = List[ProbabilityWeight]
# QuantitativeProbability generalizes probability to include both action quantities and their associated probability
QuantitativeProbability = Callable[[np.ndarray], Probability]
QuantitativeWeight = Callable[[np.ndarray], float]
QuantitativeProbabilityWeight = Tuple[QuantitativeProbability, QuantitativeWeight]
QuantitativeMOProbability = Callable[[np.ndarray], MOProbability]
QuantitativeMOProbabilityWeight = Tuple[Callable[[np.ndarray], MOProbability], Callable[[np.ndarray], float]]

# A forbidden region restricts a quantitative action's quantity space [0, 1]^d.
# Signed-margin convention: forbidden(x) > 0 => x is forbidden, <= 0 => x is allowed. A float margin (not a bare
# bool) is required so the optimizer has directional information and degrades gracefully near the boundary.
ForbiddenRegion = Callable[[np.ndarray], float]
# forbidden_actions generalizes whole-arm blocking to per-arm hypercube-region blocking:
#   - Set[ActionId]: forbid whole arms (legacy form).
#   - Dict[ActionId, None]: forbid the whole arm (equivalent to set membership).
#   - Dict[ActionId, ForbiddenRegion | List[ForbiddenRegion]]: forbid region(s) of a quantitative arm
#     (multiple regions are OR-combined: a quantity is forbidden if any region forbids it).
ForbiddenActions = Union[
    Set[ActionId],
    Dict[ActionId, Optional[Union[ForbiddenRegion, List[ForbiddenRegion]]]],
]

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
    Union[
        Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        Dict[ActionId, Union[Probability, Callable[[np.ndarray], Probability]]],
        Dict[ActionId, Union[List[Probability], Callable[[np.ndarray], List[Probability]]]],
    ],
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

    model_config = ConfigDict(extra="forbid")

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

    def __eq__(self, other: Any) -> bool:
        """Compare equality based on serializable fields only, excluding private attributes."""
        if type(self) is not type(other):
            return False
        return self.model_dump() == other.model_dump()

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
        return self.model_copy(update={argument_name: argument_value})

    @classmethod
    def _get_value_with_default(cls, key: str, values: Dict[str, Any]) -> Any:
        return values.get(key, cls.model_fields[key].default)

    @classmethod
    def _get_field_type(cls, key: str) -> Any:
        annotation = cls.model_fields[key].annotation
        if isinstance(annotation, _GenericAlias) and get_origin(annotation) is dict:
            annotation = get_args(annotation)[1]  # refer to the type of the Dict values
        if get_origin(annotation) is Union:
            annotation = get_args(annotation)
        return annotation

    @classmethod
    def _normalize_field(cls, v: Any, field_name: str) -> Any:
        """
        Normalize a field value to its default if None.

        This utility method ensures that optional fields receive their default
        values when not explicitly provided.

        Parameters
        ----------
        v : Any
            The field value to normalize.
        field_name : str
            Name of the field in the model.

        Returns
        -------
        Any
            The original value if not None, otherwise the field's default value.
        """
        return v if v is not None else cls.model_fields[field_name].default
