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


from typing import Any, Dict, List, NewType, Tuple, Union, _GenericAlias, get_args, get_origin

from pydantic import (
    BaseModel,
    confloat,
    conint,
    constr,
)
from typing_extensions import Self

ActionId = NewType("ActionId", constr(min_length=1))
Float01 = NewType("Float_0_1", confloat(ge=0, le=1))
Probability = NewType("Probability", Float01)
PositiveProbability = NewType("PositiveProbability", confloat(gt=0, le=1))
# SmabPredictions is a tuple of two lists: the first list contains the selected action ids,
# and the second list contains their associated probabilities
SmabPredictions = NewType("SmabPredictions", Tuple[List[ActionId], List[Dict[ActionId, Probability]]])
# CmabPredictions is a tuple of three lists: the first list contains the selected action ids,
# the second list contains their associated probabilities,
# and the third list contains their associated weighted sums
CmabPredictions = NewType(
    "CmabPredictions", Tuple[List[ActionId], List[Dict[ActionId, Probability]], List[Dict[ActionId, float]]]
)
Predictions = NewType("Predictions", Union[SmabPredictions, CmabPredictions])
BinaryReward = NewType("BinaryReward", conint(ge=0, le=1))
ActionRewardLikelihood = NewType(
    "ActionRewardLikelihood",
    Union[Dict[ActionId, float], Dict[ActionId, Probability], Dict[ActionId, List[Probability]]],
)
ACTION_IDS_PREFIX = "action_ids_"
ACTIONS = "actions"


class _classproperty(property):
    def __get__(self, instance, owner):
        return self.fget(owner)


class PyBanditsBaseModel(BaseModel):
    """
    BaseModel of the PyBandits library.
    """

    model_config = {"extra": "forbid"}

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
        mutated_strategy = self.model_copy(update={argument_name: argument_value})
        return mutated_strategy

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
