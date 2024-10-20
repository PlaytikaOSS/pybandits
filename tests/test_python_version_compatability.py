from typing import List

import pytest

from pybandits.base import PyBanditsBaseModel
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    field_validator,
    model_validator,
    pydantic_version,
    validate_call,
)


def test_pydantic_version_compatibility():
    assert pydantic_version in [PYDANTIC_VERSION_1, PYDANTIC_VERSION_2]


def test_dummy_pybandits_model():
    class DummyPyBanditsModel(PyBanditsBaseModel):
        a: int
        b: float

        @field_validator("a")
        @classmethod
        def check_even_a(cls, value):
            if value % 2 != 0:
                raise ValueError("a must be even")
            return value

        @model_validator(mode="before")
        @classmethod
        def check_a_ge_b(cls, values):
            if values["b"] > values["a"]:
                raise ValueError("a must be greater than or equal to b")
            return values

        @validate_call
        def add(self, addends: List[float]):
            return self.a + sum(addends)

    with pytest.raises(ValueError):
        DummyPyBanditsModel(a=1, b=0.0)
    with pytest.raises(ValueError):
        DummyPyBanditsModel(a=0, b=1.0)

    dummy_model = DummyPyBanditsModel(a=0, b=0.0)
    with pytest.raises(ValueError):
        dummy_model.add(addends=0.5)

    dummy_model.add(addends=[1.0])
