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
