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

from typing import List

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pydantic import NonNegativeInt, ValidationError

from pybandits.base import ActionId, BaseMab
from pybandits.model import Beta
from pybandits.strategy import ClassicBandit


class DummyMab(BaseMab):
    def update(
        self,
        actions: List[ActionId],
        rewards: List[NonNegativeInt],
    ):
        super().update(actions=actions, rewards=rewards)
        pass

    def predict():
        pass


def test_base_mab_raise_on_less_than_2_actions():
    with pytest.raises(ValidationError):
        DummyMab(actions={"a1": Beta(), "a2": Beta()})
    with pytest.raises(ValidationError):
        DummyMab(actions={"": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    with pytest.raises(AttributeError):
        DummyMab(actions={}, strategy=ClassicBandit())
    with pytest.raises(AttributeError):
        DummyMab(actions={"a1": None}, strategy=ClassicBandit())
    with pytest.raises(ValidationError):
        DummyMab(actions={"a1": None, "a2": None}, strategy=ClassicBandit())
    with pytest.raises(AttributeError):
        DummyMab(actions={"a1": Beta()}, strategy=ClassicBandit())


def test_base_mab_check_update_params():
    dummy_mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    with pytest.raises(AttributeError):
        # actionId doesn't exist
        dummy_mab._check_update_params(actions=["a1", "a3"], rewards=[1, 1])
    with pytest.raises(AttributeError):
        # actionId cannot be empty
        dummy_mab._check_update_params(actions=[""], rewards=[1])
    with pytest.raises(AttributeError):
        dummy_mab._check_update_params(actions=["a1", "a2"], rewards=[1])


@given(
    r1=st.integers(min_value=0, max_value=1), r2=st.integers(min_value=0, max_value=1)
)
def test_base_mab_update_ok(r1, r2):
    dummy_mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    dummy_mab.update(actions=["a1", "a2"], rewards=[r1, r2])
    dummy_mab.update(actions=["a1", "a1"], rewards=[r1, r2])
