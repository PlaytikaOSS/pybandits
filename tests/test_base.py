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

from typing import Dict, List, Optional, Set

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from pydantic import NonNegativeInt, ValidationError
from pytest_mock import MockerFixture

from pybandits.base import ActionId, BaseMab, Float01, Probability
from pybandits.model import Beta
from pybandits.strategy import ClassicBandit


class DummyMab(BaseMab):
    epsilon: Optional[Float01] = None
    default_action: Optional[ActionId] = None

    def update(self, actions: List[ActionId], rewards: List[NonNegativeInt]):
        super().update(actions=actions, rewards=rewards)
        pass

    def predict(
        self,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ):
        valid_actions = self._get_valid_actions(forbidden_actions)
        return np.random.choice(valid_actions)

    def get_state(self) -> (str, dict):
        model_name = self.__class__.__name__
        state: dict = {"actions": self.actions}
        return model_name, state


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


@given(r1=st.integers(min_value=0, max_value=1), r2=st.integers(min_value=0, max_value=1))
def test_base_mab_update_ok(r1, r2):
    dummy_mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    dummy_mab.update(actions=["a1", "a2"], rewards=[r1, r2])
    dummy_mab.update(actions=["a1", "a1"], rewards=[r1, r2])


########################################################################################################################


# Epsilon-greedy functionality tests


@pytest.fixture
def p() -> Dict[ActionId, Probability]:
    return {"a1": 0.5, "a2": 0.5}


def test_valid_epsilon_value(mocker: MockerFixture, p: Dict[ActionId, Probability]):
    mocker.patch.object(ClassicBandit, "select_action", return_value="a2")
    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.1, default_action="a1")
    selected_action = mab._select_epsilon_greedy_action(p)
    assert selected_action in p.keys()


def test_epsilon_boundary_values(mocker: MockerFixture, p: Dict[ActionId, Probability]):
    mocker.patch.object(ClassicBandit, "select_action", return_value="a2")

    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.0)
    selected_action = mab._select_epsilon_greedy_action(p)
    assert selected_action == "a2"

    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=1.0, default_action="a1")
    selected_action = mab._select_epsilon_greedy_action(p)
    assert selected_action == "a1"


def test_default_action_not_in_actions(p: Dict[ActionId, Probability]):
    with pytest.raises(AttributeError):
        DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=1.0, default_action="a3")


def test_select_action_raises_exception(mocker: MockerFixture, p: Dict[ActionId, Probability]):
    mocker.patch.object(ClassicBandit, "select_action", side_effect=Exception("Test Exception"))
    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.0, default_action=None)

    with pytest.raises(Exception) as excinfo:
        mab._select_epsilon_greedy_action(p)

    assert str(excinfo.value) == "Test Exception"


def test_default_action_in_forbidden_actions():
    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.1, default_action="a1")
    with pytest.raises(ValueError):
        mab.predict(forbidden_actions={"a1"})
