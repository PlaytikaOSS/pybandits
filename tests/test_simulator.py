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
from random import choice
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from pybandits.base import ActionId, BinaryReward
from pybandits.mab import BaseMab
from pybandits.simulator import Simulator


class DummySimulator(Simulator):
    def _initialize_results(self):
        self._results = pd.DataFrame

    def _draw_rewards(self, actions: List[ActionId], metadata: Dict[str, List]) -> List[BinaryReward]:
        return choice([0, 1], k=len(actions))

    def _get_batch_step_kwargs_and_metadata(self, batch_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        return {}, {}

    def _finalize_step(self, batch_results: pd.DataFrame) -> pd.DataFrame:
        return batch_results

    def _finalize_results(self):
        pass


def test_mismatched_probs_reward_columns(mocker: MockerFixture):
    def check_value_error(probs_reward):
        with pytest.raises(ValueError):
            DummySimulator(mab=mab, probs_reward=probs_reward)

    mab = mocker.Mock(spec=BaseMab)
    mab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    mab.epsilon = 0.0
    mab.default_action = None
    probs_reward = pd.DataFrame({"a3": [0.5]})
    check_value_error(probs_reward)
    probs_reward = pd.DataFrame({"a1": [0.5], "a2": [2]})
    check_value_error(probs_reward)
    probs_reward = pd.DataFrame({"a1": [0.5], "a2": [0.5], "a3": [0.5]})
    check_value_error(probs_reward)
