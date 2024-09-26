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
import os
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from pybandits.model import Beta
from pybandits.smab import SmabBernoulli
from pybandits.smab_simulator import SmabSimulator


def test_mismatched_probs_reward_columns(mocker: MockerFixture):
    smab = mocker.Mock(spec=SmabBernoulli)
    smab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    smab.epsilon = 0.0
    smab.default_action = None
    probs_reward = pd.DataFrame({"a1": [0.5, 0.5], "a2": [0.5, 0.5]})
    with pytest.raises(ValueError):
        SmabSimulator(mab=smab, probs_reward=probs_reward)


def test_smab_e2e_simulation():
    mab = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
    with TemporaryDirectory() as path:
        simulator = SmabSimulator(mab=mab, visualize=True, save=True, path=path)
        simulator.run()
        assert not simulator.results.empty
        dir_list = os.listdir(path)
        assert "simulation_results.csv" in dir_list
        assert "selected_actions_count.csv" in dir_list
        assert "positive_reward_proportion.csv" in dir_list
        assert "simulation_results.html" in dir_list
