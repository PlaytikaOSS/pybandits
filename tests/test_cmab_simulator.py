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

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from pybandits.cmab import CmabBernoulli
from pybandits.cmab_simulator import CmabSimulator
from pybandits.model import BayesianLogisticRegression, StudentT


def test_mismatched_probs_reward_columns(mocker: MockerFixture, groups=[0, 1]):
    def check_value_error(probs_reward, context):
        with pytest.raises(ValueError):
            CmabSimulator(mab=cmab, probs_reward=probs_reward, groups=groups, context=context)

    num_groups = len(groups)
    cmab = mocker.Mock(spec=CmabBernoulli)
    cmab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    cmab.epsilon = 0.0
    cmab.default_action = None
    context = pd.DataFrame({"a1": [0.5] * num_groups, "a2": [0.5] * num_groups})
    probs_reward = pd.DataFrame({"a1": [0.5], "a2": [0.5]})
    check_value_error(probs_reward, context)
    probs_reward = pd.DataFrame({"a1": [0.5] * num_groups, "a2": [0.5] * num_groups})
    check_value_error(probs_reward, context[:1])


def test_cmab_e2e_simulation(n_features=3, n_updates=2, batch_size=10):
    mab = CmabBernoulli(
        actions={
            "a1": BayesianLogisticRegression(alpha=StudentT(mu=1, sigma=2), betas=n_features * [StudentT()]),
            "a2": BayesianLogisticRegression(alpha=StudentT(mu=1, sigma=2), betas=n_features * [StudentT()]),
        }
    )
    group = [0, 1] * (n_updates * batch_size // 2) + [0] * (n_updates * batch_size % 2)
    context = (
        np.repeat(np.arange(3).reshape(1, -1), n_updates * batch_size, axis=0).T * (np.array(group) - np.mean(group))
    ).T
    with TemporaryDirectory() as path:
        simulator = CmabSimulator(
            mab=mab,
            visualize=True,
            save=True,
            path=path,
            group=[str(g) for g in group],
            batch_size=batch_size,
            n_updates=n_updates,
            context=context,
        )
        simulator.run()
        assert not simulator.results.empty
        dir_list = os.listdir(path)
        assert "simulation_results.csv" in dir_list
        assert "selected_actions_count.csv" in dir_list
        assert "positive_reward_proportion.csv" in dir_list
        assert "simulation_results.html" in dir_list
