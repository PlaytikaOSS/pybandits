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
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from pybandits.cmab import CmabBernoulli
from pybandits.cmab_simulator import CmabSimulator
from pybandits.model import BayesianLogisticRegression


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


def test_cmab_e2e_simulation_with_default_arguments(
    action_ids=["a1", "a2"], n_features=3, n_updates=2, batch_size=10, num_groups=2
):
    mab = CmabBernoulli.cold_start(
        action_ids=action_ids, n_features=n_features
    )
    base_groups = list(range(num_groups))
    group = base_groups * (n_updates * batch_size // num_groups) + base_groups[: (n_updates * batch_size % num_groups)]
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


@settings(deadline=None)
@given(
    st.just(["a1", "a2"]),
    st.just(3),
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=10),
    st.booleans(),
    st.sampled_from([None, 0, 42]),
    st.booleans(),
    st.booleans(),
    st.sampled_from(["", "unit_test"]),
    st.integers(min_value=1, max_value=3),
)
def test_cmab_e2e_simulation_with_non_default_args(
    action_ids,
    n_features,
    n_updates,
    batch_size,
    save,
    random_seed,
    verbose,
    visualize,
    file_prefix,
    num_groups,
):
    base_groups = list(range(num_groups))
    group = base_groups * (n_updates * batch_size // num_groups) + base_groups[: (n_updates * batch_size % num_groups)]
    effective_base_groups = sorted(set(group))
    context = (
        np.repeat(np.arange(n_features).reshape(1, -1), n_updates * batch_size, axis=0).T
        * (np.array(group) - np.mean(group))
    ).T
    probs_reward = pd.DataFrame(
        np.random.uniform(0, 1, (len(effective_base_groups), len(action_ids))),
        columns=action_ids,
        index=[str(g) for g in effective_base_groups],
    )
    mab = CmabBernoulli.cold_start(
        action_ids=action_ids, n_features=n_features, update_method="VI"
    )
    if visualize and not save:
        with pytest.raises(ValueError):
            CmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                group=[str(g) for g in group],
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=probs_reward,
                verbose=verbose,
                file_prefix=file_prefix,
                context=context,
            )
    else:
        with TemporaryDirectory() as path:
            simulator = CmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                path=path,
                group=[str(g) for g in group],
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=probs_reward,
                verbose=verbose,
                file_prefix=file_prefix,
                context=context,
            )
            simulator.run()
            if save:
                assert not simulator.results.empty
                dir_list = os.listdir(path)
                if file_prefix:
                    assert f"{file_prefix}_simulation_results.csv" in dir_list
                    assert f"{file_prefix}_selected_actions_count.csv" in dir_list
                    assert f"{file_prefix}_positive_reward_proportion.csv" in dir_list
                    if visualize:
                        assert f"{file_prefix}_simulation_results.html" in dir_list
                else:
                    assert "simulation_results.csv" in dir_list
                    assert "selected_actions_count.csv" in dir_list
                    assert "positive_reward_proportion.csv" in dir_list
                    if visualize:
                        assert "simulation_results.html" in dir_list
