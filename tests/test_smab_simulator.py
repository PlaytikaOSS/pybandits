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
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from pybandits.model import Beta
from pybandits.quantitative_model import SmabZoomingModel
from pybandits.smab import SmabBernoulli
from pybandits.smab_simulator import SmabSimulator


def test_mismatched_probs_reward_columns(mocker: MockerFixture):
    smab = mocker.Mock(spec=SmabBernoulli)
    smab.actions = {"a1": mocker.Mock(), "a2": mocker.Mock()}
    smab.epsilon = 0.0
    smab.default_action = None
    probs_reward = {str(i): {"a1": 0.5, "a2": 0.5} for i in range(2)}
    with pytest.raises(ValueError):
        SmabSimulator(mab=smab, probs_reward=probs_reward)


@pytest.mark.parametrize(
    "probability, is_quantitative_action, should_pass",
    [
        # Valid non-quantitative actions
        (0.5, False, True),
        (0.0, False, True),
        (1.0, False, True),
        # Invalid non-quantitative actions - not float
        ("0.5", False, False),
        (None, False, False),
        (lambda x: 0.5, False, False),
        # Invalid non-quantitative actions - out of range
        (-0.1, False, False),
        (1.1, False, False),
        # Valid quantitative actions
        (lambda x: 0.5, True, True),
        (lambda quantity: 0.7, True, True),
        (lambda *args: 0.5, True, True),
        # Invalid quantitative actions - not callable
        (0.5, True, False),
        (None, True, False),
        # Invalid quantitative actions - wrong argument count
        (lambda: 0.5, True, False),
        (lambda x, y: 0.5, True, False),
    ],
)
def test_validate_probs_reward_values(probability, is_quantitative_action, should_pass):
    """
    Test the _validate_probs_reward_values method with various combinations
    of probability values and action types.

    Parameters
    ----------
    probability : Union[float, callable]
        The probability value to test
    is_quantitative_action : bool
        Whether the action is quantitative
    should_pass : bool
        Whether the validation should pass
    """
    if should_pass:
        # Should not raise any exception
        SmabSimulator._validate_probs_reward_values(probability, is_quantitative_action)
    else:
        # Should raise ValueError
        with pytest.raises(ValueError):
            SmabSimulator._validate_probs_reward_values(probability, is_quantitative_action)


@settings(deadline=None)
@given(
    action_ids=st.just(["a1", "a2"]),
    models=st.lists(st.sampled_from([Beta(), SmabZoomingModel.cold_start()]), min_size=2, max_size=2),
)
def test_smab_e2e_simulation_with_default_args(action_ids, models, monkeymodule):
    monkeymodule.setattr(SmabSimulator, "_maximize_prob_reward", lambda *args, **kwargs: np.random.random())
    mab = SmabBernoulli(actions=dict(zip(action_ids, models)))
    with TemporaryDirectory() as path:
        simulator = SmabSimulator(mab=mab, visualize=True, save=True, path=path)
        simulator.run()
        assert not simulator.results.empty
        dir_list = os.listdir(path)
        assert "simulation_results.csv" in dir_list
        assert "selected_actions_count.csv" in dir_list
        assert "positive_reward_proportion.csv" in dir_list
        assert "simulation_results.html" in dir_list


@settings(deadline=None)
@given(
    action_ids=st.just(["a1", "a2"]),
    models=st.lists(st.sampled_from([Beta(), SmabZoomingModel.cold_start()]), min_size=2, max_size=2),
    n_updates=st.integers(min_value=1, max_value=10),
    batch_size=st.integers(min_value=1, max_value=10),
    save=st.booleans(),
    random_seed=st.sampled_from([None, 0, 42]),
    verbose=st.booleans(),
    visualize=st.booleans(),
    file_prefix=st.sampled_from(["", "unit_test"]),
)
def test_smab_e2e_simulation_with_non_default_args(
    action_ids, models, n_updates, batch_size, save, random_seed, verbose, visualize, file_prefix, monkeymodule
):
    monkeymodule.setattr(
        SmabSimulator,
        "_maximize_prob_reward",
        lambda *args, **kwargs: np.random.random(),
    )
    mab = SmabBernoulli(actions=dict(zip(action_ids, models)))
    if visualize and not save:
        with pytest.raises(ValueError):
            SmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=None,
                verbose=verbose,
                file_prefix=file_prefix,
            )
    else:
        with TemporaryDirectory() as path:
            simulator = SmabSimulator(
                mab=mab,
                visualize=visualize,
                save=save,
                path=path,
                n_updates=n_updates,
                batch_size=batch_size,
                random_seed=random_seed,
                probs_reward=None,
                verbose=verbose,
                file_prefix=file_prefix,
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
