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

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pybandits.model import Beta, BetaCC, BetaMO

########################################################################################################################

# Beta


@given(st.integers(), st.integers())
def test_can_init_beta(success_counter, failure_counter):
    if (success_counter <= 0) or (failure_counter <= 0):
        with pytest.raises(ValidationError):
            Beta(n_successes=success_counter, n_failures=failure_counter)
    else:
        b = Beta(n_successes=success_counter, n_failures=failure_counter)
        assert (b.n_successes, b.n_failures) == (success_counter, failure_counter)

        b = Beta()
        assert (b.n_successes, b.n_failures) == (1, 1)


def test_both_or_neither_counters_are_defined():
    with pytest.raises(ValidationError):
        Beta(n_successes=0)
    with pytest.raises(ValidationError):
        Beta(n_failures=0)
    with pytest.raises(ValidationError):
        Beta(n_successes=1, n_failures=None)
    with pytest.raises(ValidationError):
        Beta(n_successes=None, n_failures=0)


@given(st.lists(st.integers(min_value=0, max_value=1)))
def test_beta_update(rewards):
    b = Beta(n_successes=1, n_failures=2)
    b.update(rewards=rewards)
    assert b == Beta(
        n_successes=1 + sum(rewards), n_failures=2 + (len(rewards) - sum(rewards))
    )


@given(st.builds(Beta))
def test_beta_get_stats_is_working(e: Beta):
    assert e.mean >= 0, "Mean negative"
    assert e.std >= 0, "Std negative"
    assert e.count >= 2, "Count too low"


def test_beta_sample_proba():
    b = Beta(n_successes=1, n_failures=2)

    for _ in range(1000):
        prob = b.sample_proba()
        assert prob >= 0 and prob <= 1


########################################################################################################################


# BetaCC


@given(st.floats())
def test_can_init_betaCC(a_float):
    if a_float < 0 or np.isnan(a_float):
        with pytest.raises(ValidationError):
            BetaCC(cost=a_float)
    else:
        b = BetaCC(cost=a_float)
        assert b.cost == a_float


########################################################################################################################


# BetaMO


def test_can_init_beta_mo():
    # init with default params
    b = BetaMO(counters=[Beta(), Beta()])
    assert b.counters[0].n_successes == 1 and b.counters[0].n_failures == 1
    assert b.counters[1].n_successes == 1 and b.counters[1].n_failures == 1

    # init with empty dict
    b = BetaMO(counters=[{}, {}])
    assert b.counters[0] == Beta()

    # invalid init with BetaCC instead of Beta
    with pytest.raises(ValidationError):
        BetaMO(counters=[BetaCC(cost=1), BetaCC(cost=1)])


def test_calculate_proba_beta_mo():
    b = BetaMO(counters=[Beta(), Beta()])
    b.sample_proba()


@given(
    st.lists(st.integers(min_value=0, max_value=1)),
    st.lists(st.integers(min_value=0, max_value=1)),
)
def test_beta_update_mo(rewards1, rewards2):
    min_len = min([len(rewards1), len(rewards2)])
    rewards1, rewards2 = rewards1[:min_len], rewards2[:min_len]
    rewards = [[a, b] for a, b in zip(rewards1, rewards2)]

    b = BetaMO(
        counters=[
            Beta(n_successes=11, n_failures=22),
            Beta(n_successes=33, n_failures=44),
        ]
    )

    b.update(rewards=rewards)

    assert b == BetaMO(
        counters=[
            Beta(
                n_successes=11 + sum(rewards1),
                n_failures=22 + len(rewards1) - sum(rewards1),
            ),
            Beta(
                n_successes=33 + sum(rewards2),
                n_failures=44 + len(rewards2) - sum(rewards2),
            ),
        ]
    )

    with pytest.raises(AttributeError):
        b.update(rewards=[[1, 1], [1], [0, 1]])
