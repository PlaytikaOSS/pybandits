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

from typing import Dict, List
from unittest import mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pybandits.base import ActionId, Probability
from pybandits.model import Beta, BetaCC
from pybandits.strategy import (
    BestActionIdentification,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
)

########################################################################################################################

# ClassicBandit


def test_can_init_classic_bandit():
    ClassicBandit()


@given(
    st.lists(st.text(min_size=1), min_size=2, unique=True),
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=2),
)
def test_select_action_classic_bandit(a_list_str, a_list_float):
    p = dict(zip(a_list_str, a_list_float))

    c = ClassicBandit()
    assert max(p, key=p.get) == c.select_action(p=p)


# def test_is_compatible():
#     assert ClassicBandit().is_compatible_with_model(Beta())
# assert not ClassicBandit().is_compatible_with_model(BetaCC(cost=1))


########################################################################################################################


# BestActionIdentification


@given(st.floats())
def test_can_init_best_action_identification(a_float):
    # init default params
    b = BestActionIdentification()
    assert b.exploit_p == 0.5

    # init with input arguments
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            BestActionIdentification(exploit_p=a_float)
    else:
        b = BestActionIdentification(exploit_p=a_float)
        assert b.exploit_p == a_float


@given(st.floats())
def test_set_exploit_p(a_float):
    b = BestActionIdentification()

    # set with invalid float
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            b.set_exploit_p(exploit_p=a_float)
    # set with valid float
    else:
        b.set_exploit_p(exploit_p=a_float)
        assert b.exploit_p == a_float


@given(
    st.lists(st.text(min_size=1), min_size=2, unique=True),
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=2),
)
def test_select_action(a_list_str, a_list_float):
    p = dict(zip(a_list_str, a_list_float))
    b = BestActionIdentification()
    b.select_action(p=p)


@given(
    st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False),
)
def test_select_action_logic(a_float1, a_float2, a_float3):
    p = {"a1": a_float1, "a2": a_float2, "a3": a_float3}

    b = BestActionIdentification(exploit_p=1)
    # if exploit_p factor is 1 => return the action with 1st highest prob (max)
    assert max(p, key=p.get) == b.select_action(p=p)

    # if exploit_p factor is 0 => return the action with 2nd highest prob (not 1st highest prob)
    b.set_exploit_p(exploit_p=0)
    assert max(p, key=p.get) != b.select_action(p=p)
    assert sorted(p.items(), key=lambda x: x[1], reverse=True)[1][0] == b.select_action(p=p)


def test_select_action_logic_all_probs_equal():
    p = {"a1": 0.5, "a2": 0.5, "a3": 0.5}

    b = BestActionIdentification(exploit_p=1)
    # if exploit_p is 1 and all probs are equal => return the action with 1st highest prob (max)
    assert "a1" == b.select_action(p=p)

    # if exploit_p is 0 => return the action with 2nd highest prob (not 1st highest prob)
    b.set_exploit_p(exploit_p=0)
    assert "a2" == b.select_action(p=p)


@given(st.builds(Beta), st.builds(Beta), st.builds(Beta))
def test_compare_best_action(b1, b2, b3):
    b = BestActionIdentification()
    actions = {"a1": b1, "a2": b2, "a3": b3}

    pval = b.compare_best_actions(actions=actions)

    assert pval > 0


########################################################################################################################


# CostControlBandit


@given(st.floats())
def test_can_init_cost_control(a_float):
    # init with default arguments
    c = CostControlBandit()
    assert c.subsidy_factor == 0.5

    # init with input arguments
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            CostControlBandit(subsidy_factor=a_float)
    else:
        c = CostControlBandit(subsidy_factor=a_float)
        assert c.subsidy_factor == a_float


@given(st.floats())
def test_set_subsidy_factor(a_float):
    c = CostControlBandit()

    # set with invalid float
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            c.set_subsidy_factor(subsidy_factor=a_float)
    # set with valid float
    else:
        c.set_subsidy_factor(subsidy_factor=a_float)
        assert c.subsidy_factor == a_float


@given(
    st.lists(st.text(min_size=1), min_size=1),
    st.lists(st.floats(min_value=0, allow_infinity=False, allow_nan=False), min_size=1),
)
def test_select_action_cc(a_list_str, a_list_float):
    a_list_float_0_1 = [float(i) / (sum(a_list_float) + 1) for i in a_list_float]

    p = dict(zip(a_list_str, a_list_float_0_1))
    a = dict(zip(a_list_str, [BetaCC(cost=c) for c in a_list_float]))

    c = CostControlBandit()
    c.select_action(p=p, actions=a)


def test_select_action_logic_cc():
    actions_cost = {"a1": 10, "a2": 30, "a3": 20, "a4": 10, "a5": 20}
    p = {"a1": 0.1, "a2": 0.8, "a3": 0.6, "a4": 0.2, "a5": 0.65}

    actions = {
        "a1": BetaCC(cost=actions_cost["a1"]),
        "a2": BetaCC(cost=actions_cost["a2"]),
        "a3": BetaCC(cost=actions_cost["a3"]),
        "a4": BetaCC(cost=actions_cost["a4"]),
        "a5": BetaCC(cost=actions_cost["a5"]),
    }

    c = CostControlBandit(subsidy_factor=1)
    # if subsidy_factor is 1 => return the action with min cost and the highest sampled probability
    assert "a4" == c.select_action(p=p, actions=actions)

    # if subsidy_factor is 0 => return the action with highest p (classic bandit)
    c.set_subsidy_factor(subsidy_factor=0)
    assert "a2" == c.select_action(p=p, actions=actions)

    # otherwise, return the cheapest feasible action with the highest sampled probability
    assert "a5" == c.select_action(p=p, actions=actions)


@given(
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=3, max_size=3),
    st.lists(
        st.floats(min_value=0, allow_infinity=False, allow_nan=False),
        min_size=3,
        max_size=3,
    ),
)
def test_select_action_logic_corner_cases(a_list_p, a_list_cost):
    action_ids = ["a1", "a2", "a3"]

    p = dict(zip(action_ids, a_list_p))
    actions_cost = dict(zip(action_ids, a_list_cost))

    actions = {
        "a1": BetaCC(cost=actions_cost["a1"]),
        "a2": BetaCC(cost=actions_cost["a2"]),
        "a3": BetaCC(cost=actions_cost["a3"]),
    }

    c = CostControlBandit(subsidy_factor=1)
    # if cost factor is 1 => return the action with min cost
    assert min(actions_cost, key=actions_cost.get) == c.select_action(p=p, actions=actions)

    # if cost factor is 0:
    c.set_subsidy_factor(subsidy_factor=0)
    # get the keys of the max p.values() (there might be more max_p_values)
    max_p_values = [k for k, v in p.items() if v == max(p.values())]

    # if cost factor is 0 and only 1  max_value  => return the action with highest p (classic bandit)
    # e.g. p={"a1": 0.5, "a2": 0.2} => return always "a1"
    if len(max_p_values) == 1:
        assert max(p, key=p.get) == c.select_action(p=p, actions=actions)
    # if cost factor is 0 and only 1+ max_values => return the action with highest p and min cost
    # e.g. p={"a1": 0.5, "a2": 0.5} and cost={"a1": 20, "a2": 10} => return always "a2"
    else:
        actions_cost_max = {k: actions_cost[k] for k in max_p_values}
        min(actions_cost_max, key=actions_cost_max.get) == c.select_action(p=p, actions=actions)


# def test_cc_is_compatible():
#     assert not CostControlBandit().is_compatible_with_model(Beta())
#     assert CostControlBandit().is_compatible_with_model(BetaCC(cost=1))


########################################################################################################################


# MultiObjectiveBandit


def test_can_init_multiobjective():
    MultiObjectiveBandit()


@given(
    st.dictionaries(
        st.text(min_size=1), st.lists(st.floats(min_value=0, max_value=1), min_size=3, max_size=3), min_size=3
    )
)
def test_select_action_mo(p: Dict[ActionId, List[Probability]]):
    # patch pareto_front method
    pareto_set = np.random.choice(list(p.keys()), 3, replace=False)
    pareto_front_mock = mock.Mock(return_value=pareto_set)
    with mock.patch.object(MultiObjectiveBandit, "get_pareto_front", pareto_front_mock):
        # verify the result is within pareto_set
        m = MultiObjectiveBandit()
        assert m.select_action(p) in pareto_set


def test_pareto_front():
    m = MultiObjectiveBandit()

    # works in 2D

    #    +
    # .3 |     X
    #    |
    # .2 |          X
    #    |
    # .1 |      X       X
    #    |
    #  0 | X            X
    #    +-----------------+
    #      0   .1  .2  .3

    p2d = {
        "a0": [0.1, 0.3],
        "a1": [0.1, 0.3],
        "a2": [0.0, 0.0],
        "a3": [0.1, 0.1],
        "a4": [0.3, 0.1],
        "a5": [0.2, 0.2],
        "a6": [0.3, 0.0],
        "a7": [0.1, 0.1],
    }

    assert m.get_pareto_front(p2d) == ["a0", "a1", "a4", "a5"]

    p2d = {
        "a0": [0.1, 0.1],
        "a1": [0.3, 0.3],
        "a2": [0.3, 0.3],
    }

    assert m.get_pareto_front(p2d) == ["a1", "a2"]

    # works in 3D
    p3d = {
        "a0": [0.1, 0.3, 0.1],
        "a1": [0.1, 0.3, 0.1],
        "a2": [0.0, 0.0, 0.1],
        "a3": [0.1, 0.1, 0.1],
        "a4": [0.3, 0.1, 0.1],
        "a5": [0.2, 0.2, 0.1],
        "a6": [0.3, 0.0, 0.1],
        "a7": [0.1, 0.1, 0.3],
    }

    assert m.get_pareto_front(p3d) == ["a0", "a1", "a4", "a5", "a7"]
