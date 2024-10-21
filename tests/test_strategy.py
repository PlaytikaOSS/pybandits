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

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pybandits.base import ActionId, Probability
from pybandits.model import Beta, BetaCC, BetaMOCC
from pybandits.pydantic_version_compatibility import ValidationError
from pybandits.strategy import (
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
    MultiObjectiveStrategy,
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


########################################################################################################################


# BestActionIdentificationBandit


@given(st.floats())
def test_can_init_best_action_identification(a_float):
    # init default params
    b = BestActionIdentificationBandit()
    assert b.exploit_p == 0.5

    # init with input arguments
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            BestActionIdentificationBandit(exploit_p=a_float)
    else:
        b = BestActionIdentificationBandit(exploit_p=a_float)
        assert b.exploit_p == a_float


@given(st.floats())
def test_with_exploit_p(a_float):
    b = BestActionIdentificationBandit()

    # set with invalid float
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            b.with_exploit_p(exploit_p=a_float)
    # set with valid float
    else:
        mutated_b = b.with_exploit_p(exploit_p=a_float)
        assert mutated_b.exploit_p == a_float
        assert mutated_b is not b


@given(
    st.lists(st.text(min_size=1), min_size=2, unique=True),
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=2),
)
def test_select_action(a_list_str, a_list_float):
    p = dict(zip(a_list_str, a_list_float))
    b = BestActionIdentificationBandit()
    b.select_action(p=p)


@given(
    st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False),
)
def test_select_action_logic(a_float1, a_float2, a_float3):
    p = {"a1": a_float1, "a2": a_float2, "a3": a_float3}

    b = BestActionIdentificationBandit(exploit_p=1)
    # if exploit_p factor is 1 => return the action with 1st highest prob (max)
    assert max(p, key=p.get) == b.select_action(p=p)

    # if exploit_p factor is 0 => return the action with 2nd highest prob (not 1st highest prob)
    mutated_b = b.with_exploit_p(exploit_p=0)
    assert max(p, key=p.get) != mutated_b.select_action(p=p)
    assert sorted(p.items(), key=lambda x: x[1], reverse=True)[1][0] == mutated_b.select_action(p=p)


def test_select_action_logic_all_probs_equal():
    p = {"a1": 0.5, "a2": 0.5, "a3": 0.5}

    b = BestActionIdentificationBandit(exploit_p=1)
    # if exploit_p is 1 and all probs are equal => return the action with 1st highest prob (max)
    assert "a1" == b.select_action(p=p)

    # if exploit_p is 0 => return the action with 2nd highest prob (not 1st highest prob)
    mutated_b = b.with_exploit_p(exploit_p=0)
    assert "a2" == mutated_b.select_action(p=p)


@given(st.builds(Beta), st.builds(Beta), st.builds(Beta))
def test_compare_best_action(b1, b2, b3):
    b = BestActionIdentificationBandit()
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
def test_with_subsidy_factor(a_float):
    c = CostControlBandit()

    # set with invalid float
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            c.with_subsidy_factor(subsidy_factor=a_float)
    # set with valid float
    else:
        mutated_c = c.with_subsidy_factor(subsidy_factor=a_float)
        assert mutated_c.subsidy_factor == a_float
        assert mutated_c is not c


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
    mutated_c = c.with_subsidy_factor(subsidy_factor=0)
    assert "a2" == mutated_c.select_action(p=p, actions=actions)

    # otherwise, return the cheapest feasible action with the highest sampled probability
    mutated_c = c.with_subsidy_factor(subsidy_factor=0.5)
    assert "a5" == mutated_c.select_action(p=p, actions=actions)


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
    actions_cost_proba = [(a_cost, -a_proba, a_id) for a_id, a_cost, a_proba in zip(action_ids, a_list_cost, a_list_p)]

    actions = {
        "a1": BetaCC(cost=actions_cost["a1"]),
        "a2": BetaCC(cost=actions_cost["a2"]),
        "a3": BetaCC(cost=actions_cost["a3"]),
    }

    c = CostControlBandit(subsidy_factor=1)
    # if cost factor is 1 return:
    # - the action with the min cost, or
    # - the highest probability in case of cost equality, or
    # - the lowest action id (alphabetically) in case of equal cost and probability
    assert sorted(actions_cost_proba)[0][-1] == c.select_action(p=p, actions=actions)

    # if cost factor is 0:
    mutated_c = c.with_subsidy_factor(subsidy_factor=0)
    # get the keys of the max p.values() (there might be more max_p_values)
    max_p_values = [k for k, v in p.items() if v == max(p.values())]

    # if cost factor is 0 and only 1 max_value => return the action with highest p (classic bandit)
    # e.g. p={"a1": 0.5, "a2": 0.2} => return always "a1"
    if len(max_p_values) == 1:
        assert max(p, key=p.get) == mutated_c.select_action(p=p, actions=actions)

    # if cost factor is 0 and only 1+ max_values => return the action with highest p and min cost
    # e.g. p={"a1": 0.5, "a2": 0.5} and cost={"a1": 20, "a2": 10} => return always "a2"
    else:
        actions_cost_max = {k: actions_cost[k] for k in max_p_values}
        assert min(actions_cost_max, key=actions_cost_max.get) == mutated_c.select_action(p=p, actions=actions)


########################################################################################################################


# MultiObjectiveBandit


def test_can_init_multiobjective():
    MultiObjectiveBandit()


@given(
    st.dictionaries(
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cc", "Cs"))),
        st.lists(st.floats(min_value=0, max_value=1), min_size=3, max_size=3),
        min_size=3,
    )
)
def test_select_action_mo(p: Dict[ActionId, List[Probability]]):
    m = MultiObjectiveBandit()
    assert m.select_action(p=p) in m.get_pareto_front(p=p)


def test_pareto_front():
    # works in 2D
    #
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

    assert MultiObjectiveStrategy.get_pareto_front(p2d) == ["a0", "a1", "a4", "a5"]

    p2d = {
        "a0": [0.1, 0.1],
        "a1": [0.3, 0.3],
        "a2": [0.3, 0.3],
    }

    assert MultiObjectiveStrategy.get_pareto_front(p2d) == ["a1", "a2"]

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

    assert MultiObjectiveStrategy.get_pareto_front(p3d) == ["a0", "a1", "a4", "a5", "a7"]


########################################################################################################################


# MultiObjectiveCostControlBandit


def test_can_init_multiobjective_mo_cc():
    MultiObjectiveCostControlBandit()


def test_select_action_mo_cc():
    m = MultiObjectiveCostControlBandit()

    actions = {
        "a1": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=8),
        "a2": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=2),
        "a3": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=5),
        "a4": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=1),
        "a5": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=7),
    }
    p = {
        "a1": [0.1, 0.3, 0.5],
        "a2": [0.1, 0.3, 0.5],
        "a3": [0.0, 0.4, 0.4],
        "a4": [0.5, 0.3, 0.7],
        "a5": [0.6, 0.1, 0.5],
    }
    # within the pareto front ("a3", "a4", "a5") select the action with min cost ("a4")
    assert m.get_pareto_front(p) == ["a3", "a4", "a5"]
    assert m.select_action(p=p, actions=actions) == "a4"

    actions = {
        "a1": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=2),
        "a2": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=2),
        "a3": BetaMOCC(counters=[Beta(), Beta(), Beta()], cost=5),
    }
    p = {
        "a1": [0.6, 0.1, 0.1],
        "a2": [0.5, 0.8, 0.8],
        "a3": [0.0, 0.1, 0.9],
    }
    # within the actions with the min cost ("a1" or "a2") select the action the highest mean of probabilities ("a2")
    assert m.get_pareto_front(p) == ["a1", "a2", "a3"]
    assert m.select_action(p=p, actions=actions) == "a2"
