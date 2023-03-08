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

from copy import deepcopy
from typing import List

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pybandits.base import BinaryReward
from pybandits.model import Beta, BetaCC, BetaMO
from pybandits.smab import (
    SmabBernoulli,
    SmabBernoulliBAI,
    SmabBernoulliCC,
    SmabBernoulliMO,
    create_smab_bernoulli_bai_cold_start,
    create_smab_bernoulli_cc_cold_start,
    create_smab_bernoulli_cold_start,
    create_smab_bernoulli_mo_cold_start,
)
from pybandits.strategy import ClassicBandit, CostControlBandit, MultiObjectiveBandit

########################################################################################################################


# SmabBernoulli with strategy=ClassicBandit()


def test_create_smab_bernoulli_cold_start():
    assert create_smab_bernoulli_cold_start(action_ids=["a1", "a2"]) == SmabBernoulli(
        actions={"a1": Beta(), "a2": Beta()},
    )


@given(st.integers(min_value=0, max_value=1), st.integers(min_value=0, max_value=1))
def test_base_smab_update_ok(r1, r2):
    mab = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
    mab.update(actions=["a1", "a2"], rewards=[r1, r2])
    mab.update(actions=["a1", "a1"], rewards=[r1, r2])


def test_can_instantiate_smab():
    with pytest.raises(TypeError):
        SmabBernoulli()
    with pytest.raises(AttributeError):
        SmabBernoulli(actions={})
    with pytest.raises(AttributeError):
        SmabBernoulli(actions={"action1": Beta()})
    with pytest.raises(TypeError):  # strategy is not an argument of init
        SmabBernoulli(
            actions={
                "action1": Beta(),
                "action2": Beta(),
            },
            strategy=ClassicBandit(),
        )
    with pytest.raises(ValidationError):
        SmabBernoulli(
            actions={
                "action1": None,
                "action2": None,
            },
        )
    smab = SmabBernoulli(
        actions={
            "action1": Beta(),
            "action2": Beta(),
        },
    )

    assert smab.actions["action1"] == Beta()
    assert smab.actions["action2"] == Beta()


@given(
    st.integers(min_value=1),
    st.integers(min_value=1),
)
def test_can_instantiate_smab_with_params(a, b):
    s = SmabBernoulli(
        actions={
            "action1": Beta(n_successes=a, n_failures=b),
            "action2": Beta(n_successes=a, n_failures=b),
        },
    )
    assert (s.actions["action1"].n_successes == a) and (s.actions["action1"].n_failures == b)
    assert s.actions["action1"] == s.actions["action2"]


@given(st.integers(max_value=0))
def test_smab_predict_raise_when_samples_low(n_samples):
    s = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
    with pytest.raises(ValidationError):
        s.predict(n_samples=n_samples)


def test_smab_predict_raise_when_all_actions_forbidden():
    s = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
    with pytest.raises(ValueError):
        s.predict(n_samples=10, forbidden_actions=["a1", "a2"])


def test_smab_predict():
    s = SmabBernoulli(
        actions={
            "a0": Beta(),
            "a1": Beta(n_successes=5, n_failures=5),
            "forb_1": Beta(n_successes=10, n_failures=1),
            "best": Beta(n_successes=10, n_failures=5),
            "forb_2": Beta(n_successes=100, n_failures=4),
            "a5": Beta(),
        },
    )
    forbidden_actions = set(["forb_1", "forb_2"])

    best_actions, probs = s.predict(n_samples=1000, forbidden_actions=forbidden_actions)
    assert ["forb1" not in p.keys() for p in probs], "forbidden actions weren't removed from the output"

    valid_actions = set(s.actions.keys()) - forbidden_actions
    for probas, best_action in zip(probs, best_actions):
        assert set(probas.keys()) == valid_actions, "restituted actions don't match valid actions"

        best_proba = probas[best_action]
        assert best_proba == max(probas.values()), "best action hasn't the best probability"


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=1),
    st.lists(st.integers(min_value=0, max_value=1), min_size=1),
)
def test_smab_update(rewards: List[BinaryReward], rewards_1: List[BinaryReward]):
    updated = SmabBernoulli(
        actions={
            "a0": Beta(),
            "a1": Beta(),
        },
    )
    batch_updated = deepcopy(updated)

    # update the model sequentially
    [updated.update(actions=["a0"], rewards=[reward]) for reward in rewards]
    [updated.update(actions=["a1"], rewards=[reward]) for reward in rewards_1]

    # update the model in batch
    batch_updated.update(actions=["a0"] * len(rewards) + ["a1"] * len(rewards_1), rewards=rewards + rewards_1)

    assert updated == batch_updated, "update() has different result when each item is applied separately"

    sum_failures = sum([1 - x for x in rewards])
    assert updated.actions["a0"] == Beta(
        n_successes=1 + sum(rewards), n_failures=1 + sum_failures
    ), "Unexpected results in counter"

    sum_failures_1 = sum([1 - x for x in rewards_1])
    assert updated.actions["a1"] == Beta(
        n_successes=1 + sum(rewards_1), n_failures=1 + sum_failures_1
    ), "Unexpected results in counter"


@given(st.text())
def test_smab_accepts_only_valid_actions(s):
    if s == "":
        with pytest.raises(ValidationError):
            SmabBernoulli(
                actions={
                    s: Beta(),
                    s + "_": Beta(),
                }
            )
    else:
        SmabBernoulli(actions={s: Beta(), s + "_": Beta()})


########################################################################################################################


# SmabBernoulli with strategy=BestActionIdentification()


def test_create_smab_bernoulli_bai():
    assert create_smab_bernoulli_bai_cold_start(action_ids=["a1", "a2"], exploit_p=0.2) == SmabBernoulliBAI(
        actions={"a1": Beta(), "a2": Beta()},
        exploit_p=0.2,
    )

    assert create_smab_bernoulli_bai_cold_start(action_ids=["a1", "a2"]) == SmabBernoulliBAI(
        actions={"a1": Beta(), "a2": Beta()},
    )


def test_can_init_smabbai():
    # init default params
    s = SmabBernoulliBAI(
        actions={
            "a1": Beta(),
            "a2": Beta(),
        },
    )

    assert s.actions["a1"] == Beta()
    assert s.actions["a2"] == Beta()
    assert s.strategy.exploit_p == 0.5

    # init input params
    s = SmabBernoulliBAI(
        actions={
            "a1": Beta(n_successes=1, n_failures=2),
            "a2": Beta(n_successes=3, n_failures=4),
        },
        exploit_p=0.3,
    )
    assert s.actions["a1"] == Beta(n_successes=1, n_failures=2)
    assert s.actions["a2"] == Beta(n_successes=3, n_failures=4)
    assert s.strategy.exploit_p == 0.3


def test_smabbai_predict():
    s = SmabBernoulliBAI(actions={"a1": Beta(), "a2": Beta()})
    _, _ = s.predict(n_samples=1000)


def test_smabbai_update():
    s = SmabBernoulliBAI(actions={"a1": Beta(), "a2": Beta()})
    s.update(actions=["a1", "a1"], rewards=[1, 0])


def test_smabbai_with_betacc():
    # Fails because smab bernoulli with BAI shouldn't support BetaCC
    with pytest.raises(ValidationError):
        SmabBernoulliBAI(
            actions={
                "a1": BetaCC(cost=10),
                "a2": BetaCC(cost=20),
            },
        )


########################################################################################################################


# SmabBernoulli with strategy=CostControlBandit()


def test_create_smab_bernoulli_cc():
    assert create_smab_bernoulli_cc_cold_start(
        dict_action_ids_cost={"a1": 10, "a2": 20},
        subsidy_factor=0.2,
    ) == SmabBernoulliCC(
        actions={"a1": BetaCC(cost=10), "a2": BetaCC(cost=20)},
        subsidy_factor=0.2,
    )

    assert create_smab_bernoulli_cc_cold_start(dict_action_ids_cost={"a1": 10, "a2": 20}) == SmabBernoulliCC(
        actions={"a1": BetaCC(cost=10), "a2": BetaCC(cost=20)},
    )


def test_can_init_smabcc():
    # init default arguments
    s = SmabBernoulliCC(
        actions={
            "a1": BetaCC(cost=10),
            "a2": BetaCC(cost=20),
        },
    )
    assert s.actions["a1"] == BetaCC(cost=10)
    assert s.actions["a2"] == BetaCC(cost=20)
    assert s.strategy.subsidy_factor == 0.5

    # init with input args
    s = SmabBernoulliCC(
        actions={
            "a1": BetaCC(n_successes=1, n_failures=2, cost=10),
            "a2": BetaCC(n_successes=3, n_failures=4, cost=20),
        },
        subsidy_factor=0.7,
    )
    assert s.actions["a1"] == BetaCC(n_successes=1, n_failures=2, cost=10)
    assert s.actions["a2"] == BetaCC(n_successes=3, n_failures=4, cost=20)
    assert s.strategy == CostControlBandit(subsidy_factor=0.7)
    assert s.strategy.subsidy_factor == 0.7


def test_smabcc_predict():
    s = SmabBernoulliCC(
        actions={
            "a1": BetaCC(n_successes=1, n_failures=2, cost=10),
            "a2": BetaCC(n_successes=3, n_failures=4, cost=20),
        },
        subsidy_factor=0.7,
    )
    _, _ = s.predict(n_samples=1000)


def test_smabcc_update():
    s = SmabBernoulliCC(actions={"a1": BetaCC(cost=10), "a2": BetaCC(cost=10)})
    s.update(actions=["a1", "a1"], rewards=[1, 0])


########################################################################################################################


# SmabBernoulli with strategy=MultiObjectiveBandit()


@given(st.lists(st.integers(min_value=1), min_size=6, max_size=6))
def test_can_init_smab_mo(a_list):
    a, b, c, d, e, f = a_list

    s = SmabBernoulliMO(
        actions={
            "a1": BetaMO(
                counters=[
                    Beta(n_successes=a, n_failures=b),
                    Beta(n_successes=c, n_failures=d),
                    Beta(n_successes=e, n_failures=f),
                ]
            ),
            "a2": BetaMO(
                counters=[
                    Beta(n_successes=d, n_failures=a),
                    Beta(n_successes=e, n_failures=b),
                    Beta(n_successes=f, n_failures=c),
                ]
            ),
        },
    )
    assert s.actions["a1"] == BetaMO(
        counters=[
            Beta(n_successes=a, n_failures=b),
            Beta(n_successes=c, n_failures=d),
            Beta(n_successes=e, n_failures=f),
        ]
    )
    assert s.actions["a2"] == BetaMO(
        counters=[
            Beta(n_successes=d, n_failures=a),
            Beta(n_successes=e, n_failures=b),
            Beta(n_successes=f, n_failures=c),
        ]
    )
    assert s.strategy == MultiObjectiveBandit()


def test_all_actions_must_have_same_number_of_objectives():
    with pytest.raises(ValueError):
        SmabBernoulliMO(
            actions={
                "action 1": BetaMO(counters=[Beta(), Beta()]),
                "action 2": BetaMO(counters=[Beta(), Beta()]),
                "action 3": BetaMO(counters=[Beta(), Beta(), Beta()]),
            },
        )


def test_init_smab_mo_predict():
    n_samples = 3

    s = create_smab_bernoulli_mo_cold_start(action_ids=["a1", "a2"], n_objectives=3)

    forbidden = None
    s.predict(n_samples=n_samples, forbidden_actions=forbidden)

    forbidden = ["a1"]
    predicted_actions, _ = s.predict(n_samples=n_samples, forbidden_actions=forbidden)

    assert "a1" not in predicted_actions

    forbidden = ["a1", "a2"]
    with pytest.raises(ValueError):
        s.predict(n_samples=n_samples, forbidden_actions=forbidden)

    forbidden = ["a1", "a2", "a3"]
    with pytest.raises(ValueError):
        s.predict(n_samples=n_samples, forbidden_actions=forbidden)

    forbidden = ["a1", "a3"]
    with pytest.raises(ValueError):
        s.predict(n_samples=n_samples, forbidden_actions=forbidden)
