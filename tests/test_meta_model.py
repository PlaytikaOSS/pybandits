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

"""Tests for the meta-model abstraction layer (pybandits.meta_model)."""

import pickle
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

import pybandits.model as model_mod
from pybandits.base import ACTION_IDS_PREFIX, QUANTITATIVE_ACTION_IDS_PREFIX
from pybandits.meta_model import (
    BaseMetaModel,
    SmabMetaModel,
    SmabMetaModelMO,
    SmabMetaModelSO,
)
from pybandits.model import Beta, BetaCC, BetaMO
from pybandits.utils import classproperty
from tests.utils import make_action_ids

########################################################################################################################
# Shared fixtures


@pytest.fixture
def make_beta_actions():
    """Factory returning a fresh ``{action_id: Beta()}`` dict each call.

    Defaults to 2 actions; pass ``n`` to vary. Each call builds *new* model
    instances so tests stay independent.
    """

    def _make(n: int = 2, factory=Beta):
        return {a: factory() for a in make_action_ids(n)}

    return _make


@pytest.fixture
def make_beta_meta(make_beta_actions):
    """Factory returning a fresh ``SmabMetaModel`` over N Beta actions."""

    def _make(n: int = 2, factory=Beta):
        return SmabMetaModel(actions=make_beta_actions(n, factory))

    return _make


@pytest.fixture
def spy_beta_update(monkeypatch) -> List[Dict[str, Any]]:
    """Replace ``BaseBeta.update`` with a kwargs-capturing spy; return the captured list."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(model_mod.BaseBeta, "update", lambda self, **kw: captured.append(kw), raising=True)
    return captured


########################################################################################################################
# SmabMetaModel — core behaviour


def test_base_meta_model_is_abstract():
    """BaseMetaModel cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        BaseMetaModel()


@pytest.mark.parametrize("n", [2, 4, 8])
def test_independent_actions_meta_model_construction(make_beta_meta, n):
    meta = make_beta_meta(n)
    assert isinstance(meta, BaseMetaModel)
    # Typed alias must also construct.
    SmabMetaModelSO(actions=meta.actions.copy())


@pytest.mark.parametrize("n", [2, 4, 8])
def test_action_ids_property(make_beta_meta, n):
    meta = make_beta_meta(n)
    assert set(meta.action_ids) == set(make_action_ids(n))


@pytest.mark.parametrize("n", [2, 4])
def test_actions_field_preserves_model_instances(make_beta_actions, n):
    """The ``actions`` field exposes the same model instances stored by the meta-model.

    Callers that access per-action state directly (e.g. cost retrieval,
    change-point detection) must see the same model objects via the manager's
    ``actions`` property as the meta-model holds internally.
    """
    actions = make_beta_actions(n)
    meta = SmabMetaModel(actions=actions)
    for a, beta in actions.items():
        assert meta.actions[a] is beta


@pytest.mark.parametrize("n_actions, n_samples", [(2, 1), (3, 3), (5, 7)])
def test_sample_proba_dispatches_to_each_action(make_beta_meta, rng, n_actions, n_samples):
    meta = make_beta_meta(n_actions)
    result = meta.sample_proba(rng=rng, n_samples=n_samples)
    assert set(result.keys()) == set(meta.action_ids)
    assert all(len(v) == n_samples for v in result.values())


@pytest.mark.parametrize("n_actions, valid_ratio, n_samples", [(4, 0.5, 2), (4, 1.0, 3), (8, 0.25, 5)])
def test_sample_proba_restricts_to_valid_action_ids(make_beta_meta, rng, n_actions, valid_ratio, n_samples):
    meta = make_beta_meta(n_actions)
    valid = set(list(meta.action_ids)[: max(1, int(len(meta.action_ids) * valid_ratio))])
    result = meta.sample_proba(rng=rng, valid_action_ids=valid, n_samples=n_samples)
    assert set(result.keys()) == valid


@pytest.mark.parametrize("rewards_per_action", [(2, 0), (3, 1), (5, 2)])
def test_update_dispatches_per_action_with_no_context(make_beta_meta, rewards_per_action):
    """smab-flavored update path: rewards only, no context.

    For each action we issue ``rewards_per_action[i]`` successful pulls;
    other actions should remain at their cold-start state.
    """
    meta = make_beta_meta(len(rewards_per_action))
    actions_arg, rewards_arg = [], []
    expected_delta = {}
    for action_id, n in zip(meta.action_ids, rewards_per_action):
        actions_arg.extend([action_id] * n)
        rewards_arg.extend([1] * n)
        expected_delta[action_id] = n
    initial = {a: meta.actions[a].n_successes for a in meta.action_ids}

    meta.update(actions=actions_arg, rewards=rewards_arg)

    for a in meta.action_ids:
        assert meta.actions[a].n_successes == initial[a] + expected_delta[a]


@pytest.mark.parametrize("n", [2, 4])
def test_update_resets_via_reset_method(make_beta_meta, n):
    meta = make_beta_meta(n)
    action_ids = list(meta.action_ids)
    meta.update(actions=action_ids, rewards=[1] * n)
    assert all(meta.actions[a].n_successes > 1 for a in action_ids)
    meta.reset()
    assert all(meta.actions[a].n_successes == 1 for a in action_ids)
    assert all(meta.actions[a].n_failures == 1 for a in action_ids)


@pytest.mark.parametrize("n_actions, batch_size", [(2, 5), (3, 1), (4, 10)])
def test_update_with_only_some_actions_in_batch_leaves_others_untouched(make_beta_actions, n_actions, batch_size):
    """Update only the first action's model; others must remain at cold-start state."""
    actions = make_beta_actions(n_actions)
    targeted = next(iter(actions))
    others_before = {a: (m.n_successes, m.n_failures) for a, m in actions.items() if a != targeted}
    meta = SmabMetaModel(actions=actions)

    meta.update(actions=[targeted] * batch_size, rewards=[1] * batch_size)

    assert actions[targeted].n_successes > 1
    for a, before in others_before.items():
        assert (actions[a].n_successes, actions[a].n_failures) == before


@pytest.mark.parametrize(
    "quantities, forwarded",
    [
        # All-None quantities → quantities arg should NOT be forwarded.
        ([None, None], False),
        # Any truthy quantity → full list passed through unchanged.
        ([None, 1.5], True),
        ([0.5, 2.0], True),
    ],
)
def test_update_with_quantities_only_passes_quantities_when_any_present(
    make_beta_meta, spy_beta_update, quantities, forwarded
):
    """Mirrors the historical cmab behaviour for the ``any(quantities)`` guard."""
    meta = make_beta_meta(1)
    a0 = meta.action_ids[0]
    meta.update(actions=[a0] * len(quantities), rewards=[1] * len(quantities), quantities=quantities)
    assert len(spy_beta_update) == 1
    if forwarded:
        assert spy_beta_update[0]["quantities"] == quantities
    else:
        assert "quantities" not in spy_beta_update[0]


@pytest.mark.parametrize("costs", [(1.0, 2.0), (0.1, 5.0, 10.0)])
def test_independent_actions_meta_model_accepts_cc_models(costs):
    """Cost-controlled models (BetaCC) satisfy the BaseModel constraint."""
    actions = {a: BetaCC(cost=c) for a, c in zip(make_action_ids(len(costs)), costs)}
    meta = SmabMetaModel(actions=actions)
    for action_id, expected_cost in zip(meta.action_ids, costs):
        assert meta.actions[action_id].cost == expected_cost


def test_per_action_meta_model_rejects_empty_actions():
    with pytest.raises(AttributeError, match="At least one action"):
        SmabMetaModel[Beta](actions={})


def test_per_action_meta_model_warns_on_single_action(make_beta_actions):
    with pytest.warns(UserWarning, match="deterministic"):
        SmabMetaModel[Beta](actions=make_beta_actions(1))


def test_per_action_meta_model_rejects_wrong_action_type():
    """SmabMetaModel[Beta] rejects BetaMO instances (wrong concrete type)."""
    with pytest.raises((ValidationError, TypeError)):
        SmabMetaModel[Beta](actions={"a0": Beta(), "a1": BetaMO()})


@pytest.mark.parametrize(
    "alias, factory",
    [
        (SmabMetaModelSO, lambda: Beta()),
        (SmabMetaModelMO, lambda: BetaMO(models=[Beta(), Beta()])),
    ],
    ids=["smab-SO", "smab-MO"],
)
@pytest.mark.parametrize("n_actions", [2, 3])
def test_pickle_roundtrip_meta_model_aliases(alias, factory, n_actions):
    """Module-level aliases are pickle-safe (require module-level name for qualname lookup)."""
    meta = alias(actions={a: factory() for a in make_action_ids(n_actions)})
    restored = pickle.loads(pickle.dumps(meta))
    assert set(restored.actions.keys()) == set(meta.action_ids)
    assert type(restored) is type(meta)


########################################################################################################################
# SmabMetaModel._extract_action_specific_kwargs


@pytest.mark.parametrize(
    "kwargs, expected_action, expected_quant, expected_remaining",
    [
        # No action-specific keys: returns empty dicts, kwargs untouched.
        ({"param1": 1, "param2": 2}, {}, {}, {"param1": 1, "param2": 2}),
        # Non-dict value at an action-prefixed key: silently ignored.
        ({f"{ACTION_IDS_PREFIX}param1": "not_a_dict"}, {}, {}, {f"{ACTION_IDS_PREFIX}param1": "not_a_dict"}),
        # Empty dicts at action-prefixed keys: consumed, return empty.
        ({f"{ACTION_IDS_PREFIX}param1": {}, f"{ACTION_IDS_PREFIX}param2": {}}, {}, {}, {}),
        # Valid action-prefixed keys: pivoted by action id.
        (
            {
                f"{ACTION_IDS_PREFIX}param1": {"action1": 1, "action2": 2},
                f"{ACTION_IDS_PREFIX}param2": {"action1": 3, "action2": 4},
            },
            {"action1": {"param1": 1, "param2": 3}, "action2": {"param1": 2, "param2": 4}},
            {},
            {},
        ),
        # Valid quantitative-action-prefixed keys: pivoted into the second dict.
        (
            {
                f"{QUANTITATIVE_ACTION_IDS_PREFIX}param1": {"action1": 1, "action2": 2},
                f"{QUANTITATIVE_ACTION_IDS_PREFIX}param2": {"action1": 3, "action2": 4},
            },
            {},
            {"action1": {"param1": 1, "param2": 3}, "action2": {"param1": 2, "param2": 4}},
            {},
        ),
    ],
)
def test_extract_action_specific_kwargs(kwargs, expected_action, expected_quant, expected_remaining):
    action, quant = SmabMetaModel._extract_action_specific_kwargs(kwargs)
    assert action == expected_action
    assert quant == expected_quant
    assert kwargs == expected_remaining


########################################################################################################################
# BaseMetaModel._extract_action_model_class_and_attributes


class _MockActionModel:
    """Stand-in action-model class for kwargs-extraction tests.

    Constructor signature carries the ``param1``/``param2`` names that the
    extractor introspects via ``extract_argument_names_from_function``.
    """

    def __init__(self, param1, param2):
        pass

    @classmethod
    def cold_start(cls):
        pass


class _MockMetaModel(SmabMetaModel):
    """Concrete subclass that exposes ``_MockActionModel`` as the only action-model class.

    Avoids patching the ``_action_model_classes`` classproperty (which now raises on
    unparameterised classes by design); instead we override it directly here so the
    kwargs-extractor tests can exercise the routing logic with a controlled mock class.
    """

    @classproperty
    def _action_model_classes(cls):  # noqa: D401
        return (_MockActionModel,)


@pytest.mark.parametrize(
    "issubclass_side_effect, expected_branch",
    [
        # The extractor's first `issubclass(...)` against (Model, ModelMO) returns True →
        # the mock is treated as a non-quantitative ("regular") action model.
        ({"return_value": True}, "regular"),
        # First call returns False, second returns True → quantitative branch.
        ({"side_effect": [False, True]}, "quantitative"),
    ],
)
@pytest.mark.parametrize(
    "kwargs, arg_names",
    [
        ({}, []),
        ({"irrelevant_param": 1}, []),
        ({"param1": 1, "param2": 2}, ["param1", "param2"]),
    ],
)
def test_extract_action_model_class_and_attributes(
    mocker: MockerFixture, issubclass_side_effect, expected_branch, kwargs, arg_names
):
    """The extractor routes kwargs into the regular or quantitative bucket based on ``issubclass``."""
    mocker.patch("pybandits.meta_model.base.extract_argument_names_from_function", return_value=arg_names)
    mocker.patch("pybandits.meta_model.base.issubclass", **issubclass_side_effect)

    kwargs_copy = kwargs.copy()
    (_, _, action_general_kwargs, quantitative_action_general_kwargs) = (
        _MockMetaModel._extract_action_model_class_and_attributes(kwargs_copy)
    )

    # Only kwargs whose names are in `arg_names` are consumed; others remain untouched.
    expected_consumed = {k: v for k, v in kwargs.items() if k in arg_names}
    if expected_branch == "regular":
        assert action_general_kwargs == expected_consumed
        assert quantitative_action_general_kwargs is None
    else:
        assert action_general_kwargs is None
        assert quantitative_action_general_kwargs == expected_consumed
    # The non-consumed kwargs must remain in the input dict.
    assert kwargs_copy == {k: v for k, v in kwargs.items() if k not in expected_consumed}


def test_extract_action_model_class_and_attributes_raises_without_subclass_match(mocker: MockerFixture):
    """When ``_MockActionModel`` matches neither Model/ModelMO nor QuantitativeModel, the extractor raises."""
    mocker.patch("pybandits.meta_model.base.extract_argument_names_from_function", return_value=["param1", "param2"])
    with pytest.raises(TypeError):
        _MockMetaModel._extract_action_model_class_and_attributes({"param1": 1, "param2": 2})


def test_action_model_classes_raises_on_unparameterised_meta_model():
    """The classproperty refuses to return a fallback on unparameterised subclasses."""
    with pytest.raises(TypeError, match="parameterise"):
        _ = SmabMetaModel._action_model_classes
