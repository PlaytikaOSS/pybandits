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

from typing import Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from pydantic import ValidationError
from pytest_mock import MockerFixture

from pybandits.base import ActionId, BinaryReward, Float01, PositiveFloat01, Probability, UnifiedActionId
from pybandits.mab import BaseMab
from pybandits.model import Beta, BetaCC
from pybandits.quantitative_model import Zooming
from pybandits.strategy import ClassicBandit
from tests.test_actions_manager import REFERENCE_DELTA, DummyActionsManager

_rng = np.random.default_rng(seed=42)


class DummyMab(BaseMab):
    epsilon: Optional[Float01] = None
    default_action: Optional[UnifiedActionId] = None
    actions_manager: DummyActionsManager

    def _update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
        **kwargs,
    ):
        pass

    def predict(
        self,
        forbidden_actions: Optional[Set[ActionId]] = None,
    ):
        valid_actions = self._get_valid_actions(forbidden_actions)
        return _rng.choice(valid_actions)

    def get_state(self) -> Tuple[str, dict]:
        model_name = self.__class__.__name__
        state: dict = {"actions": self.actions}
        return model_name, state


def test_base_mab_raise_on_bad_actions(cost=0.0):
    with pytest.raises(TypeError):
        DummyMab(actions={"a1": Beta(), "a2": Beta()})
    with pytest.raises(ValidationError):
        DummyMab(actions={"": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    with pytest.raises(AttributeError):
        DummyMab(actions={}, strategy=ClassicBandit())
    with pytest.raises((ValidationError, TypeError)):
        DummyMab(actions={"a1": None}, strategy=ClassicBandit())
    with pytest.raises((ValidationError, TypeError)):
        DummyMab(actions={"a1": None, "a2": None}, strategy=ClassicBandit())
    with pytest.warns(UserWarning):
        DummyMab(actions={"a1": Beta()}, strategy=ClassicBandit())
    with pytest.raises(ValidationError):
        DummyMab(actions={"a1": Beta(), "a2": BetaCC(cost=cost)}, strategy=ClassicBandit())


def test_base_mab_check_update_params():
    dummy_mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    with pytest.raises(AttributeError):
        # actionId doesn't exist
        dummy_mab.update(actions=["a1", "a3"], rewards=[1, 1], quantities=None)
    with pytest.raises(ValidationError):
        # actionId cannot be empty
        dummy_mab.update(actions=[""], rewards=[1], quantities=None)
    with pytest.raises(AttributeError):
        dummy_mab._validate_params_lengths(actions=["a1", "a2"], rewards=[1], quantities=None)

    with pytest.raises(AttributeError):
        # quantities of different length
        dummy_mab._validate_params_lengths(actions=["a1", "a2"], rewards=[1, 1], quantities=[1])

    with pytest.raises(AttributeError):
        # context of different length
        dummy_mab._validate_params_lengths(actions=["a1", "a2"], rewards=[1, 1], quantities=None, context=[1])


@given(r1=st.integers(min_value=0, max_value=1), r2=st.integers(min_value=0, max_value=1))
def test_base_mab_update_ok(r1, r2):
    dummy_mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit())
    dummy_mab.update(actions=["a1", "a2"], rewards=[r1, r2], quantities=None)
    dummy_mab.update(actions=["a1", "a1"], rewards=[r1, r2], quantities=None)


########################################################################################################################


# Epsilon-greedy functionality tests


@pytest.fixture
def p() -> Dict[ActionId, Probability]:
    return {"a1": 0.5, "a2": 0.5}


def test_valid_epsilon_value(mocker: MockerFixture, p: Dict[ActionId, Probability]):
    mocker.patch.object(ClassicBandit, "select_action", return_value="a2")
    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.1, default_action="a1")
    selected_action = mab._select_epsilon_greedy_action(p)
    assert selected_action in p.keys()


def test_epsilon_boundary_values(mocker: MockerFixture, p: Dict[ActionId, Probability]):
    mocker.patch.object(ClassicBandit, "select_action", return_value="a2")

    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.0)
    selected_action = mab._select_epsilon_greedy_action(p)
    assert selected_action == "a2"

    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=1.0, default_action="a1")
    selected_action = mab._select_epsilon_greedy_action(p)
    assert selected_action == "a1"


def test_default_action_not_in_actions(p: Dict[ActionId, Probability]):
    with pytest.raises(AttributeError):
        DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=1.0, default_action="a3")


def test_select_action_raises_exception(mocker: MockerFixture, p: Dict[ActionId, Probability]):
    mocker.patch.object(ClassicBandit, "select_action", side_effect=Exception("Test Exception"))
    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.0, default_action=None)

    with pytest.raises(Exception) as excinfo:
        mab._select_epsilon_greedy_action(p)

    assert str(excinfo.value) == "Test Exception"


def test_default_action_in_forbidden_actions():
    mab = DummyMab(actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), epsilon=0.1, default_action="a1")
    with pytest.raises(ValueError):
        mab.predict(forbidden_actions={"a1"})


@given(st.sampled_from([10, "inf"]), st.just(0.1))
def test_adaptive_window_without_epsilon_fails(adaptive_window_size, epsilon):
    with pytest.raises(ValueError):
        DummyMab(
            actions={"a1": Beta(), "a2": Beta()}, strategy=ClassicBandit(), adaptive_window_size=adaptive_window_size
        )
    with pytest.raises(ValueError):
        DummyMab(
            actions={"a1": Beta(), "a2": Beta()},
            strategy=ClassicBandit(),
            adaptive_window_size=adaptive_window_size,
            epsilon=epsilon,
            default_action="a1",
        )


########################################################################################################################


_EPSILON: Float01 = 0.1
_DEFAULT_FRACTION: PositiveFloat01 = 0.5
_MIX_FRACTION: PositiveFloat01 = 0.3
_MIX_TOLERANCE: float = 0.03
_N_DRAWS: int = 4_000
_N_REPEATS: int = 20


class TestDefaultActionFraction:
    """Tests for BaseMab.default_action_fraction mixed epsilon-greedy behavior."""

    @pytest.mark.parametrize("fraction", [0.3, 0.5, 0.7])
    def test_requires_epsilon(self, fraction: float, epsilon=None):
        """default_action_fraction is rejected when epsilon is not set, for any valid fraction."""
        with pytest.raises(AttributeError, match=r"default_action_fraction requires epsilon to be defined\."):
            DummyMab(
                actions={"a1": Beta(), "a2": Beta()},
                strategy=ClassicBandit(),
                epsilon=epsilon,
                default_action_fraction=fraction,
            )

    @pytest.mark.parametrize("fraction", [0.3, 0.5, 0.7])
    def test_requires_default_action(self, fraction: float, epsilon=_EPSILON):
        """default_action_fraction is rejected when default_action is not set, for any valid fraction."""
        with pytest.raises(AttributeError, match=r"default_action_fraction requires default_action to be defined\."):
            DummyMab(
                actions={"a1": Beta(), "a2": Beta()},
                strategy=ClassicBandit(),
                epsilon=epsilon,
                default_action=None,
                default_action_fraction=fraction,
            )

    def test_zero_fraction_is_invalid(self, epsilon=_EPSILON):
        """default_action_fraction=0.0 must be rejected — equivalent to having no default_action."""
        with pytest.raises(ValidationError):
            DummyMab(
                actions={"a1": Beta(), "a2": Beta()},
                strategy=ClassicBandit(),
                epsilon=epsilon,
                default_action="a1",
                default_action_fraction=0.0,
            )

    @pytest.mark.parametrize("fraction", [None, 1.0])
    def test_always_returns_default_action(
        self,
        p: Dict[ActionId, Probability],
        fraction: Optional[float],
        epsilon=1.0,
    ):
        """When fraction is None or 1.0, exploration must always return default_action.

        epsilon=1.0 guarantees the explore branch always fires.
        fraction=None always picks default; fraction=1.0 makes _rng.binomial(1, 1.0) always return 1.
        """
        mab = DummyMab(
            actions={"a1": Beta(), "a2": Beta()},
            strategy=ClassicBandit(),
            epsilon=epsilon,
            default_action="a1",
            default_action_fraction=fraction,
        )
        for _ in range(_N_REPEATS):
            assert mab._select_epsilon_greedy_action(p) == "a1"

    def test_mix_distribution(self, p: Dict[ActionId, Probability], epsilon=1.0):
        """Empirical share of default_action under explore must approximate default_action_fraction.

        epsilon=1.0 always fires the explore branch. A seeded real rng is used for the fraction
        Bernoulli while integers is stubbed to always return the non-default arm index, isolating
        the fraction draw as the sole source of variance in the default_action share.
        """
        mab = DummyMab(
            actions={"a1": Beta(), "a2": Beta()},
            strategy=ClassicBandit(),
            epsilon=epsilon,  # always explore
            default_action="a1",
            default_action_fraction=_MIX_FRACTION,
        )
        # Replace _rng with a mock: binomial delegates to a seeded real generator for realistic
        # fraction sampling; integers always returns 1 (index of "a2") so the random branch never
        # accidentally picks "a1", keeping the empirical share unconfounded.
        real_rng = np.random.default_rng(42)
        mock_rng = MagicMock()
        mock_rng.binomial.side_effect = real_rng.binomial
        mock_rng.integers.return_value = 1
        mab._rng = mock_rng

        selections = [mab._select_epsilon_greedy_action(p) for _ in range(_N_DRAWS)]
        default_share = sum(1 for s in selections if s == "a1") / _N_DRAWS
        assert "a1" in selections and "a2" in selections
        assert abs(default_share - _MIX_FRACTION) < _MIX_TOLERANCE

    def test_quantitative_random_branch(self, epsilon=1.0):
        """When the random branch fires for a quantitative model, the result is (action_id, quantity_tuple)."""
        actions = {"a1": Zooming.cold_start(), "a2": Zooming.cold_start()}
        mab = DummyMab(
            actions=actions,
            strategy=ClassicBandit(),
            epsilon=epsilon,
            default_action=("a1", (0.5,)),
            default_action_fraction=_DEFAULT_FRACTION,
        )
        # Replace _rng with a mock: binomial returns [1, 0] so epsilon fires (explore) but the
        # fraction check fails (random branch). integers picks "a2" (index 1). random supplies
        # the quantity tuple for the quantitative action.
        mock_rng = MagicMock()
        mock_rng.binomial.side_effect = [1, 0]
        mock_rng.integers.return_value = 1
        mock_rng.random.return_value = np.array([0.5])
        mab._rng = mock_rng

        p_quant = {"a1": 0.5, "a2": 0.5}
        selected = mab._select_epsilon_greedy_action(p_quant)

        assert isinstance(selected, tuple)
        assert selected[0] == "a2"
        assert isinstance(selected[1], tuple)
        assert len(selected[1]) == actions["a2"].dimension


########################################################################################################################


# MAB model_post_init validation tests


def test_mab_model_post_init_adaptive_window_epsilon_validation():
    """Test model_post_init validation for adaptive window with epsilon greedy requirements."""
    actions = {"action1": Beta(), "action2": Beta()}

    # Test case 1: delta is set but epsilon is None - should raise ValueError
    with pytest.raises(
        ValueError, match="Adaptive window requires epsilon greedy super strategy with not default action."
    ):
        DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=None, default_action=None, delta=REFERENCE_DELTA)

    # Test case 2: delta is set, epsilon is provided, but default_action is also provided - should raise ValueError
    with pytest.raises(
        ValueError, match="Adaptive window requires epsilon greedy super strategy with not default action."
    ):
        DummyMab(
            actions=actions, strategy=ClassicBandit(), epsilon=0.1, default_action="action1", delta=REFERENCE_DELTA
        )


def test_mab_model_post_init_default_action_validation():
    """Test model_post_init validation for default action requirements."""
    actions = {"action1": Beta(), "action2": Beta()}

    # Test case 1: epsilon is not provided but default_action is provided - should raise AttributeError
    with pytest.raises(AttributeError, match="A default action should only be defined when epsilon is defined."):
        DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=None, default_action="action1")


def test_mab_model_post_init_invalid_default_action(epsilon=_EPSILON):
    """Test model_post_init validation for invalid default action."""
    actions = {"action1": Beta(), "action2": Beta()}

    with pytest.raises(AttributeError, match="The default action must be valid action defined in the actions set."):
        DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=epsilon, default_action="invalid_action")


def test_mab_model_post_init_quantitative_default_action_validation(epsilon=_EPSILON):
    """Test model_post_init validation for quantitative default action requirements."""
    actions = {"action1": Beta(), "action2": Beta()}

    with pytest.raises(AttributeError, match="Quantitative default action requires a quantitative action model."):
        DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=epsilon, default_action=("action1", (0.5, 0.5)))


def test_mab_model_post_init_standard_default_action_validation(epsilon=_EPSILON):
    """Test model_post_init validation for standard default action requirements."""
    actions = {"action1": Zooming.cold_start(), "action2": Zooming.cold_start()}

    with pytest.raises(AttributeError, match="Standard default action requires a standard action model."):
        DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=epsilon, default_action="action1")


def test_mab_model_post_init_valid_configurations(epsilon=_EPSILON):
    """Test model_post_init validation with valid configurations."""
    actions = {"action1": Beta(), "action2": Beta()}

    # Valid case 1: No epsilon, no default action, no delta
    mab = DummyMab(actions=actions, strategy=ClassicBandit())
    assert mab.epsilon is None
    assert mab.default_action is None

    # Valid case 2: Epsilon with valid default action
    mab = DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=epsilon, default_action="action1")
    assert mab.epsilon == epsilon
    assert mab.default_action == "action1"

    # Valid case 3: Epsilon without default action and delta (adaptive window)
    mab = DummyMab(actions=actions, strategy=ClassicBandit(), epsilon=epsilon, delta=REFERENCE_DELTA)
    assert mab.epsilon == epsilon
    assert mab.default_action is None
    assert mab.actions_manager.delta == REFERENCE_DELTA


class TestLimitedActions:
    """Tests for BaseMab.limited_actions throttling via limited_action_fraction."""

    other_arm = "a1"
    limited_arm = "a2"
    unknown_arm = "a3"
    other_p = 0.4
    limited_p = 0.6  # the limited arm strictly wins whenever it is allowed to compete
    limited_epsilon = 0.1
    mix_fraction = 0.3
    mix_tolerance = 0.03
    mix_seed = 42
    n_draws = 4_000
    fraction_min = 0.05
    fraction_max = 0.95

    @pytest.fixture(scope="class")
    def make_mab(self):
        """Factory: builds a DummyMab over the two-arm action set with the given limited-action config."""

        def _factory(**kwargs) -> DummyMab:
            return DummyMab(
                actions={self.other_arm: Beta(), self.limited_arm: Beta()},
                strategy=ClassicBandit(),
                **kwargs,
            )

        return _factory

    @pytest.fixture(scope="class")
    def p(self) -> Dict[ActionId, Probability]:
        """Sampled probabilities where the limited arm strictly wins whenever it is allowed to compete."""
        return {self.other_arm: self.other_p, self.limited_arm: self.limited_p}

    @given(fraction=st.floats(min_value=fraction_min, max_value=fraction_max))
    def test_requires_limited_actions(self, make_mab, fraction: float) -> None:
        """limited_action_fraction is rejected when limited_actions is not set, for any valid fraction."""
        with pytest.raises(AttributeError, match="must be defined together"):
            make_mab(limited_action_fraction=fraction)

    def test_requires_fraction(self, make_mab, limited_arm: ActionId = limited_arm) -> None:
        """limited_actions is rejected when limited_action_fraction is not set."""
        with pytest.raises(AttributeError, match="must be defined together"):
            make_mab(limited_actions={limited_arm})

    @given(
        fraction=st.floats(min_value=fraction_min, max_value=fraction_max),
        unknown_arm=st.just(unknown_arm),
    )
    def test_rejects_unknown_action(self, make_mab, fraction: float, unknown_arm: ActionId) -> None:
        """A limited action outside the action set is rejected, for any valid fraction."""
        with pytest.raises(AttributeError, match="must be a subset of the action set"):
            make_mab(limited_actions={unknown_arm}, limited_action_fraction=fraction)

    @given(
        fraction=st.floats(min_value=fraction_min, max_value=fraction_max),
        epsilon=st.just(limited_epsilon),
        limited_arm=st.just(limited_arm),
    )
    def test_rejects_default_action_in_limited(
        self, make_mab, fraction: float, epsilon: float, limited_arm: ActionId
    ) -> None:
        """A default_action that is also a limited action is rejected, for any valid fraction."""
        with pytest.raises(AttributeError, match="default_action must not be a limited action"):
            make_mab(
                epsilon=epsilon,
                default_action=limited_arm,
                limited_actions={limited_arm},
                limited_action_fraction=fraction,
            )

    @given(
        fraction=st.floats(min_value=fraction_min, max_value=fraction_max),
        limited_arm=st.just(limited_arm),
        other_arm=st.just(other_arm),
    )
    def test_masks_when_gate_closed(
        self, make_mab, p: Dict[ActionId, Probability], fraction: float, limited_arm: ActionId, other_arm: ActionId
    ) -> None:
        """When the gate draw is 0 the limited arm is masked out, so the strategy picks the other arm."""
        mab = make_mab(limited_actions={limited_arm}, limited_action_fraction=fraction)
        mab._rng = MagicMock()
        mab._rng.binomial.return_value = 0
        assert mab._select_epsilon_greedy_action(p, actions=mab.actions) == other_arm

    @given(
        fraction=st.floats(min_value=fraction_min, max_value=fraction_max),
        limited_arm=st.just(limited_arm),
    )
    def test_competes_when_gate_open(
        self, make_mab, p: Dict[ActionId, Probability], fraction: float, limited_arm: ActionId
    ) -> None:
        """When the gate draw is 1 the limited arm competes normally and wins on probability."""
        mab = make_mab(limited_actions={limited_arm}, limited_action_fraction=fraction)
        mab._rng = MagicMock()
        mab._rng.binomial.return_value = 1
        assert mab._select_epsilon_greedy_action(p, actions=mab.actions) == limited_arm

    def test_never_masks_entire_set(
        self,
        make_mab,
        p: Dict[ActionId, Probability],
        fraction: float = mix_fraction,
        limited_arm: ActionId = limited_arm,
        other_arm: ActionId = other_arm,
    ) -> None:
        """If every action is limited, a closed gate leaves the full set intact rather than emptying it."""
        mab = make_mab(limited_actions={other_arm, limited_arm}, limited_action_fraction=fraction)
        mab._rng = MagicMock()
        mab._rng.binomial.return_value = 0
        assert mab._select_epsilon_greedy_action(p, actions=mab.actions) == limited_arm

    def test_mix_distribution(
        self,
        make_mab,
        p: Dict[ActionId, Probability],
        fraction: float = mix_fraction,
        seed: int = mix_seed,
        n_draws: int = n_draws,
        tolerance: float = mix_tolerance,
        limited_arm: ActionId = limited_arm,
        other_arm: ActionId = other_arm,
    ) -> None:
        """Empirical share of the limited arm approximates limited_action_fraction.

        No epsilon is set, so selection routes straight through the gate to the strategy; a seeded
        real rng drives the gate Bernoulli, making the fraction draw the sole source of variance.
        """
        mab = make_mab(limited_actions={limited_arm}, limited_action_fraction=fraction, random_seed=seed)

        selections = [mab._select_epsilon_greedy_action(p, actions=mab.actions) for _ in range(n_draws)]
        limited_share = sum(1 for s in selections if s == limited_arm) / n_draws
        assert other_arm in selections and limited_arm in selections
        assert abs(limited_share - fraction) < tolerance


class TestActionKindDisjoint:
    """An action id may be regular xor quantitative, never both (meta_model._instantiate_actions guard)."""

    regular_arm = "a1"
    quantitative_arm = "a2"
    shared_arm = "a3"
    overlap_match = "both regular and quantitative"

    @pytest.mark.parametrize(
        "action_ids, quantitative_action_ids",
        [
            ({shared_arm}, {shared_arm}),  # full overlap
            ({regular_arm, shared_arm}, {shared_arm}),  # overlap alongside a distinct regular arm
            ({shared_arm}, {shared_arm, quantitative_arm}),  # overlap alongside a distinct quantitative arm
            ({regular_arm, shared_arm}, {shared_arm, quantitative_arm}),  # overlap with distinct arms on both sides
        ],
    )
    def test_rejects_shared_action_id(
        self, action_ids: Set[ActionId], quantitative_action_ids: Set[ActionId], match: str = overlap_match
    ) -> None:
        """An id present in both the regular and quantitative sets is rejected, whatever else is declared."""
        with pytest.raises(AttributeError, match=match):
            DummyMab(
                action_ids=action_ids,
                quantitative_action_ids=quantitative_action_ids,
                strategy=ClassicBandit(),
            )
