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

"""Tests for transfer learning functionality."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest import MonkeyPatch

import pybandits
from pybandits.cmab import CmabBernoulli, CmabBernoulliMO
from pybandits.model import BaseBayesianNeuralNetwork
from pybandits.smab import SmabBernoulli, SmabBernoulliCC, SmabBernoulliMO
from pybandits.strategy import (
    ClassicBandit,
    CostControlBandit,
    MultiObjectiveBandit,
)
from pybandits.transfer import (
    _merge_mabs,
    edit_model_on_the_fly,
)
from tests.utils import mock_update

# ---------------------------------------------------------------------------
# Module-level constants shared across tests
# ---------------------------------------------------------------------------
_N_FEATURES = 5
_ACTION_ID = "a1"

# Contrasting value pairs for each structural key — both must produce valid CmabBernoulli models.
# Keyed by the attribute name as it appears in transfer_structural_keys AND in cold_start kwargs.
_BNN_STRUCTURAL_KEY_VALUES: Dict[str, Tuple[Any, Any]] = {
    "activation": ("relu", "tanh"),
    "use_residual_connections": (False, True),
}

# Contrasting value pairs for each extendable key.
_BNN_EXTENDABLE_KEY_VALUES: Dict[str, Tuple[Any, Any]] = {
    "update_method": ("VI", "MCMC"),
}

# Guard: these dicts must stay in sync with the actual ClassVar declarations.
assert set(_BNN_STRUCTURAL_KEY_VALUES) == set(BaseBayesianNeuralNetwork.transfer_structural_keys), (
    "_BNN_STRUCTURAL_KEY_VALUES must cover exactly BaseBayesianNeuralNetwork.transfer_structural_keys. "
    f"Missing: {set(BaseBayesianNeuralNetwork.transfer_structural_keys) - set(_BNN_STRUCTURAL_KEY_VALUES)}, "
    f"Extra: {set(_BNN_STRUCTURAL_KEY_VALUES) - set(BaseBayesianNeuralNetwork.transfer_structural_keys)}"
)
assert set(_BNN_EXTENDABLE_KEY_VALUES) == set(BaseBayesianNeuralNetwork.transfer_extendable_keys), (
    "_BNN_EXTENDABLE_KEY_VALUES must cover exactly BaseBayesianNeuralNetwork.transfer_extendable_keys. "
    f"Missing: {set(BaseBayesianNeuralNetwork.transfer_extendable_keys) - set(_BNN_EXTENDABLE_KEY_VALUES)}, "
    f"Extra: {set(_BNN_EXTENDABLE_KEY_VALUES) - set(BaseBayesianNeuralNetwork.transfer_extendable_keys)}"
)

# Hypothesis strategies for test data generation
single_action_id_strategy = st.text(
    min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
)
action_ids_strategy = st.lists(
    single_action_id_strategy,
    min_size=1,
    max_size=5,
    unique=True,
).map(set)
n_features_strategy = st.integers(min_value=2, max_value=10)
n_objectives_strategy = st.integers(min_value=2, max_value=4)
epsilon_strategy = st.floats(min_value=0.0, max_value=0.5)
subsidy_factor_strategy = st.floats(min_value=0.1, max_value=1.0)
cost_strategy = st.floats(min_value=0.1, max_value=10.0)


class TestMergeMABs:
    """Test suite for _merge_mabs function."""

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
    )
    def test_merge_compatible_smabs(self, action_ids1: set, action_ids2: set) -> None:
        """Test merging two compatible SmabBernoulli instances uses mab2 as template."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        mab1 = SmabBernoulli.cold_start(action_ids=action_ids1, strategy=ClassicBandit())
        mab2 = SmabBernoulli.cold_start(action_ids=action_ids2, strategy=ClassicBandit())

        merged = _merge_mabs(mab1, mab2)

        # Result should only have actions from mab2 (template)
        assert len(merged.actions) == len(action_ids2)
        assert set(merged.actions.keys()) == action_ids2
        assert isinstance(merged, SmabBernoulli)
        assert isinstance(merged.strategy, ClassicBandit)

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
        n_features=n_features_strategy,
    )
    def test_merge_compatible_cmabs(self, action_ids1: set, action_ids2: set, n_features: int) -> None:
        """Test merging two compatible CmabBernoulli instances uses mab2 as template."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        mab1 = CmabBernoulli.cold_start(
            action_ids=action_ids1, n_features=n_features, strategy=ClassicBandit(), update_method="VI"
        )
        mab2 = CmabBernoulli.cold_start(
            action_ids=action_ids2, n_features=n_features, strategy=ClassicBandit(), update_method="VI"
        )

        merged = _merge_mabs(mab1, mab2)

        # Result should only have actions from mab2 (template)
        assert len(merged.actions) == len(action_ids2)
        assert set(merged.actions.keys()) == action_ids2
        assert isinstance(merged, CmabBernoulli)

    @given(
        common_actions=action_ids_strategy,
        unique_actions1=action_ids_strategy,
        unique_actions2=action_ids_strategy,
    )
    def test_merge_with_overlapping_actions(
        self, common_actions: set, unique_actions1: set, unique_actions2: set
    ) -> None:
        """Test that merge uses mab2 as template and transfers learned state for overlapping actions."""
        # Ensure we have at least one common action
        if not common_actions:
            pytest.skip("Need at least one common action")

        # Create distinct action sets with overlap
        action_ids1 = common_actions.union({f"a_{aid}" for aid in unique_actions1})
        action_ids2 = common_actions.union({f"b_{aid}" for aid in unique_actions2})

        mab1 = SmabBernoulli.cold_start(action_ids=action_ids1, strategy=ClassicBandit())
        mab2 = SmabBernoulli.cold_start(action_ids=action_ids2, strategy=ClassicBandit())

        # Train overlapping actions in mab1
        common_action = next(iter(common_actions))
        mab1.update(actions=[common_action], rewards=[1])

        # Merge: should use mab2 as template
        merged = _merge_mabs(mab1, mab2)

        # Result should only have actions from mab2
        assert set(merged.actions.keys()) == action_ids2

        # Overlapping action should have learned state from mab1
        assert merged.actions[common_action].n_successes == 2  # 1 (cold start) + 1 (update)

    @given(
        action_id1=single_action_id_strategy,
        action_id2=single_action_id_strategy,
        n_features=n_features_strategy,
    )
    def test_merge_incompatible_types_raises(self, action_id1: str, action_id2: str, n_features: int) -> None:
        """Test that merging different MAB types raises TypeError."""
        mab1 = SmabBernoulli.cold_start(action_ids={action_id1}, strategy=ClassicBandit())
        mab2 = CmabBernoulli.cold_start(action_ids={action_id2}, n_features=n_features, strategy=ClassicBandit())

        with pytest.raises(TypeError, match="Cannot merge MABs"):
            _merge_mabs(mab1, mab2)

    @given(
        action_id1=single_action_id_strategy,
        action_id2=single_action_id_strategy,
        cost=cost_strategy,
        subsidy_factor=subsidy_factor_strategy,
    )
    def test_merge_cc_and_noncc_is_allowed(
        self, action_id1: str, action_id2: str, cost: float, subsidy_factor: float
    ) -> None:
        """Test that merging CC and non-CC MABs is allowed (CC is a config, not a structural constraint)."""
        mab1 = SmabBernoulli.cold_start(action_ids={action_id1}, strategy=ClassicBandit())
        mab2 = SmabBernoulliCC.cold_start(
            action_ids_cost={action_id2: cost}, strategy=CostControlBandit(subsidy_factor=subsidy_factor)
        )

        # CC ↔ non-CC merging is allowed: result has mab2's type (template)
        merged = _merge_mabs(mab1, mab2)
        assert isinstance(merged, SmabBernoulliCC)
        merged_rev = _merge_mabs(mab2, mab1)
        assert isinstance(merged_rev, SmabBernoulli)

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
    )
    def test_merge_strategy_from_first_mab(self, action_ids1: set, action_ids2: set) -> None:
        """Test that merged MAB uses strategy from first MAB."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        mab1 = SmabBernoulli.cold_start(action_ids=action_ids1, strategy=ClassicBandit())
        mab2 = SmabBernoulli.cold_start(action_ids=action_ids2, strategy=ClassicBandit())

        merged = _merge_mabs(mab1, mab2)

        assert isinstance(merged.strategy, ClassicBandit)

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
        epsilon1=epsilon_strategy,
        epsilon2=epsilon_strategy,
    )
    def test_merge_epsilon_from_first_mab(
        self, action_ids1: set, action_ids2: set, epsilon1: float, epsilon2: float
    ) -> None:
        """Test that merged MAB uses epsilon from second MAB (template)."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        mab1 = SmabBernoulli.cold_start(action_ids=action_ids1, strategy=ClassicBandit(), epsilon=epsilon1)
        mab2 = SmabBernoulli.cold_start(action_ids=action_ids2, strategy=ClassicBandit(), epsilon=epsilon2)

        merged = _merge_mabs(mab1, mab2)

        assert merged.epsilon == epsilon2

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
        n_objectives=n_objectives_strategy,
    )
    def test_merge_mo_mabs(self, action_ids1: set, action_ids2: set, n_objectives: int) -> None:
        """Test merging multi-objective MABs uses mab2 as template."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        mab1 = SmabBernoulliMO.cold_start(
            action_ids=action_ids1, n_objectives=n_objectives, strategy=MultiObjectiveBandit()
        )
        mab2 = SmabBernoulliMO.cold_start(
            action_ids=action_ids2, n_objectives=n_objectives, strategy=MultiObjectiveBandit()
        )

        merged = _merge_mabs(mab1, mab2)

        # Result should only have actions from mab2 (template)
        assert len(merged.actions) == len(action_ids2)
        assert isinstance(merged, SmabBernoulliMO)

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
        subsidy_factor=subsidy_factor_strategy,
    )
    def test_merge_cc_mabs(self, action_ids1: set, action_ids2: set, subsidy_factor: float) -> None:
        """Test merging cost-control MABs uses mab2 as template."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        # Generate random costs for each action
        action_ids_cost1 = {aid: np.random.uniform(0.5, 5.0) for aid in action_ids1}
        action_ids_cost2 = {aid: np.random.uniform(0.5, 5.0) for aid in action_ids2}

        mab1 = SmabBernoulliCC.cold_start(
            action_ids_cost=action_ids_cost1, strategy=CostControlBandit(subsidy_factor=subsidy_factor)
        )
        mab2 = SmabBernoulliCC.cold_start(
            action_ids_cost=action_ids_cost2, strategy=CostControlBandit(subsidy_factor=subsidy_factor)
        )

        merged = _merge_mabs(mab1, mab2)

        # Result should only have actions from mab2 (template)
        assert len(merged.actions) == len(action_ids2)
        assert isinstance(merged, SmabBernoulliCC)

    @given(
        action_id1=single_action_id_strategy,
        action_id2=single_action_id_strategy,
        n_updates1=st.integers(min_value=1, max_value=10),
        n_updates2=st.integers(min_value=1, max_value=10),
        rewards1=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=5),
        rewards2=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=5),
    )
    def test_merge_preserves_learned_state(
        self,
        action_id1: str,
        action_id2: str,
        n_updates1: int,
        n_updates2: int,
        rewards1: list,
        rewards2: list,
    ) -> None:
        """Test that merge uses mab2 as template and preserves learned state for overlapping actions."""
        # Create MABs with overlapping actions
        mab1 = SmabBernoulli.cold_start(action_ids={action_id1, action_id2}, strategy=ClassicBandit())
        mab2 = SmabBernoulli.cold_start(action_ids={action_id2}, strategy=ClassicBandit())

        # Train both MABs (use limited updates to match rewards list size)
        n_updates1 = min(n_updates1, len(rewards1))
        n_updates2 = min(n_updates2, len(rewards2))

        mab1.update(actions=[action_id2] * n_updates1, rewards=rewards1[:n_updates1])
        mab2.update(actions=[action_id2] * n_updates2, rewards=rewards2[:n_updates2])

        merged = _merge_mabs(mab1, mab2)

        # Result should only have actions from mab2 (template)
        assert set(merged.actions.keys()) == {action_id2}
        # action_id2 should have learned state from mab1 (not mab2)
        assert merged.actions[action_id2].n_successes == 1 + sum(rewards1[:n_updates1])
        assert merged.actions[action_id2].n_failures == 1 + n_updates1 - sum(rewards1[:n_updates1])


class TestEditModelOnTheFly:
    """Test suite for edit_model_on_the_fly function."""

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
    )
    def test_edit_uses_new_as_template(self, action_ids1: set, action_ids2: set) -> None:
        """Test that edit_model_on_the_fly uses new_mab as template."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        current = SmabBernoulli.cold_start(action_ids=action_ids1, strategy=ClassicBandit())
        new = SmabBernoulli.cold_start(action_ids=action_ids2, strategy=ClassicBandit())

        merged = edit_model_on_the_fly(current, new)

        # Result should only have actions from new_mab (template)
        assert len(merged.actions) == len(action_ids2)
        assert set(merged.actions.keys()) == action_ids2

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
        n_features=n_features_strategy,
        n_iterations=st.integers(min_value=1, max_value=10),
    )
    def test_edit_uses_new_config(self, action_ids1: set, action_ids2: set, n_features: int, n_iterations: int) -> None:
        """Test that edit_model_on_the_fly uses new_mab's configuration."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        current = CmabBernoulli.cold_start(
            action_ids=action_ids1,
            n_features=n_features,
            strategy=ClassicBandit(),
            update_kwargs={"num_steps": n_iterations},
        )
        new = CmabBernoulli.cold_start(
            action_ids=action_ids2,
            n_features=n_features,
            strategy=ClassicBandit(),
            update_kwargs={"num_steps": n_iterations},
        )

        merged = edit_model_on_the_fly(current, new)

        # Result should only have actions from new_mab with their configuration
        assert set(merged.actions.keys()) == action_ids2
        for action_id in action_ids2:
            assert merged.actions[action_id].update_kwargs["num_steps"] == n_iterations

    @given(
        action_ids1=action_ids_strategy,
        action_ids2=action_ids_strategy,
    )
    def test_edit_uses_new_as_template_simple(self, action_ids1: set, action_ids2: set) -> None:
        """Test that edit_model_on_the_fly uses new_mab as template."""
        # Ensure no overlap
        action_ids2 = {f"b_{aid}" for aid in action_ids2}

        current = SmabBernoulli.cold_start(action_ids=action_ids1, strategy=ClassicBandit())
        new = SmabBernoulli.cold_start(action_ids=action_ids2, strategy=ClassicBandit())

        merged = edit_model_on_the_fly(current, new)

        # Result should only have actions from new_mab (template)
        assert len(merged.actions) == len(action_ids2)
        assert set(merged.actions.keys()) == action_ids2

    @given(
        action_id1=single_action_id_strategy,
        action_id2=single_action_id_strategy,
        n_updates=st.integers(min_value=1, max_value=10),
        rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=5),
    )
    def test_edit_preserves_learned_state(
        self, action_id1: str, action_id2: str, n_updates: int, rewards: list
    ) -> None:
        """Test that learned state is preserved during edit for overlapping actions."""
        # Ensure distinct action IDs
        if action_id1 == action_id2:
            action_id2 = f"b_{action_id2}"

        # Create MABs with one overlapping action
        current = SmabBernoulli.cold_start(action_ids={action_id1, action_id2}, strategy=ClassicBandit())
        new = SmabBernoulli.cold_start(action_ids={action_id1}, strategy=ClassicBandit())

        # Train current (use limited updates to match rewards list size)
        n_updates = min(n_updates, len(rewards))
        current.update(actions=[action_id1] * n_updates, rewards=rewards[:n_updates])

        merged = edit_model_on_the_fly(current, new)

        # Result should only have actions from new (template)
        assert set(merged.actions.keys()) == {action_id1}
        # action_id1 should have learned state from current
        assert merged.actions[action_id1].n_successes == 1 + sum(rewards[:n_updates])
        assert merged.actions[action_id1].n_failures == 1 + n_updates - sum(rewards[:n_updates])

    @given(
        action_id1=single_action_id_strategy,
        action_id2=single_action_id_strategy,
        n_features=n_features_strategy,
    )
    def test_edit_incompatible_raises(self, action_id1: str, action_id2: str, n_features: int) -> None:
        """Test that editing incompatible MABs raises error."""
        current = SmabBernoulli.cold_start(action_ids={action_id1}, strategy=ClassicBandit())
        new = CmabBernoulli.cold_start(action_ids={action_id2}, n_features=n_features, strategy=ClassicBandit())

        with pytest.raises(TypeError, match="Cannot merge MABs"):
            edit_model_on_the_fly(current, new)

    @given(
        action_id1=single_action_id_strategy,
        action_id2=single_action_id_strategy,
        cost=cost_strategy,
        subsidy_factor=subsidy_factor_strategy,
    )
    def test_edit_cc_to_standard_allowed(
        self, action_id1: str, action_id2: str, cost: float, subsidy_factor: float
    ) -> None:
        """Test that changing CC model to standard (non-CC) and vice versa is allowed."""
        current_cc = SmabBernoulliCC.cold_start(
            action_ids_cost={action_id1: cost}, strategy=CostControlBandit(subsidy_factor=subsidy_factor)
        )
        current_std = SmabBernoulli.cold_start(action_ids={action_id2}, strategy=ClassicBandit())

        # CC → non-CC: result is SmabBernoulli (new_mab is the template)
        result = edit_model_on_the_fly(current_cc, current_std)
        assert isinstance(result, SmabBernoulli)

        # non-CC → CC: result is SmabBernoulliCC (new_mab is the template)
        result_rev = edit_model_on_the_fly(current_std, current_cc)
        assert isinstance(result_rev, SmabBernoulliCC)


class TestIntegration:
    """Integration tests for transfer learning workflows."""

    @given(
        old_action=single_action_id_strategy,
        new_action=single_action_id_strategy,
        n_updates=st.integers(min_value=1, max_value=10),
        rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=30),
    )
    def test_transfer_learning_workflow(self, old_action: str, new_action: str, n_updates: int, rewards: list) -> None:
        """Test a complete transfer learning workflow with overlapping action."""
        # Ensure distinct action IDs
        if old_action == new_action:
            new_action = f"new_{new_action}"

        # Create source MAB and train it
        source = SmabBernoulli.cold_start(action_ids={old_action, new_action}, strategy=ClassicBandit())
        n_updates = min(n_updates, len(rewards))
        source.update(actions=[old_action] * n_updates, rewards=rewards[:n_updates])

        # Create target MAB with overlapping action and one new action
        target = SmabBernoulli.cold_start(action_ids={old_action}, strategy=ClassicBandit())

        # Merge using target as template
        combined = _merge_mabs(source, target)

        # Result should only have actions from target (template)
        assert set(combined.actions.keys()) == {old_action}
        # old_action should have learned state from source
        assert combined.actions[old_action].n_successes == 1 + sum(rewards[:n_updates])
        assert combined.actions[old_action].n_failures == 1 + n_updates - sum(rewards[:n_updates])

    @given(
        action_ids=action_ids_strategy.filter(lambda x: len(x) >= 2),
        n_features=n_features_strategy,
        n_initial=st.integers(min_value=1, max_value=10),
        n_updated=st.integers(min_value=11, max_value=20),
        learning_rate=st.floats(min_value=0.0001, max_value=0.01),
    )
    def test_hyperparameter_tuning_workflow(
        self,
        action_ids: set,
        n_features: int,
        n_initial: int,
        n_updated: int,
        learning_rate: float,
        monkeymodule: MonkeyPatch,
    ) -> None:
        """Test using transfer learning for hyperparameter tuning."""
        # Mock the VI/MCMC fitting
        monkeymodule.setattr(
            pybandits.model.BaseBayesianNeuralNetwork,
            "_update",
            mock_update,
        )

        # Create MAB with initial hyperparameters
        mab = CmabBernoulli.cold_start(
            action_ids=action_ids,
            n_features=n_features,
            strategy=ClassicBandit(),
            update_kwargs={"num_steps": n_initial},
        )

        # Create new MAB with updated hyperparameters
        new_mab = CmabBernoulli.cold_start(
            action_ids=action_ids,
            n_features=n_features,
            strategy=ClassicBandit(),
            update_kwargs={"num_steps": n_updated, "optimizer_kwargs": {"step_size": learning_rate}},
        )

        # Use edit_model_on_the_fly to merge (preserves learned state, updates config)
        tuned_mab = edit_model_on_the_fly(mab, new_mab)

        # Verify hyperparameters updated
        for action in tuned_mab.actions.values():
            assert action.update_kwargs["num_steps"] == n_updated
            assert action.update_kwargs["optimizer_kwargs"]["step_size"] == learning_rate

    @given(
        n_actions_per_exp=st.lists(st.integers(min_value=1, max_value=3), min_size=3, max_size=3),
    )
    def test_model_consolidation_workflow(self, n_actions_per_exp: list) -> None:
        """Test updating action set while preserving learned state."""
        # Create an initial experiment with some actions
        initial_actions = {f"exp1_a{i}" for i in range(n_actions_per_exp[0])}

        current = SmabBernoulli.cold_start(action_ids=initial_actions, strategy=ClassicBandit())

        # Train one action
        if initial_actions:
            trained_action = next(iter(initial_actions))
            current.update(actions=[trained_action], rewards=[1])

        # Create new template with some overlapping and some new actions
        new_actions = {f"exp1_a{i}" for i in range(min(1, n_actions_per_exp[0]))}  # Keep at least one
        new_actions.update({f"exp2_a{i}" for i in range(n_actions_per_exp[1])})  # Add new ones

        new_template = SmabBernoulli.cold_start(action_ids=new_actions, strategy=ClassicBandit())

        # Update action set using new template
        updated = _merge_mabs(current, new_template)

        # Result should only have actions from new_template
        assert set(updated.actions.keys()) == new_actions

        # If there was overlap, learned state should be preserved
        overlap = initial_actions & new_actions
        if overlap and trained_action in overlap:
            assert updated.actions[trained_action].n_successes == 2  # 1 (cold start) + 1 (update)


class TestModelCompatibilityValidation:
    """Test suite for model compatibility validation during transfer learning."""

    @pytest.mark.parametrize(
        "key,val1,val2",
        [(k, v[0], v[1]) for k, v in _BNN_STRUCTURAL_KEY_VALUES.items()],
    )
    def test_transfer_structural_key_mismatch_raises_error(self, key: str, val1: Any, val2: Any) -> None:
        """Test that a mismatched structural key raises ValueError (parametrized over transfer_structural_keys)."""
        current = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, **{key: val1}, strategy=ClassicBandit()
        )
        template = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, **{key: val2}, strategy=ClassicBandit()
        )
        with pytest.raises(ValueError, match=key):
            edit_model_on_the_fly(current, template)

    @pytest.mark.parametrize(
        "key,val1,val2",
        [(k, v[0], v[1]) for k, v in _BNN_EXTENDABLE_KEY_VALUES.items()],
    )
    def test_transfer_extendable_key_change_warns(
        self, key: str, val1: Any, val2: Any, caplog: LogCaptureFixture
    ) -> None:
        """Test that changing an extendable key emits a warning but succeeds (parametrized over transfer_extendable_keys)."""
        caplog.set_level(logging.WARNING)
        current = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, **{key: val1}, strategy=ClassicBandit()
        )
        template = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, **{key: val2}, strategy=ClassicBandit()
        )
        result = edit_model_on_the_fly(current, template)
        assert result is not None
        assert getattr(result.actions[_ACTION_ID], key) == val2

    def test_transfer_dist_type_change_allowed(self) -> None:
        """Test that changing distribution type (StudentT vs Normal) is allowed."""
        current = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, dist_type="studentt", strategy=ClassicBandit()
        )
        template = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, dist_type="normal", strategy=ClassicBandit()
        )
        result = edit_model_on_the_fly(current, template)
        assert set(result.actions.keys()) == {_ACTION_ID}

    def test_transfer_hidden_dims_change_allowed(self) -> None:
        """Test that changing hidden layer dimensions is allowed (shape is not a structural constraint)."""
        current = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, hidden_dim_list=[16], strategy=ClassicBandit()
        )
        template = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, hidden_dim_list=[32], strategy=ClassicBandit()
        )
        result = edit_model_on_the_fly(current, template)
        assert _ACTION_ID in result.actions

    def test_transfer_valid_same_structure_succeeds(self) -> None:
        """Test that transfer with same structure succeeds without error."""
        action_ids = {_ACTION_ID, "a2"}
        shared_kwargs = dict(n_features=_N_FEATURES, activation="relu", hidden_dim_list=[16], strategy=ClassicBandit())
        current = CmabBernoulli.cold_start(action_ids=action_ids, **shared_kwargs)
        template = CmabBernoulli.cold_start(action_ids=action_ids, **shared_kwargs)
        result = edit_model_on_the_fly(current, template)
        assert action_ids.issubset(result.actions)

    def test_transfer_update_kwargs_change_allowed(self) -> None:
        """Test that changing update_kwargs is allowed (configurable hyperparameter)."""
        shared_kwargs = dict(n_features=_N_FEATURES, activation="relu", strategy=ClassicBandit())
        current = CmabBernoulli.cold_start(action_ids={_ACTION_ID}, **shared_kwargs, update_kwargs={"num_steps": 50})
        template = CmabBernoulli.cold_start(action_ids={_ACTION_ID}, **shared_kwargs, update_kwargs={"num_steps": 200})
        result = edit_model_on_the_fly(current, template)
        assert result.actions[_ACTION_ID].update_kwargs["num_steps"] == 200

    def test_transfer_beta_models_no_validation(self) -> None:
        """Test that Beta models skip structural validation."""
        current = SmabBernoulli.cold_start(action_ids={_ACTION_ID, "a2"}, strategy=ClassicBandit())
        current.update(actions=[_ACTION_ID], rewards=[1])
        template = SmabBernoulli.cold_start(action_ids={_ACTION_ID, "a3"}, strategy=ClassicBandit())
        result = edit_model_on_the_fly(current, template)
        assert set(result.actions.keys()) == {_ACTION_ID, "a3"}
        assert result.actions[_ACTION_ID].n_successes == 2

    def test_transfer_mo_validates_all_objectives(self) -> None:
        """Test that multi-objective models validate each objective separately."""
        val1, val2 = _BNN_STRUCTURAL_KEY_VALUES["activation"]
        current = CmabBernoulliMO.cold_start(
            action_ids={_ACTION_ID},
            n_features=_N_FEATURES,
            n_objectives=2,
            activation=val1,
            strategy=MultiObjectiveBandit(),
        )
        template = CmabBernoulliMO.cold_start(
            action_ids={_ACTION_ID},
            n_features=_N_FEATURES,
            n_objectives=2,
            activation=val2,
            strategy=MultiObjectiveBandit(),
        )
        with pytest.raises(ValueError, match="activation"):
            edit_model_on_the_fly(current, template)

    def test_transfer_mo_objective_count_mismatch_raises_error(self) -> None:
        """Test that changing number of objectives raises ValueError."""
        shared_kwargs = dict(n_features=_N_FEATURES, strategy=MultiObjectiveBandit())
        current = CmabBernoulliMO.cold_start(action_ids={_ACTION_ID}, n_objectives=2, **shared_kwargs)
        template = CmabBernoulliMO.cold_start(action_ids={_ACTION_ID}, n_objectives=3, **shared_kwargs)
        with pytest.raises(ValueError, match="number of objectives mismatch"):
            edit_model_on_the_fly(current, template)

    def test_transfer_multiple_actions_validates_each(self) -> None:
        """Test that validation applies to each overlapping action independently."""
        val1, val2 = _BNN_STRUCTURAL_KEY_VALUES["activation"]
        action_ids = {_ACTION_ID, "a2"}
        current = CmabBernoulli.cold_start(
            action_ids=action_ids, n_features=_N_FEATURES, activation=val1, strategy=ClassicBandit()
        )
        template = CmabBernoulli.cold_start(
            action_ids=action_ids, n_features=_N_FEATURES, activation=val2, strategy=ClassicBandit()
        )
        with pytest.raises(ValueError, match="activation"):
            edit_model_on_the_fly(current, template)

    def test_transfer_non_overlapping_actions_no_validation(self) -> None:
        """Test that non-overlapping actions don't trigger validation."""
        val1, val2 = _BNN_STRUCTURAL_KEY_VALUES["activation"]
        current = CmabBernoulli.cold_start(
            action_ids={_ACTION_ID}, n_features=_N_FEATURES, activation=val1, strategy=ClassicBandit()
        )
        template = CmabBernoulli.cold_start(
            action_ids={"a2"}, n_features=_N_FEATURES, activation=val2, strategy=ClassicBandit()
        )
        result = edit_model_on_the_fly(current, template)
        assert set(result.actions.keys()) == {"a2"}
