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

"""Transfer learning functionality for Multi-Armed Bandits.

This module provides the edit_model_on_the_fly() function for transfer learning,
which enables updating MAB configuration while preserving learned state and
automatically handling input dimension changes for contextual MABs.
"""

import json
from typing import Any, Dict, Tuple, Type

from loguru import logger

from pybandits.base_model import BaseModelMO, BaseModelSO
from pybandits.cmab import BaseCmabBernoulli
from pybandits.mab import BaseMab
from pybandits.model import BaseBayesianNeuralNetwork
from pybandits.pydantic_version_compatibility import validate_call
from pybandits.quantitative_model import QuantitativeModel


def _pick_learned_state(source_state: Dict[str, Any], model_cls: Type[BaseModelSO]) -> Dict[str, Any]:
    """Extract transferable learned parameters from a source state dict.

    Parameters
    ----------
    source_state : Dict[str, Any]
        Source action state dictionary.
    model_cls : Type[BaseModelSO]
        Model class whose transfer_learned_keys declares which keys to extract.

    Returns
    -------
    Dict[str, Any]
        Subset of source_state containing only the declared learned keys.
    """
    return {k: source_state[k] for k in model_cls.transfer_learned_keys if k in source_state}


def _get_state_dict(mab: BaseMab) -> Dict[str, Any]:
    """Convert MAB to dictionary via JSON serialization.

    Parameters
    ----------
    mab : BaseMab
        MAB instance to convert.

    Returns
    -------
    state_dict : Dict[str, Any]
        Dictionary representation of MAB state.
    """
    _, state_json = mab.get_state()
    return json.loads(state_json)


def _reconstruct_mab(mab_class: Type[BaseMab], state_dict: Dict[str, Any]) -> BaseMab:
    """Reconstruct MAB from state dictionary.

    Parameters
    ----------
    mab_class : Type[BaseMab]
        MAB class to instantiate.
    state_dict : Dict[str, Any]
        State dictionary to reconstruct from.

    Returns
    -------
    mab : BaseMab
        Reconstructed MAB instance.
    """
    state_json = json.dumps(state_dict)
    return mab_class.from_state(state_json)


_VARIANT_LABELS = (
    ("CMAB", "SMAB", "context type (CMAB vs SMAB)"),
    ("MO", "SO", "objective type (MO vs SO)"),
    ("quantitative", "standard", "action type (quantitative vs standard)"),
)


def _mab_variant(mab: BaseMab) -> Tuple[bool, bool, bool]:
    """Return (is_cmab, is_mo, is_quantitative) for a MAB based on its type and action models."""
    first_model = next(iter(mab.actions.values()), None)
    return (
        isinstance(mab, BaseCmabBernoulli),
        isinstance(first_model, BaseModelMO),
        isinstance(first_model, QuantitativeModel),
    )


def _validate_mab_compatibility(mab1: BaseMab, mab2: BaseMab) -> None:
    """Validate that two MABs can be merged.

    Checks that both MABs share the same three variant dimensions:
    - Context type: CMAB (contextual) vs SMAB (stochastic)
    - Objective type: MO (multi-objective) vs SO (single-objective)
    - Action type: quantitative (zooming) vs standard (Bernoulli/Beta)

    Parameters
    ----------
    mab1 : BaseMab
        First MAB to check.
    mab2 : BaseMab
        Second MAB to check.

    Raises
    ------
    TypeError
        If MABs have incompatible variant dimensions.
    """

    v1 = _mab_variant(mab1)
    v2 = _mab_variant(mab2)

    for (val1, val2), (pos_label, neg_label, dimension) in zip(zip(v1, v2), _VARIANT_LABELS):
        if val1 != val2:
            raise TypeError(
                f"Cannot merge MABs with different {dimension}: "
                f"mab1 is {pos_label if val1 else neg_label}, "
                f"mab2 is {pos_label if val2 else neg_label}."
            )


def _validate_single_objective_model(src: Any, tgt: Any, action_id: str, model_label: str = "") -> None:
    """Validate a single SO model or one objective within MO."""
    prefix = f" (objective {model_label})" if model_label else ""

    for key in type(src).transfer_structural_keys:
        src_val = getattr(src, key)
        tgt_val = getattr(tgt, key, None)
        if src_val != tgt_val:
            raise ValueError(
                f"Cannot transfer learned state for action '{action_id}'{prefix}: "
                f"{key} mismatch (source: {src_val!r}, template: {tgt_val!r})."
            )

    for key in type(src).transfer_extendable_keys:
        src_val = getattr(src, key)
        tgt_val = getattr(tgt, key, None)
        if src_val != tgt_val:
            logger.warning(
                f"Transfer learning for action '{action_id}'{prefix}: "
                f"{key} changing from {src_val!r} to {tgt_val!r}. "
                f"This changes the inference method but does not affect learned weights."
            )


def _validate_model_compatibility(
    source_model: Any,
    target_model: Any,
    action_id: str,
) -> None:
    """Validate that source and target model instances have compatible structures for transfer learning.

    Structural params are compared via getattr using each class's transfer_structural_keys.
    Extendable params are compared via getattr using each class's transfer_extendable_keys.

    For MO (multi-objective) models, validates each objective's model separately.

    Parameters
    ----------
    source_model : Any
        Source model instance (from mab1.actions[action_id]).
    target_model : Any
        Target model instance (from mab2.actions[action_id]).
    action_id : str
        Action ID for error messages.

    Raises
    ------
    ValueError
        If models have incompatible structures that prevent weight transfer.
    """

    # MO models have a .models list of SO models; validate per-objective
    if isinstance(source_model, BaseModelMO):
        if len(source_model.models) != len(target_model.models):
            raise ValueError(
                f"Cannot transfer learned state for action '{action_id}': "
                f"number of objectives mismatch "
                f"(source: {len(source_model.models)}, template: {len(target_model.models)})."
            )
        for obj_idx, (src_obj, tgt_obj) in enumerate(zip(source_model.models, target_model.models)):
            _validate_single_objective_model(src_obj, tgt_obj, action_id, str(obj_idx))
    else:
        _validate_single_objective_model(source_model, target_model, action_id)


def _transfer_learned_state(
    target_action_state: Dict[str, Any],
    source_action_state: Dict[str, Any],
    model_instance: Any,
) -> Dict[str, Any]:
    """Transfer learned state from source to target action.

    Takes the target action's structure and configuration,
    but replaces learned state with values from source. The set of parameters
    that counts as learned state is determined by the model class itself via
    get_transfer_learned_params().

    Since we work with state dicts from JSON serialization, objects are already
    independent copies, so we don't need deepcopy.

    Parameters
    ----------
    target_action_state : Dict[str, Any]
        Target action state (defines structure and configuration).
    source_action_state : Dict[str, Any]
        Source action state (provides learned state to transfer).
    model_instance : Any
        The actual model object from the source MAB for this action. Used to dispatch
        to the correct model class without relying on field presence.

    Returns
    -------
    merged_action_state : Dict[str, Any]
        Action state with target's configuration and source's learned state.
    """
    result = target_action_state.copy()

    if isinstance(model_instance, BaseModelMO):
        # MO: transfer learned state per objective using each objective's model class
        result["models"] = [
            {**tgt_obj, **_pick_learned_state(src_obj, type(obj_model))}
            for obj_model, src_obj, tgt_obj in zip(
                model_instance.models, source_action_state["models"], target_action_state["models"]
            )
        ]
    else:
        # SO: use the model class's declared transferable keys
        result.update(_pick_learned_state(source_action_state, type(model_instance)))

    return result


@validate_call
def _merge_mabs(
    mab1: BaseMab,
    mab2: BaseMab,
) -> BaseMab:
    """Merge two MABs using mab2 as template with learned state transfer.

    Uses mab2 to define the structure of the resulting MAB (which actions exist and their
    configuration), while transferring learned state from mab1 for overlapping actions.
    This enables transfer learning scenarios like updating hyperparameters or changing the
    action set while preserving learned knowledge.

    The resulting MAB contains:
    - Actions: Only actions from mab2 (mab2 defines the final action set)
    - Configuration: All parameters (epsilon, delta, update_kwargs, etc.) from mab2
    - Learned state: For overlapping actions, transferable parameters (n_successes,
      n_failures, model_params) from mab1
    - Strategy: From mab2

    Parameters
    ----------
    mab1 : BaseMab
        Source MAB providing learned state for overlapping actions.
    mab2 : BaseMab
        Template MAB defining which actions exist and their configuration.

    Returns
    -------
    merged_mab : BaseMab
        New MAB with mab2's structure and mab1's learned state for overlapping actions.

    Raises
    ------
    TypeError
        If MAB classes don't match (e.g., SmabBernoulli vs CmabBernoulli).

    Examples
    --------
    >>> from pybandits.smab import SmabBernoulli
    >>> from pybandits.strategy import ClassicBandit
    >>>
    >>> # Example 1: No overlapping actions - simple combination
    >>> mab1 = SmabBernoulli.cold_start(action_ids={'a1', 'a2'}, strategy=ClassicBandit())
    >>> mab2 = SmabBernoulli.cold_start(action_ids={'a3', 'a4'}, strategy=ClassicBandit())
    >>> merged = merge_mabs(mab1, mab2)
    >>> sorted(merged.actions.keys())  # Only actions from mab2
    ['a3', 'a4']
    >>>
    >>> # Example 2: Overlapping actions - transfer learned state
    >>> mab1 = SmabBernoulli.cold_start(action_ids={'a1', 'a2'}, strategy=ClassicBandit())
    >>> mab1.update(actions=['a1'], rewards=[1])  # Train a1 in mab1
    >>> mab2 = SmabBernoulli.cold_start(action_ids={'a1', 'a3'}, strategy=ClassicBandit())
    >>> merged = merge_mabs(mab1, mab2)
    >>> # Result has only actions from mab2: a1 (with learned state from mab1) and a3
    >>> sorted(merged.actions.keys())  # a2 is discarded (not in mab2)
    ['a1', 'a3']
    >>> merged.actions['a1'].n_successes  # Learned state from mab1
    2
    >>> merged.actions['a3'].n_successes  # Cold start from mab2
    1
    >>>
    >>> # Example 3: Update hyperparameters while keeping learned weights
    >>> from pybandits.cmab import CmabBernoulli
    >>> mab1 = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5,
    ...                                  update_kwargs={'fit': {'n': 50}})
    >>> # ... train mab1 ...
    >>> mab2 = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5,
    ...                                  update_kwargs={'fit': {'n': 200}})
    >>> merged = merge_mabs(mab1, mab2)
    >>> merged.actions['a1'].update_kwargs['fit']['n']  # New config from mab2
    200
    >>> # But model_params (learned weights) come from mab1
    """
    _validate_mab_compatibility(mab1, mab2)
    state1 = _get_state_dict(mab1)
    state2 = _get_state_dict(mab2)

    # Extract actions
    actions1 = state1["actions_manager"]["actions"]
    actions2 = state2["actions_manager"]["actions"]

    # Count overlapping actions
    overlapping = set(actions1.keys()) & set(actions2.keys())

    # Use mab2 as template (defines which actions exist and their configuration)
    # Transfer learned state from mab1 for overlapping actions
    merged_actions = {}
    for action_id, action_state in actions2.items():
        if action_id in actions1:
            source_model = mab1.actions[action_id]
            target_model = mab2.actions[action_id]

            # Validate model compatibility before transfer
            _validate_model_compatibility(source_model, target_model, action_id)

            # Action exists in both: use mab2's config, mab1's learned state
            merged_actions[action_id] = _transfer_learned_state(
                target_action_state=action_state,
                source_action_state=actions1[action_id],
                model_instance=source_model,
            )
        else:
            # Action only in mab2: use as-is (cold start)
            # No need to copy - already from JSON deserialization
            merged_actions[action_id] = action_state

    logger.info(
        f"Merged {type(mab1).__name__}: used mab2 as template with {len(actions2)} action(s), "
        f"transferred learned state from mab1 for {len(overlapping)} overlapping action(s)."
    )

    # Create merged state using mab2 as base (provides all configuration)
    # Then replace actions with merged versions (which have learned state from mab1)
    merged_state = state2
    merged_state["actions_manager"]["actions"] = merged_actions

    # Merge actions_with_change sets (for adaptive windowing)
    actions_with_change_1 = set(state1["actions_manager"].get("actions_with_change", []))
    actions_with_change_2 = set(state2["actions_manager"].get("actions_with_change", []))
    merged_state["actions_manager"]["actions_with_change"] = list(actions_with_change_1 | actions_with_change_2)

    # Reconstruct MAB using mab2's type (mab2 is the template defining final structure)
    return _reconstruct_mab(type(mab2), merged_state)


def _append_template_bnn_features(
    current_model_params: Dict[str, Any],
    template_model_params: Dict[str, Any],
    current_dim: int,
) -> None:
    """Append template's new feature weights to current model parameters (in-place).

    Parameters
    ----------
    current_model_params : Dict[str, Any]
        Model parameters dict to expand (modified in-place).
    template_model_params : Dict[str, Any]
        Template model parameters with additional features.
    current_dim : int
        Current input dimension (number of features in current model).
    """
    # For both learned and init params
    for params_key in ["bnn_layer_params", "bnn_layer_params_init"]:
        if params_key not in current_model_params or params_key not in template_model_params:
            continue

        current_layer = current_model_params[params_key][0]
        template_layer = template_model_params[params_key][0]

        # Expand weight distribution parameters
        current_weight = current_layer[BaseBayesianNeuralNetwork.weight_var_name]
        template_weight = template_layer[BaseBayesianNeuralNetwork.weight_var_name]

        for field_name in current_weight:
            if isinstance(current_weight[field_name], list) and current_weight[field_name]:
                if isinstance(current_weight[field_name][0], list):
                    # This is 2D array (input_dim, output_dim)
                    # Append the new rows from template (from current_dim onward)
                    new_rows = template_weight[field_name][current_dim:]
                    current_weight[field_name].extend(new_rows)


def _expand_with_template_weights(
    current_cmab: BaseCmabBernoulli,
    template_cmab: BaseCmabBernoulli,
) -> BaseCmabBernoulli:
    """Expand current CMAB to match template's input dimension using template's weights for new features.

    Parameters
    ----------
    current_cmab : BaseCmabBernoulli
        CMAB with smaller input dimension.
    template_cmab : BaseCmabBernoulli
        Template CMAB with larger input dimension.

    Returns
    -------
    expanded_cmab : BaseCmabBernoulli
        Expanded CMAB with current's existing feature weights and template's new feature weights.
    """
    current_dim = current_cmab.input_dim
    template_dim = template_cmab.input_dim
    n_new_features = template_dim - current_dim

    logger.info(
        f"Expanding current CMAB from {current_dim} to {template_dim} features "
        f"using template's weights for {n_new_features} new feature(s)"
    )

    # Get state dicts
    current_state = _get_state_dict(current_cmab)
    template_state = _get_state_dict(template_cmab)

    current_actions = current_state["actions_manager"]["actions"]
    template_actions = template_state["actions_manager"]["actions"]

    # For each overlapping action, expand by appending template's new features
    for action_id in set(current_actions.keys()) & set(template_actions.keys()):
        current_action = current_actions[action_id]
        template_action = template_actions[action_id]
        action_model = current_cmab.actions[action_id]

        if isinstance(action_model, BaseModelMO):
            # Multi-objective: each objective has its own BNN model
            for model_idx in range(len(current_action["models"])):
                if model_idx >= len(template_action["models"]):
                    continue
                _append_template_bnn_features(
                    current_action["models"][model_idx]["model_params"],
                    template_action["models"][model_idx]["model_params"],
                    current_dim,
                )
        elif isinstance(action_model, BaseBayesianNeuralNetwork):
            # Single-objective BNN
            _append_template_bnn_features(
                current_action["model_params"],
                template_action["model_params"],
                current_dim,
            )
        else:
            raise TypeError(
                f"Unsupported model type for feature expansion: {type(action_model).__name__}. "
                "Only BNN-based CMAB models support input dimension expansion."
            )

    return _reconstruct_mab(type(current_cmab), current_state)


@validate_call
def edit_model_on_the_fly(
    current_mab: BaseMab,
    new_mab: BaseMab,
) -> BaseMab:
    """Update MAB configuration and actions using new_mab as template.

    Uses new_mab to define which actions exist and their configuration (editable parameters),
    while preserving learned state (non-editable parameters) from current_mab for overlapping
    actions. This is useful for transfer learning scenarios where you want to:
    - Update hyperparameters while keeping learned weights
    - Add or remove actions while preserving knowledge of existing ones
    - Change model configuration without retraining from scratch
    - Add new input features to CMAB models while preserving existing feature weights

    Model Compatibility Validation
    -------------------------------
    Transfer learning requires structural compatibility between models. The function validates
    that overlapping actions have compatible model structures using a 3-tier system:

    **STRUCTURAL_IMMUTABLE (must match - raises ValueError):**
        - activation: Activation function (tanh, relu, sigmoid, gelu)
        - use_residual_connections: Whether skip connections are used

    **STRUCTURAL_EXTENDABLE (warns but allows):**
        - update_method: Inference method (VI vs MCMC) - emits warning if changed

    **CONFIGURABLE_MUTABLE (safe to change):**
        - update_kwargs: Training hyperparameters (epochs, learning rate, etc.)
        - use_layerwise_scaling: Only affects initialization
        - epsilon, delta: Exploration parameters

    For Beta models (simple Bernoulli), no structural validation is needed as they only
    store n_successes and n_failures counts.

    For CMABs with different input dimensions:
    - If template has MORE features: automatically expands current MAB using template's weights for new features
    - If template has FEWER features: raises ValueError (cannot reduce dimensions)

    The resulting MAB will have:
    - Actions: All actions from new_mab
    - Configuration: All parameters from new_mab (epsilon, delta, update_kwargs, etc.)
    - Learned state: For actions that exist in both MABs, transferable parameters
      (n_successes, n_failures, model_params) from current_mab
    - Strategy: From new_mab
    - Input features (CMAB): Existing features from current_mab + new features from template

    Parameters
    ----------
    current_mab : BaseMab
        MAB with current actions and learned state.
    new_mab : BaseMab
        MAB defining desired actions and configuration.

    Returns
    -------
    updated_mab : BaseMab
        New MAB with actions from new_mab, configuration from new_mab,
        and learned state from current_mab for overlapping actions.

    Raises
    ------
    TypeError
        If MAB classes don't match.
    ValueError
        If template CMAB has fewer features than current CMAB, or if overlapping actions
        have incompatible model structures (different activation or residual connections).

    Examples
    --------
    >>> from pybandits.smab import SmabBernoulli
    >>> from pybandits.cmab import CmabBernoulli
    >>> from pybandits.strategy import ClassicBandit
    >>>
    >>> # Example 1: Simple MAB - update actions
    >>> current = SmabBernoulli.cold_start(action_ids={'a1', 'a2'}, strategy=ClassicBandit())
    >>> current.update(actions=['a1', 'a1'], rewards=[1, 1])  # Train a1
    >>> new = SmabBernoulli.cold_start(action_ids={'a1', 'a3'}, strategy=ClassicBandit())
    >>> merged = edit_model_on_the_fly(current, new)
    >>> len(merged.actions)
    2
    >>> merged.actions['a1'].n_successes  # Learned state from current
    3
    >>> merged.actions['a3'].n_successes  # Cold start from new
    1
    >>>
    >>> # Example 2: CMAB - update hyperparameters (ALLOWED)
    >>> current = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5,
    ...                                      activation='relu', update_kwargs={'fit': {'n': 50}})
    >>> # ... train current ...
    >>> new = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5,
    ...                                 activation='relu', update_kwargs={'fit': {'n': 200}})
    >>> merged = edit_model_on_the_fly(current, new)  # Success - only hyperparameters changed
    >>>
    >>> # Example 3: CMAB - incompatible activation (RAISES ValueError)
    >>> current = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5, activation='relu')
    >>> new = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5, activation='tanh')
    >>> try:
    ...     merged = edit_model_on_the_fly(current, new)
    ... except ValueError as e:
    ...     print("Error: activation mismatch prevents transfer")
    >>>
    >>> # Example 4: CMAB - update_method change (WARNS but allows)
    >>> current = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5, update_method='VI')
    >>> new = CmabBernoulli.cold_start(action_ids={'a1'}, n_features=5, update_method='MCMC')
    >>> merged = edit_model_on_the_fly(current, new)  # Success with warning
    """
    _validate_mab_compatibility(current_mab, new_mab)

    # Check if we're dealing with CMABs and handle input dimension mismatches
    if isinstance(current_mab, BaseCmabBernoulli) and isinstance(new_mab, BaseCmabBernoulli):
        # Get input dimensions via the property on BaseCmabBernoulli
        try:
            current_dim = current_mab.input_dim
            new_dim = new_mab.input_dim

            if current_dim != new_dim:
                if new_dim < current_dim:
                    raise ValueError(
                        f"Template MAB has fewer features ({new_dim}) than current MAB ({current_dim}). "
                        "Cannot reduce feature dimensions. Template must have same or more features."
                    )
                else:
                    # new_dim > current_dim: expand current_mab to match template's dimension
                    logger.info(
                        f"Template has more features ({new_dim}) than current ({current_dim}). "
                        f"Expanding current MAB to match template's dimension."
                    )
                    current_mab = _expand_with_template_weights(current_mab, new_mab)
        except (StopIteration, AttributeError):
            # No actions or not a CMAB with input_dim, proceed normally
            pass

    # Merge using new_mab as template with learned state transfer from current_mab
    merged_mab = _merge_mabs(current_mab, new_mab)

    logger.info(
        f"Updated MAB using new_mab as template. "
        f"Final MAB has {len(merged_mab.actions)} action(s) with new_mab's configuration "
        f"and current_mab's learned state for overlapping actions."
    )

    return merged_mab
