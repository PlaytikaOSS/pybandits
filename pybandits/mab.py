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

import importlib.metadata
import json
import re
from abc import ABC, abstractmethod
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Self, Set, Tuple, Union, get_origin

import numpy as np
from packaging import version
from pydantic import (
    NonNegativeInt,
    PrivateAttr,
    validate_call,
)

from pybandits.actions_manager import ActionsManager
from pybandits.base import (
    ActionId,
    ActionRewardLikelihood,
    BinaryReward,
    Float01,
    ForbiddenActions,
    MOProbability,
    MOProbabilityWeight,
    PositiveFloat01,
    Predictions,
    Probability,
    ProbabilityWeight,
    PyBanditsBaseModel,
    QuantitativeMOProbability,
    QuantitativeMOProbabilityWeight,
    QuantitativeProbability,
    QuantitativeProbabilityWeight,
    Serializable,
    UnifiedActionId,
)
from pybandits.base_model import BaseModel
from pybandits.model import Model, ModelMO
from pybandits.quantitative_model import QuantitativeModel
from pybandits.strategy import BaseStrategy
from pybandits.utils import extract_argument_names_from_function


def _get_pybandits_version() -> str:
    """Get pybandits version from installed metadata or pyproject.toml."""
    # Try installed metadata first (works in wheels)
    try:
        return importlib.metadata.version("pybandits")
    except importlib.metadata.PackageNotFoundError:
        # Fall back to parsing pyproject.toml (development only)
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path) as f:
                content = f.read()
            match = re.search(r'version = "([^"]+)"', content)
            if match:
                return match.group(1)
    raise RuntimeError("Could not determine pybandits version")


class BaseMab(PyBanditsBaseModel, ABC):
    """
    Multi-armed bandit superclass.

    Parameters
    ----------
    actions : Dict[ActionId, Model]
        The list of possible actions, and their associated Model.
    strategy : Strategy
        The strategy used to select actions.
    epsilon : Optional[Float01], 0 if not specified.
        The probability of selecting a random action.
    default_action : Optional[ActionId], None if not specified.
        The default action to select with a probability of epsilon when using the epsilon-greedy approach.
        If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
    default_action_fraction : Optional[PositiveFloat01], None if not specified.
        Probability of picking `default_action` (vs a uniformly-random action) when the epsilon-greedy
        coin flip selects the explore branch. Only meaningful together with `epsilon` and `default_action`.
        Must be in the range (0, 1] — use ``None`` to preserve legacy behavior (always default when set).
        Boundary semantics:
            - `default_action_fraction = 1.0`: explore always returns `default_action` (same as omitting this field).
            - `None` (default): legacy behavior; explore returns `default_action` deterministically when set,
              otherwise a uniformly-random action.
    limited_actions : Optional[Set[ActionId]], None if not specified.
        A set of actions whose selection is throttled, e.g. newly-introduced arms that Thompson Sampling
        would otherwise over-explore. On each selection they are allowed to compete only with probability
        `limited_action_fraction`; otherwise they are masked out. Must be used together with
        `limited_action_fraction`, must be a subset of the action set, and must not contain `default_action`.
    limited_action_fraction : Optional[Float01], None if not specified.
        Probability that the `limited_actions` are allowed to compete on a given selection (mirrors
        `epsilon`: higher = more exploration). Only meaningful together with `limited_actions`.
    current_supported_version_th : ClassVar[str]
        The threshold of the supported version of PyBandits which don't require any changes to the state.
    strategy_kwargs : Dict[str, Any]
        Relevant only if strategy was not provided. This argument contains the parameters for the strategy,
        which in turn will be used to instantiate the strategy.
    """

    actions_manager: ActionsManager
    strategy: BaseStrategy
    epsilon: Optional[Float01] = None
    default_action: Optional[UnifiedActionId] = None
    default_action_fraction: Optional[PositiveFloat01] = None
    limited_actions: Optional[Set[ActionId]] = None
    limited_action_fraction: Optional[Float01] = None
    version: Optional[str] = None
    random_seed: Optional[NonNegativeInt] = None
    _current_supported_version_th: ClassVar[str] = _get_pybandits_version()
    _rng: Any = PrivateAttr(default=None)

    def __init__(
        self,
        epsilon: Optional[Float01] = None,
        default_action: Optional[UnifiedActionId] = None,
        default_action_fraction: Optional[PositiveFloat01] = None,
        limited_actions: Optional[Set[ActionId]] = None,
        limited_action_fraction: Optional[Float01] = None,
        version: Optional[str] = None,
        random_seed: Optional[NonNegativeInt] = None,
        **kwargs,
    ):
        # Inject random_seed into kwargs so it flows through ActionsManager into BNN cold_start.
        if "random_seed" not in kwargs:
            kwargs["random_seed"] = random_seed
        class_attributes = {
            attribute_name: self._get_instantiated_class_attribute(attribute_name, kwargs)
            for attribute_name in self._get_class_type_attributes()
        }
        # Pop random_seed in case it was not consumed by any sub-model constructor.
        kwargs.pop("random_seed", None)
        if kwargs:
            raise ValueError(f"Unknown arguments: {kwargs.keys()}")

        version = _get_pybandits_version()
        super().__init__(
            **class_attributes,
            epsilon=epsilon,
            default_action=default_action,
            default_action_fraction=default_action_fraction,
            limited_actions=limited_actions,
            limited_action_fraction=limited_action_fraction,
            version=version,
            random_seed=random_seed,
        )

    @classmethod
    def _get_instantiated_class_attribute(cls, attribute_name: str, kwargs: Dict[str, Any]) -> PyBanditsBaseModel:
        attribute_class = cls._get_attribute_type(attribute_name)
        if attribute_name in kwargs:
            attribute = (
                kwargs[attribute_name]
                if isinstance(kwargs[attribute_name], attribute_class)
                else attribute_class(**kwargs.pop(attribute_name))
            )
        else:
            required_sub_attributes = extract_argument_names_from_function(attribute_class.__init__)
            if required_sub_attributes == extract_argument_names_from_function(
                PyBanditsBaseModel.__init__
            ):  # case of no native __init__ method, just pydantic generic __init__
                required_sub_attributes = list(attribute_class.model_fields.keys())
                sub_attributes = {k: kwargs.pop(k) for k in required_sub_attributes if k in kwargs}
            else:
                sub_attributes = {k: kwargs.pop(k) for k in required_sub_attributes if k in kwargs}
                if "kwargs" in required_sub_attributes:
                    sub_attributes["kwargs"] = kwargs

            attribute = attribute_class(**sub_attributes)
        kwargs.pop(attribute_name, None)
        return attribute

    ############################################ Instance Input Validators #############################################

    def model_post_init(self, __context: Any) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        if self.actions_manager.delta is not None and (not self.epsilon or self.default_action is not None):
            raise ValueError("Adaptive window requires epsilon greedy super strategy with not default action.")
        if self.default_action_fraction is not None and not self.epsilon:
            raise AttributeError("default_action_fraction requires epsilon to be defined.")
        if self.default_action_fraction is not None and self.default_action is None:
            raise AttributeError("default_action_fraction requires default_action to be defined.")
        if not self.epsilon and self.default_action:
            raise AttributeError("A default action should only be defined when epsilon is defined.")
        if self.default_action:
            action_id = self.default_action[0] if isinstance(self.default_action, tuple) else self.default_action
            if action_id not in self.actions:
                raise AttributeError("The default action must be valid action defined in the actions set.")
        if (
            self.default_action
            and isinstance(self.default_action, tuple)
            and not isinstance(self.actions[self.default_action[0]], QuantitativeModel)
        ):
            raise AttributeError("Quantitative default action requires a quantitative action model.")
        if (
            self.default_action
            and isinstance(self.default_action, str)
            and not isinstance(self.actions[self.default_action], (Model, ModelMO))
        ):
            raise AttributeError("Standard default action requires a standard action model.")
        if bool(self.limited_actions) != (self.limited_action_fraction is not None):
            raise AttributeError("limited_actions and limited_action_fraction must be defined together.")
        if self.limited_actions:
            if not self.limited_actions.issubset(self.actions):
                raise AttributeError("limited_actions must be a subset of the action set.")
            if self.default_action in self.limited_actions:
                raise AttributeError("default_action must not be a limited action.")

    ############################################# Method Input Validators ##############################################

    @staticmethod
    def _to_feasibility_constraint(region: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        """
        Convert a forbidden-region margin into an optimizer feasibility constraint.

        The optimizer (``maximize_by_quantity``) treats a constraint ``g`` as feasible when ``g(x) >= 0``.
        A forbidden region uses the opposite, signed-margin convention (``region(x) > 0`` means forbidden), so the
        feasibility constraint is simply the negation ``g(x) = -region(x)``. With this mapping, ``g(x) < 0``
        consistently means "x is forbidden by this region", which the explore branch reuses for rejection sampling.

        Parameters
        ----------
        region: Callable[[np.ndarray], float]
            The forbidden-region margin (``region(x) > 0`` => forbidden).

        Returns
        -------
        Callable[[np.ndarray], float]
            The feasibility constraint (``>= 0`` feasible).
        """

        def feasibility_constraint(x: np.ndarray) -> float:
            return -float(region(x))

        return feasibility_constraint

    def _normalize_forbidden_actions(
        self, forbidden_actions: Optional[ForbiddenActions]
    ) -> Tuple[Set[ActionId], Dict[ActionId, List[Callable[[np.ndarray], float]]]]:
        """
        Normalize ``forbidden_actions`` into valid action IDs and per-arm region feasibility constraints.

        Accepts both the legacy ``Set[ActionId]`` form (whole-arm blocking) and the generalized
        ``Dict[ActionId, None | ForbiddenRegion | List[ForbiddenRegion]]`` form, where a ``None`` value forbids
        the whole arm and region callable(s) forbid part of a quantitative arm's quantity space.

        Parameters
        ----------
        forbidden_actions: Optional[ForbiddenActions]
            The whole-arm and/or per-arm region restrictions.

        Returns
        -------
        valid_actions: Set[ActionId]
            Action IDs that remain selectable (region-forbidden arms stay valid; only their quantity space shrinks).
        region_constraints: Dict[ActionId, List[Callable[[np.ndarray], float]]]
            Per-arm feasibility constraints (``>= 0`` feasible) derived from the forbidden regions.
        """
        action_ids = set(self.actions.keys())
        whole_forbidden: Set[ActionId] = set()
        region_constraints: Dict[ActionId, List[Callable[[np.ndarray], float]]] = {}

        if forbidden_actions is None:
            forbidden_actions = set()
        if isinstance(forbidden_actions, dict):
            for action_id, regions in forbidden_actions.items():
                if action_id not in action_ids:
                    raise ValueError("forbidden_actions contains invalid action IDs.")
                if regions is None:
                    whole_forbidden.add(action_id)
                    continue
                if not isinstance(self.actions[action_id], QuantitativeModel):
                    raise ValueError(
                        f"Forbidden regions can only be specified for quantitative actions; '{action_id}' is not one."
                    )
                region_list = regions if isinstance(regions, list) else [regions]
                region_constraints[action_id] = [self._to_feasibility_constraint(r) for r in region_list]
        else:
            if not all(a in action_ids for a in forbidden_actions):
                raise ValueError("forbidden_actions contains invalid action IDs.")
            whole_forbidden = set(forbidden_actions)

        valid_actions = action_ids - whole_forbidden
        if len(valid_actions) == 0:
            raise ValueError("All actions are forbidden. You must allow at least 1 action.")
        if self.default_action:
            action_id = self.default_action[0] if isinstance(self.default_action, tuple) else self.default_action
            if action_id not in valid_actions:
                raise ValueError("The default action is forbidden.")
            # A quantitative default action must not point into one of its own forbidden regions.
            if isinstance(self.default_action, tuple) and action_id in region_constraints:
                point = np.array(self.default_action[1])
                if any(constraint(point) < 0 for constraint in region_constraints[action_id]):
                    raise ValueError("The default action is in a forbidden region.")

        return valid_actions, region_constraints

    def _get_valid_actions(self, forbidden_actions: Optional[ForbiddenActions]) -> Set[ActionId]:
        """
        Given forbidden actions, return the set of valid (selectable) action IDs.

        Parameters
        ----------
        forbidden_actions: Optional[ForbiddenActions]
            The whole-arm and/or per-arm region restrictions.

        Returns
        -------
        valid_actions: Set[ActionId]
            The set of valid (i.e. not wholly forbidden) action IDs.
        """
        return self._normalize_forbidden_actions(forbidden_actions)[0]

    ####################################################################################################################

    @property
    def actions(self) -> Dict[ActionId, BaseModel]:
        return self.actions_manager.actions

    @validate_call
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        **kwargs,
    ):
        """
        Update the multi-armed bandit model.

        Parameters
        ----------
        actions: List[ActionId]
            The selected action for each sample.
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]] of shape (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        quantities: Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        self.actions_manager.update(
            actions=actions,
            rewards=rewards,
            quantities=quantities,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
            **kwargs,
        )

    @staticmethod
    def _transform_nested_list(lst: List[List[Dict]]):
        return [{k: v for d in single_action_dicts for k, v in d.items()} for single_action_dicts in zip(*lst)]

    def _get_action_probabilities(
        self, valid_actions: Set[ActionId], **kwargs
    ) -> Union[
        List[Dict[ActionId, Union[Probability, QuantitativeProbability]]],
        List[Dict[ActionId, Union[ProbabilityWeight, QuantitativeProbabilityWeight]]],
        List[Dict[ActionId, Union[MOProbability, QuantitativeMOProbability]]],
        List[Dict[ActionId, Union[MOProbabilityWeight, QuantitativeMOProbabilityWeight]]],
    ]:
        """
        Get the probability of getting a positive reward for each action.

        Parameters
        ----------
        valid_actions : Set[ActionId]
            The action IDs to sample. Wholly-forbidden arms are excluded by the caller; region-forbidden
            quantitative arms remain valid here (their quantity space is restricted later, at selection time).

        Returns
        -------
        action_probabilities: Union[List[Dict[UnifiedActionId, Probability]], List[Dict[UnifiedActionId, ProbabilityWeight]], List[Dict[UnifiedActionId, MOProbability]], List[Dict[UnifiedActionId, MOProbabilityWeight]]]
            The probability of getting a positive reward for each action.
        """

        action_probabilities = self.actions_manager.sample_proba(
            rng=self._rng, valid_action_ids=valid_actions, **kwargs
        )
        # Handle standard actions for which the value is a (probability, weight) tuple
        actions_transformations = [[{key: proba} for proba in value] for key, value in action_probabilities.items()]
        action_probabilities = self._transform_nested_list(actions_transformations)

        return action_probabilities

    @abstractmethod
    @validate_call
    def predict(self, forbidden_actions: Optional[ForbiddenActions] = None) -> Predictions:
        """
        Predict actions.

        Parameters
        ----------
        forbidden_actions : Optional[ForbiddenActions], default=None
            Actions to forbid. Either a ``Set[ActionId]`` of wholly-forbidden arms, or a
            ``Dict[ActionId, None | ForbiddenRegion | List[ForbiddenRegion]]`` where ``None`` forbids the whole arm
            and region callable(s) forbid part of a quantitative arm's quantity space (``region(x) > 0`` => forbidden).
            By default, the model considers all actions as allowed_actions.
            Note that: actions = allowed_actions U forbidden_actions.

        Returns
        -------
        actions: List[ActionId] of shape (n_samples,)
            The actions selected by the multi-armed bandit model.
        probs: List[Dict[ActionId, Probability]] of shape (n_samples,)
            The probabilities of getting a positive reward for each action
        ws : List[Dict[ActionId, float]], only relevant for some of the MABs
            The weighted sum of logistic regression logits..
        """

    def get_state(self) -> (str, str):
        """
        Access the complete model internal state, enough to create an exact copy of the same model from it.
        Returns
        -------
        model_class_name: str
            The name of the class of the model.
        model_state: dict
            The internal state of the model (actions, scores, etc.).
        """
        model_name = self.__class__.__name__
        state = self.model_dump_json()
        return model_name, state

    def _sample_allowed_quantity(
        self,
        action_id: ActionId,
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]],
        max_tries: int = 100,
    ) -> Optional[Tuple[float, ...]]:
        """
        Uniformly sample a quantity in ``[0, 1]^d`` that lies outside the arm's forbidden regions.

        Used by the epsilon-greedy explore branch so a random quantitative action never lands in a
        forbidden region. Sampling is by rejection; if no allowed point is found within ``max_tries``,
        ``None`` is returned and the caller drops the arm from the explore candidate set.

        Parameters
        ----------
        action_id: ActionId
            The quantitative arm to sample a quantity for.
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]]
            Per-arm feasibility constraints (``>= 0`` feasible); a point is forbidden where any constraint is ``< 0``.
        max_tries: int, default=100
            Maximum number of rejection-sampling attempts before giving up.

        Returns
        -------
        Optional[Tuple[float, ...]]
            An allowed quantity vector, or ``None`` if none was found within ``max_tries``.
        """
        dimension = self.actions[action_id].dimension
        constraints = forbidden_regions.get(action_id) if forbidden_regions else None
        for _ in range(max_tries):
            candidate = self._rng.random(dimension)
            if constraints is None or all(constraint(candidate) >= 0 for constraint in constraints):
                return tuple(candidate)
        return None

    @validate_call
    def _select_epsilon_greedy_action(
        self,
        p: ActionRewardLikelihood,
        actions: Optional[Dict[ActionId, BaseModel]] = None,
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]] = None,
    ) -> ActionId:
        """
        Wraps self.strategy.select_action function with epsilon-greedy strategy,
        such that with probability epsilon a default_action is selected,
        and with probability 1-epsilon the select_action function is triggered to choose action.
        If no default_action is provided, a random action is selected.
        When both default_action and default_action_fraction are provided, the explore branch
        picks default_action with probability default_action_fraction and a random action
        with probability 1 - default_action_fraction.

        References
        ----------
        Reinforcement Learning: An Introduction, Ch. 2 (Sutton and Burto, 2018)
        https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf&ved=2ahUKEwjMy8WV9N2HAxVe0gIHHVjjG5sQFnoECEMQAQ&usg=AOvVaw3bKK-Y_1kf6XQVwR-UYrBY

        Parameters
        ----------
        p: Union[Dict[ActionId, float], Dict[ActionId, Probability], Dict[ActionId, List[Probability]]]
            The dictionary or actions and their sampled probability of getting a positive reward.
            For MO strategy, the sampled probability is a list with elements corresponding to the objectives.
        actions: Optional[Dict[ActionId, Model]]
            The dictionary of actions and their associated Model.
        forbidden_regions: Optional[Dict[ActionId, List[Callable[[np.ndarray], float]]]]
            Per-arm feasibility constraints restricting quantitative arms' quantity space. Honored by both the
            strategy (exploit) and the random explore branch, so a forbidden region is never selected by any path.

        Returns
        -------
        selected_action: ActionId
            The selected action.

        Raises
        ------
        KeyError
            If self.default_action is not present as a key in the probabilities dictionary.
        """

        if self.limited_actions and not self._rng.binomial(1, self.limited_action_fraction):
            masked = {k: v for k, v in p.items() if (k[0] if isinstance(k, tuple) else k) not in self.limited_actions}
            if masked:  # never mask away the entire action set
                p = masked

        if self.epsilon:
            if self.default_action:
                if isinstance(self.default_action, tuple):
                    # For quantitative models, check if any key has the same action_id
                    default_action_id = self.default_action[0]
                    if not any(
                        (isinstance(key, tuple) and key[0] == default_action_id) or key == default_action_id
                        for key in p.keys()
                    ):
                        raise KeyError(f"Default action {self.default_action} not in actions.")
                elif self.default_action not in p.keys():
                    raise KeyError(f"Default action {self.default_action} not in actions.")
            if self._rng.binomial(1, self.epsilon):
                # Decide whether to use the default action (if any) or a random action.
                # When default_action_fraction is set, it acts as the probability of picking
                # default_action vs a uniformly-random action on the explore branch.
                use_default = self.default_action and (
                    self.default_action_fraction is None or self._rng.binomial(1, self.default_action_fraction)
                )
                if use_default:
                    selected_action = self.default_action
                else:
                    # Randomly pick an arm; for a quantitative arm, rejection-sample a quantity outside its
                    # forbidden regions. An arm whose quantity space is fully forbidden is dropped and re-picked.
                    _keys = list(p.keys())
                    selected_action = None
                    while _keys:
                        idx = self._rng.integers(len(_keys))
                        candidate_action = _keys[idx]
                        if isinstance(self.actions[candidate_action], QuantitativeModel):
                            quantity = self._sample_allowed_quantity(candidate_action, forbidden_regions)
                            if quantity is None:
                                _keys.pop(idx)
                                continue
                            selected_action = (candidate_action, quantity)
                        else:
                            selected_action = candidate_action
                        break
                    if selected_action is None:
                        # Every candidate quantitative arm is fully region-forbidden; defer to the strategy.
                        selected_action = self.strategy.select_action(
                            p=p, actions=actions, forbidden_regions=forbidden_regions, rng=self._rng
                        )
            else:
                selected_action = self.strategy.select_action(
                    p=p, actions=actions, forbidden_regions=forbidden_regions, rng=self._rng
                )
        else:
            selected_action = self.strategy.select_action(
                p=p, actions=actions, forbidden_regions=forbidden_regions, rng=self._rng
            )
        return selected_action

    @classmethod
    def from_state(cls, state: str) -> "BaseMab":
        """
        Create a new instance of the class from a given model state.
        The state can be obtained by applying get_state() to a model.

        Parameters
        ----------
        state: dict
            The internal state of a model (actions, strategy, etc.) of the same type.

        Returns
        -------
        model: BaseMab
            The new model instance.

        """
        return cls.model_validate_json(state)

    @classmethod
    def update_old_state(cls, state: Dict[str, Serializable]) -> Dict[str, Serializable]:
        """
        Update the model state to the current version if needed.

        Parameters
        ----------
        state : str
            The internal state of a model (actions, strategy, etc.) of the same type.
            The state is expected to be in the old format of PyBandits below the current supported version.

        Returns
        -------
        state : Dict[str, Serializable]
            The updated state of the model.
            The state is in the current format of PyBandits, with actions_manager and delta added if needed.
        """

        return state

    @classmethod
    def from_old_state(cls, state: str) -> "BaseMab":
        """
        Create a new instance of the class from previous versions of the model state.
        (The state can be obtained by applying get_state() to a model.)

        Parameters
        ----------
        state : str
            The internal state of a model (actions, strategy, etc.) of the same type.
            The state is expected to be in the old format of PyBandits below the current supported version (cls.current_supported_version_th).

        Returns
        -------
        model : BaseMab
            The new model instance.
        """

        state_dict = json.loads(state)
        if ("version" in state_dict) and (
            version.parse(state_dict["version"]) >= version.parse(cls._current_supported_version_th)
        ):
            raise ValueError(
                f"The state is expected to be in the old format of PyBandits < {cls._current_supported_version_th}."
            )
        state_dict = cls.update_old_state(state_dict)
        state = json.dumps(state_dict)
        return cls.from_state(state)

    @classmethod
    def _get_class_type_attributes(cls) -> List[str]:
        return [
            attribute_name
            for attribute_name in cls.model_fields.keys()
            if isclass(class_ := cls._get_attribute_type(attribute_name))
            and issubclass(
                class_,
                PyBanditsBaseModel,
            )
        ]

    @classmethod
    def _get_attribute_type(cls, attribute_name: str) -> PyBanditsBaseModel:
        attribute_type = cls._get_field_type(attribute_name)
        return get_origin(attribute_type) or attribute_type

    @classmethod
    @validate_call
    def cold_start(
        cls,
        epsilon: Optional[Float01] = None,
        default_action: Optional[UnifiedActionId] = None,
        default_action_fraction: Optional[PositiveFloat01] = None,
        limited_actions: Optional[Set[ActionId]] = None,
        limited_action_fraction: Optional[Float01] = None,
        random_seed: Optional[NonNegativeInt] = None,
        **kwargs,
    ) -> Self:
        """
        Factory method to create a Multi-Armed Bandit with Thompson Sampling, with default
        parameters.

        Parameters
        ----------
        epsilon : Optional[Float01]
            epsilon for epsilon-greedy approach. If None, epsilon-greedy is not used.
        default_action : Optional[ActionId]
            The default action to select with a probability of epsilon when using the epsilon-greedy approach.
            If `default_action` is None, a random action from the action set will be selected with a probability of epsilon.
        default_action_fraction : Optional[PositiveFloat01]
            Probability of picking `default_action` (vs a uniformly-random action) when the explore
            branch of epsilon-greedy fires. Requires both `epsilon` and `default_action` to be set.
            `1.0` always returns `default_action`; `None` (default) preserves legacy behavior.
        limited_actions : Optional[Set[ActionId]]
            Actions whose selection is throttled (e.g. newly-introduced arms). On each selection they
            are allowed to compete only with probability `limited_action_fraction`. Requires
            `limited_action_fraction`, must be a subset of the action set, and must not contain
            `default_action`.
        limited_action_fraction : Optional[Float01]
            Probability that `limited_actions` are allowed to compete on a given selection (higher =
            more exploration). Requires `limited_actions`.
        random_seed : Optional[NonNegativeInt]
            Seed for the MAB's central numpy RNG (used for epsilon-greedy and Thompson sampling).
            Propagated automatically to BNN action models so the full pipeline is reproducible.
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model, e.g. ``decay_factor`` for
            per-update forgetting in the action models.

        Returns
        -------
        mab: BaseMab
            Multi-Armed Bandit
        """
        # Instantiate the MAB
        mab = cls(
            epsilon=epsilon,
            default_action=default_action,
            default_action_fraction=default_action_fraction,
            limited_actions=limited_actions,
            limited_action_fraction=limited_action_fraction,
            random_seed=random_seed,
            **kwargs,
        )
        return mab
