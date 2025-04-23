import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from inspect import isclass
from typing import Any, Callable, ClassVar, Dict, Generic, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from pybandits.base import ACTION_IDS_PREFIX, ACTIONS, ActionId, BinaryReward, PositiveProbability, PyBanditsBaseModel
from pybandits.model import (
    BaseModel,
    BayesianLogisticRegression,
    BayesianLogisticRegressionCC,
    Beta,
    BetaCC,
    BetaMO,
    BetaMOCC,
    CmabModelType,
    Model,
    ModelMO,
    SmabModelType,
)
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    GenericModel,
    NonNegativeInt,
    NonPositiveInt,
    field_validator,
    pydantic_version,
    validate_call,
)
from pybandits.utils import extract_argument_names_from_function


class ActionsManager(PyBanditsBaseModel, ABC):
    """
    Base class for managing actions and their associated models.
    The class allows to account for non-stationarity by providing an adaptive window scheme for action update.
    The change point detection is based on the adaptive windowing scheme.

    References
    ----------
    Scaling Multi-Armed Bandit Algorithms (Fouché et al., 2019)
    https://edouardfouche.com/publications/S-MAB_FOUCHE_KDD19.pdf

    Parameters
    ----------
    actions : Dict[ActionId, Model]
        The list of possible actions, and their associated Model.
    delta : Optional[PositiveProbability]
        The confidence level for the adaptive window. None for skipping the change point detection.
    """

    actions: Dict[ActionId, BaseModel]
    delta: Optional[PositiveProbability] = None
    _no_change_point: ClassVar[NonNegativeInt] = -1
    _min_adaptive_window_size: ClassVar[NonPositiveInt] = 10000
    _memory_parameters_suffix: ClassVar[str] = "_memory"

    if pydantic_version == PYDANTIC_VERSION_1:

        class Config:
            arbitrary_types_allowed = True
            json_encoders = {deque: list}

    elif pydantic_version == PYDANTIC_VERSION_2:
        model_config = {"arbitrary_types_allowed": True, "json_encoders": {deque: list}}
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @field_validator("actions", mode="after")
    @classmethod
    def at_least_one_action_is_defined(cls, v):
        # validate number of actions
        if len(v) == 0:
            raise AttributeError("At least one action should be defined.")
        elif len(v) == 1:
            warnings.warn("Only a single action was supplied. This MAB will be deterministic.")
        # validate that all actions are of the same configuration
        action_models = list(v.values())
        action_type = cls._get_field_type("actions")
        if any(not isinstance(action, action_type) for action in action_models):
            raise TypeError(f"All actions should follow {action_type} type.")
        return v

    @classmethod
    def _get_expected_memory_length(cls, actions: Dict[ActionId, BaseModel]) -> NonNegativeInt:
        """
        Get the expected memory length for the adaptive window.

        Parameters
        ----------
        actions : Dict[ActionId, BaseModel]
            The list of possible actions, and their associated Model.

        Returns
        -------
        NonNegativeInt
            The expected memory length.
        """
        if not actions:
            raise AttributeError("At least one action should be defined.")
        reference_model = list(actions.values())[0]
        if isinstance(reference_model, Model):
            expected_memory_length_for_inf = sum(
                [action_model.n_successes + action_model.n_failures - 2 for action_model in actions.values()]
            )
        elif isinstance(reference_model, ModelMO):
            expected_memory_length_for_inf = sum(
                [
                    action_model.models[0].n_successes + action_model.models[0].n_failures - 2
                    for action_model in actions.values()
                ]
            )
        else:
            raise ValueError(f"Model type {type(reference_model)} not supported.")
        return expected_memory_length_for_inf

    def __init__(
        self,
        delta: Optional[PositiveProbability] = None,
        actions: Optional[Dict[ActionId, Model]] = None,
        action_ids: Optional[Set[ActionId]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        actions = self._instantiate_actions(actions=actions, action_ids=action_ids, kwargs=kwargs)
        super().__init__(actions=actions, delta=delta)

    def _validate_update_params(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], **kwargs
    ):
        """
        Verify that the given list of action IDs is a subset of the currently defined actions and that
         the rewards type matches the strategy type.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """
        invalid = set(actions) - set(self.actions.keys())
        if invalid:
            raise AttributeError(f"The following invalid action(s) were specified: {invalid}.")
        self._validate_params_lengths(actions=actions, rewards=rewards, **kwargs)

    @classmethod
    def _validate_matching_keys(cls, update_kwargs: Dict[str, Any], memory_kwargs: Dict[str, Any]):
        """
        Verify that the keys in the update kwargs and memory kwargs match.

        Parameters
        ----------
        update_kwargs : Dict[str, Any]
            The update kwargs.
        memory_kwargs : Dict[str, Any]
            The memory kwargs.
        """
        stripped_memory_keys = {k.split(cls._memory_parameters_suffix)[0] for k in memory_kwargs.keys()}
        update_keys = set(update_kwargs.keys())
        if stripped_memory_keys != update_keys:
            raise AttributeError(f"Update and memory kwargs should have the same keys: {update_keys}.")

    @classmethod
    def _to_memory_key(cls, key: str) -> str:
        return f"{key}{cls._memory_parameters_suffix}"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        **kwargs,
    ):
        """
        Update the models associated with the given actions using the provided rewards.
        For adaptive window size, the update by resetting the action models and retraining them on the new data.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        if self.delta is None and (actions_memory or rewards_memory):
            raise AttributeError("Adaptive window size is not set, so memory should not be provided.")
        if self.delta is not None and (actions_memory is None or rewards_memory is None):
            warnings.warn("Adaptive window size is set, but memory was not provided.")
            actions_memory = []
            rewards_memory = []

        update_kwargs = {k: v for k, v in kwargs.items() if not k.endswith(self._memory_parameters_suffix)}
        memory_kwargs = {k: v for k, v in kwargs.items() if k.endswith(self._memory_parameters_suffix)}
        self._validate_update_params(actions, rewards, **update_kwargs)
        self._validate_params_lengths(actions_memory=actions_memory, rewards_memory=rewards_memory, **memory_kwargs)
        update_keys = tuple(update_kwargs.keys())

        if actions_memory is not None:
            actions_memory, rewards_memory, memory_kwargs = self._maybe_trim_memory(
                actions_memory, rewards_memory, memory_kwargs
            )
            residual_memory_len = len(actions_memory)
            if residual_memory_len < self._min_adaptive_window_size:
                warnings.warn("The adaptive window size too small value. Consider increasing it for better results.")
            actions_memory.extend(actions)
            rewards_memory.extend(rewards)
            for key in update_keys:
                memory_key = self._to_memory_key(key)
                if isinstance(update_kwargs[key], list):
                    memory_kwargs[memory_key].extend(update_kwargs[key])
                elif isinstance(update_kwargs[key], np.ndarray):
                    memory_kwargs[memory_key] = (
                        np.concatenate(
                            (memory_kwargs[f"{key}{self._memory_parameters_suffix}"], update_kwargs[key]), axis=0
                        )
                        if memory_kwargs[f"{key}{self._memory_parameters_suffix}"] is not None
                        else update_kwargs[key]
                    )

            if (
                last_change_point := self._get_last_change_point(residual_memory_len, actions_memory, rewards_memory)
            ) != self._no_change_point:
                actions_memory, rewards_memory, memory_kwargs = self._slice_memory(
                    len(actions_memory) - last_change_point, actions_memory, rewards_memory, memory_kwargs
                )

                for action_model in self.actions.values():
                    action_model.reset()
                stripped_memory_kwargs = {
                    k[: -len(self._memory_parameters_suffix)]: v for k, v in memory_kwargs.items()
                }
                self._update_actions(actions_memory, rewards_memory, **stripped_memory_kwargs)
            else:
                self._update_actions(actions, rewards, **update_kwargs)
        else:
            self._update_actions(actions, rewards, **update_kwargs)

    @staticmethod
    def _slice_memory(
        memory_len: NonNegativeInt,
        actions_memory: List[ActionId],
        rewards_memory: List[BinaryReward],
        memory_kwargs: Dict[str, Any],
    ) -> Tuple[List[ActionId], List[BinaryReward], Dict[str, Any]]:
        """
        Slice all memory parameters to memory_len length.

        Parameters
        ----------
        memory_len : NonNegativeInt
            Expected memory length after the slicing.
        actions_memory : List[ActionId]
            List of previously selected actions.
        rewards_memory : List[BinaryReward]
            List of previously collected rewards.
        memory_kwargs : Dict[str, Any]
            The memory kwargs.

        Returns
        -------
        actions_memory : List[ActionId]
            List of previously selected actions with maximum length of memory_len.
        rewards_memory : List[BinaryReward]
            List of previously collected rewards with maximum length of memory_len.
        memory_kwargs : Dict[str, Any]
            The memory kwargs with values of maximum length of memory_len.
        """
        if len(actions_memory) > memory_len:
            actions_memory = actions_memory[-memory_len:]
            rewards_memory = rewards_memory[-memory_len:]
            for memory_key, memory_value in memory_kwargs.items():
                if memory_value is not None:
                    memory_kwargs[memory_key] = memory_value[-memory_len:]
        return actions_memory, rewards_memory, memory_kwargs

    def _maybe_trim_memory(
        self,
        actions_memory: List[ActionId],
        rewards_memory: Union[List[BinaryReward], List[List[BinaryReward]]],
        memory_kwargs: Dict[str, Any],
    ) -> Tuple[List[ActionId], List[BinaryReward], Dict[str, Any]]:
        """
        Trim the memory to the adaptive window size.

        Parameters
        ----------
        actions_memory : List[ActionId]
            List of previously selected actions.
        rewards_memory : Union[List[BinaryReward], List[List[BinaryReward]]]
            List of previously collected rewards.
        memory_kwargs : Dict[str, Any]
            The memory kwargs.

        Returns
        -------
        actions_memory : List[ActionId]
            List of previously selected actions with maximum length of memory_len.
        rewards_memory : List[BinaryReward]
            List of previously collected rewards with maximum length of memory_len.
        memory_kwargs : Dict[str, Any]
            The memory kwargs with values of maximum length of memory_len.
        """
        action_stats = self._action_stats
        maximum_memory_length = self._get_memory_len_from_action_stats(action_stats)
        if len(actions_memory) > maximum_memory_length:
            warnings.warn(f"Input memory is longer then expected. Leaving only last {maximum_memory_length} elements.")
            actions_memory, rewards_memory, memory_kwargs = self._slice_memory(
                maximum_memory_length, actions_memory, rewards_memory, memory_kwargs
            )
        for action_id, (expected_successes, expected_trials) in action_stats.items():
            actual_trials = np.sum([1 for a in actions_memory if a == action_id])
            actual_successes = np.sum(
                np.array([r for a, r in zip(actions_memory, rewards_memory) if a == action_id]).reshape(
                    (-1, expected_successes.shape[1])
                ),
                axis=0,
                keepdims=True,
            )

            if np.any(actual_trials > expected_trials):
                raise ValueError(f"Memory for action {action_id} is larger than expected.")
            elif actual_trials == expected_trials[0][0]:
                if not np.array_equal(actual_successes, expected_successes):
                    raise ValueError(f"Memory for action {action_id} is not consistent with the expected stats.")
            else:
                if np.any(actual_successes > expected_successes):
                    raise ValueError(f"Memory for action {action_id} is not consistent with the expected stats.")

        return actions_memory, rewards_memory, memory_kwargs

    def _get_memory_len_from_action_stats(
        self, action_stats: Dict[ActionId, Tuple[ArrayLike, ArrayLike]]
    ) -> NonNegativeInt:
        """
        Calculate total memory length from action statistics.

        Parameters
        ----------
        action_stats : Dict[ActionId, Tuple[ArrayLike, ArrayLike]]
            Dictionary mapping action IDs to tuples of (successes, trials) arrays.

        Returns
        -------
        NonNegativeInt
            Total number of trials across all actions.
        """

        return sum([v[1][0][0] for v in action_stats.values()])

    @property
    def _action_stats(self) -> Dict[ActionId, Tuple[np.ndarray, np.ndarray]]:
        """
        Get current statistics for all actions.

        Returns
        -------
        action_stats : Dict[ActionId, Tuple[np.ndarray, np.ndarray]]
            Dictionary mapping action IDs to tuples of (successes, trials) arrays.
        """
        action_stats = {action_id: self._extract_current_stats_for_action(action_id) for action_id in self.actions}
        return action_stats

    @property
    def maximum_memory_length(self) -> NonNegativeInt:
        """
        Get maximum possible memory length based on current action statistics.

        Returns
        -------
        NonNegativeInt
            Maximum memory length allowed.
        """
        return self._get_memory_len_from_action_stats(self._action_stats)

    @abstractmethod
    def _update_actions(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], **kwargs
    ):
        """
        Update the models associated with the given actions using the provided rewards.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """
        pass

    def _get_last_change_point(
        self,
        residual_memory_len: NonNegativeInt,
        actions_memory: List[ActionId],
        rewards_memory: Union[List[BinaryReward], List[List[BinaryReward]]],
    ) -> NonNegativeInt:
        """
        Get the last change point among all actions.

        Parameters
        ----------
        residual_memory_len : NonNegativeInt
            The length of the residual memory.
        actions_memory : List[ActionId]
            List of previously selected actions.
        rewards_memory : List[BinaryReward]
            List of previously collected rewards.

        Returns
        -------
        NonNegativeInt
            The last change point. 0 if no change point is found.
        """
        change_points = [
            self._get_last_change_point_for_action(
                action_id=action_id,
                residual_memory_len=residual_memory_len,
                actions_memory=actions_memory,
                rewards_memory=rewards_memory,
            )
            for action_id in self.actions.keys()
        ]

        return max(change_points)

    def _get_threshold(self, past_trials: np.ndarray, present_trials: np.ndarray) -> np.ndarray:
        """
        Get the threshold for the given past window and present window.

        Parameters
        ----------
        past_trials : np.ndarray
            The number of trials in the past window.
        present_trials : np.ndarray
            The number of trials in the present window.

        Returns
        -------
        threshold : np.ndarray
            The threshold value.
        """
        full_trials = past_trials + present_trials
        harmonic_sum = 1 / past_trials + 1 / present_trials
        threshold = past_trials * present_trials * np.sqrt((harmonic_sum / 2) * np.log(4 * full_trials / self.delta))
        return threshold

    def _get_last_change_point_for_action(
        self,
        action_id: ActionId,
        residual_memory_len: NonNegativeInt,
        actions_memory: List[ActionId],
        rewards_memory: Union[List[BinaryReward], List[List[BinaryReward]]],
    ) -> int:
        """
        Get the last change point for the given action.

        Parameters
        ----------
        action_id : ActionId
            The action ID.
        actions_memory : List[ActionId]
            List of previously selected actions.
        rewards_memory : List[BinaryReward]
            List of previously collected rewards.

        Returns
        -------
        NonNegativeInt
            The last change point for the given action. -1 if no change point is found.
        """
        action_index = np.nonzero([a == action_id for a in actions_memory])[0].tolist()

        rewards_window = [rewards_memory[i] for i in action_index]
        window_length = len(rewards_window)
        if window_length < 2:
            return self._no_change_point
        cumulative_reward = np.cumsum(np.array(rewards_window), axis=0)
        if cumulative_reward.ndim == 1:
            cumulative_reward = cumulative_reward[:, np.newaxis]

        current_sum, current_trials = self._extract_current_stats_for_action(action_id)

        # n_successes and n_failures already take into account the statistics of remaining elements from last
        # memory update, so their statistics are removed for consistency.
        if residual_memory_len:
            projected_residual_memory_len = len([index for index in action_index if index < residual_memory_len])
            current_sum -= cumulative_reward[projected_residual_memory_len - 1]
            current_trials -= projected_residual_memory_len
        initial_start_index = 0 if np.sum(current_trials) else 1

        base_range = np.arange(initial_start_index, window_length).reshape(-1, 1)
        past_sums = np.concatenate((current_sum, current_sum + cumulative_reward[:-1]))
        present_sums = cumulative_reward[-1] - np.concatenate(
            (np.zeros((1, cumulative_reward.shape[-1])), cumulative_reward[:-1])
        )
        start_index = initial_start_index
        while start_index < window_length:
            if initial_start_index == 0 and start_index == 1:  # After first iteration, dismiss the current memory
                past_sums -= current_sum
                current_trials = np.zeros_like(current_trials)

            relevant_range = base_range[(start_index - initial_start_index) :]

            past_trials = relevant_range + current_trials
            present_trials = window_length - relevant_range

            thresholds = self._get_threshold(past_trials, present_trials)
            change_points = np.nonzero(
                np.any(
                    np.abs(past_sums[start_index:] * present_trials - present_sums[start_index:] * past_trials)
                    > thresholds,
                    axis=1,
                )
            )[0]

            if not change_points.size:
                break
            start_index += 1

        if start_index == initial_start_index:
            return self._no_change_point

        return action_index[min(start_index, window_length - 1)]

    def _extract_current_stats_for_action(self, action_id: ActionId) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the current statistics for the given action.
        The statistics include the number of successes and the number of trials for each action.
        Since `n_successes` and `n_failures` are initialized as 1 in Model class,
        we reduce 1 from `n_successes` to get the actual number of successes.
        Similarly, we reduce 2 from `count` to get the actual number of trials.

        Parameters
        ----------
        action_id : ActionId
            The action ID.

        Returns
        -------
        current_sum : np.ndarray
            The number of successes for the given action for each of the objectives.
        current_trials : np.ndarray
            The number of trials for the given action for each of the objectives.
        """
        action_model = self.actions[action_id]
        if isinstance(action_model, Model):
            current_sum = np.array([action_model.n_successes - 1]).reshape((1, -1))
            current_trials = np.array([action_model.count - 2]).reshape((1, -1))

        elif isinstance(action_model, ModelMO):
            current_sum = np.array([model.n_successes - 1 for model in action_model.models]).reshape((1, -1))
            current_trials = np.array([model.count - 2 for model in action_model.models]).reshape((1, -1))
        else:
            raise TypeError(f"Model type {type(action_model)} not supported.")
        return current_sum, current_trials

    @classmethod
    def _instantiate_actions(
        cls, actions: Optional[Dict[ActionId, Model]], action_ids: Optional[Set[ActionId]], kwargs
    ):
        """
        Utility function to instantiate the action models based on the provided kwargs.

        Parameters
        ----------
        actions : Optional[Dict[ActionId, Model]]
            The list of possible actions and their associated models.
        action_ids : Optional[Set[ActionId]]
            The list of possible actions.
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        actions : Dict[ActionId, Model]
            Dictionary of actions and the parameters of their associated model.
        """
        if actions is None:
            action_specific_kwargs = cls._extract_action_specific_kwargs(kwargs)

            # Extract inner_action_ids
            inner_action_ids = action_ids or set(action_specific_kwargs.keys())
            if not inner_action_ids:
                raise ValueError(
                    "inner_action_ids should be provided either directly or via keyword argument in the form of "
                    "action_id_{model argument name} = {action_id: value}."
                )
            action_model_start = cls._get_action_model_start_method(True)
            action_general_kwargs = cls._extract_action_model_class_and_attributes(kwargs, action_model_start)
            actions = {}
            for a in inner_action_ids:
                actions[a] = action_model_start(**action_general_kwargs, **action_specific_kwargs.get(a, {}))

        if all(isinstance(potential_model, Dict) for potential_model in actions.values()):
            action_model_start = cls._get_action_model_start_method(False)
            state_actions = actions.copy()
            actions = {}
            for action_id, action_state in state_actions.items():
                actions[action_id] = action_model_start(**action_state)

        return actions

    @staticmethod
    def _extract_action_specific_kwargs(kwargs) -> Dict[ActionId, Dict]:
        """
        Utility function to extract kwargs that are specific for each action when constructing the action model.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.

        Returns
        -------
        action_specific_kwargs : Dict[str, Dict]
            Dictionary of actions and the parameters of their associated model.
        kwargs : Dict[str, Any]
            Dictionary of parameters and their values, without the action_specific_kwargs.
        """
        action_specific_kwargs = defaultdict(dict)
        for keyword in list(kwargs.keys()):
            argument = kwargs[keyword]
            if keyword.startswith(ACTION_IDS_PREFIX) and type(argument) is dict:
                kwargs.pop(keyword)
                inner_keyword = keyword.split(ACTION_IDS_PREFIX)[1]
                for action_id, value in argument.items():
                    action_specific_kwargs[action_id][inner_keyword] = value
            if keyword == ACTIONS and type(argument) is dict:
                kwargs.pop(keyword)
                action_specific_kwargs.update(argument)
        return dict(action_specific_kwargs)

    @classmethod
    def _extract_action_model_class_and_attributes(
        cls, kwargs: Dict[str, Any], action_model_start: Callable
    ) -> Dict[str, Dict]:
        """
        Utility function to extract kwargs that are specific for each action when constructing the action model.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Additional parameters for the mab and for the action model.
        action_model_start : Callable
            Function handle for the action model start: either cold start or init.

        Returns
        -------
        action_model_cold_start : Callable
            Function handle for factoring the required action model.
        action_general_kwargs : Dict[str, any]
            Dictionary of parameters and their values for the action model.
        """
        if isclass(action_model_start):
            action_model_attributes = list(action_model_start.model_fields.keys())
        else:
            action_model_attributes = extract_argument_names_from_function(action_model_start, True)

        action_general_kwargs = {k: kwargs.pop(k) for k in action_model_attributes if k in kwargs.keys()}
        return action_general_kwargs

    @classmethod
    def _get_action_model_start_method(cls, cold_start_mode: bool) -> Callable:
        action_model_class = cls._get_field_type("actions")
        if cold_start_mode and hasattr(action_model_class, "cold_start"):
            action_model_start = action_model_class.cold_start
        else:
            action_model_start = action_model_class
        return action_model_start


class SmabActionsManager(ActionsManager, GenericModel, Generic[SmabModelType]):
    """
    Manages actions and their associated models for sMAB models.
    The class allows to account for non-stationarity by providing an adaptive window scheme for action update.

    Parameters
    ----------
    actions : Dict[ActionId, BaseBeta]
        The list of possible actions, and their associated Model.
    delta : Optional[PositiveProbability], 0.1 if not specified.
        The confidence level for the adaptive window.
    """

    actions: Dict[ActionId, SmabModelType]

    @field_validator("actions", mode="after")
    @classmethod
    def all_actions_have_same_number_of_objectives(cls, actions: Dict[ActionId, SmabModelType]):
        n_objs_per_action = [len(beta.models) if hasattr(beta, "models") else None for beta in actions.values()]
        if len(set(n_objs_per_action)) != 1:
            raise ValueError("All actions should have the same number of objectives")
        return actions

    @validate_call
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
    ):
        """
        Update the models associated with the given actions using the provided rewards.
        For adaptive window size, the update by resetting the action models and retraining them on the new data.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        super().update(actions, rewards, actions_memory, rewards_memory)

    def _update_actions(self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]]):
        """
        Update the models associated with the given actions using the provided rewards.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """

        rewards_dict = defaultdict(list)

        for a, r in zip(actions, rewards):
            rewards_dict[a].append(r)

        for a in set(actions):
            self.actions[a].update(rewards=rewards_dict[a])


class CmabActionsManager(ActionsManager, GenericModel, Generic[CmabModelType]):
    """
    Manages actions and their associated models for cMAB models.
    The class allows to account for non-stationarity by providing an adaptive window scheme for action update.

    Parameters
    ----------
    actions : Dict[ActionId, BayesianLogisticRegression]
        The list of possible actions, and their associated Model.
    delta : Optional[PositiveProbability], 0.1 if not specified.
        The confidence level for the adaptive window.
    """

    actions: Dict[ActionId, CmabModelType]

    @field_validator("actions", mode="after")
    @classmethod
    def check_bayesian_logistic_regression_models(cls, v):
        action_models = list(v.values())
        first_action = action_models[0]
        first_action_type = type(first_action)
        for action in action_models[1:]:
            if not isinstance(action, first_action_type):
                raise TypeError("All actions should follow the same type.")
            if not len(action.betas) == len(first_action.betas):
                raise AttributeError("All actions should have the same number of betas.")
            if not action.update_method == first_action.update_method:
                raise AttributeError("All actions should have the same update method.")
            if not action.update_kwargs == first_action.update_kwargs:
                raise AttributeError("All actions should have the same update kwargs.")
        return v

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        context: ArrayLike,
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        context_memory: Optional[ArrayLike] = None,
    ):
        """
        Update the models associated with the given actions using the provided rewards.
        For adaptive window size, the update by resetting the action models and retraining them on the new data.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        context_memory : Optional[ArrayLike] of shape (n_samples, n_features)
            Matrix of contextual features.
        """

        context = self._check_context_matrix(context)
        if context_memory is not None:
            context_memory = self._check_context_matrix(context_memory)
            if context.shape[1] != context_memory.shape[1]:
                raise ValueError("Context memory must have the same number of features as the context.")
        super().update(actions, rewards, actions_memory, rewards_memory, context=context, context_memory=context_memory)

    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _check_context_matrix(context: ArrayLike):
        """
        Check and cast context matrix.

        Parameters
        ----------
        context : np.ndarray of shape (n_samples, n_features)
            Matrix of contextual features.

        Returns
        -------
        context : pandas DataFrame of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        try:
            context = np.asarray(context, dtype=float)
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike that can transform to float numpy array: {e}.")
        return context

    def _update_actions(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        context: np.ndarray,
    ):
        """
        Update the models associated with the given actions using the provided rewards.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        context: np.ndarray of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        # cast inputs to numpy arrays to facilitate their manipulation
        context = context[-len(actions) :]
        rewards_dict = defaultdict(list)
        context_dict = defaultdict(list)

        for a, r, c in zip(actions, rewards, context):
            rewards_dict[a].append(r)
            context_dict[a].append(c)

        for a in set(actions):
            self.actions[a].update(rewards=rewards_dict[a], context=np.array(context_dict[a]))


# For pickling purposes
SmabActionsManagerSO = SmabActionsManager[Beta]
SmabActionsManagerCC = SmabActionsManager[BetaCC]
SmabActionsManagerMO = SmabActionsManager[BetaMO]
SmabActionsManagerMOCC = SmabActionsManager[BetaMOCC]

CmabActionsManagerSO = CmabActionsManager[BayesianLogisticRegression]
CmabActionsManagerCC = CmabActionsManager[BayesianLogisticRegressionCC]
