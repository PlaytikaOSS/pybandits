import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from inspect import isclass
from typing import Any, Callable, Deque, Dict, Generic, List, Literal, Optional, Set, Union

import numpy as np
from numpy.typing import ArrayLike

from pybandits.base import ACTION_IDS_PREFIX, ACTIONS, ActionId, BinaryReward, PositiveProbability, PyBanditsBaseModel
from pybandits.model import (
    BaseModel,
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
    PositiveInt,
    field_validator,
    model_validator,
    pydantic_version,
    validate_call,
)
from pybandits.utils import extract_argument_names_from_function

_NO_CHANGE_POINT = -1


class ActionsManager(PyBanditsBaseModel, ABC):
    """
    Base class for managing actions and their associated models.
    The class allows to account for non-stationarity by providing an adaptive window scheme for action update.
    The change point detection is based on the adaptive windowing scheme.

    Reference: Scaling Multi-Armed Bandit Algorithms (Fouché et al., 2019)
               https://edouardfouche.com/publications/S-MAB_FOUCHE_KDD19.pdf

    Parameters
    ----------
    actions : Dict[ActionId, Model]
        The list of possible actions, and their associated Model.
    adaptive_window_size : Optional[Union[PositiveInt, Literal["inf"]]]
        The size of the adaptive window for action update. If None, no adaptive window is used.
    delta : Optional[PositiveProbability], 0.1 if not specified.
        The confidence level for the adaptive window.
    """

    actions: Dict[ActionId, BaseModel]
    adaptive_window_size: Optional[Union[PositiveInt, Literal["inf"]]] = None
    delta: Optional[PositiveProbability] = None

    actions_memory: Optional[Deque] = None
    rewards_memory: Optional[Deque] = None

    if pydantic_version == PYDANTIC_VERSION_1:

        class Config:
            arbitrary_types_allowed = True
            json_encoders = {deque: list}

    elif pydantic_version == PYDANTIC_VERSION_2:
        model_config = {"arbitrary_types_allowed": True, "json_encoders": {deque: list}}
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @field_validator("actions", mode="before")
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

    @field_validator("adaptive_window_size", mode="before")
    @classmethod
    def check_window_size(cls, v):
        if v is not None and isinstance(v, int) and v < 10000:
            warnings.warn(
                "The adaptive window size is set to a small value. Consider increasing it for better results."
            )
        return v

    if pydantic_version == PYDANTIC_VERSION_1:

        @model_validator(mode="before")
        @classmethod
        def check_delta(cls, values):
            delta = cls._get_value_with_default("delta", values)
            adaptive_window_size = cls._get_value_with_default("adaptive_window_size", values)
            if delta is not None and not adaptive_window_size:
                raise AttributeError("Delta should only be defined when adaptive_window_size is defined.")
            if adaptive_window_size and delta is None:
                values["delta"] = 0.1
            return values

        @model_validator(mode="before")
        @classmethod
        def maybe_initialize_memory(cls, values):
            reference_memory_len = None
            expected_memory_length_for_inf = cls._get_expected_memory_length(actions=values["actions"])
            for memory_name in ["actions_memory", "rewards_memory"]:
                if values["adaptive_window_size"] is None and values.get(memory_name, None) is not None:
                    raise AttributeError(f"{memory_name} should only be defined when adaptive_window_size is defined.")
                if values["adaptive_window_size"] is not None:
                    if memory_name not in values or values[memory_name] is None:
                        values[memory_name] = deque(maxlen=cls._get_max_len(values["adaptive_window_size"]))
                    else:
                        memory_len = len(values[memory_name])
                        if reference_memory_len is not None and memory_len != reference_memory_len:
                            raise AttributeError(f"{memory_name} should have the same length as the other memory.")
                        else:
                            reference_memory_len = memory_len
                        if values["adaptive_window_size"] is int:
                            if memory_len > values["adaptive_window_size"]:
                                raise AttributeError(
                                    f"{memory_name} should have a length less than or equal to adaptive_window_size."
                                )
                        else:  # adaptive_window_size == "inf"
                            if memory_len > expected_memory_length_for_inf:
                                raise AttributeError(
                                    f"{memory_name} should have a length less than or equal to the expected memory length."
                                )
                        if isinstance(values[memory_name], list):  # serialization from json
                            maxlen = cls._get_max_len(values["adaptive_window_size"])
                            values[memory_name] = deque(values[memory_name], maxlen=maxlen)

            return values

    elif pydantic_version == PYDANTIC_VERSION_2:

        @model_validator(mode="after")
        def check_delta(self):
            if self.delta is not None and not self.adaptive_window_size:
                raise AttributeError("Delta should only be defined when adaptive_window_size is defined.")
            if self.adaptive_window_size and self.delta is None:
                self.delta = 0.1
            return self

        @model_validator(mode="after")
        def maybe_initialize_memory(self):
            reference_memory_len = None
            expected_memory_length_for_inf = self._get_expected_memory_length(actions=self.actions)
            for memory_name in ["actions_memory", "rewards_memory"]:
                if self.adaptive_window_size is None and getattr(self, memory_name, None) is not None:
                    raise AttributeError(f"{memory_name} should only be defined when adaptive_window_size is defined.")
                if self.adaptive_window_size is not None:
                    if not hasattr(self, memory_name) or getattr(self, memory_name) is None:
                        setattr(
                            self,
                            memory_name,
                            deque(maxlen=self.max_len),
                        )
                    else:
                        if reference_memory_len is not None and len(getattr(self, memory_name)) != reference_memory_len:
                            raise AttributeError(f"{memory_name} should have the same length as the other memory.")
                        else:
                            reference_memory_len = len(getattr(self, memory_name))
                        if self.adaptive_window_size is int:
                            if len(getattr(self, memory_name)) > self.adaptive_window_size:
                                raise AttributeError(
                                    f"{memory_name} should have a length less than or equal to adaptive_window_size."
                                )
                        else:  # adaptive_window_size == "inf"
                            if len(getattr(self, memory_name)) > expected_memory_length_for_inf:
                                raise AttributeError(
                                    f"{memory_name} should have a length less than or equal to the expected memory length."
                                )
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @classmethod
    def _get_max_len(cls, adaptive_window_size: Union[PositiveInt, Literal["inf"]]) -> Optional[PositiveInt]:
        """
        Get the maximum length for the memory.

        Parameters
        ----------
        adaptive_window_size : Union[PositiveInt, Literal["inf"]]
            The size of the adaptive window for action update.

        Returns
        -------
        Optional[PositiveInt]
            The maximum length for the memory.
        """
        if adaptive_window_size == "inf":
            return None
        return adaptive_window_size

    @property
    def max_len(self) -> Optional[PositiveInt]:
        """
        Get the maximum length for the memory.

        Returns
        -------
        Optional[PositiveInt]
            The maximum length for the memory.
        """
        return self._get_max_len(self.adaptive_window_size)

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
        adaptive_window_size: Optional[Union[PositiveInt, Literal["inf"]]] = None,
        delta: Optional[PositiveProbability] = None,
        actions: Optional[Dict[ActionId, Model]] = None,
        action_ids: Optional[Set[ActionId]] = None,
        actions_memory: Optional[Deque] = None,
        rewards_memory: Optional[Deque] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        actions = self._instantiate_actions(actions=actions, action_ids=action_ids, kwargs=kwargs)
        super().__init__(
            actions=actions,
            adaptive_window_size=adaptive_window_size,
            delta=delta,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
        )

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
        if len(actions) != len(rewards):
            raise AttributeError(f"Shape mismatch: actions and rewards should have the same length {len(actions)}.")

    @validate_call
    def update(self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], **kwargs):
        """
        Update the models associated with the given actions using the provided rewards.
        For adaptive window size, the update by resetting the action models and retraining them on the new data.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """
        self._validate_update_params(actions, rewards, **kwargs)

        if self.adaptive_window_size is not None:
            memory_len = len(self.actions_memory)
            new_samples_len = len(actions)
            if type(self.adaptive_window_size) is int:
                residual_memory_len = max(
                    0, memory_len - max(memory_len + new_samples_len - self.adaptive_window_size, 0)
                )
            else:
                residual_memory_len = None
            self.actions_memory.extend(actions)
            self.rewards_memory.extend(rewards)
            if (
                memory_len
                and (last_change_point := self._get_last_change_point(residual_memory_len)) != _NO_CHANGE_POINT
            ):
                relative_change_point = self._get_relative_change_point(last_change_point, **kwargs)
                self.actions_memory = deque(
                    (self.actions_memory[i] for i in range(relative_change_point, 0)),
                    maxlen=self.max_len,
                )
                self.rewards_memory = deque(
                    (self.rewards_memory[i] for i in range(relative_change_point, 0)),
                    maxlen=self.max_len,
                )

                for action_model in self.actions.values():
                    action_model.reset()
                self._update_actions(self.actions_memory, self.rewards_memory, **kwargs)
            else:
                self._update_actions(actions, rewards, **kwargs)
        else:
            self._update_actions(actions, rewards, **kwargs)

    @abstractmethod
    def _get_relative_change_point(self, last_change_point: NonNegativeInt, **kwargs) -> NonPositiveInt:
        """
        Refine the last change point for the given action.

        Parameters
        ----------
        last_change_point : NonNegativeInt
            The last change point for the given action.

        Returns
        -------
        NonNegativeInt
            The refined last change point for the given action.
        """
        pass

    @abstractmethod
    def _update_actions(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], *args, **kwargs
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

    def _get_last_change_point(self, residual_memory_len: Optional[NonNegativeInt]) -> NonNegativeInt:
        """
        Get the last change point among all actions.

        Parameters
        ----------
        residual_memory_len : Optional[NonNegativeInt]
            Remaining number of elements from last memory state after current update.

        Returns
        -------
        NonNegativeInt
            The last change point. 0 if no change point is found.
        """
        change_points = [
            self._get_last_change_point_for_action(action_id=action_id, residual_memory_len=residual_memory_len)
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
        self, action_id: ActionId, residual_memory_len: Optional[NonNegativeInt]
    ) -> int:
        """
        Get the last change point for the given action.

        Parameters
        ----------
        action_id : ActionId
            The action ID.
        residual_memory_len : Optional[NonNegativeInt]
            Remaining number of elements from last memory state after current update.

        Returns
        -------
        NonNegativeInt
            The last change point for the given action. -1 if no change point is found.
        """
        action_index = np.where([a == action_id for a in self.actions_memory])[0].tolist()

        rewards_window = [self.rewards_memory[i] for i in action_index]
        window_length = len(rewards_window)
        if window_length < 2:
            return _NO_CHANGE_POINT
        cumulative_reward = np.cumsum(np.array(rewards_window), axis=0)
        if cumulative_reward.ndim == 1:
            cumulative_reward = cumulative_reward[:, np.newaxis]
        reference_model = self.actions[action_id]
        if self.adaptive_window_size == "inf" and self._get_expected_memory_length(
            actions={action_id: reference_model}
        ) == len(self.actions_memory):
            current_sum = 0
            current_trials = 0
            initial_start_index = 1
        else:
            action_model = self.actions[action_id]
            if isinstance(action_model, Model):
                current_sum = np.array([action_model.n_successes - 1]).reshape((1, -1))
                current_trials = np.array([action_model.n_successes + action_model.n_failures - 2]).reshape((1, -1))

            elif isinstance(action_model, ModelMO):
                current_sum = np.array([model.n_successes - 1 for model in action_model.models]).reshape((1, -1))
                current_trials = np.array(
                    [model.n_successes + model.n_failures - 2 for model in action_model.models]
                ).reshape((1, -1))
            else:
                raise TypeError(f"Model type {type(action_model)} not supported.")
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
            change_points = np.where(
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
            return _NO_CHANGE_POINT

        return action_index[min(start_index, window_length - 1)]

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
    adaptive_window_size : Optional[Union[PositiveInt, Literal["inf"]]]
        The size of the adaptive window for action update. If None, no adaptive window is used.
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

    def _get_relative_change_point(self, last_change_point: NonNegativeInt) -> NonPositiveInt:
        """
        Refine the last change point for the given action.

        Parameters
        ----------
        last_change_point : NonNegativeInt
            The last change point for the given action.

        Returns
        -------
        NonPositiveInt
            The refined last change point for the given action.
        """
        return last_change_point - len(self.actions_memory)

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
    adaptive_window_size : Optional[Union[PositiveInt, Literal["inf"]]]
        The size of the adaptive window for action update. If None, no adaptive window is used.
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

    def _validate_update_params(
        self, actions: List[ActionId], rewards: Union[List[BinaryReward], List[List[BinaryReward]]], context: ArrayLike
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
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        super()._validate_update_params(actions, rewards)
        if len(context) != len(actions):
            raise AttributeError(f"Shape mismatch: actions and context should have the same length {len(actions)}.")

    def _get_relative_change_point(self, last_change_point: NonNegativeInt, context: ArrayLike) -> NonPositiveInt:
        """
        Refine the last change point for the given action. Since context memory is not stored, the relative change point
        shall not exceed the length of the context.

        Parameters
        ----------
        last_change_point : NonNegativeInt
            The last change point for the given action.
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features

        Returns
        -------
        NonPositiveInt
            The refined last change point for the given action.
        """
        return max(last_change_point - len(self.actions_memory), -context.shape[0])

    def _update_actions(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        context: ArrayLike,
    ):
        """
        Update the models associated with the given actions using the provided rewards.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        # cast inputs to numpy arrays to facilitate their manipulation
        context, actions, rewards = np.array(context), np.array(actions), np.array(rewards)
        context = context[-len(actions) :]
        for a in set(actions):
            # get context and rewards of the samples associated to action a
            context_of_a = context[actions == a]
            rewards_of_a = rewards[actions == a].tolist()

            # update model associated to action a
            self.actions[a].update(context=context_of_a, rewards=rewards_of_a)
