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

import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import numpy as np
from numpy.typing import ArrayLike
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeInt,
    NonPositiveInt,
    PositiveInt,
    model_validator,
    validate_call,
)

from pybandits.base import (
    ActionId,
    BinaryReward,
    PositiveProbability,
    PyBanditsBaseModel,
)
from pybandits.base_model import BaseModel, BaseModelMO, BaseModelSO
from pybandits.meta_model import (
    BaseMetaModel,
    CmabPerActionMetaModel,
    PerActionMetaModel,
    SampleProbaResult,
)
from pybandits.model import (
    BaseBayesianNeuralNetwork,
    BaseBayesianNeuralNetworkMO,
    BaseBeta,
    BaseBetaMO,
    BayesianNeuralNetwork,
    BayesianNeuralNetworkCC,
    BayesianNeuralNetworkDP,
    BayesianNeuralNetworkMO,
    BayesianNeuralNetworkMOCC,
    Beta,
    BetaCC,
    BetaDP,
    BetaMO,
    BetaMOCC,
    Model,
    ModelMO,
)
from pybandits.quantitative_model import (
    BaseQuantitativeBayesianNeuralNetwork,
    BaseZooming,
    QuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetworkCC,
    QuantitativeBayesianNeuralNetworkDP,
    QuantitativeModel,
    Zooming,
    ZoomingCC,
    ZoomingDP,
)

SmabModelType = TypeVar("SmabModelType", bound=Union[BaseBeta, BaseBetaMO, BaseZooming])
CmabModelType = TypeVar(
    "CmabModelType",
    bound=Union[
        BaseBayesianNeuralNetwork,
        BaseBayesianNeuralNetworkMO,
        BaseQuantitativeBayesianNeuralNetwork,
    ],
)


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
    meta_model : BaseMetaModel
        The meta-model that owns per-action state and dispatches ``sample_proba``,
        ``update``, and ``reset``. Constructed automatically from an ``actions``
        dict when the manager is instantiated via the ``actions=`` kwarg.
    delta : Optional[PositiveProbability]
        The confidence level for the adaptive window. None for skipping the change point detection.
    """

    meta_model: BaseMetaModel
    delta: Optional[PositiveProbability] = None
    _no_change_point: ClassVar[NonPositiveInt] = -1
    _min_adaptive_window_size: ClassVar[PositiveInt] = 10000
    _memory_parameters_suffix: ClassVar[str] = "_memory"
    actions_with_change: Set[Tuple[ActionId, NonNegativeInt]] = Field(default_factory=set)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def actions(self) -> Dict[ActionId, BaseModel]:
        """Per-action models, delegated to ``meta_model.actions``."""
        return self.meta_model.actions

    @classmethod
    def _get_meta_model_cls(cls) -> Type[BaseMetaModel]:
        """Return the concrete meta-model class from the manager's ``meta_model`` field annotation."""
        meta_model_field = cls.model_fields.get("meta_model")
        if meta_model_field is not None and meta_model_field.annotation is not None:
            annotation = meta_model_field.annotation
            type_args = get_args(annotation)
            if type_args and not isinstance(type_args[0], TypeVar):
                return annotation  # type: ignore[return-value]
            return get_origin(annotation) or annotation  # type: ignore[return-value]
        return PerActionMetaModel

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
        if isinstance(reference_model, BaseModelSO):
            expected_memory_length_for_inf = sum([action_model.count - 2 for action_model in actions.values()])
        elif isinstance(reference_model, BaseModelMO):
            expected_memory_length_for_inf = sum(
                [action_model.models[0].count - 2 for action_model in actions.values()]
            )
        else:
            raise ValueError(f"Model type {type(reference_model)} not supported.")
        return expected_memory_length_for_inf

    def __init__(
        self,
        delta: Optional[PositiveProbability] = None,
        actions: Optional[Dict[ActionId, Model]] = None,
        action_ids: Optional[Set[ActionId]] = None,
        quantitative_action_ids: Optional[Set[ActionId]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        actions_with_change: Optional[Set[Tuple[ActionId, NonNegativeInt]]] = None,
        meta_model: Optional[BaseMetaModel] = None,
    ):
        action_args = (actions, action_ids, quantitative_action_ids)
        if meta_model is not None and any(a is not None for a in action_args):
            raise ValueError(
                "Provide either 'meta_model' or action-construction arguments "
                "('actions', 'action_ids', 'quantitative_action_ids'), not both."
            )
        if meta_model is None:
            meta_model = self._get_meta_model_cls()(
                actions=actions,
                action_ids=action_ids,
                quantitative_action_ids=quantitative_action_ids,
                kwargs=kwargs or {},
            )
        actions_with_change = actions_with_change or set()
        super().__init__(meta_model=meta_model, delta=delta, actions_with_change=actions_with_change)

    def _validate_update_params(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
        **kwargs,
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
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        """
        invalid = set(actions) - set(self.actions.keys())
        if invalid:
            raise AttributeError(f"The following invalid action(s) were specified: {invalid}.")
        self._validate_params_lengths(actions=actions, rewards=rewards, quantities=quantities, **kwargs)
        if quantities is None:
            if not all(isinstance(self.actions[action], (Model, ModelMO)) for action in actions):
                raise ValueError("Quantitative actions require defined quantities.")
        else:
            if not all(
                q is not None for a, q in zip(actions, quantities) if isinstance(self.actions[a], QuantitativeModel)
            ):
                raise ValueError("Quantitative actions require defined quantities.")
            if not all(q is None for a, q in zip(actions, quantities) if isinstance(self.actions[a], (Model, ModelMO))):
                raise ValueError("regular actions should not have defined quantities.")

    @classmethod
    def _to_memory_key(cls, key: str) -> str:
        return f"{key}{cls._memory_parameters_suffix}"

    @classmethod
    def _to_key(cls, key: str) -> str:
        return key.replace(cls._memory_parameters_suffix, "")

    def sample_proba(
        self,
        rng: np.random.Generator,
        valid_action_ids: Optional[Set[ActionId]] = None,
        **kwargs: Any,
    ) -> Dict[ActionId, SampleProbaResult]:
        """Sample per-action probabilities/scores for the bandit's predict path.

        Delegates to ``self.meta_model.sample_proba``. With the default
        ``PerActionMetaModel`` this is a per-action dispatch loop; alternative
        meta-models (e.g. a shared backbone) may evaluate all actions jointly.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central random generator from the bandit.
        valid_action_ids : Optional[Set[ActionId]]
            If provided, restrict sampling to these action ids; otherwise
            sample for all actions.
        **kwargs
            Forwarded to per-action ``sample_proba`` (e.g. ``context`` for
            cmab, ``n_samples`` for smab).
        """
        return self.meta_model.sample_proba(rng=rng, valid_action_ids=valid_action_ids, **kwargs)

    @validate_call(config=dict(arbitrary_types_allowed=True))
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
        Update the models associated with the given actions using the provided rewards.
        For adaptive window size, the update by resetting the action models and retraining them on the new data.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        self.actions_with_change.clear()
        if self.delta is None and (actions_memory or rewards_memory):
            raise AttributeError("Adaptive window size is not set, so memory should not be provided.")
        if self.delta is not None and (actions_memory is None or rewards_memory is None):
            warnings.warn("Adaptive window size is set, but memory was not provided.")
            actions_memory = []
            rewards_memory = []

        update_kwargs = {k: v for k, v in kwargs.items() if not k.endswith(self._memory_parameters_suffix)}
        memory_kwargs = {k: v for k, v in kwargs.items() if k.endswith(self._memory_parameters_suffix)}
        self._validate_update_params(actions, rewards, quantities, **update_kwargs)
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
                    if not isinstance(action_model, QuantitativeModel):
                        action_model.reset()
                # Non-quantitative models were just reset; retrain on the trimmed memory window.
                # Filter from `actions_memory` so the action labels align with `rewards_memory`
                # / `memory_kwargs` (which are filtered from the same source below).
                regular_actions = [a for a in actions_memory if not isinstance(self.actions[a], QuantitativeModel)]
                # Quantitative models keep their state (no reset), so they only need the new batch.
                quantitative_actions = [a for a in actions if isinstance(self.actions[a], QuantitativeModel)]

                if regular_actions:
                    regular_rewards = [
                        r
                        for a, r in zip(actions_memory, rewards_memory)
                        if not isinstance(self.actions[a], QuantitativeModel)
                    ]
                    filtered_update_kwargs = {
                        self._to_key(k): [
                            v
                            for a, v in zip(actions_memory, values)
                            if not isinstance(self.actions[a], QuantitativeModel)
                        ]
                        if isinstance(values, list)
                        else values
                        for k, values in memory_kwargs.items()
                    }
                    self._update_actions(regular_actions, regular_rewards, None, **filtered_update_kwargs)

                if quantitative_actions:
                    filtered_quantitative_kwargs = {
                        k: [v for a, v in zip(actions, values) if isinstance(self.actions[a], QuantitativeModel)]
                        if isinstance(values, list)
                        else values
                        for k, values in update_kwargs.items()
                    }
                    quantitative_rewards = [
                        r for a, r in zip(actions, rewards) if isinstance(self.actions[a], QuantitativeModel)
                    ]
                    quantitative_quantities = [
                        q for a, q in zip(actions, quantities) if isinstance(self.actions[a], QuantitativeModel)
                    ]
                    self._update_actions(
                        quantitative_actions,
                        quantitative_rewards,
                        quantitative_quantities,
                        **filtered_quantitative_kwargs,
                    )
            else:
                self._update_actions(actions, rewards, quantities, **update_kwargs)
        else:
            self._update_actions(actions, rewards, quantities, **update_kwargs)

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
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
        **kwargs,
    ):
        """
        Update the models associated with the given actions using the provided rewards.

        Parameters
        ----------
        actions : List[ActionId]
            The selected action for each sample.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        """

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
        last_change_point = self._no_change_point
        for action_id, action_model in self.actions.items():
            if not isinstance(action_model, QuantitativeModel):
                change_point = self._get_last_change_point_for_action(
                    action_id=action_id,
                    residual_memory_len=residual_memory_len,
                    actions_memory=actions_memory,
                    rewards_memory=rewards_memory,
                )
                if change_point != self._no_change_point:
                    self.actions_with_change.add((action_id, change_point))
                    last_change_point = max(last_change_point, change_point)
        return last_change_point

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
        if isinstance(action_model, BaseModelSO):
            current_sum = np.array([action_model.n_successes - 1]).reshape((1, -1))
            current_trials = np.array([action_model.count - 2]).reshape((1, -1))

        elif isinstance(action_model, BaseModelMO):
            current_sum = np.array([model.n_successes - 1 for model in action_model.models]).reshape((1, -1))
            current_trials = np.array([model.count - 2 for model in action_model.models]).reshape((1, -1))
        else:
            raise TypeError(f"Model type {type(action_model)} not supported.")
        return current_sum, current_trials


class SmabActionsManager(ActionsManager, Generic[SmabModelType]):
    """
    Manages actions and their associated models for sMAB models.
    The class allows to account for non-stationarity by providing an adaptive window scheme for action update.

    Parameters
    ----------
    meta_model : PerActionMetaModel[SmabModelType]
        The meta-model owning per-action state. Constructed automatically from an ``actions`` dict
        when the manager is instantiated via the ``actions=`` kwarg.
    delta : Optional[PositiveProbability]
        The confidence level for the adaptive window. ``None`` disables change-point detection.
    """

    meta_model: PerActionMetaModel[SmabModelType]  # type: ignore[valid-type]

    @model_validator(mode="after")
    def all_actions_have_same_number_of_objectives(self) -> "SmabActionsManager":
        n_objs_per_action = [
            len(beta.models) if isinstance(beta, BaseBetaMO) else None for beta in self.meta_model.actions.values()
        ]
        if len(set(n_objs_per_action)) != 1:
            raise ValueError("All actions should have the same number of objectives")
        return self

    @validate_call
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]] = None,
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
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        actions_memory : Optional[List[ActionId]]
            List of previously selected actions.
        rewards_memory : Optional[Union[List[BinaryReward], List[List[BinaryReward]]]]
            List of previously collected rewards.
        """
        super().update(actions, rewards, quantities, actions_memory, rewards_memory)

    def _update_actions(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
    ):
        """
        Update the stochastic Bernoulli bandit given the list of selected actions and their corresponding binary
        rewards. Delegates to ``self.meta_model.update``.

        Parameters
        ----------
        actions : List[ActionId] of shape (n_samples,), e.g. ['a1', 'a2', 'a3', 'a4', 'a5']
            The selected action for each sample.
        rewards : Union[List[BinaryReward], List[List[BinaryReward]]],
            if nested list, len() should follow shape of (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        """
        self.meta_model.update(actions=actions, rewards=rewards, quantities=quantities)


class CmabActionsManager(ActionsManager, Generic[CmabModelType]):
    """
    Manages actions and their associated models for cMAB models.
    The class allows to account for non-stationarity by providing an adaptive window scheme for action update.

    Parameters
    ----------
    meta_model : CmabPerActionMetaModel[CmabModelType]
        The cmab-specific meta-model owning per-action state and the cross-action consistency
        validator (input dim / update method / update kwargs). Constructed automatically from an
        ``actions`` dict when the manager is instantiated via the ``actions=`` kwarg.
    delta : Optional[PositiveProbability]
        The confidence level for the adaptive window. ``None`` disables change-point detection.
    """

    meta_model: CmabPerActionMetaModel[CmabModelType]  # type: ignore[valid-type]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
        context: np.ndarray,
        actions_memory: Optional[List[ActionId]] = None,
        rewards_memory: Optional[Union[List[BinaryReward], List[List[BinaryReward]]]] = None,
        context_memory: Optional[np.ndarray] = None,
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
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
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
        super().update(
            actions=actions,
            rewards=rewards,
            quantities=quantities,
            context=context,
            actions_memory=actions_memory,
            rewards_memory=rewards_memory,
            context_memory=context_memory,
        )

    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _check_context_matrix(context: np.ndarray):
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
            raise AttributeError(f"Context must be an eligible to transform to float numpy array: {e}.")
        return context

    def _update_actions(
        self,
        actions: List[ActionId],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        quantities: Optional[List[Union[float, List[float], None]]],
        context: np.ndarray,
    ):
        """
        Update the models associated with the given actions using the provided rewards.

        Delegates to ``self.meta_model.update``. With the default
        ``PerActionMetaModel`` this preserves the historical per-action
        loop; alternative meta-models (introduced in follow-up PRs) may
        run a joint update across actions instead.

        Parameters
        ----------
        actions : List[UnifiedActionId] of shape (n_samples,), e.g. ['a1', 'a2', 'a3', 'a4', 'a5']
            The selected action for each sample.
        rewards : List[Union[BinaryReward, List[BinaryReward]]] of shape (n_samples, n_objectives)
            The binary reward for each sample.
                If strategy is not MultiObjectiveBandit, rewards should be a list, e.g.
                    rewards = [1, 0, 1, 1, 1, ...]
                If strategy is MultiObjectiveBandit, rewards should be a list of list, e.g. (with n_objectives=2):
                    rewards = [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], ...]
        quantities : Optional[List[Union[float, List[float], None]]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        context: np.ndarray of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        self.meta_model.update(
            actions=actions,
            rewards=rewards,
            quantities=quantities,
            context=context,
        )


# For pickling purposes
SmabActionsManagerSO = SmabActionsManager[Union[Beta, Zooming]]
SmabActionsManagerCC = SmabActionsManager[Union[BetaCC, ZoomingCC]]
SmabActionsManagerDP = SmabActionsManager[Union[BetaDP, ZoomingDP]]
SmabActionsManagerMO = SmabActionsManager[BetaMO]
SmabActionsManagerMOCC = SmabActionsManager[BetaMOCC]

CmabActionsManagerSO = CmabActionsManager[Union[BayesianNeuralNetwork, QuantitativeBayesianNeuralNetwork]]
CmabActionsManagerCC = CmabActionsManager[Union[BayesianNeuralNetworkCC, QuantitativeBayesianNeuralNetworkCC]]
CmabActionsManagerDP = CmabActionsManager[Union[BayesianNeuralNetworkDP, QuantitativeBayesianNeuralNetworkDP]]
CmabActionsManagerMO = CmabActionsManager[BayesianNeuralNetworkMO]
CmabActionsManagerMOCC = CmabActionsManager[BayesianNeuralNetworkMOCC]
