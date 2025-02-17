from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta
from typing_extensions import Self

from pybandits.base import BinaryReward, Probability, PyBanditsBaseModel, QuantitativeProbability
from pybandits.base_model import BaseModel, BaseModelCC
from pybandits.model import BayesianLogisticRegression, Beta, Model
from pybandits.pydantic_version_compatibility import (
    NonNegativeFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    validate_call,
)


class QuantitativeModel(BaseModel, ABC):
    """
    Base class for quantitative models.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the _model.
    """

    dimension: PositiveInt

    @abstractmethod
    def sample_proba(self) -> float:
        """
        Sample the _model.
        """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        quantities: List[Union[float, List[float]]],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        **kwargs,
    ):
        """
        Update the _model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        context : Optional[ArrayLike]
            Context for each sample.
        """

        self._validate_params_lengths(quantities=quantities, rewards=rewards, **kwargs)
        if quantities:
            self._update(quantities, rewards, **kwargs)

    @abstractmethod
    def _update(
        self,
        quantities: Optional[List[Union[float, List[float], None]]],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        **kwargs,
    ):
        """
        Update the _model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """


class QuantitativeModelCC(BaseModelCC, ABC):
    """
    Class to _model quantitative action cost.

    Parameters
    ----------
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """

    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]


class Segment(PyBanditsBaseModel):
    """
    Segment class.

    Parameters
    ----------
    intervals: Tuple[Tuple[Probability, Probability], ...]
        Intervals of the segment.
    """

    intervals: Tuple[Tuple[Probability, Probability], ...]

    @property
    def mins(self) -> np.ndarray:
        return self.intervals_array[:, 0]

    @property
    def maxs(self) -> np.ndarray:
        return self.intervals_array[:, 1]

    @property
    def intervals_array(self) -> np.ndarray:
        array_form = np.array(self.intervals)
        if array_form.size == 0:
            return np.array([[], []]).T
        return np.array(self.intervals)

    @field_validator("intervals", mode="before")
    @classmethod
    def segment_intervals_to_tuple(cls, value):
        if isinstance(value, np.ndarray):
            if value.shape[1] != 2:
                raise ValueError("Intervals must have shape (n, 2).")
            return tuple(tuple(v) for v in value)
        return value

    def split(self) -> Tuple["Segment", "Segment"]:
        middles = (self.mins + self.maxs) / 2
        left_intervals = np.concatenate(np.atleast_2d(self.mins, middles), axis=1)
        right_intervals = np.concatenate(np.atleast_2d(middles, self.maxs), axis=1)
        return Segment(intervals=left_intervals), Segment(intervals=right_intervals)

    def __add__(self, other: "Segment") -> "Segment":
        """
        Add two adjacent segments.

        Parameters
        ----------
        other : Segment
            Segment to add.

        Returns
        -------
        Segment
            The merged segment.
            The merged segment.
        """
        if not self.is_adjacent(other):
            raise ValueError("Segments must be adjacent.")
        to_concatenate = (self.mins, other.maxs) if self.maxs == other.mins else (other.mins, self.maxs)
        new_intervals = np.concatenate(np.atleast_2d(*to_concatenate), axis=1)
        return Segment(intervals=new_intervals)

    def __hash__(self) -> int:
        return tuple(self.intervals_array.flatten()).__hash__()

    def __contains__(self, value: Union[float, np.ndarray]) -> bool:
        """
        Check if a value is contained in segment.

        Parameters
        ----------
        value : Union[float, np.ndarray]
            Value to check.

        Returns
        -------
        bool
            Whether the value is contained in the segment.
        """
        if (isinstance(value, np.ndarray) and value.shape != self.intervals_array.shape[1]) or (
            isinstance(value, float) and len(self.intervals_array) != 1
        ):
            raise ValueError("Tested value must have the same shape as the intervals.")
        return bool(
            np.all(
                np.logical_and(
                    (self.mins <= value),
                    np.logical_or((value < self.maxs), np.logical_and(value == self.maxs, self.maxs == 1)),
                )
            )
        )

    def __eq__(self, other) -> bool:
        return np.all(self.intervals_array == other.intervals_array)

    def is_adjacent(self, other: "Segment") -> bool:
        """
        Check if two segments are adjacent.

        Parameters
        ----------
        other : Segment
            Segment to check.

        Returns
        -------
        bool
            Whether the segments are adjacent.
        """
        if self.intervals_array.shape[0] != other.intervals_array.shape[0]:
            raise ValueError("Segments must have the same shape.")
        return np.all(self.maxs == other.mins) or np.all(self.mins == other.maxs)


class ZoomingModel(QuantitativeModel, ABC):
    """
    Base class for zooming models.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the _model.
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Segment, Optional[Model]]
        Mapping of segments to models.
    """

    dimension: PositiveInt
    comparison_threshold: Probability = 0.1
    n_comparison_points: PositiveInt = 1000
    n_max_segments: Optional[PositiveInt] = 32
    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[Model]]
    _base_model: Model = PrivateAttr()

    def _validate_segments(self):
        if self.n_max_segments is not None and len(self.sub_actions) > self.n_max_segments:
            raise ValueError("Number of segments must be less than the maximum number of segments.")
        dimensions = {len(segment) for segment in self.sub_actions.keys()}
        if dimensions != {self.dimension}:
            raise ValueError(f"All segments must have the same dimension {self.dimension}.")

    def model_post_init(self, __context: Any) -> None:
        self._validate_segments()
        self._init_base_model()
        segment_models_types = set(type(model) if model is not None else None for model in self.sub_actions.values())
        if None in segment_models_types:
            if len(segment_models_types) > 1:
                raise ValueError("All segments must either have a _model or miss a _model.")
            self.sub_actions = dict(zip(self.sub_actions, [self._base_model.model_copy()] * len(self.sub_actions)))

    @property
    def segmented_actions(self) -> Dict[Segment, Optional[Model]]:
        return {Segment(intervals=segment): model for segment, model in self.sub_actions.items()}

    @abstractmethod
    def _init_base_model(self):
        """
        Initialize the base _model.
        """

    @classmethod
    @validate_call
    def cold_start(
        cls,
        dimension: PositiveInt = 1,
        n_1d_segments: PositiveInt = 2,
        comparison_threshold: Probability = 0.1,
        n_comparison_points: PositiveInt = 1000,
        n_max_segments: Optional[PositiveInt] = 32,
        **kwargs,
    ) -> Self:
        """
        Create a cold start _model.

        Returns
        -------
        ZoomingModel
            Cold start _model.
        """
        interval_points = np.linspace(0, 1, n_1d_segments + 1)
        intervals = [(interval_points[i], interval_points[i + 1]) for i in range(n_1d_segments)]
        sub_actions = {tuple(segment): None for segment in product(intervals, repeat=dimension)}
        return cls(
            dimension=dimension,
            comparison_threshold=comparison_threshold,
            n_comparison_points=n_comparison_points,
            n_max_segments=n_max_segments,
            sub_actions=sub_actions,
            **kwargs,
        )

    def sample_proba(self, **kwargs) -> List[QuantitativeProbability]:
        """
        Sample an action value from each of the intervals.
        """
        result = []
        for segment, model in self.segmented_actions.items():
            sampled_proba = model.sample_proba(**kwargs)
            random_point = np.random.random((len(sampled_proba), len(segment.intervals)))
            scaled_quantity = segment.mins.T + random_point * (segment.maxs.T - segment.mins.T)

            result.append(tuple((tuple(quantity), prob) for quantity, prob in zip(scaled_quantity, sampled_proba)))
        result = list(zip(*result))
        return result

    def _update(self, quantities: List[Union[float, np.ndarray]], rewards: List[BinaryReward], **kwargs):
        """
        Update the _model parameters.

        Parameters
        ----------
        quantities : List[Union[float, np.ndarray]]
            The value associated with each action.
        rewards: List[BinaryReward]
            The reward for each sample.
        context : Optional[ArrayLike]
            Context for each sample.
        """

        segments = self._map_and_update_segment_models(quantities, rewards, **kwargs)
        self._update_segmentation(quantities, segments, rewards, **kwargs)

    def _map_and_update_segment_models(
        self, quantities: List[Union[float, np.ndarray]], rewards: List[BinaryReward], **kwargs
    ) -> List[Segment]:
        """
        Map and update the segment models.

        Parameters
        ----------
        quantities : List[Union[float, np.ndarray]]
            The value associated with each action.
        rewards: List[BinaryReward]
            The reward for each sample.

        Returns
        -------
        List[Segment]
            Segments to update.
        """
        segments = self._map_values_to_segments(quantities)
        self._inner_update(segments, rewards, **kwargs)
        return segments

    @abstractmethod
    def _inner_update(self, segments: List[Segment], rewards: List[BinaryReward], **kwargs):
        """
        Update the segments models.

        Parameters
        ----------
        segments : List[Segment]
            Segments to update.
        rewards : List[BinaryReward]
            Rewards for update.
        context : Optional[ArrayLike]
            Context for update.
        """

    def _map_values_to_segments(self, quantities: List[Union[float, np.ndarray]]) -> List[Segment]:
        segments = [segment for value in quantities for segment in self.segmented_actions.keys() if value in segment]
        return segments

    def _update_segmentation(
        self,
        quantities: List[Union[float, np.ndarray]],
        segments: List[Segment],
        rewards: List[BinaryReward],
        **kwargs,
    ):
        segment_scores = {segment: model.mean for segment, model in self.segmented_actions.items()}
        ordered_segments = sorted(segment_scores, key=segment_scores.get)
        best_segment = ordered_segments[-1]
        del self.sub_actions[best_segment.intervals]

        # Consider merging adjacent segments
        worst_segments = ordered_segments[:-1]
        i = 0
        while i < len(worst_segments) - 1:
            segment = worst_segments[i]
            j = i + 1
            while j < len(worst_segments):
                other_segment = worst_segments[j]
                if segment.is_adjacent(other_segment) and self.is_similar_performance(segment, other_segment):
                    del self.sub_actions[segment.intervals]
                    del self.sub_actions[other_segment.intervals]
                    worst_segments.remove(segment)
                    worst_segments.remove(other_segment)
                    merged_segment = segment + other_segment
                    self.sub_actions[merged_segment.intervals] = self._base_model.model_copy()
                    filtered_quantities, filtered_rewards, filtered_kwargs = self._filter_by_segment(
                        merged_segment, quantities, segments, rewards, **kwargs
                    )
                    self._map_and_update_segment_models(filtered_quantities, filtered_rewards, **filtered_kwargs)
                    break
                j += 1
            i += 1

        # Split best segment if possible
        if self.n_max_segments is None or len(self.sub_actions) < self.n_max_segments:
            sub_best_segments = best_segment.split()
            self.sub_actions[sub_best_segments[0].intervals] = self._base_model.model_copy()
            self.sub_actions[sub_best_segments[1].intervals] = self._base_model.model_copy()
            filtered_quantities, filtered_rewards, filtered_kwargs = self._filter_by_segment(
                best_segment, quantities, segments, rewards, **kwargs
            )
            self._map_and_update_segment_models(filtered_quantities, filtered_rewards, **filtered_kwargs)

    def is_similar_performance(self, segment1: Segment, segment2: Segment) -> bool:
        """
        Check if two segments have similar performance.

        Parameters
        ----------
        segment1 : Segment
            First segment.
        segment2 : Segment
            Second segment.

        Returns
        -------
        bool
            Whether the segments have similar performance.
        """
        x = np.linspace(0, 1, self.n_comparison_points)
        model1 = self.sub_actions[segment1.intervals]
        model2 = self.sub_actions[segment2.intervals]
        p1 = beta.pdf(x, model1.n_successes, model1.n_failures)
        p2 = beta.pdf(x, model2.n_successes, model2.n_failures)
        return jensenshannon(p1, p2) < self.comparison_threshold

    def _filter_by_segment(
        self,
        reference_segment: Segment,
        quantities: List[Union[float, np.ndarray]],
        segments: List[Segment],
        rewards: List[BinaryReward],
        **kwargs,
    ) -> Tuple[List[Union[float, np.ndarray]], List[BinaryReward], Dict[str, Any]]:
        """
        Filter and update the segments models.

        Parameters
        ----------
        reference_segment : Segment
            Reference segment to filter upon.
        segments : List[Segment]
            Segments to filter.
        quantities : List[Union[float, np.ndarray]]
            Values to filter.
        rewards : List[BinaryReward]
            Rewards to filter.

        Returns
        -------
        filtered_values : List[Union[float, np.ndarray]]
            Filtered quantities.
        filtered_rewards : List[BinaryReward]
            Filtered rewards.
        filtered_kwargs : Dict[str, Any]
            Filtered context.
        """
        filtered_values_rewards_kwargs = [
            (value, reward, *[kwarg[i] for kwarg in kwargs.values()])
            for i, (value, reward, segment) in enumerate(zip(quantities, rewards, segments))
            if segment == reference_segment
        ]
        if filtered_values_rewards_kwargs:
            filtered_values, filtered_rewards, *filtered_kwargs = zip(*filtered_values_rewards_kwargs)
            filtered_kwargs = dict(zip(kwargs.keys(), filtered_kwargs))
        else:
            filtered_values, filtered_rewards, filtered_kwargs = [], [], {k: [] for k in kwargs.keys()}
        filtered_kwargs = {
            k: np.array(v) if isinstance(kwargs[k], np.ndarray) else v for k, v in filtered_kwargs.items()
        }
        return filtered_values, filtered_rewards, filtered_kwargs


class BaseSmabZoomingModel(ZoomingModel, ABC):
    """
    Zooming _model for sMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the _model.
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[Beta]]
        Mapping of segments to Beta models.
    """

    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[Beta]]

    def _init_base_model(self):
        """
        Initialize the base _model.
        """
        self._base_model = Beta()

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        quantities: Optional[List[Union[float, List[float], None]]],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
    ):
        """
        Update the _model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        context : Optional
            Placeholder for context.
        """
        super().update(quantities, rewards)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _inner_update(self, segments: List[Segment], rewards: List[BinaryReward]):
        """
        Update the segments models.

        Parameters
        ----------
        segments : List[Segment]
            Segments to update.
        rewards : List[BinaryReward]
            Rewards for update.
        """
        segments = np.array(segments)
        rewards = np.array(rewards)
        for segment in set(segments):
            rewards_of_segment = rewards[segments == segment].tolist()
            self.sub_actions[segment.intervals].update(rewards=rewards_of_segment)


class SmabZoomingModel(BaseSmabZoomingModel):
    """
    Zooming _model for sMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the _model.
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[Beta]]
        Mapping of segments to Beta models.
    """


class SmabZoomingModelCC(BaseSmabZoomingModel, QuantitativeModelCC):
    """
    Zooming _model for sMAB with cost control.

    Parameters
    ----------
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """


class BaseCmabZoomingModel(ZoomingModel, ABC):
    """
    Zooming _model for CMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the _model.
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[BayesianLogisticRegression]]
        Mapping of segments to Bayesian Logistic Regression models.
    base_model_cold_start_kwargs: Dict[str, Any]
        Keyword arguments for the base _model cold start.
    """

    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[BayesianLogisticRegression]]
    base_model_cold_start_kwargs: Dict[str, Any]

    @field_validator("base_model_cold_start_kwargs", mode="before")
    @classmethod
    def validate_n_features(cls, value):
        if "n_features" not in value:
            raise KeyError("n_features must be in base_model_cold_start_kwargs.")
        return value

    def _init_base_model(self):
        """
        Initialize the base _model.
        """
        self._base_model = BayesianLogisticRegression.cold_start(**self.base_model_cold_start_kwargs)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _inner_update(self, segments: List[Segment], rewards: List[BinaryReward], context: ArrayLike):
        """
        Update the segments models.

        Parameters
        ----------
        segments : List[Segment]
            Segments to update.
        rewards : List[BinaryReward]
            Rewards for update.
        context : Optional[ArrayLike]
            Context for update.
        """
        segments = np.array(segments)
        rewards = np.array(rewards)
        context = np.array(context)
        for segment in set(segments):
            rewards_of_segment = rewards[segments == segment].tolist()
            context_of_segment = context[segments == segment]
            if rewards_of_segment:
                self.sub_actions[segment.intervals].update(rewards=rewards_of_segment, context=context_of_segment)


class CmabZoomingModel(BaseCmabZoomingModel):
    """
    Zooming _model for CMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the _model.
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Probability, Probability], ...], Optional[BayesianLogisticRegression]]
        Mapping of segments to Bayesian Logistic Regression models.
    base_model_cold_start_kwargs: Dict[str, Any]
        Keyword arguments for the base _model cold start.
    """


class CmabZoomingModelCC(BaseCmabZoomingModel, QuantitativeModelCC):
    """
    Zooming _model for CMAB with cost control.

    Parameters
    ----------
    comparison_threshold: Probability
        Comparison threshold.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    base_model_cold_start_kwargs: Dict[str, Any]
        Keyword arguments for the base _model cold start.
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """