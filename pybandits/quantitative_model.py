import functools
import inspect
import json
from abc import ABC, abstractmethod
from collections import Counter
from itertools import product
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta
from typing_extensions import Self

from pybandits.base import BinaryReward, Float01, PyBanditsBaseModel, QuantitativeProbability
from pybandits.base_model import BaseModelCC, BaseModelSO
from pybandits.model import BayesianNeuralNetwork, Beta, Model
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    NonNegativeFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    pydantic_version,
    validate_call,
)

if pydantic_version == PYDANTIC_VERSION_2:
    from pydantic import field_serializer


class QuantitativeModel(BaseModelSO, ABC):
    """
    Base class for quantitative models.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    """

    dimension: PositiveInt

    @abstractmethod
    def sample_proba(self, **kwargs) -> List[QuantitativeProbability]:
        """
        Sample the model.
        """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _update(
        self,
        quantities: List[Union[float, List[float]]],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        **kwargs,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """

        if quantities:
            self._quantitative_update(quantities, rewards, **kwargs)

    @abstractmethod
    def _quantitative_update(
        self,
        quantities: Optional[List[Union[float, List[float], None]]],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
        **kwargs,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """


class QuantitativeModelCC(BaseModelCC, ABC):
    """
    Class to model quantitative action cost.

    Parameters
    ----------
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """

    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]

    @staticmethod
    def _serialize_function(func: Callable) -> str:
        """
        Serialize a function to its source code.

        Parameters
        ----------
        func : Callable
            Function to serialize as string.

        Returns
        -------
        str
            The serialized function code.
        """
        if inspect.isfunction(func) and not func.__name__ == "<lambda>":
            try:
                return inspect.getsource(func).strip()
            except OSError:  # Dynamically evaluated functions may not have source code available
                return globals()[func.__name__].__source__  # Fallback to the global scope if source is not available
        return str(func)

    @staticmethod
    def _deserialize_function(code: str) -> Callable:
        """
        Deserialize a function from its source code.

        Parameters
        ----------
        code : str
            python code representing a function or a callable object.

        Returns
        -------
        Callable
            The deserialized function or callable object.
        """
        if code.startswith("def "):
            exec(code, globals())
            func_name = code.split("(")[0][4:].strip()
            globals()[func_name].__source__ = code.strip()  # Register the function in the global scope
        else:
            func_name = code.strip()
        return eval(func_name)

    @staticmethod
    def serialize_cost(cost_value) -> str:
        """Serialize cost value to string representation."""
        if isinstance(cost_value, functools.partial):
            return f"functools.partial({QuantitativeModelCC._serialize_function(cost_value.func)}, {cost_value.args}, {cost_value.keywords})"
        elif callable(cost_value):
            return QuantitativeModelCC._serialize_function(cost_value)
        else:
            raise ValueError(f"Unrecognized cost for serialization: {cost_value}")

    @classmethod
    def deserialize_cost(cls, value):
        """Deserialize cost from string representation if needed."""
        if isinstance(value, str):
            if value.startswith("functools.partial"):
                inner_func_split = "(".join(value.split("(")[1:]).split(",")
                # Extract function and arguments from pattern: functools.partial(func_name, args, kwargs)
                func_str = ",".join(inner_func_split[:-2]).strip()
                func = cls._deserialize_function(func_str)
                args_str = ")".join(",".join(inner_func_split[-2:]).split(")")[:-1]).strip()
                args_parts = eval(args_str) if args_str else ((), {})
                return functools.partial(func, *args_parts[0], **args_parts[1])
            else:
                return cls._deserialize_function(value)
        return value

    if pydantic_version == PYDANTIC_VERSION_1:

        def dict(self, **kwargs):
            d = super().dict(**kwargs)
            # Handle cost field serialization for CC models
            if "cost" in d:
                d["cost"] = self.serialize_cost(d["cost"])
            return d

    elif pydantic_version == PYDANTIC_VERSION_2:

        @field_serializer("cost")
        def encode_cost(self, value):
            return self.serialize_cost(value).encode("ascii")

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @field_validator("cost", mode="before")
    @classmethod
    def validate_cost(cls, value):
        """
        Deserialize cost from string representation if needed.
        """
        return cls.deserialize_cost(value)


class Segment(PyBanditsBaseModel):
    """
    This class is used to represent a segment of the quantities space.
    A segment is defined by a list of intervals, thus representing a hyper rectangle.

    Parameters
    ----------
    intervals: Tuple[Tuple[Float01, Float01], ...]
        Intervals of the segment.
    """

    intervals: Tuple[Tuple[Float01, Float01], ...]

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
        left_intervals = np.concatenate([np.atleast_2d(self.mins).T, np.atleast_2d(middles).T], axis=1)
        right_intervals = np.concatenate([np.atleast_2d(middles).T, np.atleast_2d(self.maxs).T], axis=1)
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
        """
        if not self.is_adjacent(other):
            raise ValueError("Segments must be adjacent.")

        if np.array_equal(self.maxs, other.mins):
            new_intervals = np.column_stack((self.mins, other.maxs))
        else:
            new_intervals = np.column_stack((other.mins, self.maxs))

        return Segment(intervals=new_intervals)

    def __hash__(self) -> int:
        return hash(tuple(tuple(interval) for interval in self.intervals))

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
            Check if two segments are adjacent. Segments are adjacent if they share a face,
        meaning they have identical intervals in all dimensions except one, where they touch.

        Parameters
        ----------
        other : Segment
            Segment to check for adjacency.

        Returns
        -------
        bool
            Whether the segments are adjacent.
        """

        if self.intervals_array.shape[0] != other.intervals_array.shape[0]:
            raise ValueError("Segments must have the same shape.")

        # Create a mask for dimensions where intervals differ between segments
        diff_mask = ~np.all(self.intervals_array == other.intervals_array, axis=1)
        # Count how many dimensions have different intervals
        n_differences = np.sum(diff_mask)

        # Check if the differing dimensions are adjacent
        if n_differences == 1:
            adjacent_mask = np.logical_or(
                (self.maxs[diff_mask] == other.mins[diff_mask]), (self.mins[diff_mask] == other.maxs[diff_mask])
            )
            # Segments are adjacent if exactly one dimension differs and it's adjacent
            return bool(adjacent_mask[0])
        else:
            return False


class ZoomingModel(QuantitativeModel, ABC):
    """
    This class is used to implement the zooming method. The approach is based on adaptive discretization of the
    quantitative action space. The space is represented s a hyper cube with a dimension number of dimensions.
    After each update step, the model checks if the segments are interesting or nuisance based on segment_update_factor.
    If a segment is interesting, it can be split to two segments.
    In contrast, adjacent nuisance segments can be merged based on comparison_threshold.
    The number of segments can be limited using n_max_segments.

    References
    ----------
    Multi-Armed Bandits in Metric Spaces (Kleinberg, Slivkins, and Upfal, 2008)
    https://arxiv.org/pdf/0809.4882

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Model]]
        Mapping of segments to models.
    """

    dimension: PositiveInt
    comparison_threshold: Float01 = 0.1
    segment_update_factor: Float01 = 0.1
    n_comparison_points: PositiveInt = 1000
    n_max_segments: Optional[PositiveInt] = 32
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Model]]
    _base_model: Model = PrivateAttr()
    _n_initial_segments: ClassVar = 4

    if pydantic_version == PYDANTIC_VERSION_1:

        def dict(self, **kwargs):
            d = super().dict(**kwargs)
            # Convert tuple keys to strings for serialization
            d["sub_actions"] = {str(k): v for k, v in d["sub_actions"].items()}
            return d

        def json(self, **kwargs) -> str:
            d = self.dict()
            # Convert tuple keys to strings for serialization
            return json.dumps(d, **kwargs)

    elif pydantic_version == PYDANTIC_VERSION_2:

        @field_serializer("sub_actions")
        def serialize_sub_actions(self, value):
            return {str(k): v for k, v in value.items()}

    @field_validator("sub_actions", mode="before")
    @classmethod
    def deserialize_sub_actions(cls, value):
        """
        Convert sub_actions from a dict with string keys (json representation) to tuple (object representation).
        """
        if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
            value = {cls._deserialize_sub_action_key(k): v for k, v in value.items()}

        return value

    @staticmethod
    def _deserialize_sub_action_key(key: str) -> Tuple[Tuple[Float01, Float01], ...]:
        key = eval(key)
        if isinstance(key, tuple):
            if not isinstance(key[0], tuple):  # case of dimension = 1
                key = (key,)
        elif isinstance(key, list):
            key = tuple(tuple(interval) for interval in key)
        else:
            raise ValueError(f"Invalid key type: for {key}. Expected tuple or list of lists.")
        return key

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
                raise ValueError("All segments must either have a model or miss a model.")
            self.sub_actions = dict(
                zip(self.sub_actions, [self._base_model.model_copy(deep=True) for _ in range(len(self.sub_actions))])
            )

    @property
    def segmented_actions(self) -> Dict[Segment, Optional[Model]]:
        return {Segment(intervals=segment): model for segment, model in self.sub_actions.items()}

    @abstractmethod
    def _init_base_model(self):
        """
        Initialize the base model.
        """

    @classmethod
    @validate_call
    def cold_start(
        cls,
        dimension: PositiveInt = 1,
        comparison_threshold: Float01 = 0.1,
        n_comparison_points: PositiveInt = 1000,
        n_max_segments: Optional[PositiveInt] = 32,
        **kwargs,
    ) -> Self:
        """
        Create a cold start model.

        Returns
        -------
        ZoomingModel
            Cold start model.
        """
        sub_actions = dict(zip(cls._generate_initial_segments(dimension), [None] * cls._n_initial_segments**dimension))
        return cls(
            dimension=dimension,
            comparison_threshold=comparison_threshold,
            n_comparison_points=n_comparison_points,
            n_max_segments=n_max_segments,
            sub_actions=sub_actions,
            **kwargs,
        )

    @classmethod
    def _generate_initial_segments(cls, dimension: PositiveInt) -> List[Tuple[Tuple[Float01, Float01], ...],]:
        interval_points = np.linspace(0, 1, cls._n_initial_segments + 1)
        intervals = [(interval_points[i], interval_points[i + 1]) for i in range(cls._n_initial_segments)]
        return [tuple(segment) for segment in product(intervals, repeat=dimension)]

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

    def _quantitative_update(self, quantities: List[Union[float, np.ndarray]], rewards: List[BinaryReward], **kwargs):
        """
        Update the model parameters.

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
        """
        Sort segments into three categories: interest (good), nuisance (bad), and all others (neutral).
        Segments of interest are to be splitted; adjucent nuisance segments to be merged; and reminder remain untouched.
        The segment classification is based on the rate of exploitation using self.segment_update_factor.

        Parameters
        ----------
        quantities
        segments
        rewards
        kwargs

        Returns
        -------

        """
        segments_counts = Counter(segments)
        num_segments = len(self.sub_actions)
        interest_segments = []
        nuisance_segments = []
        for segment in segments_counts:
            if segments_counts[segment] > (len(segments) / num_segments) * (1 + self.segment_update_factor):
                interest_segments.append(segment)
            elif segments_counts[segment] < (len(segments) / num_segments) * (1 - self.segment_update_factor):
                nuisance_segments.append(segment)
        interest_segments = sorted(interest_segments, key=lambda x: segments_counts[x], reverse=True)

        self._merge_adjacent_nuisance_segments(nuisance_segments, quantities, segments, rewards, **kwargs)
        self._split_segments_of_interest(interest_segments, quantities, segments, rewards, **kwargs)

    def _merge_adjacent_nuisance_segments(
        self,
        nuisance_segments: List[Segment],
        quantities: List[Union[float, np.ndarray]],
        segments: List[Segment],
        rewards: List[BinaryReward],
        **kwargs,
    ):
        """
        Merge adjacent segments that have similar performance.

        Parameters
        ----------
        nuisance_segments : List[Segment]
            List of segments to consider for merging.
        quantities : List[Union[float, np.ndarray]]
            The value associated with each action.
        segments : List[Segment]
            All segments in the model.
        rewards : List[BinaryReward]
            The reward for each sample.
        """
        i = 0
        while i < len(nuisance_segments) - 1:
            segment = nuisance_segments[i]
            j = i + 1
            while j < len(nuisance_segments):
                other_segment = nuisance_segments[j]
                if segment.is_adjacent(other_segment) and self.is_similar_performance(segment, other_segment):
                    del self.sub_actions[segment.intervals]
                    del self.sub_actions[other_segment.intervals]
                    nuisance_segments.remove(segment)
                    nuisance_segments.remove(other_segment)
                    merged_segment = segment + other_segment
                    self.sub_actions[merged_segment.intervals] = self._base_model.model_copy(deep=True)
                    filtered_quantities, filtered_rewards, filtered_kwargs = self._filter_by_segment(
                        merged_segment, quantities, segments, rewards, **kwargs
                    )
                    self._map_and_update_segment_models(filtered_quantities, filtered_rewards, **filtered_kwargs)
                    break
                j += 1
            i += 1

    def _split_segments_of_interest(
        self,
        interest_segments: List[Segment],
        quantities: List[Union[float, np.ndarray]],
        segments: List[Segment],
        rewards: List[BinaryReward],
        **kwargs,
    ):
        """
        Split segments of interest into two sub-segments if possible.

        Parameters
        ----------
        interest_segments : List[Segment]
            List of segments to consider for splitting.
        quantities : List[Union[float, np.ndarray]]
            The value associated with each action.
        segments : List[Segment]
            All segments in the model.
        rewards : List[BinaryReward]
            The reward for each sample.
        """
        i = 0
        while i < len(interest_segments) - 1 and (
            self.n_max_segments is None or len(self.sub_actions) < self.n_max_segments
        ):
            best_segment = interest_segments[i]
            del self.sub_actions[best_segment.intervals]
            sub_best_segments = best_segment.split()
            self.sub_actions[sub_best_segments[0].intervals] = self._base_model.model_copy(deep=True)
            self.sub_actions[sub_best_segments[1].intervals] = self._base_model.model_copy(deep=True)
            filtered_quantities, filtered_rewards, filtered_kwargs = self._filter_by_segment(
                best_segment, quantities, segments, rewards, **kwargs
            )
            self._map_and_update_segment_models(filtered_quantities, filtered_rewards, **filtered_kwargs)
            i += 1

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

    def _reset(self):
        self.sub_actions = dict(
            zip(
                self._generate_initial_segments(self.dimension),
                [self._base_model.model_copy(deep=True) for _ in range(self._n_initial_segments**self.dimension)],
            )
        )


class BaseSmabZoomingModel(ZoomingModel, ABC):
    """
    Zooming model for sMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Beta]]
        Mapping of segments to Beta models.
    """

    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Beta]]

    def _init_base_model(self):
        """
        Initialize the base model.
        """
        self._base_model = Beta()

    @validate_call
    def _quantitative_update(
        self,
        quantities: Optional[List[Union[float, List[float], None]]],
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]],
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: Union[List[BinaryReward], List[List[BinaryReward]]]
            The reward for each sample.
        """
        super()._quantitative_update(quantities, rewards)

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
        rewards = np.array(rewards)
        for segment in set(segments):
            rewards_of_segment = [r for r, s in zip(rewards, segments) if s == segment]
            self.sub_actions[segment.intervals].update(rewards=rewards_of_segment)


class SmabZoomingModel(BaseSmabZoomingModel):
    """
    Zooming model for sMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Beta]]
        Mapping of segments to Beta models.
    """


class SmabZoomingModelCC(BaseSmabZoomingModel, QuantitativeModelCC):
    """
    Zooming model for sMAB with cost control.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Beta]]
        Mapping of segments to Beta models.
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """


class BaseCmabZoomingModel(ZoomingModel, ABC):
    """
    Zooming model for CMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[BayesianNeuralNetwork]]
        Mapping of segments to Bayesian Logistic Regression models.
    base_model_cold_start_kwargs: Dict[str, Any]
        Keyword arguments for the base model cold start.
    """

    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[BayesianNeuralNetwork]]
    base_model_cold_start_kwargs: Dict[str, Any]

    @field_validator("base_model_cold_start_kwargs", mode="before")
    @classmethod
    def validate_n_features(cls, value):
        if "n_features" not in value:
            raise KeyError("n_features must be in base_model_cold_start_kwargs.")
        return value

    def _init_base_model(self):
        """
        Initialize the base model.
        """
        self._base_model = BayesianNeuralNetwork.cold_start(**self.base_model_cold_start_kwargs)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _quantitative_update(
        self,
        quantities: Optional[List[Union[float, List[float], None]]],
        rewards: List[BinaryReward],
        context: ArrayLike,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Optional[List[Union[float, List[float], None]]
            The value associated with each action. If none, the value is not used, i.e. non-quantitative action.
        rewards: List[BinaryReward]
            The reward for each sample.
        context : ArrayLike
            Context for each sample
        """
        super()._quantitative_update(quantities, rewards, context=context)

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
        context = np.array(context)
        for segment in set(segments):
            rewards_of_segment = [r for r, s in zip(rewards, segments) if s == segment]
            context_of_segment = context[[s == segment for s in segments]]
            if rewards_of_segment:
                self.sub_actions[segment.intervals].update(rewards=rewards_of_segment, context=context_of_segment)


class CmabZoomingModel(BaseCmabZoomingModel):
    """
    Zooming model for CMAB.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[BayesianNeuralNetwork]]
        Mapping of segments to Bayesian Logistic Regression models.
    base_model_cold_start_kwargs: Dict[str, Any]
        Keyword arguments for the base model cold start.
    """


class CmabZoomingModelCC(BaseCmabZoomingModel, QuantitativeModelCC):
    """
    Zooming model for CMAB with cost control.

    Parameters
    ----------
    dimension: PositiveInt
        Number of parameters of the model.
    comparison_threshold: Float01
        Comparison threshold.
    segment_update_factor: Float01
        Segment update factor. If the number of samples in a segment is more than the average number of samples in all
        segments by this factor, the segment is considered interesting. If the number of samples in a segment is less
        than the average number of samples in all segments by this factor, the segment is considered a nuisance.
        Interest segments can be split, while nuisance segments can be merged.
    n_comparison_points: PositiveInt
        Number of comparison points.
    n_max_segments: PositiveInt
        Maximum number of segments.
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[BayesianNeuralNetwork]]
        Mapping of segments to Bayesian Logistic Regression models.
    base_model_cold_start_kwargs: Dict[str, Any]
        Keyword arguments for the base model cold start.
    cost: Callable[[Union[float, NonNegativeFloat]], NonNegativeFloat]
        Cost associated to the Beta distribution.
    """
