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

import ast
from abc import ABC
from collections import Counter, defaultdict
from itertools import product
from typing import Any, ClassVar, Dict, List, Optional, Self, Tuple, Union

import numpy as np
from pydantic import (
    Field,
    PositiveInt,
    field_serializer,
    field_validator,
    validate_call,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta

from pybandits.base import (
    BinaryReward,
    Float01,
    Probability,
    QuantitativeProbability,
)
from pybandits.model import Beta
from pybandits.quantitative_model.base import QuantitativeModel, QuantitativeModelCC, QuantitativeModelDP
from pybandits.quantitative_model.segment import Segment


class BaseZooming(QuantitativeModel, ABC):
    """
    Zooming model for sMAB. The approach is based on adaptive discretization of the
    quantitative action space. The space is represented as a hyper cube with a dimension number of dimensions.
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
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Beta]]
        Mapping of segments to Beta models.
    base_model: Beta
        Template Beta model copied for every new segment. Carries the per-segment configuration
        (e.g. ``decay_factor``); built from ``base_model_cold_start_kwargs`` at cold start.
    """

    dimension: PositiveInt
    comparison_threshold: Float01 = 0.1
    segment_update_factor: Float01 = 0.1
    n_comparison_points: PositiveInt = 1000
    n_max_segments: Optional[PositiveInt] = 32
    sub_actions: Dict[Tuple[Tuple[Float01, Float01], ...], Optional[Beta]]
    base_model: Beta = Field(default_factory=Beta)
    _n_initial_segments: ClassVar = 4
    _transfer_learned_keys: ClassVar[Tuple[str, ...]] = ("sub_actions",)

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
        try:
            key = ast.literal_eval(key)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid sub-action key: {key!r}. Expected a tuple or list literal.") from e
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
        segment_models_types = set(type(model) if model is not None else None for model in self.sub_actions.values())
        if None in segment_models_types:
            if len(segment_models_types) > 1:
                raise ValueError("All segments must either have a model or miss a model.")
            self.sub_actions = dict(
                zip(self.sub_actions, [self.base_model.model_copy(deep=True) for _ in range(len(self.sub_actions))])
            )

    @property
    def segmented_actions(self) -> Dict[Segment, Optional[Beta]]:
        return {Segment(intervals=segment): model for segment, model in self.sub_actions.items()}

    @classmethod
    @validate_call
    def cold_start(
        cls,
        dimension: PositiveInt = 1,
        comparison_threshold: Float01 = 0.1,
        n_comparison_points: PositiveInt = 1000,
        n_max_segments: Optional[PositiveInt] = 32,
        base_model_cold_start_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Self:
        """
        Create a cold start model.

        Parameters
        ----------
        base_model_cold_start_kwargs : Optional[Dict[str, Any]]
            Keyword arguments forwarded to the per-segment Beta model (e.g. ``decay_factor``).

        Returns
        -------
        BaseZooming
            Cold start model.
        """
        sub_actions = dict(zip(cls._generate_initial_segments(dimension), [None] * cls._n_initial_segments**dimension))
        return cls(
            dimension=dimension,
            comparison_threshold=comparison_threshold,
            n_comparison_points=n_comparison_points,
            n_max_segments=n_max_segments,
            sub_actions=sub_actions,
            base_model=Beta(**(base_model_cold_start_kwargs or {})),
            **kwargs,
        )

    @classmethod
    def _generate_initial_segments(cls, dimension: PositiveInt) -> List[Tuple[Tuple[Float01, Float01], ...],]:
        interval_points = np.linspace(0, 1, cls._n_initial_segments + 1)
        intervals = [(interval_points[i], interval_points[i + 1]) for i in range(cls._n_initial_segments)]
        return [tuple(segment) for segment in product(intervals, repeat=dimension)]

    def sample_proba(self, n_samples: PositiveInt, rng: np.random.Generator) -> List[QuantitativeProbability]:
        """
        Sample probability functions from the model.

        Parameters
        ----------
        rng : numpy.random.Generator
            Central numpy random generator provided by the MAB.

        Returns
        -------
        List[QuantitativeProbability]
            A list of functions that evaluate probability at any given location.
        """
        # Get sampled probabilities from each segment model
        segment_probabilities = {}
        for segment, model in self.segmented_actions.items():
            segment_probabilities[segment] = model.sample_proba(n_samples=n_samples, rng=rng)
        return self._to_quantitative_probabilities(segment_probabilities)

    def _to_quantitative_probabilities(
        self, segment_probabilities: Dict[Segment, List[Probability]]
    ) -> List[QuantitativeProbability]:
        """
        Convert the segment probabilities to quantitative probabilities.

        Parameters
        ----------
        segment_probabilities : Dict[Segment, List[Probability]]
            The probabilities of each segment.

        Returns
        -------
        List[QuantitativeProbability]
            The quantitative probabilities.
        """
        result = []
        max_samples = max(len(probas) for probas in segment_probabilities.values())
        for sample_idx in range(max_samples):

            def create_probability_function(sample_idx: int) -> QuantitativeProbability:
                def probability_function(quantity: np.ndarray) -> Probability:
                    """
                    Evaluate probability at the given quantity.
                    """
                    for segment in segment_probabilities.keys():
                        if quantity in segment:
                            segment_probas_for_segment = segment_probabilities[segment]
                            return segment_probas_for_segment[sample_idx]
                    return 0.0

                return probability_function

            result.append(create_probability_function(sample_idx))
        return result

    @validate_call
    def _quantitative_update(self, quantities: Union[List[float], List[List[float]]], rewards: List[BinaryReward]):
        """
        Update the model parameters.

        Parameters
        ----------
        quantities : Union[List[float], List[List[float]]],
            The value associated with each action.
        rewards: List[BinaryReward]
            The reward for each sample.
        """

        segments = self._map_and_update_segment_models(quantities, rewards)
        self._update_segmentation(quantities, segments, rewards)

    def _map_and_update_segment_models(
        self, quantities: Union[List[float], List[List[float]]], rewards: List[BinaryReward]
    ) -> List[Segment]:
        """
        Map and update the segment models.

        Parameters
        ----------
        quantities : Union[List[float], List[List[float]]]
            The value associated with each action.
        rewards: List[BinaryReward]
            The reward for each sample.

        Returns
        -------
        List[Segment]
            Segments to update.
        """
        segments = self._map_values_to_segments(quantities)
        self._inner_update(segments, rewards)
        return segments

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
        rewards_by_segment = defaultdict(list)
        for segment, reward in zip(segments, rewards):
            rewards_by_segment[segment].append(reward)

        for segment, rewards_of_segment in rewards_by_segment.items():
            self.sub_actions[segment.intervals].update(rewards=rewards_of_segment)

    def _map_values_to_segments(
        self,
        quantities: Union[List[float], List[List[float]]],
    ) -> List[Segment]:
        segments = [segment for value in quantities for segment in self.segmented_actions.keys() if value in segment]
        return segments

    def _update_segmentation(
        self,
        quantities: Union[List[float], List[List[float]]],
        segments: List[Segment],
        rewards: List[BinaryReward],
    ):
        """
        Sort segments into three categories: interest (good), nuisance (bad), and all others (neutral).
        Segments of interest are to be split; adjacent nuisance segments to be merged; and reminder remain untouched.
        The segment classification is based on the rate of exploitation using self.segment_update_factor.

        Parameters
        ----------
        quantities : Union[List[float], List[List[float]]]
            The value associated with each action.
        segments : List[Segment]
            All segments in the model.
        rewards : List[BinaryReward]
            Rewards for update.
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

        self._merge_adjacent_nuisance_segments(nuisance_segments, quantities, segments, rewards)
        self._split_segments_of_interest(interest_segments, quantities, segments, rewards)

    def _merge_adjacent_nuisance_segments(
        self,
        nuisance_segments: List[Segment],
        quantities: Union[List[float], List[List[float]]],
        segments: List[Segment],
        rewards: List[BinaryReward],
    ):
        """
        Merge adjacent segments that have similar performance.

        Parameters
        ----------
        nuisance_segments : List[Segment]
            List of segments to consider for merging.
        quantities : Union[List[float], List[List[float]]]
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
                    self.sub_actions[merged_segment.intervals] = self.base_model.model_copy(deep=True)
                    filtered_quantities, filtered_rewards = self._filter_by_segment(
                        [segment, other_segment], quantities, segments, rewards
                    )
                    self._map_and_update_segment_models(filtered_quantities, filtered_rewards)
                    break
                j += 1
            i += 1

    def _split_segments_of_interest(
        self,
        interest_segments: List[Segment],
        quantities: Union[List[float], List[List[float]]],
        segments: List[Segment],
        rewards: List[BinaryReward],
    ):
        """
        Split segments of interest into two sub-segments if possible.

        Parameters
        ----------
        interest_segments : List[Segment]
            List of segments to consider for splitting.
        quantities : Union[List[float], List[List[float]]]
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
            self.sub_actions[sub_best_segments[0].intervals] = self.base_model.model_copy(deep=True)
            self.sub_actions[sub_best_segments[1].intervals] = self.base_model.model_copy(deep=True)
            filtered_quantities, filtered_rewards = self._filter_by_segment(best_segment, quantities, segments, rewards)
            self._map_and_update_segment_models(filtered_quantities, filtered_rewards)
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
        reference_segment: Union[Segment, List[Segment]],
        quantities: Union[List[float], List[List[float]]],
        segments: List[Segment],
        rewards: List[BinaryReward],
    ) -> Tuple[Union[List[float], List[List[float]]], List[BinaryReward]]:
        """
        Filter and update the segments models.

        Parameters
        ----------
        reference_segment : Union[Segment, List[Segment]]
            Reference segment(s) to filter upon. Pass a list when multiple source
            segments should be included (e.g. after a merge).
        segments : List[Segment]
            Segments to filter.
        quantities : Union[List[float], List[List[float]]]
            Values to filter.
        rewards : List[BinaryReward]
            Rewards to filter.

        Returns
        -------
        filtered_values : Union[List[float], List[List[float]]]
            Filtered quantities.
        filtered_rewards : List[BinaryReward]
            Filtered rewards.
        """
        reference_segments = reference_segment if isinstance(reference_segment, list) else [reference_segment]
        filtered_values_rewards = [
            (value, reward)
            for value, reward, segment in zip(quantities, rewards, segments)
            if segment in reference_segments
        ]
        if filtered_values_rewards:
            filtered_values, filtered_rewards = zip(*filtered_values_rewards)
            return list(filtered_values), list(filtered_rewards)
        return [], []

    def _reset(self):
        self.sub_actions = dict(
            zip(
                self._generate_initial_segments(self.dimension),
                [self.base_model.model_copy(deep=True) for _ in range(self._n_initial_segments**self.dimension)],
            )
        )


class Zooming(BaseZooming):
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


class ZoomingCC(BaseZooming, QuantitativeModelCC):
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


class ZoomingDP(BaseZooming, QuantitativeModelDP):
    """
    Zooming model for sMAB with dynamic pricing.

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
    price: Callable[[Union[float, np.ndarray]], NonNegativeFloat]
        Price associated to the Beta distribution.
    """
