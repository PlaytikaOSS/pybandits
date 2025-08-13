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

import functools
import json
from copy import deepcopy
from typing import List, Union

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import pybandits
from pybandits.base import BinaryReward
from pybandits.model import Beta
from pybandits.pydantic_version_compatibility import NonNegativeFloat
from pybandits.quantitative_model import (
    CmabZoomingModel,
    Segment,
    SmabZoomingModel,
    SmabZoomingModelCC,
    ZoomingModel,
)
from tests.utils import FakeApproximation


def tuple_of_tuples_strategy(n, m, elements=st.floats(min_value=0, max_value=1)):
    return st.tuples(*[st.tuples(*[elements for _ in range(m)]) for _ in range(n)])


# Create segment with valid intervals array of shape (n,2)
@given(tuple_of_tuples_strategy(2, 2))
def test_create_valid_segment(intervals):
    segment = Segment(intervals=intervals)
    assert isinstance(segment, Segment)
    assert len(segment.intervals) == 2
    assert all(len(interval) == 2 for interval in segment.intervals)


# Access minimum and maximum quantities via mins and maxs properties
@given(tuple_of_tuples_strategy(3, 2))
def test_mins_maxs_properties(intervals):
    segment = Segment(intervals=intervals)
    assert np.all(segment.mins == np.array([interval[0] for interval in intervals]))
    assert np.all(segment.maxs == np.array([interval[1] for interval in intervals]))


# Create segment with empty intervals array
def test_create_empty_segment():
    empty_intervals = np.empty((0, 2))
    segment = Segment(intervals=empty_intervals)
    assert len(segment.intervals) == 0
    assert len(segment.mins) == 0
    assert len(segment.maxs) == 0


# Create segment with invalid interval shape
@given(arrays(np.float64, shape=(2, 3), elements=st.floats(min_value=0, max_value=100)))
def test_invalid_interval_shape(intervals):
    with pytest.raises(ValueError, match="Intervals must have shape .n, 2."):
        Segment(intervals=intervals)


# Add non-adjacent segments
def test_add_nonadjacent_segments():
    seg1 = Segment(intervals=np.array([[0, 0.1], [0, 0.1]]))
    seg2 = Segment(intervals=np.array([[0.2, 0.5], [0.15, 0.2]]))
    with pytest.raises(ValueError, match="Segments must be adjacent."):
        seg1 + seg2


class DummyZoomingModel(ZoomingModel):
    def _init_base_model(self):
        self._base_model = Beta()

    def _inner_update(self, segments: List[Segment], rewards: List[BinaryReward], **kwargs):
        pass


# Model initialization with valid parameters creates correct number of segments
@given(dimension=st.integers(min_value=1, max_value=3))
def test_init_creates_correct_segments(dimension):
    model = DummyZoomingModel.cold_start(dimension=dimension, n_max_segments=None)
    expected_segments = DummyZoomingModel._n_initial_segments**dimension
    assert len(model.sub_actions) == expected_segments


# Update method correctly processes rewards and quantities for existing segments
@given(
    st.integers(min_value=2, max_value=5).flatmap(
        lambda size: st.tuples(
            st.lists(st.integers(min_value=0, max_value=1), min_size=size, max_size=size),
            st.lists(st.floats(min_value=0, max_value=1), min_size=size, max_size=size),
        )
    )
)
def test_update_processes_rewards_correctly(data):
    rewards, quantities = data
    model = DummyZoomingModel.cold_start(dimension=1)
    model.update(quantities=quantities, rewards=rewards)
    assert model.n_successes != 1 or model.n_failures != 1


# Best performing segment gets split when below max segments limit
def test_best_segment_splits():
    model = DummyZoomingModel.cold_start(dimension=1, n_max_segments=8)
    quantities = [0.25, 0.75]
    rewards = [1, 0]
    model.update(quantities=quantities, rewards=rewards)
    assert len(model.sub_actions) == 5
    assert any(interval[0][1] - interval[0][0] < 0.25 for interval in model.sub_actions.keys())  # split
    assert any(np.isclose(interval[0][1] - interval[0][0], 0.25) for interval in model.sub_actions.keys())


# Adjacent segments with similar performance get merged correctly
def test_similar_segments_merge():
    model = DummyZoomingModel.cold_start(dimension=1, comparison_threshold=0.5)
    initial_segmented_actions = model.sub_actions.copy()
    quantities = [0.75, 0.75]
    rewards = [1, 1]
    model.update(quantities=quantities, rewards=rewards)
    assert initial_segmented_actions.keys() != model.sub_actions
    assert len(model.sub_actions) == 4


# Sample_proba returns valid probability for each segment
def test_sample_proba_returns_valid_probabilities(n_samples=100):
    model = DummyZoomingModel.cold_start(dimension=1)
    probs = model.sample_proba(n_samples=n_samples)
    assert all(len(prob) == len(model.sub_actions) for prob in probs)
    assert len(probs) == n_samples
    assert all(0 <= prob[1] <= 1 for sample in probs for prob in sample)
    assert all(0 <= v <= 1 for sample in probs for prob in sample for v in prob[0])


# Update with empty rewards/quantities list
def test_update_with_empty_lists():
    model = DummyZoomingModel.cold_start(dimension=1)
    initial_segments = len(model.sub_actions)
    model.update(quantities=[], rewards=[])
    assert len(model.sub_actions) == initial_segments


# Update when at maximum number of segments
def test_update_at_max_segments():
    model = DummyZoomingModel.cold_start(dimension=1, n_max_segments=4)
    quantities = [0.5]
    rewards = [1]
    model.update(quantities=quantities, rewards=rewards)
    assert len(model.sub_actions) <= model.n_max_segments


# Merging segments when only 2 segments remain
def test_merge_with_two_segments():
    model = DummyZoomingModel.cold_start(dimension=1, comparison_threshold=1.0)
    quantities = [0.25, 0.75]
    rewards = [1, 1]
    model.update(quantities=quantities, rewards=rewards)
    assert len(model.sub_actions) >= 1


# Comparing non-adjacent segments for merging
def test_non_adjacent_segments_comparison():
    model = DummyZoomingModel.cold_start(dimension=1)
    segments = list(model.segmented_actions.keys())
    non_adjacent = [segments[0], segments[2]]
    assert not non_adjacent[0].is_adjacent(non_adjacent[1])


@given(arrays(np.float64, shape=(2, 2), elements=st.floats(min_value=0, max_value=1)))
def test_adjacent_segments_different_shape(intervals):
    segment1 = Segment(intervals=intervals)
    segment2 = Segment(intervals=np.array([[0.5, 1.0]]))
    with pytest.raises(ValueError, match="Segments must have the same shape."):
        segment1.is_adjacent(segment2)


def test_segments_adjacent_in_one_dimension():
    # Segments touching on the first dimension
    segment1 = Segment(intervals=np.array([[0, 0.5], [0, 0.5]]))
    segment2 = Segment(intervals=np.array([[0.5, 1], [0, 0.5]]))
    assert segment1.is_adjacent(segment2)
    assert segment2.is_adjacent(segment1)

    # Segments touching on the second dimension
    segment3 = Segment(intervals=np.array([[0, 0.5], [0, 0.5]]))
    segment4 = Segment(intervals=np.array([[0, 0.5], [0.5, 1]]))
    assert segment3.is_adjacent(segment4)
    assert segment4.is_adjacent(segment3)


def test_segments_not_adjacent():
    # Segments different in multiple dimensions
    segment1 = Segment(intervals=np.array([[0, 0.5], [0, 0.5]]))
    segment2 = Segment(intervals=np.array([[0.5, 1], [0.5, 1]]))
    assert not segment1.is_adjacent(segment2)
    assert not segment2.is_adjacent(segment1)

    # Segments not touching
    segment3 = Segment(intervals=np.array([[0, 0.4], [0, 0.5]]))
    segment4 = Segment(intervals=np.array([[0.5, 1], [0, 0.5]]))
    assert not segment3.is_adjacent(segment4)
    assert not segment4.is_adjacent(segment3)


def test_identical_segments_not_adjacent():
    segment = Segment(intervals=np.array([[0, 0.5], [0, 0.5]]))
    assert not segment.is_adjacent(segment)


@given(
    st.integers(min_value=1, max_value=5),
    st.tuples(st.tuples(st.floats(min_value=0, max_value=0.49), st.floats(min_value=0.5, max_value=1))),
)
def test_higher_dimension_adjacency(dimension, base_interval):
    # Create two segments that are adjacent in exactly one dimension
    intervals1 = np.tile(base_interval, (dimension, 1))
    intervals2 = np.tile(base_interval, (dimension, 1))
    intervals2[0, 0] = intervals1[0, 1]  # Make them adjacent in first dimension

    segment1 = Segment(intervals=intervals1)
    segment2 = Segment(intervals=intervals2)
    assert segment1.is_adjacent(segment2)


# Values that fall on segment boundaries
@given(st.integers(min_value=4, max_value=8))
def test_boundary_values(n_segments):
    model = DummyZoomingModel.cold_start(dimension=1, n_max_segments=n_segments)
    boundary = 1.0 / n_segments
    quantities = [boundary]
    rewards = [1]
    model.update(quantities=quantities, rewards=rewards)
    mapped_segments = model._map_values_to_segments(quantities)
    assert len(mapped_segments) >= 1


# Test SmabZoomingModel initialization with valid parameters
@given(dimension=st.integers(min_value=1, max_value=3), n_max_segments=st.just(None))
def test_initializes_smab_zooming_model_correctly(dimension, n_max_segments):
    model = SmabZoomingModel.cold_start(dimension=dimension, n_max_segments=n_max_segments)
    expected_segments = SmabZoomingModel._n_initial_segments**dimension
    assert len(model.segmented_actions) == expected_segments


# Test SmabZoomingModel update with valid rewards and quantities
@given(
    rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=5),
    quantities=st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=5),
    dimension=st.just(1),
)
def test_updates_smab_zooming_model_correctly(rewards, quantities, dimension):
    model = SmabZoomingModel.cold_start(dimension=dimension)
    initial_segments = deepcopy(model.segmented_actions)
    model.update(quantities=quantities, rewards=rewards)
    assert model.segmented_actions != initial_segments


# Test SmabZoomingModel sample_proba returns valid probabilities
def test_sample_proba_returns_valid_probabilities_smab(dimension=1, n_samples=100):
    model = SmabZoomingModel.cold_start(dimension=dimension)
    probas = model.sample_proba(n_samples=n_samples)
    for proba in probas:
        for (q,), p in proba:
            assert 0 <= q <= 1
            assert 0 <= p <= 1


# Test CmabZoomingModel initialization with valid parameters
@given(dimension=st.integers(min_value=1, max_value=3), n_max_segments=st.just(None))
def test_initializes_cmab_zooming_model_correctly(dimension, n_max_segments):
    model = CmabZoomingModel.cold_start(
        dimension=dimension, n_max_segments=n_max_segments, base_model_cold_start_kwargs={"n_features": 1}
    )
    expected_segments = CmabZoomingModel._n_initial_segments**dimension
    assert len(model.segmented_actions) == expected_segments


# Test CmabZoomingModel update with valid rewards, quantities, and context
@given(
    rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=5),
    quantities=st.lists(st.floats(min_value=0, max_value=1), min_size=5, max_size=5),
    context=arrays(np.float64, shape=(5, 1), elements=st.floats(min_value=0, max_value=1)),
    dimension=st.just(1),
    n_features=st.just(1),
)
def test_updates_cmab_zooming_model_correctly(rewards, quantities, context, dimension, n_features, monkeymodule):
    model = CmabZoomingModel.cold_start(dimension=dimension, base_model_cold_start_kwargs={"n_features": n_features})
    initial_segments = deepcopy(model.segmented_actions)
    monkeymodule.setattr(
        pybandits.model,
        "fit",
        lambda *args, **kwargs: FakeApproximation(n_features=n_features),
    )
    monkeymodule.setattr(
        pybandits.model,
        "sample",
        FakeApproximation(n_features=n_features).sample,
    )
    model.update(quantities=quantities, rewards=rewards, context=context)
    assert model.segmented_actions != initial_segments


# Test CmabZoomingModel sample_proba returns valid probabilities
@given(
    context=arrays(np.float64, shape=(5, 1), elements=st.floats(min_value=0, max_value=1)),
    dimension=st.just(1),
    n_features=st.just(1),
)
def test_sample_proba_returns_valid_probabilities_cmab(context, dimension, n_features):
    model = CmabZoomingModel.cold_start(dimension=dimension, base_model_cold_start_kwargs={"n_features": n_features})
    probas = model.sample_proba(context=context)
    for proba in probas:
        for (q,), (p, _) in proba:
            assert 0 <= q <= 1
            assert 0 <= p <= 1


########################################################################################################################


# QuantitativeModelCC tests via SmabZoomingModelCC


def simple_cost(value: float) -> float:
    return value * 2


def complex_cost(value: Union[float, NonNegativeFloat], factor: float = 1.0) -> NonNegativeFloat:
    return value * factor


partial_cost = functools.partial(complex_cost, factor=1.5)


@pytest.mark.parametrize(
    "cost_function",
    [
        simple_cost,
        complex_cost,
        partial_cost,
    ],
)
def test_cost_serialization_deserialization(cost_function):
    """Test serialization and deserialization of cost field with different callables."""

    # Create model with the test callable as cost
    model = SmabZoomingModelCC.cold_start(cost=cost_function)
    serialized = model._apply_version_adjusted_method("model_dump_json", "json")
    serialized_dict = json.loads(serialized)

    assert "cost" in serialized_dict
    assert isinstance(serialized_dict["cost"], str)

    deserialized_model = SmabZoomingModelCC.model_validate_json(serialized)
    # Check callable was properly restored
    assert callable(deserialized_model.cost)

    reserialized = deserialized_model._apply_version_adjusted_method("model_dump_json", "json")
    assert reserialized == serialized
