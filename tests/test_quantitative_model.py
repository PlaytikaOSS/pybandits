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
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pydantic import NonNegativeFloat

import pybandits
from pybandits.base import BinaryReward, QuantitativeProbability, UnifiedProbability
from pybandits.model import Beta
from pybandits.quantitative_model import (
    BaseZooming,
    QuantitativeBayesianNeuralNetwork,
    QuantitativeBayesianNeuralNetworkCC,
    QuantitativeBayesianNeuralNetworkDP,
    Segment,
    Zooming,
    ZoomingCC,
    ZoomingDP,
)
from tests.utils import mock_update


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


class DummyZooming(BaseZooming):
    cost: Callable[[float | NonNegativeFloat], NonNegativeFloat] | None = None

    def _init_base_model(self):
        self._base_model = Beta()

    def _inner_update(self, segments: list[Segment], rewards: list[BinaryReward], **kwargs):
        pass

    def _to_quantitative_probabilities(
        self, segment_probabilities: dict[Segment, list[UnifiedProbability]]
    ) -> list[QuantitativeProbability]:
        max_samples = max(len(probas) for probas in segment_probabilities.values())
        return [lambda x: np.random.uniform(0, 1) for _ in range(max_samples)]


# Model initialization with valid parameters creates correct number of segments
@given(dimension=st.integers(min_value=1, max_value=3))
def test_init_creates_correct_segments(dimension):
    model = DummyZooming.cold_start(dimension=dimension, n_max_segments=None)
    expected_segments = DummyZooming._n_initial_segments**dimension
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
    model = DummyZooming.cold_start(dimension=1)
    model.update(quantities=quantities, rewards=rewards)
    assert model.n_successes != 1 or model.n_failures != 1


# Best performing segment gets split when below max segments limit
def test_best_segment_splits():
    model = DummyZooming.cold_start(dimension=1, n_max_segments=8)
    quantities = [0.25, 0.75]
    rewards = [1, 0]
    model.update(quantities=quantities, rewards=rewards)
    assert len(model.sub_actions) == 5
    assert any(interval[0][1] - interval[0][0] < 0.25 for interval in model.sub_actions.keys())  # split
    assert any(np.isclose(interval[0][1] - interval[0][0], 0.25) for interval in model.sub_actions.keys())


# Adjacent segments with similar performance get merged correctly
def test_similar_segments_merge():
    model = DummyZooming.cold_start(dimension=1, comparison_threshold=0.5)
    initial_segmented_actions = model.sub_actions.copy()
    quantities = [0.75, 0.75]
    rewards = [1, 1]
    model.update(quantities=quantities, rewards=rewards)
    assert initial_segmented_actions.keys() != model.sub_actions
    assert len(model.sub_actions) == 4


# Sample_proba returns valid probability functions
def test_sample_proba_returns_valid_probabilities(rng, n_samples=100, test_locations=((0.1,), (0.5,), (0.9,))):
    model = DummyZooming.cold_start(dimension=1)
    prob_functions = model.sample_proba(n_samples=n_samples, rng=rng)
    assert len(prob_functions) == n_samples

    # Test that each function is callable and returns valid probabilities

    for prob_func in prob_functions:
        assert callable(prob_func)
        for location in test_locations:
            prob = prob_func(location)
            assert 0 <= prob <= 1


# Same-seed RNG must yield identical sampled probabilities; different seeds must differ.
@given(
    dimension=st.integers(min_value=1, max_value=3),
    n_samples=st.integers(min_value=10, max_value=30),
    seeds=st.lists(st.integers(min_value=0, max_value=2**31 - 1), min_size=2, max_size=2, unique=True),
    n_locations=st.just(10),
)
def test_sample_proba_reproducible_with_same_rng(rng, dimension, n_samples, seeds, n_locations):
    model = Zooming.cold_start(dimension=dimension, n_max_segments=None)
    locations = [tuple([q] * dimension) for q in rng.uniform(size=n_locations)]

    def draw(seed):
        functions = model.sample_proba(n_samples=n_samples, rng=np.random.default_rng(seed=seed))
        return [[fn(loc) for loc in locations] for fn in functions]

    seed_a, seed_b = seeds
    assert draw(seed_a) == draw(seed_a)
    assert draw(seed_a) != draw(seed_b)


# Update with empty rewards/quantities list
def test_update_with_empty_lists():
    model = DummyZooming.cold_start(dimension=1)
    initial_segments = len(model.sub_actions)
    model.update(quantities=[], rewards=[])
    assert len(model.sub_actions) == initial_segments


# Update when at maximum number of segments
def test_update_at_max_segments():
    model = DummyZooming.cold_start(dimension=1, n_max_segments=4)
    quantities = [0.5]
    rewards = [1]
    model.update(quantities=quantities, rewards=rewards)
    assert len(model.sub_actions) <= model.n_max_segments


# Merging segments when only 2 segments remain
def test_merge_with_two_segments():
    model = DummyZooming.cold_start(dimension=1, comparison_threshold=1.0)
    quantities = [0.25, 0.75]
    rewards = [1, 1]
    model.update(quantities=quantities, rewards=rewards)
    assert len(model.sub_actions) >= 1


# Comparing non-adjacent segments for merging
def test_non_adjacent_segments_comparison():
    model = DummyZooming.cold_start(dimension=1)
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
    model = DummyZooming.cold_start(dimension=1, n_max_segments=n_segments)
    boundary = 1.0 / n_segments
    quantities = [boundary]
    rewards = [1]
    model.update(quantities=quantities, rewards=rewards)
    mapped_segments = model._map_values_to_segments(quantities)
    assert len(mapped_segments) >= 1


# Test Zooming initialization with valid parameters
@given(dimension=st.integers(min_value=1, max_value=3), n_max_segments=st.just(None))
def test_initializes_smab_zooming_model_correctly(dimension, n_max_segments):
    model = Zooming.cold_start(dimension=dimension, n_max_segments=n_max_segments)
    expected_segments = Zooming._n_initial_segments**dimension
    assert len(model.segmented_actions) == expected_segments


# Test Zooming update with valid rewards and quantities
@given(
    rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=5),
    dimension=st.just(1),
)
def test_updates_smab_zooming_model_correctly(rng, rewards, dimension):
    model = Zooming.cold_start(dimension=dimension)
    initial_segments = deepcopy(model.segmented_actions)
    quantities = rng.uniform(0, 1, size=len(rewards)).tolist()
    model.update(quantities=quantities, rewards=rewards)
    assert model.segmented_actions != initial_segments


# Test Zooming sample_proba returns valid probability functions
def test_sample_proba_returns_valid_probabilities_smab(
    rng, dimension=1, n_samples=100, test_locations=((0.1,), (0.5,), (0.9,))
):
    model = Zooming.cold_start(dimension=dimension)
    prob_functions = model.sample_proba(n_samples=n_samples, rng=rng)
    assert len(prob_functions) == n_samples

    # Test that each function is callable and returns valid probabilities
    for prob_func in prob_functions:
        assert callable(prob_func)
        for location in test_locations:
            prob = prob_func(location)
            assert 0 <= prob <= 1


# Test  QuantitativeBayesianNeuralNetwork initialization with valid parameters
@given(
    dimension=st.integers(min_value=1, max_value=3),
    n_features=st.integers(min_value=1, max_value=2),
)
def test_initializes_cmab_quantitative_model_correctly(dimension, n_features):
    model = QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=n_features)
    assert model.dimension == dimension
    assert model.bnn.input_dim == dimension + n_features

    # Also verify that categorical feature_config is accepted
    categorical_features = {0: 3}
    model_emb = QuantitativeBayesianNeuralNetwork.cold_start(
        dimension=dimension, categorical_features=categorical_features
    )
    assert model_emb.dimension == dimension
    assert model_emb.bnn.feature_config.has_categorical


# Test QuantitativeBayesianNeuralNetwork update with valid rewards, quantities, and context
@st.composite
def dimension_and_quantities(draw):
    dimension = draw(st.integers(min_value=1, max_value=3))

    flat = st.lists(
        st.floats(min_value=0, max_value=1),
        min_size=5,
        max_size=5,
    )

    nested = st.lists(
        st.lists(
            st.floats(min_value=0, max_value=1),
            min_size=dimension,
            max_size=dimension,
        ),
        min_size=4,
        max_size=5,
    )
    if dimension == 1:
        quantities = draw(st.one_of(flat, nested))
    else:
        quantities = draw(nested)

    return dimension, quantities


@given(
    rewards=st.lists(st.integers(min_value=0, max_value=1), min_size=4, max_size=5),
    context=arrays(np.float64, shape=(5, 1), elements=st.floats(min_value=0, max_value=1)),
    n_features=st.just(1),
    dimension_and_quantities=dimension_and_quantities(),
)
def test_updates_cmab_quantitative_model_correctly(
    rewards, context, n_features, dimension_and_quantities, monkeymodule
):
    dimension, quantities = dimension_and_quantities
    model = QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=n_features)
    init_params = deepcopy(model.bnn.model_params)

    monkeymodule.setattr(
        pybandits.model.BaseBayesianNeuralNetwork,
        "_update",
        mock_update,
    )
    if (len(context) != len(quantities)) or (len(context) != len(rewards)):
        with pytest.raises(AttributeError):
            model.update(quantities=quantities, rewards=rewards, context=context)
        return
    model.update(quantities=quantities, rewards=rewards, context=context)
    assert model.bnn.model_params.bnn_layer_params != init_params.bnn_layer_params


# Test QuantitativeBayesianNeuralNetwork sample_proba returns valid probability functions
@pytest.mark.parametrize("use_embedding,categorical_features", [(False, None), (True, {0: 3})])
@given(
    dimension=st.just(1),
    n_features=st.just(1),
    quantity=st.floats(min_value=0, max_value=1),
)
def test_sample_proba_returns_valid_probabilities_cmab(
    rng, dimension, n_features, quantity, use_embedding, categorical_features
):
    if use_embedding:
        model = QuantitativeBayesianNeuralNetwork.cold_start(
            dimension=dimension, categorical_features=categorical_features
        )
        context = np.array([[0], [1], [2], [0], [1]], dtype=np.float64)
    else:
        model = QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=n_features)
        context = rng.uniform(low=0, high=1, size=(5, 1))

    prob_functions = model.sample_proba(context=context, rng=rng)
    assert len(prob_functions) == len(context)

    # Test that each function is callable and returns valid probabilities
    for prob_weight_func in prob_functions:
        assert all(callable(func) for func in prob_weight_func)
        prob, weight = (func(np.atleast_1d(quantity)) for func in prob_weight_func)
        assert 0 <= prob <= 1


########################################################################################################################


# Tests for _to_quantitative_probabilities determinism
@st.composite
def quantity_strategy(draw):
    array_length = draw(st.integers(min_value=1, max_value=5))
    temp_list = [
        draw(arrays(dtype=np.float64, shape=array_length, elements=st.floats(min_value=0, max_value=1)))
        for _ in range(5)
    ]
    temp_list = [arr[0] if len(arr) == 1 else arr for arr in temp_list]
    return temp_list


@given(
    n_calls=st.just(10),
    test_quantity=quantity_strategy(),
    context=arrays(dtype=np.float64, shape=(5, 2), elements=st.floats(min_value=0, max_value=1)),
)
def test_quantitative_bnn_to_quantitative_probabilities_consistency(rng, n_calls, test_quantity, context):
    """Test that QuantitativeBayesianNeuralNetwork._to_quantitative_probabilities returns consistent results when called repeatedly."""
    if isinstance(test_quantity[0], float):
        dimension = 1
    else:
        dimension = len(test_quantity[0])
    model = QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, n_features=2)
    n_samples = len(context)
    sampled_weights = model.bnn.sample_weights(n_samples, rng=rng)
    # Create context and sample probability functions
    prob_functions = model._to_quantitative_probabilities(context=context, sampled_weights=sampled_weights)

    # Call each probability/weight function multiple times with the same quantity
    for prob_weight_funcs in prob_functions:
        for sample_idx in range(n_samples):
            # Get first results for both probability and weight
            first_results = tuple(func(test_quantity[sample_idx]) for func in prob_weight_funcs)
            for _ in range(n_calls - 1):
                subsequent_results = tuple(func(test_quantity[sample_idx]) for func in prob_weight_funcs)
                assert first_results == subsequent_results, (
                    f"Inconsistent results: first={first_results}, subsequent={subsequent_results}"
                )


@pytest.mark.parametrize("dimension,categorical_features", [(1, {0: 3}), (2, {0: 3})])
@given(
    n_calls=st.just(10),
    tier_indices=arrays(dtype=np.int64, shape=(5, 1), elements=st.integers(min_value=0, max_value=2)),
    test_quantities=arrays(dtype=np.float64, shape=(5,), elements=st.floats(min_value=0, max_value=1)),
)
def test_quantitative_bnn_to_quantitative_probabilities_consistency_with_categorical(
    rng, dimension, categorical_features, n_calls, tier_indices, test_quantities
):
    """Pre-sampled embeddings make repeated closure calls deterministic for QBNN with categorical features.

    Without the fix (embeddings re-sampled inside forward_pass), calling the same closure with
    the same quantity would yield a different result on each call.  This test verifies the fix
    by asserting exact equality across ``n_calls`` evaluations of each closure.
    """
    model = QuantitativeBayesianNeuralNetwork.cold_start(dimension=dimension, categorical_features=categorical_features)
    context = tier_indices.astype(np.float64)
    n_samples = len(context)

    # Replicate exactly what sample_proba does: sample weights and pre-sample embeddings once.
    sampled_weights = model.bnn.sample_weights(n_samples, rng=rng)
    dummy_full = np.column_stack([np.zeros((n_samples, dimension)), context])
    sampled_embeddings = model.bnn.sample_embeddings(dummy_full, rng=rng)

    prob_functions = model._to_quantitative_probabilities(
        context=context,
        sampled_weights=sampled_weights,
        sampled_embeddings=sampled_embeddings,
    )

    for prob_weight_funcs in prob_functions:
        for sample_idx in range(n_samples):
            # Build a quantity vector of the right dimension from a single hypothesis float.
            q = np.full(dimension, test_quantities[sample_idx])
            first_results = tuple(func(q) for func in prob_weight_funcs)
            for _ in range(n_calls - 1):
                subsequent_results = tuple(func(q) for func in prob_weight_funcs)
                assert first_results == subsequent_results, (
                    f"Inconsistent results: first={first_results}, subsequent={subsequent_results}"
                )


########################################################################################################################


# QuantitativeModelCC tests via ZoomingCC


def simple_cost(value: float) -> float:
    return value * 2


def complex_cost(value: float | NonNegativeFloat, factor: float = 1.0) -> NonNegativeFloat:
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
    model = ZoomingCC.cold_start(cost=cost_function)
    serialized = model.model_dump_json()
    serialized_dict = json.loads(serialized)

    assert "cost" in serialized_dict
    assert isinstance(serialized_dict["cost"], str)

    deserialized_model = ZoomingCC.model_validate_json(serialized)
    # Check callable was properly restored
    assert callable(deserialized_model.cost)

    reserialized = deserialized_model.model_dump_json()
    assert reserialized == serialized


# QuantitativeModelDP tests via ZoomingDP


def simple_price(value: float) -> float:
    return value * 10


def complex_price(value: float | NonNegativeFloat, factor: float = 1.0) -> NonNegativeFloat:
    return value * factor


partial_price = functools.partial(complex_price, factor=2.5)


@pytest.mark.parametrize(
    "price_function",
    [
        simple_price,
        complex_price,
        partial_price,
    ],
)
def test_price_serialization_deserialization(price_function):
    """Test serialization and deserialization of price field with different callables."""

    # Create model with the test callable as price
    model = ZoomingDP.cold_start(price=price_function)
    serialized = model.model_dump_json()
    serialized_dict = json.loads(serialized)

    assert "price" in serialized_dict
    assert isinstance(serialized_dict["price"], str)

    deserialized_model = ZoomingDP.model_validate_json(serialized)
    # Check callable was properly restored
    assert callable(deserialized_model.price)

    reserialized = deserialized_model.model_dump_json()
    assert reserialized == serialized


########################################################################################################################


# QuantitativeBayesianNeuralNetworkCC tests


@pytest.mark.parametrize(
    "cost_function",
    [
        simple_cost,
        complex_cost,
        partial_cost,
    ],
)
def test_bnncc_cost_serialization_deserialization(cost_function):
    """Test serialization and deserialization of cost field in BNNCC with different callables."""

    # Create model with the test callable as cost
    model = QuantitativeBayesianNeuralNetworkCC.cold_start(dimension=1, cost=cost_function)
    serialized = model.model_dump_json()
    serialized_dict = json.loads(serialized)

    assert "cost" in serialized_dict
    assert isinstance(serialized_dict["cost"], str)

    deserialized_model = QuantitativeBayesianNeuralNetworkCC.model_validate_json(serialized)
    # Check callable was properly restored
    assert callable(deserialized_model.cost)

    reserialized = deserialized_model.model_dump_json()
    assert reserialized == serialized


# QuantitativeBayesianNeuralNetworkDP tests


@pytest.mark.parametrize(
    "price_function",
    [
        simple_price,
        complex_price,
        partial_price,
    ],
)
def test_bnndp_price_serialization_deserialization(price_function):
    """Test serialization and deserialization of price field in BNNDP with different callables."""

    # Create model with the test callable as price
    model = QuantitativeBayesianNeuralNetworkDP.cold_start(dimension=1, price=price_function)
    serialized = model.model_dump_json()
    serialized_dict = json.loads(serialized)

    assert "price" in serialized_dict
    assert isinstance(serialized_dict["price"], str)

    deserialized_model = QuantitativeBayesianNeuralNetworkDP.model_validate_json(serialized)
    # Check callable was properly restored
    assert callable(deserialized_model.price)

    reserialized = deserialized_model.model_dump_json()
    assert reserialized == serialized
