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

"""Tests for the benchmark harness in ``pybandits/benchmark_models.py``.

These cover the harness itself — the argument parsing, the model registry, the reward
shaping, the step accounting and the CSV writer — plus one end-to-end sMAB run.

The properties that hold for *any* action count, row count or repeat count are stated with
hypothesis; the discrete case sets (the two model families, the two cMAB passes, the two
sweep tables, single- versus multi-objective) are parametrized. Both keep the stubs cheap:
no cMAB is ever *updated* here, because a single SVI update takes seconds and covering it
would turn the test suite into the benchmark. Only the cold-start path is exercised on a
real cMAB, to check that the hidden-layer configuration reaches its network. The sMAB
end-to-end run reaches the same ``benchmark_predict`` / ``benchmark_update`` code the cMAB
path uses.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pybandits import benchmark_models
from pybandits.cmab import (
    CmabBernoulli,
    CmabBernoulliBAI,
    CmabBernoulliCC,
    CmabBernoulliMO,
    CmabBernoulliMOCC,
)
from pybandits.smab import (
    SmabBernoulli,
    SmabBernoulliBAI,
    SmabBernoulliCC,
    SmabBernoulliMO,
    SmabBernoulliMOCC,
)

# --- test configuration constants ---
#: Action-set sizes explored by the property tests. The properties are about bookkeeping
#: rather than scale, so the range stays small enough that a full hypothesis run is instant.
MIN_ACTIONS = 1
MAX_ACTIONS = 6
#: Row counts for the reward-shaping and step-accounting properties.
MIN_SAMPLES = 1
MAX_SAMPLES = 12
#: Smallest call the row-to-call partitioning can be split into.
MIN_CHUNK = 1
#: Repeat counts ``time_call`` must accept, and the largest one it must reject.
MIN_REPEATS = 1
MAX_REPEATS = 4
MAX_REJECTED_REPEATS = 0
#: Seed for every RNG in this module, so failures are reproducible.
SEED = 0

#: Bounds for the cMAB pass configuration, whose values are carried rather than interpreted.
MIN_PASS_VALUE = 1
MAX_PASS_VALUE = 512
#: Pass names and the ``cold_start`` kwargs keys the two passes must carry.
FIXED_PASS_NAME = "fixed"
EPOCHS_PASS_NAME = "epochs"
UPDATE_KWARGS_KEY = "update_kwargs"
NUM_STEPS_KEY = "num_steps"
EARLY_STOPPING_KEY = "early_stopping_kwargs"
EPOCHS_KEY = "epochs"
BATCH_SIZE_KEY = "batch_size"

#: Sweep values the argparse type hooks must accept, and the largest they must reject.
MIN_SWEEP_VALUE = 1
MAX_SWEEP_VALUE = 10_000
MIN_SWEEP_LENGTH = 1
MAX_SWEEP_LENGTH = 4
MAX_REJECTED_SWEEP_VALUE = 0
MAX_REJECTED_SEED = -1
MIN_ACCEPTED_SEED = 0
SWEEP_SEPARATOR = ","
#: Sweeps that carry no usable value at all and must be rejected rather than run empty.
MALFORMED_SWEEPS = ("", " ", ",", " , ")

#: Stub-network configuration. One step per row makes the expected total budget the row
#: count itself, which is derivable without restating the function under test.
ONE_STEP_PER_ROW = 1
SINGLE_EPOCH = 1
MIN_ROWS_PER_STEP = 1
MAX_ROWS_PER_STEP = 8
MIN_EPOCHS = 1
MAX_EPOCHS = 4
#: What a model with no step budget — every closed-form sMAB — must report.
NO_STEP_BUDGET = 0
#: Lower bounds that hold by construction rather than by configuration.
MIN_COST = 0.0
MIN_SECONDS = 0.0

#: Model classes that must each be registered as a benchmark spec.
SMAB_MODEL_CLASSES = (SmabBernoulli, SmabBernoulliBAI, SmabBernoulliCC, SmabBernoulliMO, SmabBernoulliMOCC)
CMAB_MODEL_CLASSES = (CmabBernoulli, CmabBernoulliBAI, CmabBernoulliCC, CmabBernoulliMO, CmabBernoulliMOCC)

#: Benchmark configuration the tests assert against, aliased so no test body reads the
#: module directly.
N_OBJECTIVES = benchmark_models.DEFAULT_N_OBJECTIVES
N_FEATURES = benchmark_models.DEFAULT_N_FEATURES
FULL_DEFAULTS = benchmark_models.FULL_DEFAULTS
QUICK_DEFAULTS = benchmark_models.QUICK_DEFAULTS
CMAB_HIDDEN_DIMS = benchmark_models.CMAB_HIDDEN_DIMS
CMAB_HIDDEN_SPEC_NAME = benchmark_models.CMAB_HIDDEN_SPEC_NAME
#: Action-set size used for the cold-start checks — the smallest that is still a bandit.
HIDDEN_SPEC_ACTIONS = 2

#: Scalar settings the parser must reject below one, rather than let a nonsense value reach
#: a network fit.
POSITIVE_INT_FLAGS = ("--n-features", "--repeats", "--fixed-num-steps", "--epochs", "--epoch-batch-size")

#: CLI fragments used by the parser and end-to-end tests.
QUICK_FLAG = "--quick"
FAMILY_FLAG = "--family"
SMAB_FAMILY = "smab"
END_TO_END_ARGV = (FAMILY_FLAG, SMAB_FAMILY, QUICK_FLAG)
#: The per-request predict shape; ``--quick`` must not drop it.
SINGLE_REQUEST_SIZE = 1
CSV_FILENAME = "benchmark.csv"

#: ``Result`` payload defaults, and the fields whose accepted domains differ.
UPDATE_METHOD = "update"
PREDICT_METHOD = "predict"
BENCHMARKED_METHODS = (UPDATE_METHOD, PREDICT_METHOD)
NO_PASS_NAME = "n/a"
STUB_SPEC_NAME = "stub"
VALID_COUNT = 1
VALID_SECONDS = 1.0
ZERO_BUDGET = 0
POSITIVE_COUNT_FIELDS = ("n_samples", "n_actions", "n_calls")
NON_NEGATIVE_COUNT_FIELDS = ("batch_size", "steps")
MAX_REJECTED_POSITIVE = 0
MAX_REJECTED_NON_NEGATIVE = -1


class _StepCountingNetwork:
    """Stand-in for a Bayesian network, reporting a step budget derived from the row count."""

    def __init__(self, rows_per_step: int, epochs: int) -> None:
        self._rows_per_step = rows_per_step
        self._epochs = epochs

    def compute_epoch_steps(self, n_samples: int) -> List[int]:
        """Steps per epoch, in the shape the real networks report them."""
        return [n_samples // self._rows_per_step] * self._epochs


class _NestedActionModel:
    """Stand-in for a multi-objective action model, which nests one network per objective."""

    def __init__(self, models: Sequence[object]) -> None:
        self.models = list(models)


class _StubMab:
    """Stand-in MAB exposing one action model per action id."""

    def __init__(self, actions: Dict[str, object]) -> None:
        self.actions = actions


class _StatefulModel:
    """Stand-in model counting its own updates, so leaked state between repeats is visible."""

    def __init__(self) -> None:
        self.updates = 0

    def update(self) -> None:
        """Record one update."""
        self.updates += 1


SpecFactory = Callable[..., benchmark_models.ModelSpec]
StubMabFactory = Callable[..., _StubMab]
ResultPayloadFactory = Callable[..., Dict[str, Any]]


@pytest.fixture(scope="module")
def make_spec() -> SpecFactory:
    """Factory: a placeholder ``ModelSpec`` whose model factory is never invoked."""

    def _factory(
        multi_objective: bool = False,
        contextual: bool = False,
        name: str = STUB_SPEC_NAME,
    ) -> benchmark_models.ModelSpec:
        return benchmark_models.ModelSpec(
            name=name,
            factory=lambda **kwargs: None,
            contextual=contextual,
            multi_objective=multi_objective,
        )

    return _factory


@pytest.fixture(scope="module")
def make_stub_mab() -> StubMabFactory:
    """Factory: a stand-in MAB whose action models mimic the real per-action networks."""

    def _factory(
        action_ids: Iterable[str],
        rows_per_step: int = ONE_STEP_PER_ROW,
        epochs: int = SINGLE_EPOCH,
        closed_form: bool = False,
        nested: bool = False,
    ) -> _StubMab:
        def build_action_model() -> object:
            if closed_form:
                return object()
            network = _StepCountingNetwork(rows_per_step=rows_per_step, epochs=epochs)
            return _NestedActionModel([network]) if nested else network

        return _StubMab({action_id: build_action_model() for action_id in action_ids})

    return _factory


@pytest.fixture(scope="module")
def make_result_payload() -> ResultPayloadFactory:
    """Factory: a valid ``Result`` payload, with individual fields overridable."""

    def _factory(**overrides: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "family": SMAB_FAMILY,
            "model": STUB_SPEC_NAME,
            "method": UPDATE_METHOD,
            "pass_name": NO_PASS_NAME,
            "n_samples": VALID_COUNT,
            "n_actions": VALID_COUNT,
            "batch_size": ZERO_BUDGET,
            "n_calls": VALID_COUNT,
            "steps": ZERO_BUDGET,
            "seconds": VALID_SECONDS,
            "seconds_per_obs": VALID_SECONDS,
        }
        payload.update(overrides)
        return payload

    return _factory


@st.composite
def action_assignments(draw: st.DrawFn) -> Tuple[List[str], List[slice]]:
    """Draw a row-to-action assignment together with the call slices it is split into."""
    n_actions = draw(st.integers(min_value=MIN_ACTIONS, max_value=MAX_ACTIONS))
    action_ids = benchmark_models._action_ids(n_actions)
    actions = draw(st.lists(st.sampled_from(action_ids), min_size=MIN_SAMPLES, max_size=MAX_SAMPLES))
    chunk = draw(st.integers(min_value=MIN_CHUNK, max_value=len(actions)))
    slices = [slice(start, min(start + chunk, len(actions))) for start in range(0, len(actions), chunk)]
    return actions, slices


def _sweep_size(value: Union[int, Tuple[int, ...]]) -> int:
    """How much work one sweep setting asks for: its length, or its value when it is scalar."""
    return len(value) if isinstance(value, tuple) else value


@given(
    n_actions=st.integers(min_value=MIN_ACTIONS, max_value=MAX_ACTIONS),
    min_cost=st.just(MIN_COST),
)
def test_action_ids_and_costs_are_consistent(n_actions: int, min_cost: float) -> None:
    """Costs are keyed by exactly the generated action ids, for any action-set size."""
    action_ids = benchmark_models._action_ids(n_actions)
    costs = benchmark_models._action_costs(n_actions)

    assert len(action_ids) == n_actions
    assert len(set(action_ids)) == n_actions
    assert action_ids == benchmark_models._action_ids(n_actions), "ids must be stable across calls"
    assert set(costs) == set(action_ids)
    assert all(cost > min_cost for cost in costs.values())


@pytest.mark.parametrize(
    ("build_specs", "model_classes", "contextual"),
    [
        pytest.param(benchmark_models.build_smab_specs, SMAB_MODEL_CLASSES, False, id="smab"),
        pytest.param(benchmark_models.build_cmab_specs, CMAB_MODEL_CLASSES, True, id="cmab"),
    ],
)
def test_every_model_family_is_registered(
    build_specs: Callable[[], List[benchmark_models.ModelSpec]],
    model_classes: Tuple[type, ...],
    contextual: bool,
) -> None:
    """The issue asks for all sMABs and cMABs; guard against a variant being dropped."""
    specs = build_specs()
    names = {spec.name for spec in specs}

    assert {model_class.__name__ for model_class in model_classes} <= names
    assert len(names) == len(specs), "spec names are the CSV model column and must be unique"
    assert all(spec.contextual is contextual for spec in specs)


def test_hidden_layer_spec_configures_its_own_network(
    spec_name: str = CMAB_HIDDEN_SPEC_NAME,
    hidden_dims: Tuple[int, ...] = CMAB_HIDDEN_DIMS,
    n_actions: int = HIDDEN_SPEC_ACTIONS,
    n_features: int = N_FEATURES,
    seed: int = SEED,
) -> None:
    """The extra cMAB configuration reaches every one of its networks with a hidden layer.

    The default ``CmabBernoulli`` is a single linear layer, so without this contrast the
    extra row would time the same network twice under two names.
    """
    specs = {spec.name: spec for spec in benchmark_models.build_cmab_specs()}
    build = dict(n_actions=n_actions, random_seed=seed, n_features=n_features)

    with_hidden_layer = specs[spec_name].factory(**build)
    default = specs[CmabBernoulli.__name__].factory(**build)

    assert [model.hidden_dim_list for model in with_hidden_layer.actions.values()] == [list(hidden_dims)] * n_actions
    assert all(not model.hidden_dim_list for model in default.actions.values())


@pytest.mark.parametrize("multi_objective", [False, True])
@given(
    n_samples=st.integers(min_value=MIN_SAMPLES, max_value=MAX_SAMPLES),
    n_objectives=st.just(N_OBJECTIVES),
    seed=st.just(SEED),
)
def test_reward_shape_matches_the_model(
    multi_objective: bool,
    n_samples: int,
    n_objectives: int,
    seed: int,
    make_spec: SpecFactory,
) -> None:
    """Multi-objective models take one reward per objective; the others take a scalar."""
    spec = make_spec(multi_objective=multi_objective)
    rewards = benchmark_models.make_rewards(spec, n_samples=n_samples, rng=np.random.default_rng(seed))

    assert len(rewards) == n_samples
    if multi_objective:
        assert all(len(reward) == n_objectives for reward in rewards)
    else:
        assert all(isinstance(reward, int) for reward in rewards)


@given(
    fixed_num_steps=st.integers(min_value=MIN_PASS_VALUE, max_value=MAX_PASS_VALUE),
    epochs=st.integers(min_value=MIN_PASS_VALUE, max_value=MAX_PASS_VALUE),
    epoch_batch_size=st.integers(min_value=MIN_PASS_VALUE, max_value=MAX_PASS_VALUE),
    fixed_pass_name=st.just(FIXED_PASS_NAME),
    epochs_pass_name=st.just(EPOCHS_PASS_NAME),
    update_kwargs_key=st.just(UPDATE_KWARGS_KEY),
    num_steps_key=st.just(NUM_STEPS_KEY),
    early_stopping_key=st.just(EARLY_STOPPING_KEY),
    epochs_key=st.just(EPOCHS_KEY),
    batch_size_key=st.just(BATCH_SIZE_KEY),
)
def test_cmab_passes_are_distinct_configurations(
    fixed_num_steps: int,
    epochs: int,
    epoch_batch_size: int,
    fixed_pass_name: str,
    epochs_pass_name: str,
    update_kwargs_key: str,
    num_steps_key: str,
    early_stopping_key: str,
    epochs_key: str,
    batch_size_key: str,
) -> None:
    """The two passes must differ in the way that makes their numbers comparable.

    ``fixed`` pins the step budget and disables early stopping; ``epochs`` sets an explicit
    batch size, which is what makes the step count scale with the data. Whatever values they
    are configured with must arrive unaltered, since a substituted default would be reported
    as if it were the requested one.
    """
    passes = dict(
        benchmark_models.cmab_passes(fixed_num_steps=fixed_num_steps, epochs=epochs, epoch_batch_size=epoch_batch_size)
    )

    assert set(passes) == {fixed_pass_name, epochs_pass_name}
    assert passes[fixed_pass_name][update_kwargs_key] == {
        num_steps_key: fixed_num_steps,
        early_stopping_key: None,
    }
    assert passes[epochs_pass_name][update_kwargs_key] == {
        epochs_key: epochs,
        batch_size_key: epoch_batch_size,
    }


@given(
    repeats=st.integers(min_value=MIN_REPEATS, max_value=MAX_REPEATS),
    min_seconds=st.just(MIN_SECONDS),
)
def test_time_call_warms_up_once_and_gives_every_repeat_fresh_state(repeats: int, min_seconds: float) -> None:
    """One model is built per warm-up and per timed repeat, and none of them is reused.

    The warm-up must stay outside the timed repeats because the first call pays JIT
    compilation, and reusing one instance would time an already-fitted model being retrained
    on data it has already seen — not the cold update the row reports.
    """
    built: List[_StatefulModel] = []

    def make_call() -> Callable[[], None]:
        model = _StatefulModel()
        built.append(model)
        return model.update

    seconds = benchmark_models.time_call(make_call, repeats=repeats)

    assert len(built) == repeats + 1, "one model per warm-up and per timed repeat"
    assert all(model.updates == 1 for model in built), "state leaked between repeats"
    assert seconds >= min_seconds


@given(repeats=st.integers(max_value=MAX_REJECTED_REPEATS))
def test_time_call_rejects_non_positive_repeats(repeats: int) -> None:
    """Fewer than one repeat leaves no timings and would report NaN into the CSV."""
    with pytest.raises(ValueError):
        benchmark_models.time_call(lambda: (lambda: None), repeats=repeats)


@pytest.mark.parametrize("multi_objective", [False, True])
@given(
    assignment=action_assignments(),
    rows_per_step=st.just(ONE_STEP_PER_ROW),
    epochs=st.just(SINGLE_EPOCH),
    n_objectives=st.just(N_OBJECTIVES),
)
def test_total_update_steps_charges_each_action_only_its_own_rows(
    multi_objective: bool,
    assignment: Tuple[List[str], List[slice]],
    rows_per_step: int,
    epochs: int,
    n_objectives: int,
    make_spec: SpecFactory,
    make_stub_mab: StubMabFactory,
) -> None:
    """At one step per row the budget is the row count, however the rows split across arms.

    Charging every action for the whole call — the bug this guards — would scale with the
    number of arms present, which is exactly what makes the column worth reporting.
    """
    actions, slices = assignment
    spec = make_spec(multi_objective=multi_objective)
    mab = make_stub_mab(action_ids=set(actions), rows_per_step=rows_per_step, epochs=epochs)

    expected = len(actions) * epochs * (n_objectives if multi_objective else 1)
    assert benchmark_models.total_update_steps(mab, spec, actions, slices) == expected


@given(
    n_samples=st.integers(min_value=MIN_SAMPLES, max_value=MAX_SAMPLES),
    n_actions=st.integers(min_value=MIN_ACTIONS, max_value=MAX_ACTIONS),
    expected_steps=st.just(NO_STEP_BUDGET),
)
def test_steps_per_update_is_zero_for_closed_form_models(
    n_samples: int,
    n_actions: int,
    expected_steps: int,
    make_stub_mab: StubMabFactory,
) -> None:
    """sMABs have no step budget, and must report zero rather than a misleading number."""
    mab = make_stub_mab(action_ids=benchmark_models._action_ids(n_actions), closed_form=True)

    assert benchmark_models.steps_per_update(mab, n_samples=n_samples) == expected_steps


@pytest.mark.parametrize("nested", [False, True], ids=["single_objective", "multi_objective"])
@given(
    n_samples=st.integers(min_value=MIN_SAMPLES, max_value=MAX_SAMPLES),
    n_actions=st.integers(min_value=MIN_ACTIONS, max_value=MAX_ACTIONS),
    rows_per_step=st.integers(min_value=MIN_ROWS_PER_STEP, max_value=MAX_ROWS_PER_STEP),
    epochs=st.integers(min_value=MIN_EPOCHS, max_value=MAX_EPOCHS),
)
def test_steps_per_update_reads_the_models_own_partition(
    nested: bool,
    n_samples: int,
    n_actions: int,
    rows_per_step: int,
    epochs: int,
    make_stub_mab: StubMabFactory,
) -> None:
    """The budget comes from the model's own ``compute_epoch_steps``, MO nesting included."""
    mab = make_stub_mab(
        action_ids=benchmark_models._action_ids(n_actions),
        rows_per_step=rows_per_step,
        epochs=epochs,
        nested=nested,
    )

    assert benchmark_models.steps_per_update(mab, n_samples=n_samples) == epochs * (n_samples // rows_per_step)


@given(
    values=st.lists(
        st.integers(min_value=MIN_SWEEP_VALUE, max_value=MAX_SWEEP_VALUE),
        min_size=MIN_SWEEP_LENGTH,
        max_size=MAX_SWEEP_LENGTH,
    ),
    separator=st.just(SWEEP_SEPARATOR),
)
def test_positive_ints_round_trips_a_comma_separated_sweep(values: List[int], separator: str) -> None:
    """Any comma-separated list of positive integers parses back to exactly that tuple."""
    raw = separator.join(str(value) for value in values)

    assert benchmark_models._positive_ints(raw) == tuple(values)


@pytest.mark.parametrize(
    "parse_sweep",
    [benchmark_models._positive_int, benchmark_models._positive_ints],
    ids=["single", "list"],
)
@given(value=st.integers(max_value=MAX_REJECTED_SWEEP_VALUE))
def test_sweep_parsers_reject_non_positive_values(parse_sweep: Callable[[str], object], value: int) -> None:
    """A zero or negative sweep value is a mistake, not a silently empty benchmark."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_sweep(str(value))


@given(value=st.integers(min_value=MIN_ACCEPTED_SEED))
def test_non_negative_int_round_trips_a_usable_seed(value: int) -> None:
    """Zero is a legitimate seed, so the boundary accepts it alongside every positive value."""
    assert benchmark_models._non_negative_int(str(value)) == value


@given(value=st.integers(max_value=MAX_REJECTED_SEED))
def test_non_negative_int_rejects_negative_seeds(value: int) -> None:
    """``seed`` is annotated ``NonNegativeInt``, so a negative fails at the CLI, not at the model."""
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_models._non_negative_int(str(value))


@pytest.mark.parametrize("raw", MALFORMED_SWEEPS)
def test_positive_ints_rejects_malformed_sweeps(raw: str) -> None:
    """A sweep carrying no value at all must fail loudly rather than benchmark nothing."""
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_models._positive_ints(raw)


def test_sweep_tables_describe_the_same_settings(
    full_defaults: Dict[str, Any] = FULL_DEFAULTS,
    quick_defaults: Dict[str, Any] = QUICK_DEFAULTS,
) -> None:
    """Both tables carry the same keys, so ``--quick`` reads as a diff of the two."""
    assert set(full_defaults) == set(quick_defaults)


@pytest.mark.parametrize("defaults", [FULL_DEFAULTS, QUICK_DEFAULTS], ids=["full", "quick"])
def test_sweep_tables_hold_only_positive_values(
    defaults: Dict[str, Any],
    min_sweep_value: int = MIN_SWEEP_VALUE,
) -> None:
    """Table entries bypass the argparse type hooks, so their positivity is checked here."""
    for setting in defaults.values():
        for value in setting if isinstance(setting, tuple) else (setting,):
            assert value >= min_sweep_value


@pytest.mark.parametrize("setting", sorted(FULL_DEFAULTS))
def test_parser_defaults_come_from_the_full_sweep_table(
    setting: str,
    full_defaults: Dict[str, Any] = FULL_DEFAULTS,
) -> None:
    """The parser reads its defaults from the table rather than from inline literals."""
    assert getattr(benchmark_models.parse_args([]), setting) == full_defaults[setting]


@pytest.mark.parametrize("setting", sorted(QUICK_DEFAULTS))
def test_quick_mode_applies_the_quick_table_and_shrinks_every_sweep(
    setting: str,
    quick_flag: str = QUICK_FLAG,
    quick_defaults: Dict[str, Any] = QUICK_DEFAULTS,
) -> None:
    """``--quick`` exists to bound CI time; every setting it substitutes must get smaller."""
    full = benchmark_models.parse_args([])
    quick = benchmark_models.parse_args([quick_flag])

    assert getattr(quick, setting) == quick_defaults[setting]
    assert _sweep_size(getattr(quick, setting)) < _sweep_size(getattr(full, setting))


@pytest.mark.parametrize("flag", POSITIVE_INT_FLAGS)
@given(value=st.integers(max_value=MAX_REJECTED_SWEEP_VALUE))
def test_parser_rejects_non_positive_scalar_settings(flag: str, value: int) -> None:
    """A setting that must be at least one fails at the command line, not inside a fit."""
    with pytest.raises(SystemExit):
        benchmark_models.parse_args([flag, str(value)])


def test_quick_mode_keeps_the_single_request_predict_shape(
    quick_flag: str = QUICK_FLAG,
    single_request_size: int = SINGLE_REQUEST_SIZE,
) -> None:
    """The one-row predict is the serving shape; the reduced sweep must still cover it."""
    assert single_request_size in benchmark_models.parse_args([quick_flag]).predict_sizes


@pytest.mark.parametrize("field", POSITIVE_COUNT_FIELDS)
@given(value=st.integers(max_value=MAX_REJECTED_POSITIVE))
def test_result_rejects_non_positive_counts(
    field: str,
    value: int,
    make_result_payload: ResultPayloadFactory,
) -> None:
    """The count columns are ``PositiveInt``: a zero would describe a row measured on nothing."""
    with pytest.raises(ValidationError):
        benchmark_models.Result(**make_result_payload(**{field: value}))


@pytest.mark.parametrize("field", NON_NEGATIVE_COUNT_FIELDS)
@given(value=st.integers(max_value=MAX_REJECTED_NON_NEGATIVE))
def test_result_rejects_negative_budgets(
    field: str,
    value: int,
    make_result_payload: ResultPayloadFactory,
) -> None:
    """``batch_size`` and ``steps`` are ``NonNegativeInt``: below zero is not a measurement."""
    with pytest.raises(ValidationError):
        benchmark_models.Result(**make_result_payload(**{field: value}))


@pytest.mark.parametrize("field", NON_NEGATIVE_COUNT_FIELDS)
def test_result_accepts_a_zero_budget(
    field: str,
    make_result_payload: ResultPayloadFactory,
    zero_budget: int = ZERO_BUDGET,
) -> None:
    """``batch_size=0`` means one call with all rows, and ``steps=0`` a closed-form update."""
    result = benchmark_models.Result(**make_result_payload(**{field: zero_budget}))

    assert getattr(result, field) == zero_budget


def test_end_to_end_smab_run_writes_a_readable_csv(
    tmp_path: Path,
    argv: Tuple[str, ...] = END_TO_END_ARGV,
    filename: str = CSV_FILENAME,
    family: str = SMAB_FAMILY,
    methods: Tuple[str, ...] = BENCHMARKED_METHODS,
    min_seconds: float = MIN_SECONDS,
) -> None:
    """One real sMAB sweep, through the same code path the cMAB benchmark uses."""
    args = benchmark_models.parse_args(list(argv))
    results = benchmark_models.run(args)
    assert results, "the sMAB sweep produced no measurements"

    output = tmp_path / filename
    benchmark_models.write_csv(results, str(output))
    with open(output, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    columns = list(benchmark_models.Result.model_fields)
    assert rows == [{column: str(getattr(result, column)) for column in columns} for result in results]
    assert {result.family for result in results} == {family}
    assert {result.method for result in results} == set(methods)
    for result in results:
        assert result.seconds > min_seconds
        # The per-observation figure is what stays comparable across batch sizes.
        assert result.seconds_per_obs == result.seconds / result.n_samples
