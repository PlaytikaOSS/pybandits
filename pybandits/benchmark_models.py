"""Execution-time benchmark for every available sMAB and cMAB model.

The script times the two hot paths of the library — ``update`` and ``predict`` — across a
range of sample sizes, and writes one CSV row per measurement.

Design notes
------------
The measurements are deliberately shaped so that the numbers stay comparable *between*
models rather than only within one:

* **sMAB and cMAB are benchmarked separately.** They are not two implementations of one
  thing: the Bernoulli sMABs are conjugate and update in closed form, while the cMABs fit a
  Bayesian neural network by stochastic variational inference. Putting them on a single
  axis would make the sMABs look categorically faster for a reason unrelated to sample size.

* **The cMABs are timed in two passes**, because the cost of an SVI ``update`` is set by the
  number of optimiser steps rather than by the number of rows:

  ``fixed``
      A fixed step budget with early stopping disabled. Step count is held constant across
      sample sizes, so the per-step cost is genuinely comparable across models and across n.
  ``epochs``
      The ``epochs`` setting with an explicit ``batch_size``, which is the realistic
      configuration. Note that ``epochs`` only makes the step count grow with the data when
      a ``batch_size`` is set: ``compute_epoch_steps`` derives
      ``steps_per_epoch = n_samples // (batch_size or n_samples)``, so with the default
      ``batch_size=None`` an epoch is a single full-batch step and the step count is constant
      in n. The number of steps actually taken is reported alongside the time, since a
      duration produced by an unknown number of steps cannot be interpreted or reproduced.

* **``update`` is timed batch and incremental.** Updating once with n rows is a different
  operation from n sequential single-row updates, and serving is usually closer to the
  latter. Both are reported, along with the cost per observation.

* **``predict`` is swept over the number of actions as well as n**, and always includes
  ``n=1``, which is the shape a live serving path actually issues.

* **One cMAB is benchmarked at two network depths.** ``CmabBernoulli`` defaults to a single
  linear layer, so a sweep of the defaults alone reports the cost of the SVI machinery and
  never the cost of the network being fitted. A second configuration with one hidden layer
  is included so the two can be read against each other.

* **A warm-up call precedes every timed measurement.** The cMAB path is JAX-backed and the
  first call pays JIT compilation — measured at roughly ten seconds for a six-row update on
  a laptop. Without a warm-up the first row of every sweep reports compilation, not compute.

This is a timing benchmark only. Reward quality (regret, off-policy estimates) is out of
scope here and is tracked separately.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import contextmanager
from functools import partial, partialmethod
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import NonNegativeInt, PositiveInt
from tqdm import tqdm

from pybandits.base import PyBanditsBaseModel
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


@contextmanager
def suppressed_progress_bars() -> Iterator[None]:
    """Disable tqdm for the duration of the block, then restore it.

    The SVI training loop draws a progress bar per update, which would otherwise emit
    thousands of lines during a sweep. Patching ``__init__`` takes effect at construction
    time rather than import time, which is what lets the suppression live here instead of
    forcing the imports out of order.

    It is scoped to a context manager rather than applied at module level because this
    module now ships inside the package: a module-level patch would mutate ``tqdm`` for the
    whole process on import, silencing progress bars for unrelated library code — including
    every other test in a pytest session that collects this module. Setting ``TQDM_DISABLE``
    at import instead has exactly the same process-wide reach, so it is not an alternative.
    """
    original = tqdm.__init__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    try:
        yield
    finally:
        tqdm.__init__ = original


#: Sweep settings for the full benchmark, keyed by the argument they supply a default for.
#:
#: ``update_sizes`` is capped at 10k because beyond that the SVI path dominates the run
#: without changing the shape of the curve. ``predict_sizes`` includes ``1`` deliberately:
#: it is the per-request shape in a serving path, and where fixed overhead is visible.
#: ``action_counts`` is swept because ``predict`` cost grows with the number of arms and not
#: only with n. ``update_batches`` drives the incremental/minibatch ``update`` comparison.
FULL_DEFAULTS: Dict[str, Union[PositiveInt, Tuple[PositiveInt, ...]]] = {
    "update_sizes": (100, 1_000, 10_000),
    "predict_sizes": (1, 100, 1_000, 10_000),
    "action_counts": (2, 10),
    "update_batches": (1, 100),
    "repeats": 3,
}

#: Sweep settings ``--quick`` substitutes, so the workflow can run on every pull request
#: without the SVI sweep dominating CI time. Deliberately carries the same keys as
#: :data:`FULL_DEFAULTS`: the two tables then read as a diff, and what ``--quick`` gives up
#: in coverage is exactly what differs between them rather than being spelled out inline.
QUICK_DEFAULTS: Dict[str, Union[PositiveInt, Tuple[PositiveInt, ...]]] = {
    "update_sizes": (100,),
    "predict_sizes": (1, 100),
    "action_counts": (2,),
    "update_batches": (100,),
    "repeats": 1,
}

#: Settings that are identical in both sweeps, and so belong in neither table.
DEFAULT_N_FEATURES: PositiveInt = 5
DEFAULT_N_OBJECTIVES: PositiveInt = 2
DEFAULT_SEED: NonNegativeInt = 0

#: Hidden layer widths for the extra ``CmabBernoulli`` configuration. The default cMAB is a
#: single linear layer, so a sweep of only the defaults never measures the cost of depth;
#: this adds one hidden layer wide enough for that cost to be visible.
CMAB_HIDDEN_DIMS: Tuple[PositiveInt, ...] = (256,)

#: ``model`` column value for that configuration. Named here rather than inline so the CSV
#: label and the network it describes cannot drift apart.
CMAB_HIDDEN_SPEC_NAME: str = f"CmabBernoulliHidden{'x'.join(str(dim) for dim in CMAB_HIDDEN_DIMS)}"

#: Step budget for the ``fixed`` cMAB pass — small enough to keep a sweep tractable while
#: still dominating one-off overhead.
FIXED_NUM_STEPS: PositiveInt = 20

#: ``epochs``/``batch_size`` for the ``epochs`` cMAB pass. The batch size is what makes the
#: step count scale with n; see the module docstring.
EPOCHS: PositiveInt = 2
EPOCH_BATCH_SIZE: PositiveInt = 128


class Result(PyBanditsBaseModel):
    """One timing measurement.

    Attributes
    ----------
    family : str
        ``smab`` or ``cmab``.
    model : str
        Class name of the benchmarked model.
    method : str
        ``update`` or ``predict``.
    pass_name : str
        ``n/a`` for sMABs (closed-form, no step budget), otherwise ``fixed`` or ``epochs``.
    n_samples : PositiveInt
        Number of rows in the call.
    n_actions : PositiveInt
        Size of the action set.
    batch_size : NonNegativeInt
        Rows per call for ``update``; ``0`` means a single call with all rows, which is why
        this is the one count that is not a :class:`PositiveInt`.
    n_calls : PositiveInt
        Number of calls made to cover ``n_samples``.
    steps : NonNegativeInt
        SVI steps per network fit, summed over the calls in this measurement, or ``0`` for
        the closed-form sMABs — the other count whose zero is meaningful. See
        :func:`steps_per_update` for what "per fit" excludes.
    seconds : float
        Median wall-clock time over the repeats, for the whole measurement.
    seconds_per_obs : float
        ``seconds / n_samples`` — the number that stays comparable across batch sizes.
    """

    family: str
    model: str
    method: str
    pass_name: str
    n_samples: PositiveInt
    n_actions: PositiveInt
    batch_size: NonNegativeInt
    n_calls: PositiveInt
    steps: NonNegativeInt
    seconds: float
    seconds_per_obs: float


class ModelSpec(PyBanditsBaseModel):
    """A benchmarkable model and the arguments its constructor needs.

    Attributes
    ----------
    name : str
        Label for the ``model`` column. The class name, suffixed where several configurations
        of the same class are benchmarked.
    factory : Callable[..., object]
        Callable taking ``(n_actions, **kwargs)`` and returning a cold-start model. The
        ``kwargs`` carry whatever the pass adds — ``n_features``, ``update_kwargs``, the seed.
    contextual : bool
        Whether the model requires a context matrix.
    multi_objective : bool
        Whether rewards must be supplied as one list per objective.
    """

    name: str
    factory: Callable[..., object]
    contextual: bool
    multi_objective: bool


def _action_ids(n_actions: PositiveInt) -> List[str]:
    """Stable action ids for an action set of size ``n_actions``."""
    return [f"a{i}" for i in range(n_actions)]


def _action_costs(n_actions: PositiveInt) -> Dict[str, float]:
    """Deterministic per-action costs for the cost-control (``CC``) variants."""
    return {action_id: float(i + 1) for i, action_id in enumerate(_action_ids(n_actions))}


def build_smab_specs() -> List[ModelSpec]:
    """Every stochastic (non-contextual) Bernoulli MAB exposed by the library."""
    return [
        ModelSpec(
            name="SmabBernoulli",
            factory=lambda n_actions, **kwargs: SmabBernoulli.cold_start(
                action_ids=set(_action_ids(n_actions)), **kwargs
            ),
            contextual=False,
            multi_objective=False,
        ),
        ModelSpec(
            name="SmabBernoulliBAI",
            factory=lambda n_actions, **kwargs: SmabBernoulliBAI.cold_start(
                action_ids=set(_action_ids(n_actions)), **kwargs
            ),
            contextual=False,
            multi_objective=False,
        ),
        ModelSpec(
            name="SmabBernoulliCC",
            factory=lambda n_actions, **kwargs: SmabBernoulliCC.cold_start(
                action_ids_cost=_action_costs(n_actions), **kwargs
            ),
            contextual=False,
            multi_objective=False,
        ),
        ModelSpec(
            name="SmabBernoulliMO",
            factory=lambda n_actions, **kwargs: SmabBernoulliMO.cold_start(
                action_ids=set(_action_ids(n_actions)), n_objectives=DEFAULT_N_OBJECTIVES, **kwargs
            ),
            contextual=False,
            multi_objective=True,
        ),
        ModelSpec(
            name="SmabBernoulliMOCC",
            factory=lambda n_actions, **kwargs: SmabBernoulliMOCC.cold_start(
                action_ids_cost=_action_costs(n_actions), n_objectives=DEFAULT_N_OBJECTIVES, **kwargs
            ),
            contextual=False,
            multi_objective=True,
        ),
    ]


def build_cmab_specs() -> List[ModelSpec]:
    """Every contextual Bernoulli MAB exposed by the library."""
    return [
        ModelSpec(
            name="CmabBernoulli",
            factory=lambda n_actions, **kwargs: CmabBernoulli.cold_start(
                action_ids=set(_action_ids(n_actions)), **kwargs
            ),
            contextual=True,
            multi_objective=False,
        ),
        ModelSpec(
            # Same class as above with one hidden layer, so the sweep separates the cost of
            # the SVI machinery itself from the cost of the network it is fitting. The
            # default cMAB is a single linear layer, so without this row every cMAB timing
            # is a timing of the shallowest possible network.
            name=CMAB_HIDDEN_SPEC_NAME,
            factory=lambda n_actions, **kwargs: CmabBernoulli.cold_start(
                action_ids=set(_action_ids(n_actions)), hidden_dim_list=list(CMAB_HIDDEN_DIMS), **kwargs
            ),
            contextual=True,
            multi_objective=False,
        ),
        ModelSpec(
            name="CmabBernoulliBAI",
            factory=lambda n_actions, **kwargs: CmabBernoulliBAI.cold_start(
                action_ids=set(_action_ids(n_actions)), **kwargs
            ),
            contextual=True,
            multi_objective=False,
        ),
        ModelSpec(
            name="CmabBernoulliCC",
            factory=lambda n_actions, **kwargs: CmabBernoulliCC.cold_start(
                action_ids_cost=_action_costs(n_actions), **kwargs
            ),
            contextual=True,
            multi_objective=False,
        ),
        ModelSpec(
            name="CmabBernoulliMO",
            factory=lambda n_actions, **kwargs: CmabBernoulliMO.cold_start(
                action_ids=set(_action_ids(n_actions)), n_objectives=DEFAULT_N_OBJECTIVES, **kwargs
            ),
            contextual=True,
            multi_objective=True,
        ),
        ModelSpec(
            name="CmabBernoulliMOCC",
            factory=lambda n_actions, **kwargs: CmabBernoulliMOCC.cold_start(
                action_ids_cost=_action_costs(n_actions), n_objectives=DEFAULT_N_OBJECTIVES, **kwargs
            ),
            contextual=True,
            multi_objective=True,
        ),
    ]


def make_rewards(spec: ModelSpec, n_samples: PositiveInt, rng: np.random.Generator) -> List:
    """Binary rewards in the shape the model expects.

    Multi-objective models take one reward per objective per sample; the rest take a scalar.
    """
    if spec.multi_objective:
        return rng.integers(0, 2, size=(n_samples, DEFAULT_N_OBJECTIVES)).tolist()
    return rng.integers(0, 2, size=n_samples).tolist()


def time_call(make_call: Callable[[], Callable[[], object]], repeats: PositiveInt) -> float:
    """Median wall-clock seconds over ``repeats``, after one warm-up, on fresh state each time.

    ``make_call`` is a factory rather than the callable itself, and it is invoked *outside*
    the timer. ``update`` mutates the model, so reusing one instance would mean the second
    and later repeats time an already-fitted model being retrained on data it has already
    seen — a different operation from the cold update the row claims to report. Each
    iteration therefore gets its own model, built untimed.

    The warm-up is not optional on the cMAB path: the first JAX call compiles, and that cost
    is unrelated to the quantity being measured. The median (rather than the mean) keeps a
    single scheduling hiccup from dominating a short measurement.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    make_call()()  # warm-up: pays JIT compilation and any first-call allocation
    timings = []
    for _ in range(repeats):
        call = make_call()  # fresh model, constructed outside the timed region
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def steps_per_update(model: object, n_samples: PositiveInt) -> NonNegativeInt:
    """SVI steps a single network fit of ``n_samples`` rows will take.

    Returns ``0`` where the notion does not apply — the conjugate sMABs have no step budget,
    since their update is closed form.

    Read from the model's own ``compute_epoch_steps`` so the reported figure is what the
    library will actually run, rather than a re-derivation that could drift from it. The
    multi-objective models hold one network per objective under ``.models``; they are
    configured identically, so the per-fit budget is read from the first. Note this is the
    budget *per fit*, not per ``update`` call: an update fits a network per updated action
    (and per objective for the MO variants).
    """
    for action_model in (getattr(model, "actions", {}) or {}).values():
        candidates = list(getattr(action_model, "models", []) or []) or [action_model]
        compute = getattr(candidates[0], "compute_epoch_steps", None)
        if compute is not None:
            return int(sum(compute(n_samples)))
    return 0


def total_update_steps(
    model: object,
    spec: ModelSpec,
    actions: Sequence[str],
    slices: Sequence[slice],
) -> NonNegativeInt:
    """Total SVI steps the whole measurement will run.

    An ``update`` fits one network per action present in the call, and each of those
    networks sees only *its own* rows — so the budget is not
    ``steps_per_update(chunk) * n_calls``, which would charge every action for the whole
    chunk. This sums ``compute_epoch_steps`` over the per-action row counts of each call,
    and multiplies by the objective count for the multi-objective variants, which hold one
    network per objective.

    Returns ``0`` for the closed-form sMABs, which have no step budget.
    """
    per_objective = DEFAULT_N_OBJECTIVES if spec.multi_objective else 1
    total = 0
    for sl in slices:
        counts: Dict[str, int] = {}
        for action_id in actions[sl]:
            counts[action_id] = counts.get(action_id, 0) + 1
        for rows in counts.values():
            total += steps_per_update(model, rows) * per_objective
    return total


def benchmark_predict(
    spec: ModelSpec,
    family: str,
    pass_name: str,
    model_kwargs: Dict,
    sizes: Sequence[PositiveInt],
    action_counts: Sequence[PositiveInt],
    n_features: PositiveInt,
    repeats: PositiveInt,
    seed: NonNegativeInt,
) -> List[Result]:
    """Time ``predict`` across sample sizes and action-set sizes.

    Unlike ``update``, ``predict`` does not mutate the model, so one instance is reused
    across the repeats: rebuilding it would only add cold-start cost outside the timer
    without changing what is measured.

    The two families differ only in what ``predict`` is given — a cMAB takes the context
    matrix, whose row count *is* n, and a sMAB takes the row count itself — so that
    difference is carried in the kwargs and both are timed through one call shape.
    """

    def make_call(model: object, **kwargs) -> Callable[[], object]:
        """Zero-argument callable running one ``predict`` with the given kwargs."""
        return lambda: model.predict(**kwargs)

    results: List[Result] = []
    for n_actions in action_counts:
        rng = np.random.default_rng(seed)
        model = spec.factory(n_actions=n_actions, random_seed=seed, **model_kwargs)
        for n_samples in sizes:
            predict_kwargs = (
                {"context": rng.normal(size=(n_samples, n_features))} if spec.contextual else {"n_samples": n_samples}
            )
            seconds = time_call(partial(make_call, model, **predict_kwargs), repeats)
            results.append(
                Result(
                    family=family,
                    model=spec.name,
                    method="predict",
                    pass_name=pass_name,
                    n_samples=n_samples,
                    n_actions=n_actions,
                    batch_size=0,
                    n_calls=1,
                    steps=0,
                    seconds=seconds,
                    seconds_per_obs=seconds / n_samples,
                )
            )
    return results


def benchmark_update(
    spec: ModelSpec,
    family: str,
    pass_name: str,
    model_kwargs: Dict,
    sizes: Sequence[PositiveInt],
    batches: Sequence[PositiveInt],
    n_actions: PositiveInt,
    n_features: PositiveInt,
    repeats: PositiveInt,
    seed: NonNegativeInt,
) -> List[Result]:
    """Time ``update`` as a single batched call and as smaller repeated calls.

    ``batch_size=0`` denotes one call carrying every row; any other value splits the same
    rows into calls of that size, which is what a serving path actually does.
    """
    results: List[Result] = []
    for n_samples in sizes:
        for batch_size in (0, *batches):
            if batch_size and batch_size > n_samples:
                continue
            rng = np.random.default_rng(seed)
            action_ids = _action_ids(n_actions)
            actions = [action_ids[i % n_actions] for i in range(n_samples)]
            rewards = make_rewards(spec, n_samples, rng)
            context = rng.normal(size=(n_samples, n_features)) if spec.contextual else None

            chunk = batch_size or n_samples
            slices = [slice(i, min(i + chunk, n_samples)) for i in range(0, n_samples, chunk)]

            # A factory, not a closure over one shared model: update mutates, so every
            # repeat needs its own cold instance (built untimed inside time_call).
            def make_call(slices=slices, actions=actions, rewards=rewards, context=context):
                model = spec.factory(n_actions=n_actions, random_seed=seed, **model_kwargs)

                def call() -> None:
                    for sl in slices:
                        kwargs = {"actions": actions[sl], "rewards": rewards[sl]}
                        if spec.contextual:
                            kwargs["context"] = context[sl]
                        model.update(**kwargs)

                return call

            seconds = time_call(make_call, repeats)
            probe = spec.factory(n_actions=n_actions, random_seed=seed, **model_kwargs)
            results.append(
                Result(
                    family=family,
                    model=spec.name,
                    method="update",
                    pass_name=pass_name,
                    n_samples=n_samples,
                    n_actions=n_actions,
                    batch_size=batch_size,
                    n_calls=len(slices),
                    steps=total_update_steps(probe, spec, actions, slices),
                    seconds=seconds,
                    seconds_per_obs=seconds / n_samples,
                )
            )
    return results


def cmab_passes(
    fixed_num_steps: PositiveInt, epochs: PositiveInt, epoch_batch_size: PositiveInt
) -> List[Tuple[str, Dict]]:
    """The two cMAB configurations, as ``(pass_name, cold_start kwargs)``.

    ``fixed`` pins the step budget and disables early stopping, so per-step cost is
    comparable across models and sample sizes. ``epochs`` is the realistic setting; the
    explicit ``batch_size`` is what makes the step count grow with the data.
    """
    return [
        (
            "fixed",
            {"update_kwargs": {"num_steps": fixed_num_steps, "early_stopping_kwargs": None}},
        ),
        (
            "epochs",
            {"update_kwargs": {"epochs": epochs, "batch_size": epoch_batch_size}},
        ),
    ]


def run(args: argparse.Namespace) -> List[Result]:
    """Run every selected benchmark and return the collected rows."""
    results: List[Result] = []

    if args.family in ("smab", "all"):
        for spec in build_smab_specs():
            print(f"[smab] {spec.name}", file=sys.stderr)
            results += benchmark_predict(
                spec,
                "smab",
                "n/a",
                {},
                args.predict_sizes,
                args.action_counts,
                args.n_features,
                args.repeats,
                args.seed,
            )
            results += benchmark_update(
                spec,
                "smab",
                "n/a",
                {},
                args.update_sizes,
                args.update_batches,
                args.action_counts[0],
                args.n_features,
                args.repeats,
                args.seed,
            )

    if args.family in ("cmab", "all"):
        for spec in build_cmab_specs():
            for pass_name, pass_kwargs in cmab_passes(args.fixed_num_steps, args.epochs, args.epoch_batch_size):
                # Every contextual cold_start requires the context width up front.
                model_kwargs = {**pass_kwargs, "n_features": args.n_features}
                print(f"[cmab:{pass_name}] {spec.name}", file=sys.stderr)
                results += benchmark_predict(
                    spec,
                    "cmab",
                    pass_name,
                    model_kwargs,
                    args.predict_sizes,
                    args.action_counts,
                    args.n_features,
                    args.repeats,
                    args.seed,
                )
                results += benchmark_update(
                    spec,
                    "cmab",
                    pass_name,
                    model_kwargs,
                    args.update_sizes,
                    args.update_batches,
                    args.action_counts[0],
                    args.n_features,
                    args.repeats,
                    args.seed,
                )

    return results


def write_csv(results: Sequence[Result], path: Optional[str]) -> None:
    """Write ``results`` as CSV to ``path``, or to stdout when ``path`` is ``None``."""
    columns = list(Result.model_fields)
    handle = open(path, "w", newline="", encoding="utf-8") if path else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: getattr(result, column) for column in columns})
    finally:
        if path:
            handle.close()


def _positive_int(raw: str) -> PositiveInt:
    """Parse a single positive integer (argparse type hook).

    ``--repeats 0`` would leave no timings after the warm-up and report ``NaN``, so it is
    rejected at the boundary rather than producing an unusable CSV.
    """
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}")
    return value


def _non_negative_int(raw: str) -> NonNegativeInt:
    """Parse a single non-negative integer (argparse type hook).

    ``--seed`` is annotated ``NonNegativeInt`` on ``benchmark_predict`` and ``benchmark_update``,
    so a negative value is rejected at the boundary rather than at the model call.
    """
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {raw!r}")
    return value


def _positive_ints(raw: str) -> Tuple[PositiveInt, ...]:
    """Parse a comma-separated list of positive integers (argparse type hook)."""
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"expected comma-separated positive integers, got {raw!r}")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command-line interface.

    ``--quick`` exists so the workflow can run the benchmark on every push without the SVI
    sweep dominating CI time; it shrinks the sweeps but exercises the identical code path.

    The sweep values themselves live in :data:`FULL_DEFAULTS` and :data:`QUICK_DEFAULTS`,
    which carry the same keys: the parser takes its defaults from the first, and ``--quick``
    substitutes the second wholesale. Reading the two tables against each other is the whole
    description of what the reduced sweep gives up. The full sweep is orders of magnitude
    more expensive on the cMAB path — it is meant for a manual ``workflow_dispatch``, not
    for a pull request.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=("smab", "cmab", "all"), default="all", help="Model family to benchmark.")
    parser.add_argument(
        "--update-sizes",
        type=_positive_ints,
        default=FULL_DEFAULTS["update_sizes"],
        help="Comma-separated sample sizes for update.",
    )
    parser.add_argument(
        "--predict-sizes",
        type=_positive_ints,
        default=FULL_DEFAULTS["predict_sizes"],
        help="Comma-separated sample sizes for predict.",
    )
    parser.add_argument(
        "--action-counts",
        type=_positive_ints,
        default=FULL_DEFAULTS["action_counts"],
        help="Comma-separated action-set sizes.",
    )
    parser.add_argument(
        "--update-batches",
        type=_positive_ints,
        default=FULL_DEFAULTS["update_batches"],
        help=(
            "Comma-separated batch sizes for the incremental update comparison. A batch size of 1 "
            "means one model fit per row, so pairing it with a large --update-sizes value is "
            "expensive on the cMAB path: the full sweep's 10000 x 1 cell is 10000 sequential SVI "
            "fits per cMAB model, per repeat. --quick reduces it; see QUICK_DEFAULTS."
        ),
    )
    parser.add_argument("--n-features", type=_positive_int, default=DEFAULT_N_FEATURES, help="Context width for cMABs.")
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=FULL_DEFAULTS["repeats"],
        help="Timed repeats per measurement (median reported).",
    )
    parser.add_argument(
        "--seed", type=_non_negative_int, default=DEFAULT_SEED, help="Seed for data generation and the models."
    )
    parser.add_argument(
        "--fixed-num-steps",
        type=_positive_int,
        default=FIXED_NUM_STEPS,
        help="SVI step budget for the 'fixed' cMAB pass.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=EPOCHS, help="Epochs for the 'epochs' cMAB pass.")
    parser.add_argument(
        "--epoch-batch-size",
        type=_positive_int,
        default=EPOCH_BATCH_SIZE,
        help="Batch size for the 'epochs' cMAB pass; this is what makes steps scale with n.",
    )
    parser.add_argument("--output", default=None, help="CSV output path. Defaults to stdout.")
    parser.add_argument("--quick", action="store_true", help="Small sweeps, for CI.")

    args = parser.parse_args(argv)
    if args.quick:
        # Substitute the whole quick table rather than a hand-written block: the reduced
        # sweep is then exactly the QUICK_DEFAULTS entries, and adding a swept setting means
        # adding it to the two tables rather than remembering to shrink it here as well.
        for setting, value in QUICK_DEFAULTS.items():
            setattr(args, setting, value)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    with suppressed_progress_bars():
        results = run(args)
    write_csv(results, args.output)
    print(f"{len(results)} measurements", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
