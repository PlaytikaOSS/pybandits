---
name: pytest-writer
description: |
  Write pytest tests matching this project's style: hypothesis + @given for property tests,
  @pytest.mark.parametrize for discrete cases, configuration values injected via fixtures, st.just, or
  default arguments (never read directly from module globals in test bodies), factory fixtures for
  expensive objects, full type annotations and docstrings.
  Use when: asked to write, add, or expand pytest tests for pybandits.
license: MIT
metadata:
  author: shaharb
  version: "1.0.0"
---

# PyTest Writer

You write pytest tests that match the pybandits codebase style exactly.

## Non-Negotiable Rules

1. **No magic numbers or hardcoded string literals in test bodies.**
   Module-level constants are fine, but test functions must declare them as default arguments so the
   test signature documents its inputs and the body never reaches into module globals:
   ```python
   REGION_SPLIT = 0.5
   REGION_TOLERANCE = 1e-2

   def test_quantity_below_split(
       region_split: float = REGION_SPLIT,
       tolerance: float = REGION_TOLERANCE,
   ) -> None:
       """Optimizer stays within the allowed half-interval."""
       assert result <= region_split + tolerance
   ```
   This pattern keeps constants easy to find and change while making every test's dependencies
   explicit in its signature. Never write bare literals in test bodies, and never access module
   constants directly from inside a test function.

2. **No local imports inside test functions.**
   All imports belong at module top-level.

3. **Use hypothesis for any property that should hold across a range of inputs.**
   Use `@given` + `st.*` strategies. For complex object construction use `@st.composite`.

4. **Use `@pytest.mark.parametrize` for discrete, enumerable cases** (e.g., valid vs. invalid enum
   values, model subclass variants, update methods). Never repeat near-identical test functions —
   parametrize them.

5. **Fixtures for expensive or shared state.**
   Use `scope="module"` for objects expensive to construct (BNN, full MAB).
   Use factory fixtures (return a callable) when tests need fresh instances with varying params.

6. **Full type annotations on every function and fixture**, including return types.

7. **Docstrings on every test class and test function** — one short sentence describing what is verified.

## Combining `@given` with constants: use `st.just`

Hypothesis raises `InvalidArgument` if you apply `@given` to a function that has default arguments.
Use `st.just(CONSTANT)` to pass fixed configuration values through hypothesis — this keeps every
dependency explicit in the `@given` call and the function signature, with no module globals in the body:

```python
# plain test — use default args
def test_foo(region_split: float = REGION_SPLIT, tolerance: float = REGION_TOLERANCE) -> None:
    assert x <= region_split + tolerance

# hypothesis test — varying params use real strategies; fixed params use st.just
@given(
    region_split=st.floats(min_value=0.1, max_value=0.9),
    n_samples=st.just(EXPLORE_SAMPLES),
    tolerance=st.just(REGION_TOLERANCE),
)
def test_foo(region_split: float, n_samples: int, tolerance: float) -> None:
    assert x <= region_split + tolerance
```

Do NOT add `@settings` at all — `deadline`, `max_examples`, `suppress_health_check`, and everything
else in `@settings` belong to the engineer, not the test author.

## Constants vs Fixtures

**Module-level constants** are fine for scalar configuration values. Declare them in the test signature
as default arguments so the body never references the module name directly:

```python
MIN_ACTIONS = 2
MAX_ACTIONS = 5

def test_cold_start_action_count(
    min_actions: int = MIN_ACTIONS,
    max_actions: int = MAX_ACTIONS,
) -> None:
    """cold_start accepts any action count in the allowed range."""
    ...
```

**Class-scoped configuration** — shared by every test in one test class — goes on the class as a plain
lowercase attribute, not an uppercase constant. `UPPER_CASE` signals a module-level constant; a class
attribute is namespaced by the class and reads as configuration:

```python
class TestMLPBackbone:
    n_features = 6
    embedding_dim = 4
```

Note that a comprehension in a class body cannot see other class attributes (only its first iterable
expression is evaluated in the enclosing scope), so derive one class attribute from another with a
plain function call rather than an inline comprehension.

**Fixtures** are for objects that require construction, teardown, or should be overrideable per-test
(e.g., a pre-built MAB instance, a seeded RNG, a monkeypatched function):

```python
@pytest.fixture(scope="module")
def smab() -> SmabBernoulli:
    """Pre-built SmabBernoulli for use across the module."""
    return SmabBernoulli.cold_start(action_ids={"a", "b"}, strategy=ClassicBandit())
```

Fixtures compose via injection; constants compose via default arguments. Use each for what it's good at.

## Patterns to Follow

### Hypothesis property test
```python
@given(
    n_successes=st.integers(min_value=1, max_value=100),
    reward=st.floats(min_value=0.0, max_value=1.0),
)
def test_beta_update_increases_counts(n_successes: int, reward: float) -> None:
    """Beta n_successes increments on positive reward."""
    b = Beta(n_successes=n_successes, n_failures=1)
    before = b.n_successes
    b.update(rewards=[reward])
    assert b.n_successes >= before
```

### Composite strategy for structured objects
```python
@st.composite
def action_ids_strategy(draw: st.DrawFn) -> List[str]:
    n = draw(st.integers(min_value=MIN_ACTIONS, max_value=MAX_ACTIONS))
    return [f"action_{i}" for i in range(n)]
```

### Parametrize for variants
```python
@pytest.mark.parametrize("update_method", ["VI", "MCMC"])
def test_bnn_update_methods(update_method: UpdateMethods) -> None:
    """BNN update runs without error for both VI and MCMC."""
    ...
```

### Module-scope factory fixture
```python
@pytest.fixture(scope="module")
def make_bnn():
    """Factory: builds a BayesianNeuralNetwork via cold_start with given params."""
    def _factory(n_features: int, hidden_dim_list: list[int]) -> BayesianNeuralNetwork:
        return BayesianNeuralNetwork.cold_start(
            n_features=n_features,
            hidden_dim_list=hidden_dim_list,
        )
    return _factory
```

### ValidationError testing pattern
```python
@given(n_successes=st.integers(max_value=0))
def test_beta_rejects_non_positive_successes(n_successes: int) -> None:
    """Beta raises ValidationError when n_successes <= 0."""
    with pytest.raises(ValidationError):
        Beta(n_successes=n_successes, n_failures=1)
```

## Import Block Template

```python
from typing import List, Optional

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pybandits.<module> import <ClassUnderTest>
```

## Named-Constant Location

All constants at module top, grouped and commented:
```python
# --- test configuration constants ---
MIN_ACTIONS = 2
MAX_ACTIONS = 5
N_FEATURES = 4
HIDDEN_DIMS = [8, 4]
```

## What NOT to Do

- Never write `assert result == 42` — use a default-argument parameter or derive the expected value from inputs.
- Never reference a module-level constant directly inside a test body — pass it as a default argument (plain tests) or via `st.just` (hypothesis tests) instead.
- Never copy-paste a test and change one number — parametrize instead.
- Never use `unittest.TestCase` — pytest classes only (no inheritance from `TestCase`).
- Never write `scope="function"` for expensive objects — use `scope="module"`.
- Never skip type annotations or docstrings.
- Never import inside a test function body — all imports at module top.
- Never add `@settings` at all — `deadline`, `max_examples`, `suppress_health_check`, and everything else in `@settings` belong to the engineer.
