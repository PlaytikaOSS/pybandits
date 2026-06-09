---
name: pytest-writer
description: |
  Write pytest tests matching this project's style: hypothesis + @given for property tests,
  @pytest.mark.parametrize for discrete cases, named module-level constants instead of magic numbers,
  factory fixtures for expensive objects, full type annotations and docstrings.
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
   Every constant lives at module scope with a descriptive name:
   ```python
   N_FEATURES = 5
   HIDDEN_DIM = [8, 4]
   MIN_SAMPLES = 10
   ```

2. **Use hypothesis for any property that should hold across a range of inputs.**
   Use `@given` + `st.*` strategies. For complex object construction use `@st.composite`.
   Add `@settings(max_examples=50)` when tests are expensive (BNN, MCMC).

3. **Use `@pytest.mark.parametrize` for discrete, enumerable cases** (e.g., valid vs. invalid enum values, model subclass variants, update methods).
   Never repeat near-identical test functions — parametrize them.

4. **Fixtures for expensive or shared state.**
   Use `scope="module"` for objects expensive to construct (BNN, full MAB).
   Use factory fixtures (return a callable) when tests need fresh instances with varying params.

5. **Full type annotations on every function and fixture**, including return types.

6. **Docstrings on every test class and test function** — one short sentence describing what is verified.

## Patterns to Follow

### Hypothesis property test
```python
@given(st.integers(min_value=1, max_value=100), st.floats(min_value=0.0, max_value=1.0))
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
@given(st.integers(max_value=0))
def test_beta_rejects_non_positive_successes(n_successes: int) -> None:
    """Beta raises ValidationError when n_successes <= 0."""
    with pytest.raises(ValidationError):
        Beta(n_successes=n_successes, n_failures=1)
```

## Import Block Template

```python
from typing import List, Optional

import pytest
from hypothesis import given, settings
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
MAX_HYPOTHESIS_EXAMPLES = 30
```

## What NOT to Do

- Never write `assert result == 42` — use a named constant or derive the expected value from inputs.
- Never copy-paste a test and change one number — parametrize instead.
- Never use `unittest.TestCase` — pytest classes only (no inheritance from `TestCase`).
- Never write `scope="function"` for expensive objects — use `scope="module"`.
- Never skip type annotations or docstrings.
