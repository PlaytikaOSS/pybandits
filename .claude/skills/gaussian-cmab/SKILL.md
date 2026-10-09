---
name: gaussian-cmab
description: |
  Background and conventions for continuous (real-valued) rewards in pybandits: GaussianBayesianNeuralNetwork,
  CmabGaussian, the uplift reward R − E0(x), and why the model has one μ head with a learned scalar σ.
  Use when: working on GaussianBayesianNeuralNetwork / CmabGaussian, continuous or revenue-like rewards, uplift
  rewards against a control baseline, the extra-site likelihood hook, or extending the Gaussian model (offsets,
  hurdle models, new likelihoods).
license: MIT
metadata:
  author: ronshiff1
  version: "1.0.0"
---

# Gaussian cMAB: continuous rewards

## What exists (pybandits ≥ 8.4.0)

- **`GaussianBayesianNeuralNetwork`** (`pybandits/model/bnn/network.py`):
  - a Bayesian MLP with **one output unit, μ(x)**, and the likelihood `Normal(μ(x), σ)`;
  - **σ is one latent per model** (`log σ ~ Normal`, site `noise_log_sigma`), with its posterior stored in the state
    and carried across updates like the weights;
  - **standardized targets** `(r − reward_loc) / reward_scale`, using the user's values or else fitted on the first
    batch, then frozen. `standardize_rewards=False` trains on raw rewards (and then rejects loc / scale).
- **`CmabGaussian`** (`pybandits/cmab.py`): `ClassicBandit` strategy, `_predict_with_proba = True`. Actions are
  ranked on μ; `predict` returns `(actions, mu, sigma)`, where σ is for monitoring only.
- **Reward types** (`pybandits/base.py`): `BinaryReward`, `SoftReward`, `ContinuousReward`, `AnyReward`.
  - Entry points (`BaseMab.update`, `BaseCmabBernoulli.update`, `ActionsManager.update`) accept `AnyReward`.
  - `ActionsManager._check_rewards` enforces what the models support (`BaseModel.supports_continuous_rewards`).
  - `delta` + continuous rewards is rejected.

## Core decisions and why

1. **One σ, not σ(x).** The Gaussian NLL weights each residual by 1/σ(x)². On zero-inflated or skewed rewards
   (revenue), a learned σ(x) lets the model call large rewards noise instead of raising μ, so μ is biased low where
   rewards are most variable. In evaluations this under-predicted the level ~3×, and the epistemic sd collapsed
   (confidently wrong). With one σ, the μ fit is least squares, whose target is E[r | x] for any noise shape. The
   σ(x) head was therefore removed entirely.
2. **σ learned, not fixed.** A fixed σ mis-sets the posterior width, i.e. the exploration. The learned σ converged to
   the true residual sd and gave calibrated predictive intervals. Changing σ by hand between updates would make the
   stored posterior inconsistent: never do it. Use `noise_sigma` for the initial prior and `decay_factor` for
   forgetting.
3. **Standardize.** The weight priors are O(1). Raw revenue-scale targets need weights about 100× outside the
   prior, and the model barely moves (with tanh or ReLU alike). Putting the prior on the data scale instead is
   equivalent in theory but badly conditioned for Adam. `loc` / `scale` are frozen because the weights only mean
   something relative to them.
4. **`reset()` = back to the cold-start state:** weights, noise latent, counters, and `loc` / `scale` back to their
   cold-start values. `reward_loc_init` / `reward_scale_init` hold the user-supplied values (None if fitted), so fitted
   values are re-fitted on the next update and user values survive.
5. **Reward = uplift `R − E0(x)`** against a control baseline computed outside pybandits.
   - `E0` doesn't depend on the action, so the ranking is unchanged and the shared variance is removed (a control
     variate).
   - Per-action data volumes are too small to learn revenue from scratch; models without a baseline lost about
     a third of the explained variance.
   - The uplift can be negative, and that is expected.

## Rejected alternatives (don't re-propose without new evidence)

- **σ(x) noise head:** see decision 1.
- **A Bernoulli likelihood on a continuous reward rescaled to [0, 1]** (e.g. `(min(Y, c) − k·baseline + a) / b`):
  - The mean is consistent, but the Bernoulli variance p(1 − p) overstated the real noise ~128×, so each observation
    carried ~1/128 of its information and the model learned only an intercept.
  - Weighting the likelihood by ≈ p(1 − p) / var(z) matches a Gaussian only to second order (same minimum and
    curvature, but a different loss shape, link and per-row weighting). Use the Gaussian directly.
- **Hurdle × log-normal amount** (`p0 · exp(m + σ²/2)`):
  - The best distributional fit (flat PIT), but the back-transform around a baseline biased the level by 10–15%.
  - Averaging or drawing `exp(m + σ²/2)` over posterior samples has extreme tails (10⁸× outliers).
- **The hurdle that works, if a conversion vs. amount split is needed:** a Bernoulli with a `logit p0_control`
  offset × a Gaussian on the raw amount residual `R − E_buy_control`, for buyers only (called "H3n"). It is as
  accurate and calibrated as the single-Gaussian uplift model. It needs per-row offset support (see below).

## Code map (internals)

**Base-class likelihood hooks in `BaseBayesianNeuralNetwork`** (the Bernoulli defaults reproduce the original
behavior):
- `_output_dim` (ClassVar), `create_model_params(..., output_dim=None)`;
- `output_distribution(linear_out, extra_sites)`, `_observe_output`, `_postprocess_output(linear_out, extra_samples)`;
- `prepare_rewards(rewards)` → training targets; `_fit(context, rewards)` (the `_update` body); `_is_first_fit`.

**The extra-site hook, for head-level latents beyond the weights and embeddings:**
- `extra_site_params()` → `{name: BaseLocationScaleArray}`;
- `sample_extra_sites()` (NumPyro, KL-annealed), `sample_extra_numpy(n, rng)` (Thompson);
- `update_extra_params_from_vi()`, `_inflate_extra_params()`.

It is wired into guide init (`collect_guide_init_arrays`), VI readback, the full-rank ADVI site filter, decay and
sampling. A new likelihood latent (e.g. a Student-t ν) only needs these methods.

**Meta-model** (`pybandits/meta_model/cmab_meta_model.py`):
- `_check_models` requires one likelihood class across heads;
- `_so_model` stacks each arm's extra sites and gathers them per row;
- `_store_head` reads them back;
- `_prepare_targets` applies each head's `prepare_rewards` to its own rows (each arm keeps its own standardization).

**Aliases:** `CmabMetaModelGaussian`, `CmabActionsManagerGaussian`.

## Usage

```python
mab = CmabGaussian.cold_start(
    action_ids={...}, n_features=d, hidden_dim_list=[16],
    dist_type="normal", dist_params_init={"mu": 0, "sigma": 0.1},          # narrow prior: default sigma=1 is too slow
    update_kwargs={"num_steps": 400, "optimizer_kwargs": {"step_size": 3e-3}},
    # production: reward_loc / reward_scale / noise_sigma from history; decay_factor if drifting
)
actions, mu, sigma = mab.predict(context=X)
mab.update(actions=actions, rewards=(R - E0).tolist(), context=X)
```

## Gotchas

- **Evaluate the mean with a plug-in** (posterior mean of μ) or by averaging draws of μ. Never average
  `exp(...)` of log-scale draws.
- **A wide prior** (`sigma=1`) on an unbounded likelihood converges slowly. Narrow it (≈0.1) and raise the step size.
- **Additive residual models can predict E0 + μ < 0** on low-reward rows. That's irrelevant for ranking an uplift,
  but floor it when reading the prediction back as revenue.
- **Whale-type under-prediction** inherited from the baseline isn't fixed by the BNN. Fix the baseline (more history).

## Possible next work

- **Per-row offset support in `BaseBayesianNeuralNetwork`**, for the H3n hurdle: `logit = offset + f(x)`.
  - A working prototype treats the context's last column as the offset: it strips it in `check_context_matrix`,
    `emit_submodel` and `forward_pass`, and adds it in `_observe_output` / `_postprocess_output`. It is full-batch
    only.
  - A proper version must carry the offset through the data plate (minibatching) and the joint meta-model.
- **Gaussian variants** of CC / MO / DP / quantitative; `delta` with continuous rewards; OPE and simulators.

## Tests

- `tests/test_gaussian_bnn.py` (the project style: see the `pytest-writer` skill).
- Run with `PYTHONPATH=$PWD python -m pytest tests/test_gaussian_bnn.py -q -n 8`. Force the local checkout: some
  environments have another checkout installed as editable.
- Pre-existing flaky failures, unrelated:
  - `test_model.py::{test_bnn_svi_nan_loss_raises_error, test_vi_training_options, test_bnn_vi_update_with_categorical_features_updates_embeddings}`;
  - `test_quantitative_model.py::test_sample_proba_reproducible_with_same_rng`.
