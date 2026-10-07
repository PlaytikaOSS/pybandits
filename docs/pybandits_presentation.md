---
marp: true
theme: default
paginate: true
size: 16:9
header: 'PyBandits — Multi-Armed Bandits in Python'
footer: 'PlaytikaOSS / pybandits · v7.x'
style: |
  section { font-size: 24px; }
  section.lead h1 { font-size: 56px; }
  code { font-size: 0.85em; }
  table { font-size: 0.80em; }
  h1 { color: #1a5276; }
  h2 { color: #1f618d; }
---

<!-- _class: lead -->

# 🎰 PyBandits

### A Python library for Multi-Armed Bandits
#### sMAB · cMAB · Bayesian Neural Networks · Quantitative Bandits

A walkthrough for Data Scientists

---

## What is PyBandits?

A **Multi-Armed Bandit (MAB)** library built on **Thompson Sampling**.

- **sMAB** — *stochastic* MAB (no context): Bernoulli arms via conjugate Beta posteriors
- **cMAB** — *contextual* MAB: reward depends on features, modeled by a **Bayesian Neural Network**
- **Quantitative bandits** — actions carry a *continuous quantity* (price, dosage, discount…)

Built around three pillars: **Pydantic v2** (typed, serializable models), **NumPyro / JAX** (Bayesian inference), and a clean **predict → update** loop.

```python
smab = SmabBernoulli(actions={"a1": Beta(), "a2": Beta()})
actions, probs = smab.predict(n_samples=100)      # Thompson sampling
smab.update(actions=actions, rewards=rewards)     # posterior update
```

---

## The core loop: every bandit is `predict` → `update`

```
            ┌──────────────────────────────────────────┐
            │              BaseMab                       │
            │  predict()  → sample probabilities,        │
            │               apply strategy, ε-greedy     │
            │  update()   → feed rewards to action models│
            │  get_state()/from_state() → JSON serialize │
            └───────────────┬───────────────┬────────────┘
                            │               │
                    actions_manager      strategy
                   (per-action models)  (how to pick)
```

- `predict()` draws posterior samples → **strategy** chooses an action (with optional **ε-greedy** exploration)
- `update()` routes rewards to each action's model → refreshes the posterior
- Fully **serializable** to JSON (`get_state` / `from_state`) — train, save, reload, continue

---

## sMAB vs. cMAB

| | **sMAB** (`smab.py`) | **cMAB** (`cmab.py`) |
|---|---|---|
| Context? | ❌ No features | ✅ Feature vector per sample |
| Action model | **Beta** (Beta-Bernoulli) | **Bayesian Neural Network** |
| Posterior | Closed-form, conjugate | Approx. (Variational Inference / MCMC) |
| `predict()` input | `n_samples` | `context` matrix `(n, n_features)` |
| Speed | O(1) updates, instant | Heavier — neural inference |
| Use when | Few arms, no covariates | Reward driven by user/item features |

Both inherit the **same `BaseMab` API** — only the action models and the manager differ.

---

## The "type" suffixes — one base, five strategies

Every bandit comes in variants distinguished by a **strategy** suffix:

| Suffix | Class | Strategy | Behavior |
|---|---|---|---|
| *(none)* | `…Bernoulli` | **ClassicBandit** | Greedy on the Thompson sample — pick highest sampled reward |
| **BAI** | `…BernoulliBAI` | **BestActionIdentification** | Explore/exploit between best & 2nd-best (`exploit_p`) |
| **CC** | `…BernoulliCC` | **CostControl** | Cheapest action within `subsidy_factor` of the best reward |
| **MO** | `…BernoulliMO` | **MultiObjective** | Vector rewards → pick from the **Pareto front** |
| **MOCC** | `…BernoulliMOCC` | **MO + CostControl** | Pareto front *and* cost-aware |

> The matrix `{Smab, Cmab} × {—, BAI, CC, MO, MOCC}` gives all the concrete bandit classes.

---

## Class architecture at a glance

```
BaseMab  (predict / update / ε-greedy / serialize)
 ├── actions_manager : ActionsManager
 │    └── meta_model : BaseMetaModel          ← per-action state owner
 │         └── actions : Dict[ActionId, BaseModel]
 │              ├── BaseModelSO  (scalar reward)  → Beta · BNN
 │              ├── BaseModelMO  (vector reward)  → BetaMO · BNN-MO
 │              └── BaseModelCC  (+ cost mixin)   → …CC · …MOCC
 └── strategy : BaseStrategy
      ├── SingleObjectiveStrategy → Classic · BAI · CostControl
      └── MultiObjectiveStrategy  → MO · MOCC
```

**Design principle:** *strategy* (how to choose) is fully decoupled from the *action model* (how to estimate). Mixins (`SO` / `MO` / `CC`) compose orthogonally.

---

## Action models — the estimators behind each arm

`base_model.py` defines composable mixins:

- **`BaseModelSO`** — single objective; tracks `n_successes` / `n_failures`, exposes `mean`, `count`
- **`BaseModelMO`** — multi-objective; wraps a *list* of SO models (one per objective)
- **`BaseModelCC`** — adds a `cost` (fixed value or function of quantity)

Two families implement them:

| Family | Bandit | Idea |
|---|---|---|
| **Beta** | sMAB | Conjugate Beta-Bernoulli, count-based |
| **Bayesian Neural Network** | cMAB | Learns reward = f(context), with weight uncertainty |
| **Zooming / QBNN** | quantitative | Continuous action quantity |

---

## Beta model (sMAB) — exact & instant

The conjugate **Beta-Bernoulli** posterior. No features, just counts.

- State: `n_successes`, `n_failures` (both start at 1 → uniform prior)
- **Update:** for each binary reward, increment success/failure counts → posterior stays exact
- **Sample (Thompson):**

```python
rng.beta(n_successes, n_failures, size=n_samples)   # one draw = a plausible true success rate
```

✅ Closed-form · O(1) · interpretable · uncertainty (Beta variance) naturally drives exploration

Variants: `Beta`, `BetaCC` (+cost), `BetaMO` (vector), `BetaMOCC`.

---

## Bayesian Neural Network (cMAB) — why & what

When the reward depends on **context features**, a point-estimate classifier can't express *uncertainty* — and uncertainty is what Thompson Sampling needs to explore.

A **BNN** puts a **distribution over every weight & bias** instead of a single value.

```
context ─▶ [Hidden₁] ─▶ [Hidden₂] ─▶ … ─▶ [Output neuron] ─▶ σ(·) ─▶ P(reward=1)
            weights ~ prior     each weight is a distribution, not a scalar
```

- **One-layer BNN = Bayesian Logistic Regression** — the simplest, most common config
  (input → single output neuron, sigmoid). This is the cMAB default.
- Add `hidden_dim_list=[16, 8]` → a deeper non-linear BNN.

References in code: *Neal 1995*; *Blundell et al., "Weight Uncertainty in NNs", ICML 2015*.

---

## BNN — the building blocks

| Component | Role |
|---|---|
| **`BnnLayerParams`** | Per layer: a `weight` distribution `(in×out)` + a `bias` distribution `(out)` |
| **`BnnParams`** | Whole network: layer params + initial priors (for reset) + embeddings |
| **`NormalArray`** | Gaussian prior `N(μ,σ)` — L2-like, fast |
| **`StudentTArray`** | Heavy-tailed Student-t prior — robust to outliers (**default**) |
| **`FeaturesConfig`** | Describes mixed numerical + categorical inputs |
| **`EmbeddingParams`** | Bayesian embedding matrices for categorical features |

Priors are **location-scale** (`μ`, `σ`, optionally `ν` for Student-t). `cold_start()` builds them; optional **layerwise scaling** divides σ by √(input_dim) for smoother, GP-like behavior.

---

## BNN — how inference works (NumPyro / JAX)

`update()` fits the posterior; `sample_proba()` does Thompson sampling.

**Two inference engines** (set by `update_method`):

- **VI (default)** — Variational Inference (ADVI / full-rank ADVI)
  - Maximizes the **ELBO** with `optax` optimizers, runs inside XLA via `jax.lax.scan`
  - Fast, scalable, supports **mini-batching** and **epochs**
  - Guide: `ParameterizedScaleAutoNormal` → enables **warm-starting** from the previous posterior
- **MCMC** — NUTS sampler; asymptotically exact, slower (warmup + samples + chains)

**Forward pass** (`sample_proba`): sample weights & embeddings once → run `x·W + b` layer by layer → sigmoid → `(probability, logit)` per draw.

---

## BNN — the knobs that matter

- **`update_method`** `"VI"` / `"MCMC"`; VI sub-methods `advi` / `fullrank_advi`
- **Epochs / mini-batches** — `epochs`, `batch_size`, `num_steps`
- **Early stopping** — `patience`, `tolerance` on ELBO convergence
- **Warm start** — re-init the variational posterior from the last fit (online learning)
- **Bias priors** — `bias_std` constrains output logit; **`calibrate_output_bias`** sets the
  intercept to `logit(empirical reward rate)` on the first update → avoids over-optimistic exploration
- **Recent VI stability options** — `num_particles` (ELBO samples/step),
  **gradient clipping** (`gradient_clip_norm`), **KL temperature scaling**, LR schedules
- **Categorical embeddings** — learn dense Bayesian vectors for high-cardinality columns

---

## Quantitative bandits — actions with a *value*

A **quantitative action** = a discrete action **+ a continuous quantity**.
Examples: *promo with a discount rate*, *treatment with a dosage*, *strategy with a price*.

Two complementary approaches:

| | **Zooming** | **QBNN** (Quantitative BNN) |
|---|---|---|
| Quantity handling | Adaptively **discretized** into segments | Fed as an **input feature** to the BNN |
| Exploration | Per-segment Beta uncertainty | Full posterior over network weights |
| Scaling | Capped # of segments | Single network, high-dim friendly |
| Interpretability | Clear segment → reward map | Black-box, but flexible |

---

## Zooming — adaptive discretization

Splits the continuous space into **segments** (hyper-rectangles), each with its own Beta model, and **refines the grid where it matters**:

```
  reward
    ▲           coarse everywhere ──▶ zoom into the promising region
    │     ░░░░░░░░░░░░          ░░░░░░░░│▓▓│░░░░░░
    │                                   split "interesting" segments
    └──────────────────────────────────────────▶ quantity
```

- **Interesting** segments (sampled more than average) → **split** into finer sub-segments
- **Nuisance** segments (rarely sampled) → **merge** with similar neighbors
  (similarity via **Jensen-Shannon divergence**)
- Bounded by `n_max_segments`; tuned by `segment_update_factor`, `comparison_threshold`

---

## QBNN — quantity as a learned input

Instead of discretizing, **concatenate the quantity with the context** and let the BNN learn the response surface:

```python
bnn_input = [quantity_1 … quantity_d, context_1 … context_n]
P(reward) = BNN(bnn_input)        # end-to-end, continuous
```

- `sample_proba()` returns **callables**: give it a quantity → get a probability.
  The bandit then **optimizes over the quantity** to select the best action+value.
- Inherits all BNN machinery (VI/MCMC, embeddings, priors, warm-start).
- **`QuantitativeBayesianNeuralNetworkCC`** adds a `cost(quantity)` for reward-vs-cost trade-offs.

Key enabler (#121): `sample_proba` was split into **weight sampling** + **forward pass**, so the network can be evaluated at many quantities *without re-sampling weights*.

---

## ActionsManager & the Meta-Model

**`ActionsManager`** owns the per-action models and handles two extra concerns:

- **Adaptive windowing / change detection** — when `delta` is set, it watches for
  reward-distribution shifts (KL-based test) and **resets + retrains on a trimmed window**
  → robust to **non-stationary** environments.
- **Memory** — replays `actions_memory` / `rewards_memory` for warm updates.

**`BaseMetaModel`** (refactor #140) sits *between* the manager and the per-action models, delegating `sample_proba` / `update` / `reset`. `CmabPerActionMetaModel` enforces cross-action consistency (same `input_dim`, same update config). This abstraction opens the door to **shared backbones** in the future.

---

## Beyond the core: extra capabilities

- **Transfer learning** (`transfer.py`, #120) — merge two MABs, edit actions on the fly,
  **expand the cMAB feature dimension** or BNN architecture without losing learned state
- **Offline Policy Evaluation** (`offline_policy_evaluator.py`, #59) — estimate a policy's value
  from logged data (propensity scoring + estimators), with Optuna tuning & Bokeh visualization
- **Simulators** (`smab_simulator.py`, `cmab_simulator.py`) — synthetic environments for testing
- **Serialization everywhere** — JSON state with backward-compatible migration

---

## How the package evolved — highlights

| PR | Milestone |
|---|---|
| #59 | **Offline Policy Evaluation** module |
| #72 | **Zooming** quantitative bandit (continuous action spaces) + SO/MO/CC base classes |
| **#89** | **Bayesian Neural Network** introduced → powers cMAB (Bayesian logistic regression) |
| #107 | **Multi-objective** support across cMAB & BNN (MO / MOCC) |
| #115/#116 | BNN with Normal priors, **layerwise scaling**, early stopping |
| #120 | **Transfer learning** (merge MABs, grow feature dims) |
| #121 | **QBNN** — quantitative action selection on top of BNN |
| #122 | **Categorical embeddings** in BNN |

---

## How the package evolved — the big shift

| PR | Milestone |
|---|---|
| **#129** | 🚀 **Migration PyMC → NumPyro/JAX** — re-platformed all BNN inference |
| #133/#141 | Drop legacy state migration & Pydantic v1 → **Pydantic v2 only** |
| #140 | **`BaseMetaModel`** abstraction for cMAB action managers |
| #142/#143 | BNN bias priors (`bias_std`), `calibrate_output_bias` |
| #144 | **Python 3.13 / 3.14** support |
| latest | VI training options: `num_particles`, **gradient clipping**, **KL temperature scaling** |

### Why PyMC → NumPyro?
JAX-backed **JIT/XLA** compilation, mini-batched SVI in-graph (`lax.scan`), faster MCMC (NUTS),
and `optax` optimizers — major **speed & scalability** gains for the BNN.

---

## Takeaways

- **One API, many bandits:** `predict` / `update` on top of `BaseMab`, fully serializable.
- **Two axes of choice:** the *model* (Beta ↔ BNN ↔ Zooming/QBNN) × the *strategy*
  (Classic / BAI / CC / MO / MOCC).
- **Beta** = exact & instant for contextless problems; **BNN** = uncertainty-aware deep model for contextual ones; a **one-layer BNN is Bayesian logistic regression**.
- **Quantitative bandits** handle continuous action values — discretized (**Zooming**) or learned (**QBNN**).
- The codebase moved to a **modern Bayesian stack (NumPyro/JAX + Pydantic v2)** and gained transfer learning, OPE, multi-objective, embeddings, and non-stationarity handling.

---

<!-- _class: lead -->

# Thank you 🙏

### Questions?

`pip install pybandits` · github.com/PlaytikaOSS/pybandits
Tutorials: `docs/src/tutorials/` (smab, cmab, bnn, zooming, quant-bnn, ope, transfer learning)
