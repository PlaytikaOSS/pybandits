# Continuous rewards for cMAB

pybandits supports three kinds of reward for contextual bandits:

| reward | type | model | bandit |
|---|---|---|---|
| binary 0 / 1 | `BinaryReward` | `BayesianNeuralNetwork` (Bernoulli) | `CmabBernoulli` (and the CC / MO / DP variants) |
| soft label in [0, 1] | `SoftReward` (`use_soft_rewards=True`) | `BayesianNeuralNetwork` (Bernoulli on fractional targets) | the same |
| real-valued | `ContinuousReward` | `GaussianBayesianNeuralNetwork` | `CmabGaussian` |

The bandit entry points accept `AnyReward`, and the actions manager checks each reward against what its models
support (`ActionsManager._check_rewards`).

## The Gaussian model

Each action is a Bayesian MLP with **one output unit, μ(x)**, and the reward is modelled as `Normal(μ(x), σ)`:

- **σ is a single noise std per action,** not a function of the context. It is a latent (`log σ ~ Normal`) whose
  posterior is stored in the model state (`noise_log_sigma`) and carried across updates, like the weights. Its initial
  prior is centered on `noise_sigma` (reward units) if given, otherwise on the std of the first update batch.
- **Training is on raw rewards by default** (`standardize_rewards=False`). With `standardize_rewards=True` it is on
  standardized targets `(r − reward_loc) / reward_scale`, so the default O(1) weight priors fit rewards on any scale;
  `reward_loc` / `reward_scale` are the values given at cold start, otherwise fitted on the first batch and then frozen.
- **Thompson sampling** draws the weights from the posterior and ranks actions on μ. `predict` returns μ (in the
  "probabilities" slot) and σ (for monitoring) per action and row.
- **`reset()`** returns the model to its cold-start state, including the standardization (re-fitted on the next
  update unless it was given at cold start).

### Why one σ and not σ(x)

With a single noise level the fit of μ is a (Bayesian) least-squares fit, whose target is `E[reward | x]` whatever
the shape of the noise. A context-dependent σ(x) weights each residual by 1/σ(x)². On zero-inflated or skewed rewards
(e.g. revenue), the model can then explain large rewards as noise instead of raising μ, which biases μ low exactly
where the rewards are most variable. That bias is what Thompson sampling would rank on.

### Recommended use: uplift against a control baseline

For rewards with a lot of player-to-player variance (e.g. revenue), train on the **uplift** `r = R − E0(x)`, where
`E0(x)` is the expected reward under a control policy, from a separate baseline model. `E0` doesn't depend on the
action, so the ranking of actions is unchanged, and the shared variance is removed (a control variate). The posterior
then contracts much faster with per-action data volumes. `r` can be negative.

```python
from pybandits.cmab import CmabGaussian

mab = CmabGaussian.cold_start(
    action_ids={"a1", "a2", "a3"},
    n_features=n_features,
    hidden_dim_list=[16],
    dist_type="normal",
    dist_params_init={"mu": 0, "sigma": 0.1},   # a narrow prior fits much faster than the default sigma=1
    update_kwargs={"num_steps": 400, "optimizer_kwargs": {"step_size": 3e-3}},
    # optionally, from historical data rather than the first batch:
    # reward_loc=..., reward_scale=..., noise_sigma=...,
    # decay_factor=...,                         # if the reward process drifts
)
actions, mu, sigma = mab.predict(context=X)
mab.update(actions=actions, rewards=(R - E0).tolist(), context=X)
```

A shared backbone (`backbone_hidden_dims=...`) works as for `CmabBernoulli`; each action keeps its own σ and
standardization.

### Alternatives that were evaluated and rejected

- **σ(x) noise head:** biases μ on skewed rewards (see above).
- **Bernoulli on a [0, 1] rescaling of a continuous reward:** the mean is consistent, but the Bernoulli variance
  p(1 − p) can overstate the real noise by orders of magnitude. Each observation then carries far too little weight,
  and the model learns little beyond an intercept.
- **Hurdle × log-normal amount:** a good fit of the distribution's shape, but the back-transform `exp(m + σ²/2)` can
  bias the level, and Thompson draws of it have extreme tails. If a split of the effect into conversion vs. amount is
  needed, a Bernoulli × Gaussian amount (on the raw scale) is the better hurdle.

### Not supported (yet)

- Gaussian variants of the cost-control, multi-objective, dynamic-pricing and quantitative bandits.
- The adaptive window (`delta`) with continuous rewards.
- Offline policy evaluation and the simulators with continuous rewards.
