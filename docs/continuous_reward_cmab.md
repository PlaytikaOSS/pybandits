# Continuous rewards in [-1, 1] for cMAB

## Current state

The library does not take -1/1 rewards. It takes binary {0, 1} rewards:

- `BinaryReward = conint(ge=0, le=1)` in `pybandits/base.py`.
- Every cMAB model uses a Bernoulli likelihood on a logit:
  - `pybandits/model/bnn/network.py` (`NumpyroBernoulli(logits=logit)`)
  - `pybandits/meta_model/cmab_meta_model.py` (`NumpyroBernoulli(logits=logit)`)

So any -1/1 reward is already being mapped to 0/1 before it reaches the library.

## Option A: rescale to [0, 1] and keep the Bernoulli model (easy)

Map the reward with `y = (r + 1) / 2`, which puts it in [0, 1].

### A1. Binarization trick (no library change)

For each observation, draw `b ~ Bernoulli((r + 1) / 2)` and pass `b` as the reward.

- This is Agrawal & Goyal's standard way to extend Thompson Sampling to bounded rewards, and the regret guarantees still hold.
- Cost: extra noise, so the model learns somewhat slower.
- It can be used today with no code change.

### A2. Pass the fractional target directly

This fits a logistic regression with targets that are not 0/1. The Bernoulli log-likelihood with logits is BCE-with-logits, which is valid for any `y` in [0, 1], and numpyro does not check the support by default.

Changes needed:

- Relax `BinaryReward` to a float in [0, 1]. The type appears in many signatures, but the change is mechanical.
- Check `_calibrate_output_bias`. It should still work because it uses the mean reward.
- Update the simulators and tests.

Behaviour:

- The mean is estimated correctly: `E[y | x] = sigmoid(f)`.
- The uncertainty is not. The model assumes variance `p(1 - p)`, which is the largest possible variance for a variable in [0, 1]. The posterior is therefore too wide and the bandit **over-explores**. This errs on the safe side.

Estimated effort: 1–2 days, including tests.

## Option B: a proper continuous likelihood (harder)

Use a Gaussian, Beta or truncated-normal likelihood, with an identity or tanh output and a learned noise `σ`.

The likelihood itself is simple. The work is in what is built around it:

- `sample_proba` returns a `Probability` (`Float01`) everywhere, and the strategies and meta-model all rely on that.
- `_calibrate_output_bias` uses `logit(rate)`.
- The CostControl `subsidy_factor` is a **relative** threshold ("within X of the best"). Relative thresholds break once the expected reward can be negative, so they must be rescaled or redefined.
- The OPE estimators, the simulators and the whole test suite need changes.

Estimated effort: 1–2+ weeks. It touches every cMAB / MO / CC variant.

## Do we need to understand the reward distribution first?

**For Option A: mostly no.** Arm selection depends only on the mean, and the Bernoulli model's uncertainty is conservative whatever the shape. A histogram per arm is still worth plotting, because some shapes change the recommendation:

| Reward shape | Implication |
|---|---|
| Mostly at ±1 with a few values in between | The Bernoulli model is almost exact. Use Option A. |
| Spike at 0 (e.g. "no response" coded as 0) | Rewards are zero-inflated. Consider modelling "any response" and "sign/size given a response" separately. |
| Concentrated in the middle (e.g. 0 ± 0.2) | Option A's posterior is much too wide and exploration is slow. Option B with a learned `σ` pays off here. |
| Heteroscedastic (noise varies with context) | Only Option B can model this. |

**For Option B: yes.** The likelihood family and a sensible prior on `σ` have to be chosen from the data.

## Recommendation

1. Start with **A2** (fractional targets), or **A1** for zero code changes.
2. Back-test it on logged data with the existing offline policy evaluator.
3. Build **Option B** only if the histogram shows rewards bunched in the middle and A2 explores too slowly.
