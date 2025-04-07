# Multi-Armed Bandit Library

PyBandits is a Python library for Multi-Armed Bandit (MAB) developed by the Playtika AI lab. We developed this tool in order to provide personalised recommendation. It provides an implementation of stochastic Multi-Armed Bandit (sMAB) and contextual Multi-Armed Bandit (cMAB) based on Thompson sampling.

In a bandit problem, a learner recommends an action to a user and observes a reward from the user for the chosen action. Information is then used to improve learner prediction for the next user.

For the stochastic multi-armed bandit (sMAB),
we implemented a Bernoulli multi-armed bandit based on Thompson sampling algorithm
([Agrawal and Goyal, 2012](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf)).
If user information is available (‘context’), a generalisation of Thompson Sampling for
cMAB ([Agrawal and Goyal, 2014](https://arxiv.org/pdf/1209.3352.pdf)) has been implemented using [PyMC].

Currently, the cMAB implements a robust logistic regression for binary rewards (Bernoulli bandit) with Student-t priors.
A robust regression is less sensitive to outliers since Student’s T for prior distributions have wider tails than Normal distributions.
Despite binary reward is very common, the package will be updated in order to extend to other reward definitions.
However, as an open-source package, modification can easily be applied to our functions in order to modify priors
and/or likelihood distributions as described in [PyMC] documentation for the cMAB.
In order to observe bandit behaviours based on different assumptions (sample size, reward probabilities,
number of actions, context dimension, ...) or validate modifications, a simulation process is available. One can
simulate data using different parameters setting, apply sMAB or cMAB and observe recommendation efficiency.

[PyMC]: https://docs.pymc.io/
