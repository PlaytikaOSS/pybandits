
PyBandits
=========

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/PlaytikaOSS/pybandits/continuous_integration.yml)
![PyPI - Version](https://img.shields.io/pypi/v/pybandits)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pybandits)
![alt text](https://img.shields.io/badge/license-MIT-blue)

**PyBandits**  is a ``Python`` library for Multi-Armed Bandit. It provides an implementation of stochastic Multi-Armed Bandit (sMAB) and contextual Multi-Armed Bandit (cMAB) based on Thompson Sampling.

For the sMAB, we implemented a Bernoulli multi-armed bandit based on Thompson Sampling algorithm [Agrawal and Goyal, 2012](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf). If context information is available we provide a generalisation of Thompson Sampling for cMAB [Agrawal and Goyal, 2014](https://arxiv.org/pdf/1209.3352.pdf) implemented with [PyMC3](https://peerj.com/articles/cs-55/), an open source probabilistic programming framework  for automatic Bayesian inference on user-defined probabilistic models.

Installation
------------

This library is distributed on [PyPI](https://pypi.org/project/pybandits/) and can be installed with ``pip``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~bash
pip install pybandits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the guidelines of ``pymc3`` authors, it is highly recommended to install the library in a conda environment via the following.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~bash
conda install -c conda-forge pymc3
pip install pybandits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command above will automatically install all the dependencies listed in ``pyproject.toml``. Please visit the
[installation](https://playtikaoss.github.io/pybandits/installation.html)
page for more details.

Getting started
---------------

A short example, illustrating it use. Use the sMAB model to predict actions and update the model based on rewards from the environment.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
import numpy as np
from pybandits.model import Beta
from pybandits.smab import SmabBernoulli

n_samples=100

# define action model
actions = {
    "a1": Beta(),
    "a2": Beta(),
}

# init stochastic Multi-Armed Bandit model
smab = SmabBernoulli(actions=actions)

# predict actions
pred_actions, _ = smab.predict(n_samples=n_samples)
simulated_rewards = np.random.randint(2, size=n_samples)

# update model
smab.update(actions=pred_actions, rewards=simulated_rewards)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation
-------------

For more information please read the full
[documentation](https://playtikaoss.github.io/pybandits/pybandits.html)
and
[tutorials](https://playtikaoss.github.io/pybandits/tutorials.html).

Info for developers
-------------------

The source code of the project is available on [GitHub](https://github.com/playtikaoss/pybandits).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~bash
git clone https://github.com/playtikaoss/pybandits.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the library and the dependencies from the source code with one of the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~bash
poetry install                # install library + dependencies
poetry install --without dev     # install library + dependencies, excluding developer-dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the HTML documentation run the following commands:

~~~~~~~~~~~bash
cd docs/src
make html
~~~~~~~~~~~

Run tests
---------

Tests can be executed with ``pytest`` running the following commands. Make sure to have the library installed before to
run any tests.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~bash
cd tests
pytest -vv                                      # run all tests
pytest -vv test_testmodule.py                   # run all tests within a module
pytest -vv test_testmodule.py -k test_testname  # run only 1 test
pytest -vv -k 'not time'                        # run all tests but not exec time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

License
-------

[MIT License](LICENSE)
