[comment]: <> (Modify also docs/installation.rst if change the README.md)
[comment]: <> (Modify also LICENSE.rst if change the README.md)

PyBandits
=========

[comment]: <> (Modify also docs/badges.rst if changing the badges)
![alt text](https://img.shields.io/badge/build-passing-brightgreen)
![alt text](https://img.shields.io/badge/docs-passing-brightgreen)
![alt text](https://img.shields.io/badge/coverage-82%25-green)
![alt text](https://img.shields.io/badge/version-0.0.2-blue)
![alt text](https://img.shields.io/badge/license-MIT-blue)

**PyBandits**  is a ``Python`` library for Multi-Armed Bandit. It provides an implementation of stochastic Multi-Armed Bandit (sMAB) and contextual Multi-Armed Bandit (cMAB) based on Thompson Sampling.

<br/>

For the sMAB, we implemented a Bernoulli multi-armed bandit based on Thompson Sampling algorithm [Agrawal and Goyal, 2012](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf). If context information is available we provide a generalisation of Thompson Sampling for cMAB [Agrawal and Goyal, 2014](https://arxiv.org/pdf/1209.3352.pdf) implemented with [PyMC3](https://peerj.com/articles/cs-55/), an open source probabilistic programming framework  for automatic Bayesian inference on user-defined probabilistic models.


[comment]: <> (Paper)
[comment]: <> (-----)
[comment]: <> (Our paper is available [here: add link])

<br/>


Installation
------------

This library is distributed on [PyPI](https://pypi.org/project/pybandits/) and can be installed with ``pip``.
The latest release is version ``0.0.2``. ``pybandits`` requires a Python version ``>= 3.8``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pip install pybandits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command above will automatically install all the dependencies listed in ``requirements.txt``. Please visit the
[installation](https://playtikaresearch.github.io/pybandits/installation.html)
page for more details.

<br/>


Getting started
---------------

A short example, illustrating it use. Use the sMAB model to predict actions and update the model based on rewards from the environment.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import random
from pybandits.core.smab import Smab

# init stochastic Multi-Armed Bandit model
smab = Smab(action_ids=['Action A', 'Action B', 'Action C'])

# predict actions
pred_actions, _ = smab.predict(n_samples=100)

n_successes, n_failures = {}, {}
for a in set(pred_actions):

    # simulate rewards from environment
    n_successes[a] = random.randint(0, pred_actions.count(a))
    n_failures[a] = pred_actions.count(a) - n_successes[a]

    # update model
    smab.update(action_id=a, n_successes=n_successes[a], n_failures=n_failures[a])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<br/>

Documentation
-------------
For more information please read the full
[documentation](https://playtikaresearch.github.io/pybandits/pybandits.html)
and
[tutorials](https://playtikaresearch.github.io/pybandits/tutorials.html).

<br/>

Info for developers
-------------------

Pybandits is supported by the [AI for gaming and entertainment apps](https://www.meetup.com/ai-for-gaming-and-entertainment-apps/) community.

The source code of the project is available on [GitHub](https://github.com/playtikaresearch/pybandits).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git clone https://github.com/playtikaresearch/pybandits.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the library and the dependencies with one of the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pip install .                        # install library + dependencies
$ pip install .[develop]               # install library + dependencies + developer-dependencies
$ pip install -r requirements.txt      # install dependencies
$ pip install -r requirements-dev.txt  # install developer-dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As suggested by the authors of ``pymc3`` and ``pandoc``, we highly recommend to install these dependencies with
``conda``:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ conda install -c conda-forge pandoc
$ conda install -c conda-forge pymc3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the file ``pybandits.whl`` for the installation with ``pip`` run the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python setup.py sdist bdist_wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the HTML documentation run the following commands:

~~~~~~~~~~~
$ cd docs
$ make html
~~~~~~~~~~~

<br/>

Run tests
---------

Tests can be executed with ``pytest`` running the following commands. Make sure to have the library installed before to
run any tests.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ cd tests
$ pytest -vv                                      # run all tests
$ pytest -vv test_testmodule.py                   # run all tests within a module
$ pytest -vv test_testmodule.py -k test_testname  # run only 1 test
$ pytest -vv -k 'not time'                        # run all tests but not exec time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<br/>

License
-------

[MIT License](LICENSE)
