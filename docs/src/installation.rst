.. Modify also the the README.md if you change docs/installation.rst

============
Installation
============

Latest release
--------------

This library is distributed on PyPI_ and can be installed with ``pip``. The latest release is version ``0.0.2``. ``pybandits`` requires a Python version ``>= 8``.

.. code:: console

   $ pip install pybandits

The command above will automatically install all the dependencies listed in ``requirements.txt``.

.. _PyPI:  https://pypi.org/project/pybandits/

Info for developers
-------------------

The source code of the project is available on GitHub_.

.. code:: console

   $ git clone https://github.com/PlaytikaResearch/pybandits.git

You can install the library and the dependencies from the source code with one of the following commands:

.. code:: console

   $ pip install .                        # install library + dependencies
   $ pip install .[develop]               # install library + dependencies + developer-dependencies
   $ pip install -r requirements.txt      # install dependencies
   $ pip install -r requirements-dev.txt  # install dependencies + developer-dependencies

.. _GitHub: https://github.com/PlaytikaResearch/pybandits

As suggested by the authors of ``pymc3`` and ``pandoc``, we highly recommend to install these dependencies with
``conda``:

.. code:: console

   $ conda install -c conda-forge pandoc
   $ conda install -c conda-forge pymc3

To create the file ``pybandits.whl`` for the installation with ``pip`` run the following command:

.. code:: console

   $ python setup.py sdist bdist_wheel

To create the HTML documentation run the following commands:

.. code:: console

   $ cd docs
   $ make html

Run tests
---------

Tests can be executed with ``pytest`` running the following commands:

.. code:: console

   $ cd tests
   $ pytest                                      # run all tests
   $ pytest test_testmodule.py                   # run all tests within a module
   $ pytest test_testmodule.py -k test_testname  # run only 1 test
   $ pytest -vv -k 'not time'                    # run all tests but not exec time
