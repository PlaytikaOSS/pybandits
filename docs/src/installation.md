% Modify also the README.md if you change this file.

# Installation

```console
$ pip install pybandits
```

The command above will automatically install all the dependencies listed in `pyproject.toml`.

## Info for developers

The source code of the project is available on [GitHub].

```console
$ git clone https://github.com/PlaytikaResearch/pybandits.git
```

You can install the library and the dependencies from the source code with one of the following commands:

```console
poetry install                # install library + dependencies
poetry install --without dev     # install library + dependencies, excluding developer-dependencies
```

To create the file `pybandits.whl` for the installation with `pip` run the following command:

```console
$ poetry build
```

To create the HTML documentation run the following commands:

```console
$ cd docs/src
$ make html
```

## Run tests

Tests can be executed with `pytest` running the following commands:

```console
$ cd tests
$ pytest                                      # run all tests
$ pytest test_testmodule.py                   # run all tests within a module
$ pytest test_testmodule.py -k test_testname  # run only 1 test
$ pytest -vv -k 'not time'                    # run all tests but not exec time
```

[github]: https://github.com/PlaytikaResearch/pybandits
[pypi]: https://pypi.org/project/pybandits/
