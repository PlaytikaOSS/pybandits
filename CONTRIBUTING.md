Contributing
============

In this page we want to explain how to contribute to this library following our standards.

This library aims to be an open and collaborative project released under the MIT license. It is hosted on
[Github](https://github.com/PlaytikaResearch/pybandits) as  it intends to be supported by an open community
of contributors.

We want to ensure both code clarity and quality. *pybandits* follows *Flake8* standards. Every component, class and
function are tested with *pytest* reaching a line coverage of 90%. All contributors must follow the contributors
guidelines before to open any pull request for code merge (pre-commit hook, commits squash, *Flake8* compliance, test,
coverage, update documentation). *pybandit* also provides a detailed documentation implemented with *Sphinx* where each
class and function is described. We want to enforce a community-based collaboration with external contributors
that embrace our open source philosophy.

Guidelines
----------
We expect developers to work with the following steps before opening a pull request.

* Create a new feature branch from master. <br/> The new branch must follow the name convention
`pybandits-XXXX feature_name` where `XXXX`= number of the issue linked to this branch and `feature_name`= short
description of the new feature (e.g. add predict function to sMAB)

* Checkout on the new branch and write the code of the new feature.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git checkout feature/pybandits-XXXX feature_name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Run pre-commit hooks before any commit. This will check code style, max-line-length, etc. Code style must respect
*Flake8* standards.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pre-commit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Commit and push your code on the feature branch (as many times as you need)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git add filename
$ git commit -m 'pybandits-XXXX short_description'
$ git push origin HEAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Make sure that all tests must pass successfully. <br/> Tests can be executed with *pytest* running the following
commands:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ cd tests
$ pytest -vv                                      # run all tests
$ pytest -vv test_testmodule.py                   # run all tests within a module
$ pytest -vv test_testmodule.py -k test_testname  # run only 1 test
$ pytest -vv -k 'not time'                        # run all tests but not exec time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this stage you should be at the point where you have completed all your tasks and you are ready for other people to
review your code.

* Rebase on master with commits squash. <br/> Be careful because `rebase` is an unsafe operation, if you want to know
more please check [here](https://docs.github.com/en/get-started/using-git/about-git-rebase).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git checkout master
$ git pull
$ git checkout feature/pybandits-<XXXX><feature_name>
$ git rebase --interactive master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In `rebase` the first line must contain `pick`. In all the other lines delete `pick` and write `fixup`.
Then save the file.

* Rewrite the initial commit message with a comprehensive description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git commit --amend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Change message with the following template (Make sure to have: a white space before `### Changes`; to have an
empty line between the title and `### Changes`; to have an empty line before the bullet list. This will allow
to have the markdown correctly rendered).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pybandits-XXXX short_description

 ### Changes

* Add <filename1> description
   - sub description
   - sub description
* Add <filename1> description

<default message>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If all steps above were successfully completed, you can open a pull request.
