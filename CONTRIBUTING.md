Contributing
============

In this page we want to explain how to contribute to this library following our standards.

This library aims to be an open and collaborative project released under the MIT license. It is hosted on
[GitHub](https://github.com/PlaytikaOSS/pybandits) as  it intends to be supported by an open community
of contributors.

We want to ensure both code clarity and quality. *pybandits* follows *Ruff* standards. Every component, class and
function are tested with *pytest* reaching a line coverage of 90%. All contributors must follow the contributors
guidelines before to open any pull request for code merge (pre-commit hook, commits squash, *Ruff* compliance, test,
coverage, update documentation). *pybandit* also provides a detailed documentation implemented with *Sphinx* where each
class and function is described. We want to enforce a community-based collaboration with external contributors
that embrace our open source philosophy.

Guidelines
----------
We expect developers to work with the following steps before opening a pull request.

* Create a new feature branch from `develop`, which is the default branch and the base of every pull request.
<br/> The new branch must follow the name convention `feature/<short_description>` where `<short_description>`=
short description of the new feature (e.g. `feature/add_predict_function_to_smab`)

* Checkout on the new branch and write the code of the new feature.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git checkout develop
$ git pull
$ git checkout -b feature/<short_description>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Run pre-commit hooks before any commit. This will check code style, max-line-length, etc. Code style must respect
*Ruff* standards (`ruff-format` and `ruff`, pinned in `.pre-commit-config.yaml`). This is the same command the CI
style check runs, so a clean run here means a green check there.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ poetry run pre-commit run --all-files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Commit and push your code on the feature branch (as many times as you need). The commit subject is an imperative
sentence describing the change (e.g. `Add per-update decay_factor`); it carries no ticket prefix.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git add filename
$ git commit -m 'Add short_description'
$ git push origin HEAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Make sure that all tests must pass successfully. <br/> Tests are run from the repository root with *pytest*, as the
CI does:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ poetry run pytest -n auto -vv                                   # run all tests
$ poetry run pytest -vv tests/test_testmodule.py                  # run all tests within a module
$ poetry run pytest -vv tests/test_testmodule.py -k test_testname # run only 1 test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this stage you should be at the point where you have completed all your tasks and you are ready for other people to
review your code.

* Rebase on `develop` with commits squash. <br/> Be careful because `rebase` is an unsafe operation, if you want to
know more please check [here](https://docs.github.com/en/get-started/using-git/about-git-rebase).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git checkout develop
$ git pull
$ git checkout feature/<short_description>
$ git rebase --interactive develop
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
Add short_description

 ### Changes

* Add <filename1> description
   - sub description
   - sub description
* Add <filename1> description

<default message>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If all steps above were successfully completed, you can open a pull request against `develop`. Two conventions are
enforced on the pull request itself:

  * The title follows the same imperative form as the commit subject, and the description follows
  [the pull request template](.github/pull_request_template.md).

  * The pull request must carry at least one of the labels `bug`, `documentation`, `enhancement` or `skip-changelog`.
  The `check conventional commits labels` job fails until one is present, and re-runs whenever a label is added or
  removed. Contributors without write access cannot apply labels themselves, so ask a maintainer in the pull
  request.
