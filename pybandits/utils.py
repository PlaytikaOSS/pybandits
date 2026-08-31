# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import inspect
from abc import ABC
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from bokeh.io import curdoc, output_file, output_notebook, save, show
from bokeh.models import InlineStyleSheet, TabPanel, Tabs

try:
    from IPython import get_ipython

    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False
    get_ipython = None  # type: ignore


from pydantic import PositiveInt, validate_call
from scipy.optimize import NonlinearConstraint, differential_evolution


class OptimizationFailedError(Exception):
    """Exception raised when optimization fails to converge."""

    def __init__(self, message: str) -> None:
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Error message describing why optimization failed.
        """
        super().__init__(message)
        self.message = message


@validate_call
def extract_argument_names_from_function(handle: Callable, ignore_arguments: tuple = ("self", "cls")) -> list[str]:
    """
    Extract the argument names from a function handle.

    Parameters
    ----------
    handle : Callable
        Handle of a function or class to extract the argument names from
    ignore_arguments : Tuple
        Tuple of argument names to ignore

    Returns
    -------
    argument_names : list[str]
        List of argument names
    """

    argument_names = list(
        handle.model_fields.keys() if hasattr(handle, "model_fields") else inspect.signature(handle).parameters
    )
    for argument_name in ignore_arguments:
        if argument_name in argument_names:
            argument_names.remove(argument_name)
    return argument_names


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_non_abstract_classes(module: ModuleType) -> list[type]:
    non_abc_classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Ensure the class is defined in the module and not imported
        if obj.__module__ == module.__name__:
            # Check if the class is not an abstract class (i.e., doesn't inherit from abc.ABC)
            if not inspect.isabstract(obj) and ABC not in obj.__bases__:
                non_abc_classes.append(obj)
    return non_abc_classes


def in_jupyter_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    References
    ----------
    https://stackoverflow.com/a/39662359

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook, False otherwise.
    """
    if not _IPYTHON_AVAILABLE:
        return False

    try:
        ipython = get_ipython()
        if ipython is None:
            return False
        shell = ipython.__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (NameError, AttributeError):
        return False  # Probably standard Python interpreter


def visualize_via_bokeh(output_path: str | None, tabs: list[TabPanel]):
    """
    Visualize output to either a Jupyter notebook or an HTML file.

    Parameters
    ----------
    output_path : str | None
        Path to the output file. Required if not running in a Jupyter notebook.
    tabs : list[TabPanel]
        List of TabPanel objects to visualize.
    """

    if in_jupyter_notebook():
        output_notebook()
    else:
        if output_path is None:
            raise ValueError("output_path is required when not running in a Jupyter notebook.")
        output_file(output_path)

    # Add a Div model to the Bokeh layout for flexible tabs
    tabs_css = """
                 :host(.bk-Tabs) .bk-header {
                     flex-wrap: wrap !important;
                 }
             """

    tabs_stylesheet = InlineStyleSheet(css=tabs_css)
    curdoc().title = "Visual report"
    styled_tabs = Tabs(tabs=tabs, stylesheets=[tabs_stylesheet], sizing_mode="stretch_both")
    if in_jupyter_notebook():
        show(styled_tabs)
    else:
        save(styled_tabs)


class classproperty(property):
    def __get__(self, instance, owner):
        return self.fget(owner)


_DE_PARAMS = {
    "maxiter": 1000,  # Match scipy's default; nonlinear constraints require more iterations
    "popsize": 15,  # Population size multiplier (total pop = popsize * len(bounds))
    "atol": 1e-6,  # Relaxed tolerance for boundary convergence
    "tol": 1e-6,  # Relaxed tolerance for boundary convergence
    "strategy": "best1bin",  # Good balance of exploration/exploitation
    "mutation": (0.5, 1),  # Mutation factor range
    "recombination": 0.7,  # Crossover probability
    "polish": True,  # Local polish with L-BFGS-B
    "disp": False,
}


def maximize_by_quantity(
    quantity_score_func: Callable[[np.ndarray], float],
    dimension: PositiveInt,
    constraint: list[Callable[[np.ndarray], float]] | None = None,
    n_trials: PositiveInt = 10000,
    maxiter_factor: PositiveInt = 10,
    seed: Any | None = None,
) -> np.ndarray:
    """
    Maximize the quantity score for the given function.

    Parameters
    ----------
    quantity_score_func : Callable[[np.ndarray], float]
        The quantity score function.
    dimension : PositiveInt
        The quantity vector dimension.
    constraint : list[Callable[[np.ndarray], float]] | None
        The constraint functions. final constraint is inf >= constraint(x) >= 0
    n_trials : PositiveInt, defaults to 10000
        The number of optimization trials.
    maxiter_factor : PositiveInt, defaults to 10
        The factor to define maximum number of iterations for differential evolution.
        Relative to n_trials.
    seed : Any | None, default=None
        Seed for the differential evolution optimizer (passed directly to scipy). Accepts an int,
        a ``numpy.random.Generator``, or ``None`` (uses global numpy random state).

    Returns
    -------
    np.ndarray
        The global maxima coordinates of quantity_score_func.

    Raises
    ------
    OptimizationFailedError
        If the optimization fails to converge.
    """
    bounds = [(0, 1) for _ in range(dimension)]

    # Convert constraint to scipy format if provided
    if constraint is not None:
        constraints = []
        for constraint_func in constraint:
            constraints.append(NonlinearConstraint(constraint_func, 0, np.inf))
    else:
        constraints = None

    # Differential Evolution parameters
    de_params = {
        "func": lambda x: -quantity_score_func(x),  # Minimize negative = maximize
        "bounds": bounds,
        **_DE_PARAMS,
    }
    de_params["maxiter"] = max(de_params["maxiter"], n_trials // maxiter_factor)

    # Only add constraints if they exist
    if constraints is not None:
        de_params["constraints"] = constraints
    if seed is not None:
        de_params["seed"] = seed
    result = differential_evolution(**de_params)

    # differential_evolution flags success=False for any constraint violation > 0, so a solution sitting on the
    # feasibility boundary is rejected over floating-point noise (e.g. MAXCV ~1e-9). When constraints are present,
    # accept it if the violation is within the optimizer's own convergence tolerance (a genuine unconstrained
    # convergence failure has no violation to inspect and still raises).
    if result.success or (constraints is not None and result.constr_violation <= _DE_PARAMS["atol"]):
        return result.x
    else:
        raise OptimizationFailedError(result.message)
