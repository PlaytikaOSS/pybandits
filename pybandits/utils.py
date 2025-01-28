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
from types import ModuleType
from typing import Callable, List, Optional, Tuple

from bokeh.io import curdoc, output_file, output_notebook, save, show
from bokeh.models import InlineStyleSheet, TabPanel, Tabs
from IPython import get_ipython

from pybandits.pydantic_version_compatibility import validate_call


@validate_call
def extract_argument_names_from_function(handle: Callable, ignore_arguments: Tuple = ("self", "cls")) -> List[str]:
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
    argument_names : List[str]
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
def get_non_abstract_classes(module: ModuleType) -> List[type]:
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

    Raises
    ------
    NotImplementedError
        If the shell type is neither Jupyter notebook nor terminal.
    """

    try:
        shell = get_ipython().__class__.__name__

        return shell == "ZMQInteractiveShell"

    except NameError:
        return False  # Probably standard Python interpreter


def visualize_via_bokeh(output_path: Optional[str], tabs: List[TabPanel]):
    """
    Visualize output to either a Jupyter notebook or an HTML file.

    Parameters
    ----------
    output_path : Optional[str]
        Path to the output file. Required if not running in a Jupyter notebook.
    tabs : List[TabPanel]
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
