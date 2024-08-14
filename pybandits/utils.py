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

import json
from typing import Any, Callable, Dict, List, Optional, Union

from bokeh.io import curdoc, output_file, output_notebook, save, show
from bokeh.models import InlineStyleSheet, TabPanel, Tabs
from IPython import get_ipython

from pybandits.pydantic_version_compatibility import validate_call

JSONSerializable = Union[str, int, float, bool, None, List["JSONSerializable"], Dict[str, "JSONSerializable"]]


@validate_call
def to_serializable_dict(d: Dict[str, Any]) -> Dict[str, JSONSerializable]:
    """
    Convert a dictionary to a dictionary whose values are JSONSerializable Parameters

    ----------
    d: dictionary to convert

    Returns
    -------

    """
    return json.loads(json.dumps(d, default=dict))


@validate_call
def extract_argument_names_from_function(function_handle: Callable, is_class_method: bool = False) -> List[str]:
    """
    Extract the argument names from a function handle.

    Parameters
    ----------
    function_handle : Callable
        Handle of a function to extract the argument names from

    is_class_method : bool, defaults to False
        Whether the function is a class method

    Returns
    -------
    argument_names : List[str]
        List of argument names
    """
    start_index = int(is_class_method)
    argument_names = function_handle.__code__.co_varnames[start_index : function_handle.__code__.co_argcount]
    return argument_names


def in_jupyter_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    Reference: https://stackoverflow.com/a/39662359

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook, False otherwise.
    """

    try:
        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole

        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython

        else:
            return False  # Other type (likely shouldn't happen)

    except NameError:
        return False  # Probably standard Python interpreter


def visualize_via_bokeh(output_path: Optional[str], tabs: List[TabPanel]):
    """
    Visualize output to either a Jupyter notebook or an HTML file.

    Parameters
    ----------
    output_path : Optional[str]
        Path to the output file. Required if not running in a Jupyter notebook.
    """

    if in_jupyter_notebook():
        output_notebook()
    else:
        if output_path is None:
            raise ValueError("output_path is required when not running in a Jupyter notebook.")
        output_file(output_path)

    # Add a Div model to the Bokeh layout for flexible tabs
    css = """
                 :host(.bk-Tabs) .bk-header {
                     flex-wrap: wrap !important;
                 }
             """
    stylesheet = InlineStyleSheet(css=css)
    curdoc().title = "Visual report"
    if in_jupyter_notebook():
        show(Tabs(tabs=tabs, stylesheets=[stylesheet]))
    else:
        save(Tabs(tabs=tabs, sizing_mode="stretch_both", stylesheets=[stylesheet]))
