import json
from typing import Any, Callable, Dict, List, Union

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
