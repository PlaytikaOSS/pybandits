import json
from typing import Any, Callable, Dict, List, Union

from pydantic import validate_call

Simple = Union[str, int, float, bool, None]

JSONSerializable = Union[Simple, List["JSONSerializable"], Dict[str, "JSONSerializable"]]


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
def update_nested_struct(
    d: Union[Dict[str, Any], List, Simple], other: Union[Dict[str, Any], List, Simple]
) -> Union[Dict[str, Any], List, Simple]:
    """
    Update a nested combination of dictionaries and lists with another dictionary, recursively.

    Parameters
    ----------
    d : Union[Dict[str, Any], List, Simple]
        Nested combination of dictionaries and lists to update.
    other : Union[Dict[str, Any], List, Simple]
        Nested combination of dictionaries and lists to update with.

    Returns
    -------
    d : Union[Dict[str, Any], List, Simple]
        Updated nested combination of dictionaries and lists.

    """
    if isinstance(d, dict) and isinstance(other, dict):
        for key, value in other.items():
            if key in d:
                d[key] = update_nested_struct(d[key], value)
            else:
                d[key] = value
    elif isinstance(d, list) and isinstance(other, list):
        assert len(d) == len(other)
        for i, (d_value, other_value) in enumerate(zip(d, other)):
            d[i] = update_nested_struct(d_value, other_value)

    return d


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
