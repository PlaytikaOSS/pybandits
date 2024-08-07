import json
from typing import Any, Dict, List, Union

from pydantic import validate_call

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
