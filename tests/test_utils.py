import json


def is_serializable(something) -> bool:
    try:
        json.dumps(something, default=dict)
        return True
    except Exception:
        return False
