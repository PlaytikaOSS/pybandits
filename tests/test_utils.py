import json


def is_serializable(something) -> bool:
    try:
        json.dumps(something)
        return True
    except Exception:
        return False
