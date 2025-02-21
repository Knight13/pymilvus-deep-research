import re
import json
from json_repair import repair_json

json_re = re.compile(r"```json\n(?s:.)*\n```")

def extract_json(response: str) -> dict:
    try:
        match = json_re.search(response)
        json_results = "\n".join(match.group().splitlines()[1:-1])
    except Exception:
        return {}
    return json.loads(repair_json(json_results))

def leaves(struct) -> set:
    """Return a set of leaf values found in nested dicts and lists excluding None values."""
    values = set()

    def add_leaves(s):
        if isinstance(s, dict):
            for sub in s.values():
                add_leaves(sub)
        elif isinstance(s, list):
            for sub in s:
                add_leaves(sub)
        elif s is not None:
            values.add(s)

    add_leaves(struct)
    return values
