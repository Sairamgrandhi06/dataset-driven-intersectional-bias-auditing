"""
Failure Registry Module.

Logs failure events to 'evidence/reliability/failure_events.jsonl'.
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.reliability.failure_contract import create_failure_event


def get_failure_events_file() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "evidence", "reliability", "failure_events.jsonl")


def register_failure_event(event_dict: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Append a failure event record to JSONL evidence file.
    """
    target_path = output_path or get_failure_events_file()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict) + "\n")
    except Exception as e:
        event_dict["registry_error"] = str(e)

    return event_dict
