"""
Reliability Evidence Logger Module.

Serializes recovery and reconciliation events into JSONL evidence logs under 'evidence/reliability/'.
"""

import json
import os
from typing import Any, Dict, Optional


def get_evidence_path(filename: str) -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "evidence", "reliability", filename)


def log_recovery_event(record: Dict[str, Any], filename: str = "recovery_events.jsonl") -> Dict[str, Any]:
    """Append a recovery event record to JSONL."""
    path = get_evidence_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
    return record


def log_reconciliation_event(record: Dict[str, Any], filename: str = "reconciliation_events.jsonl") -> Dict[str, Any]:
    """Append a reconciliation event record to JSONL."""
    path = get_evidence_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
    return record
