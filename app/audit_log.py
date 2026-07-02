"""Module 4 — Stubbed execution + audit log.

"Approve & Execute" does NOT call any ad-platform API. It appends the decision
(timestamp, segments, amounts, who approved, resulting KPIs) to
`data/audit_log.csv` and the UI shows a confirmation. This is the accountability
trail that real platform execution would be built on top of, behind the exact
same approval gate.

Pure I/O — no Streamlit dependency, so it stays independently testable.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "audit_log.csv"

# Stable, ordered schema so the CSV is append-friendly and diff-able.
COLUMNS = [
    "timestamp",
    "action",            # approved | rejected | modified_approved
    "approver",
    "donor",
    "recipient",
    "optimizer_shift",   # what the optimizer originally recommended ($/day)
    "budget_shift",      # final amount the manager approved ($/day)
    "modified",          # True if the manager changed the optimizer's amount
    "expected_revenue_gain",
    "blended_roas_before",
    "blended_roas_after",
    "confidence",
    "binding_constraint",
    "note",              # approval comment (required by the UI)
    "constraints",       # JSON blob of the constraints used
    "explanation_source",  # llm | rule-based
]


def append_decision(decision: dict) -> dict:
    """Append a single approve/reject decision to the audit log.

    Accepts a loosely-typed dict; fills in a timestamp and serializes the
    constraints blob. Returns the fully-formed row that was written.
    """
    optimizer_shift = float(decision.get("optimizer_shift", decision.get("budget_shift", 0.0)))
    budget_shift = float(decision.get("budget_shift", 0.0))
    row = {
        "timestamp": decision.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
        "action": decision.get("action", "approved"),
        "approver": decision.get("approver", "demo_user"),
        "donor": decision.get("donor"),
        "recipient": decision.get("recipient"),
        "optimizer_shift": round(optimizer_shift, 2),
        "budget_shift": round(budget_shift, 2),
        "modified": bool(decision.get("modified", abs(optimizer_shift - budget_shift) > 1e-6)),
        "expected_revenue_gain": round(float(decision.get("expected_revenue_gain", 0.0)), 2),
        "blended_roas_before": decision.get("blended_roas_before"),
        "blended_roas_after": decision.get("blended_roas_after"),
        "confidence": decision.get("confidence"),
        "binding_constraint": decision.get("binding_constraint"),
        "note": decision.get("note", ""),
        "constraints": json.dumps(decision.get("constraints", {})),
        "explanation_source": decision.get("explanation_source", ""),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        LOG_PATH, mode="a", header=write_header, index=False
    )
    return row


def load_log() -> pd.DataFrame:
    """Return the full audit log (most-recent first), or an empty frame."""
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(LOG_PATH)
    # Tolerate older logs written before the schema was extended.
    df = df.reindex(columns=COLUMNS)
    return df.iloc[::-1].reset_index(drop=True)


def clear_log() -> None:
    """Delete the audit log file (used by the demo 'reset' control)."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
