"""Module 3 — Budget Optimization Engine (deterministic, one-shot).

Reads the per-channel forecast summary from Module 2 and produces a single,
constraint-bounded budget reallocation: shift money from the lowest-marginal-ROAS
channel (below the floor) to the highest-marginal-ROAS channel (with headroom).

Design reference: docs/04_optimization_design.md (the simplified one-shot
optimizer; NOT the iterative water-filling allocator).

This module contains ALL business logic and has NO Streamlit / UI dependency.
It is pure and deterministic: same inputs always produce the same recommendation.
The returned dict is the contract the Approval module (Module 4) will consume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

EPS = 1e-9


@dataclass
class Constraints:
    """Business guardrails. Every field maps to a real business risk."""
    roas_floor: float = 4.0        # marginal ROAS must stay >= this to keep funding
    max_change_pct: float = 0.20   # a channel may move at most 20% of its spend/day
    confidence_threshold: float = 0.60  # only fund a recipient we trust this much
    min_spend_pct: float = 0.10    # keep >= 10% of current spend (presence/learning)
    max_spend_mult: float = 3.0    # never exceed 3x current spend (concentration risk)


def _build_ranking(df: pd.DataFrame, c: "Constraints") -> list[dict]:
    """Per-channel eligibility table, ranked by marginal ROAS (desc).

    Makes the optimizer's decision transparent: every channel's marginal ROAS,
    confidence, and whether it qualifies as a donor or recipient under the
    current constraints.
    """
    ranked = df.sort_values("marginal_roas", ascending=False)
    out = []
    for _, r in ranked.iterrows():
        conf = float(r["confidence"]) if pd.notna(r["confidence"]) else float("nan")
        marginal = float(r["marginal_roas"])
        avg = float(r["forecast_roas"]) if pd.notna(r.get("forecast_roas")) else None
        out.append({
            "channel": r["channel"],
            "current_spend": round(float(r["current_spend"]), 2),
            "marginal_roas": round(marginal, 2),
            "average_roas": round(avg, 2) if avg is not None else None,
            "confidence": round(conf, 2) if conf == conf else None,
            "confidence_label": r.get("confidence_label"),
            "eligible_donor": bool(marginal < c.roas_floor),
            "eligible_recipient": bool(marginal > c.roas_floor and conf >= c.confidence_threshold),
            "selected": None,
        })
    return out


def _no_change(allocation_rows, blended_before, total_spend, constraints, reason, ranking):
    return {
        "status": "no_change",
        "reason": reason,
        "donor": None,
        "recipient": None,
        "donor_marginal_roas": None,
        "recipient_marginal_roas": None,
        "budget_shift": 0.0,
        "expected_revenue_gain": 0.0,
        "blended_roas_before": round(blended_before, 2),
        "blended_roas_after": round(blended_before, 2),
        "expected_blended_roas": round(blended_before, 2),
        "confidence": None,
        "confidence_score": None,
        "binding_constraint": None,
        "triggered_constraints": [
            {"name": "Budget Neutral", "status": "satisfied",
             "detail": f"Total spend unchanged at ${total_spend:,.0f}/day"},
        ],
        "allocation": allocation_rows,
        "ranking": ranking,
        "constraints_used": asdict(constraints),
        "total_spend": round(total_spend, 2),
    }


def optimize(channels: pd.DataFrame, constraints: Constraints | None = None) -> dict:
    """Run the one-shot constrained reallocation. Returns a recommendation dict."""
    c = constraints or Constraints()
    df = channels.copy()
    df = df[df["marginal_roas"].notna() & df["current_spend"].notna()].reset_index(drop=True)

    total_spend = float(df["current_spend"].sum())
    total_rev = float(df["forecast_revenue"].sum())
    blended_before = total_rev / total_spend if total_spend > 0 else float("nan")

    # Baseline allocation (everyone unchanged) — used as the no-change fallback too.
    base_alloc = [
        {
            "channel": r["channel"],
            "platform": r["platform"],
            "funnel_stage": r["funnel_stage"],
            "current_spend": round(float(r["current_spend"]), 2),
            "recommended_spend": round(float(r["current_spend"]), 2),
            "delta": 0.0,
            "marginal_roas": float(r["marginal_roas"]),
            "confidence": float(r["confidence"]) if pd.notna(r["confidence"]) else None,
        }
        for _, r in df.iterrows()
    ]

    ranking = _build_ranking(df, c)
    ranked = df.sort_values("marginal_roas").reset_index(drop=True)

    # Donor: lowest marginal ROAS that is BELOW the floor (diminishing returns).
    donor_candidates = ranked[ranked["marginal_roas"] < c.roas_floor]
    # Recipient: highest marginal ROAS ABOVE the floor AND trusted enough.
    recip_candidates = ranked[
        (ranked["marginal_roas"] > c.roas_floor) & (ranked["confidence"] >= c.confidence_threshold)
    ]

    if donor_candidates.empty:
        return _no_change(
            base_alloc, blended_before, total_spend, c,
            f"No channel is below the {c.roas_floor:.1f}x marginal-ROAS floor — "
            "no inefficient spend to reallocate. System would consider scaling total budget.",
            ranking,
        )
    if recip_candidates.empty:
        return _no_change(
            base_alloc, blended_before, total_spend, c,
            f"No channel clears the {c.roas_floor:.1f}x floor with confidence "
            f">= {c.confidence_threshold:.2f} — no trusted home for reallocated budget.",
            ranking,
        )

    donor = donor_candidates.iloc[0]
    recipient = recip_candidates.iloc[-1]

    if donor["channel"] == recipient["channel"]:
        return _no_change(
            base_alloc, blended_before, total_spend, c,
            "Donor and recipient resolved to the same channel — no move.",
            ranking,
        )

    # Mark the selected donor / recipient in the ranking table.
    for entry in ranking:
        if entry["channel"] == donor["channel"]:
            entry["selected"] = "donor"
        elif entry["channel"] == recipient["channel"]:
            entry["selected"] = "recipient"

    donor_spend = float(donor["current_spend"])
    recip_spend = float(recipient["current_spend"])

    # Candidate caps on the transfer; the smallest one binds.
    caps = {
        "20% Daily Change Limit": min(
            c.max_change_pct * donor_spend, c.max_change_pct * recip_spend
        ),
        "Minimum Spend Floor (donor)": donor_spend * (1 - c.min_spend_pct),
        "Maximum Spend Cap (recipient)": c.max_spend_mult * recip_spend - recip_spend,
    }
    binding_constraint = min(caps, key=caps.get)
    shift = max(0.0, float(caps[binding_constraint]))

    donor_marginal = float(donor["marginal_roas"])
    recip_marginal = float(recipient["marginal_roas"])
    revenue_gain = (recip_marginal - donor_marginal) * shift
    blended_after = (total_rev + revenue_gain) / total_spend if total_spend > 0 else float("nan")

    # Build the recommended allocation.
    allocation = []
    for row in base_alloc:
        rec = dict(row)
        if rec["channel"] == donor["channel"]:
            rec["recommended_spend"] = round(donor_spend - shift, 2)
            rec["delta"] = round(-shift, 2)
        elif rec["channel"] == recipient["channel"]:
            rec["recommended_spend"] = round(recip_spend + shift, 2)
            rec["delta"] = round(shift, 2)
        allocation.append(rec)

    change_binding = binding_constraint == "20% Daily Change Limit"
    triggered = [
        {"name": "Budget Neutral", "status": "satisfied",
         "detail": f"Total spend unchanged at ${total_spend:,.0f}/day"},
        {"name": "20% Daily Change Limit",
         "status": "binding" if change_binding else "satisfied",
         "detail": (f"Capped move at 20% (${shift:,.0f}/day)" if change_binding
                    else f"Move (${shift:,.0f}/day) within 20% limits")},
        {"name": "ROAS Floor", "status": "triggered",
         "detail": f"{donor['channel']} marginal {donor_marginal:.2f}x < {c.roas_floor:.1f}x floor"},
        {"name": "Confidence Threshold", "status": "satisfied",
         "detail": f"{recipient['channel']} confidence {float(recipient['confidence']):.2f} "
                   f">= {c.confidence_threshold:.2f}"},
        {"name": "Min/Max Spend per Channel",
         "status": "binding" if not change_binding else "satisfied",
         "detail": (f"Bound by {binding_constraint}" if not change_binding
                    else "Within per-channel spend bounds")},
    ]

    reason = (
        f"{donor['channel']} has the lowest marginal ROAS at {donor_marginal:.2f}x — "
        f"below the {c.roas_floor:.1f}x floor (diminishing returns). "
        f"{recipient['channel']} has the highest marginal ROAS at {recip_marginal:.2f}x "
        f"(headroom). Shifting ${shift:,.0f}/day lifts blended ROAS "
        f"{blended_before:.2f}x -> {blended_after:.2f}x (+${revenue_gain:,.0f}/day), "
        f"bounded by the {binding_constraint}."
    )

    return {
        "status": "reallocate",
        "reason": reason,
        "donor": donor["channel"],
        "recipient": recipient["channel"],
        "donor_marginal_roas": round(donor_marginal, 2),
        "recipient_marginal_roas": round(recip_marginal, 2),
        "budget_shift": round(shift, 2),
        "expected_revenue_gain": round(revenue_gain, 2),
        "blended_roas_before": round(blended_before, 2),
        "blended_roas_after": round(blended_after, 2),
        "expected_blended_roas": round(blended_after, 2),
        "confidence": recipient["confidence_label"],
        "confidence_score": round(float(recipient["confidence"]), 2),
        "binding_constraint": binding_constraint,
        "triggered_constraints": triggered,
        "allocation": allocation,
        "ranking": ranking,
        "constraints_used": asdict(c),
        "total_spend": round(total_spend, 2),
    }
