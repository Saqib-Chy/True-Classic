"""Module 4 — LLM explanation of a budget recommendation.

The LLM **never decides allocations and never invents numbers**. It receives the
optimizer's output — a *closed* set of figures (segments, spend deltas, marginal
ROAS, the binding constraint, confidence) — and narrates it in plain English for
a marketer. Hallucination control = closed inputs + visible numbers + fallback.

A deterministic, rule-based template is the fallback whenever no API key /
package is present or the call fails, so the demo never depends on the model
being up. The fallback is also what the unit story leans on: the explanation is
*grounded* either way.

All LLM access is lazy-imported so the app runs with zero extra dependencies.
Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) to enable the live call.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Load OPENAI_API_KEY (and friends) from a local .env at the project root, if
# present. Never hardcodes a key; the file is git-ignored. No-op if python-dotenv
# isn't installed or the file is absent — the deterministic fallback still works.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

# The model we call for narration + grounded Q&A (overridable via env).
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Prompt construction — the LLM only ever sees these numbers.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a paid-media analyst explaining a budget reallocation to a marketing "
    "executive at an apparel brand. You will be given a JSON object of numbers that a "
    "deterministic optimizer already computed.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY the numbers in the JSON context. Never invent, estimate, or introduce "
    "any figure that is not present in the context.\n"
    "2. You are NOT deciding the budget — the optimizer already did. You only explain its "
    "existing recommendation. Never propose a different or new allocation.\n"
    "3. The `ranking` array lists every channel with its marginal_roas, average_roas "
    "(forecast ROAS), confidence, and whether it was eligible as donor/recipient. Use it "
    "to explain why a specific channel was or wasn't chosen (e.g. average vs marginal ROAS, "
    "the ROAS floor, the confidence threshold).\n"
    "4. Lead with the decision, then WHY (marginal vs average ROAS, the ROAS floor), then "
    "the expected impact and which constraint bounded the move.\n"
    "5. Be concise: 2-4 sentences, plain English, confident but not hyped. No markdown headers.\n"
    "6. If a 'question' field is present, answer THAT question directly using only the "
    "context. If the context does not contain enough information to answer, say so plainly "
    "(e.g. \"That isn't captured in the optimizer's output\") — do not guess."
)

# A stricter prompt for the Q&A path: answer the specific question tersely, and
# refuse anything the optimizer context can't support.
QA_SYSTEM_PROMPT = (
    "You are a paid-media analyst answering a marketer's question about a budget "
    "recommendation that a deterministic optimizer already produced. You are given a JSON "
    "context of the optimizer's numbers (donor, recipient, move amount, per-channel "
    "marginal_roas and average_roas in `ranking`, blended ROAS before/after, expected "
    "revenue gain, confidence, and constraints) plus a `question`.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY the question asked, directly and concisely (1-3 sentences). Do NOT "
    "restate the whole recommendation unless the question asks for it.\n"
    "2. Use ONLY numbers present in the context. Never invent or estimate a figure.\n"
    "3. Never propose a new or different budget allocation — the optimizer decides, you "
    "only explain.\n"
    "4. Exactly ONE donor and ONE recipient are chosen. The donor is the SINGLE channel "
    "with the lowest marginal_roas among those below the ROAS floor; the recipient is the "
    "SINGLE channel with the highest marginal_roas above the floor that also clears the "
    "confidence threshold. Other channels may be eligible (see eligible_donor / "
    "eligible_recipient) yet not chosen simply because another channel was lower (for a "
    "donor) or higher (for a recipient). Average ROAS is NOT the selection criterion — "
    "marginal ROAS is.\n"
    "5. If the question is unrelated to this recommendation, or the context does not "
    "contain the information needed, reply plainly that it isn't something the optimizer's "
    "output can answer. Do not guess or use outside knowledge."
)


def _facts(rec: dict) -> dict[str, Any]:
    """Extract the closed set of figures the LLM is allowed to talk about."""
    alloc = rec.get("allocation", [])
    return {
        "decision": (
            f"Move ${rec.get('budget_shift', 0):,.0f}/day from {rec.get('donor')} "
            f"to {rec.get('recipient')}"
            if rec.get("status") == "reallocate"
            else "No reallocation recommended"
        ),
        "status": rec.get("status"),
        "donor": rec.get("donor"),
        "recipient": rec.get("recipient"),
        "donor_marginal_roas": rec.get("donor_marginal_roas"),
        "recipient_marginal_roas": rec.get("recipient_marginal_roas"),
        "budget_shift_per_day": rec.get("budget_shift"),
        "expected_revenue_gain_per_day": rec.get("expected_revenue_gain"),
        "blended_roas_before": rec.get("blended_roas_before"),
        "blended_roas_after": rec.get("blended_roas_after"),
        "confidence": rec.get("confidence"),
        "confidence_score": rec.get("confidence_score"),
        "binding_constraint": rec.get("binding_constraint"),
        "total_spend_per_day": rec.get("total_spend"),
        "constraints": rec.get("constraints_used", {}),
        "triggered_constraints": rec.get("triggered_constraints", []),
        # Full ranking so the model can explain why any channel was / wasn't chosen.
        "ranking": [
            {
                "channel": r.get("channel"),
                "current_spend": r.get("current_spend"),
                "marginal_roas": r.get("marginal_roas"),
                "average_roas": r.get("average_roas"),
                "confidence": r.get("confidence"),
                "eligible_donor": r.get("eligible_donor"),
                "eligible_recipient": r.get("eligible_recipient"),
                "selected": r.get("selected"),
            }
            for r in rec.get("ranking", [])
        ],
        "recommended_allocation": [
            {
                "channel": a["channel"],
                "current_spend": a["current_spend"],
                "recommended_spend": a["recommended_spend"],
                "marginal_roas": a["marginal_roas"],
            }
            for a in alloc
        ],
        "machine_reason": rec.get("reason"),
    }


# ---------------------------------------------------------------------------
# Deterministic fallback — always available, no dependencies.
# ---------------------------------------------------------------------------
def rule_based(rec: dict) -> str:
    """Template explanation built purely from the optimizer's numbers."""
    if rec.get("status") != "reallocate":
        floor = rec.get("constraints_used", {}).get("roas_floor", 4.0)
        return (
            f"No budget move is recommended right now. {rec.get('reason', '')} "
            f"Blended ROAS stays at {rec.get('blended_roas_before')}x and total spend is "
            f"unchanged at ${rec.get('total_spend', 0):,.0f}/day. The optimizer only acts "
            f"when a channel's marginal ROAS falls below the {floor:.1f}x floor and a "
            f"trusted, higher-return channel has headroom to absorb the dollars."
        )

    floor = rec.get("constraints_used", {}).get("roas_floor", 4.0)
    return (
        f"Recommendation: move ${rec['budget_shift']:,.0f}/day from {rec['donor']} to "
        f"{rec['recipient']}. {rec['donor']}'s next dollar is only returning "
        f"{rec['donor_marginal_roas']}x — below your {floor:.1f}x floor — so it is over-funded "
        f"despite looking healthy on average. {rec['recipient']} still earns "
        f"{rec['recipient_marginal_roas']}x on the next dollar, so the same budget works harder "
        f"there. Expected impact: +${rec['expected_revenue_gain']:,.0f}/day in revenue, lifting "
        f"blended ROAS from {rec['blended_roas_before']}x to {rec['blended_roas_after']}x at "
        f"unchanged total spend. The move was capped by the {rec['binding_constraint']}, and "
        f"confidence is {str(rec['confidence']).lower()} ({rec['confidence_score']})."
    )


# ---------------------------------------------------------------------------
# Structured executive summary — deterministic, grounded, clean formatting.
# Every value comes straight from the optimizer output; nothing is hardcoded.
# ---------------------------------------------------------------------------
_CONSTRAINT_ICON = {
    "satisfied": "✓",
    "binding": "⛔ binding",
    "triggered": "● triggered",
}


def _constraint_lines(rec: dict) -> str:
    """Bullet list of the constraints the optimizer reported for this run."""
    tcs = rec.get("triggered_constraints") or []
    lines = []
    for c in tcs:
        mark = _CONSTRAINT_ICON.get(c.get("status"), "•")
        lines.append(f"- {mark} **{c.get('name')}** — {c.get('detail')}")
    return "\n".join(lines) if lines else "- No constraints were triggered."


def structured(rec: dict) -> list[dict[str, str]]:
    """Return the explanation as ordered sections for a clean exec summary.

    Sections: Recommendation · Why this recommendation · Expected business
    impact · Constraints applied. Each body is markdown; all figures are pulled
    live from `rec` (the current optimizer output).
    """
    floor = rec.get("constraints_used", {}).get("roas_floor", 4.0)

    if rec.get("status") != "reallocate":
        return [
            {"title": "Recommendation",
             "body": "**No budget reallocation recommended.** Current allocation is left unchanged."},
            {"title": "Why this recommendation",
             "body": rec.get("reason", "No inefficient spend to move under the current constraints.")},
            {"title": "Expected business impact",
             "body": (
                 f"- Blended ROAS holds at **{rec.get('blended_roas_before', 0):.2f}x**\n"
                 f"- Total spend unchanged at **${rec.get('total_spend', 0):,.0f}/day** (budget-neutral)"
             )},
            {"title": "Constraints applied", "body": _constraint_lines(rec)},
        ]

    recommendation = (
        f"Move **${rec['budget_shift']:,.0f}/day** from **{rec['donor']}** "
        f"to **{rec['recipient']}**, holding total spend constant."
    )
    why = (
        f"- **{rec['donor']}**'s next dollar returns only **{rec['donor_marginal_roas']:.2f}x** — "
        f"below the **{floor:.1f}x** marginal-ROAS floor — so it is over-funded even if its "
        f"average ROAS still looks healthy.\n"
        f"- **{rec['recipient']}** still earns **{rec['recipient_marginal_roas']:.2f}x** on the next "
        f"dollar, so the same budget works harder there.\n"
        f"- The optimizer moves spend from the lowest marginal-ROAS channel to the highest, "
        f"because marginal — not average — ROAS decides where the next dollar belongs."
    )
    impact = (
        f"- Revenue: **+${rec['expected_revenue_gain']:,.0f}/day**\n"
        f"- Blended ROAS: **{rec['blended_roas_before']:.2f}x → {rec['blended_roas_after']:.2f}x**\n"
        f"- Total spend: **unchanged at ${rec['total_spend']:,.0f}/day** (budget-neutral)\n"
        f"- Confidence: **{rec['confidence']}** ({rec['confidence_score']:.2f})"
    )
    return [
        {"title": "Recommendation", "body": recommendation},
        {"title": "Why this recommendation", "body": why},
        {"title": "Expected business impact", "body": impact},
        {"title": "Constraints applied", "body": _constraint_lines(rec)},
    ]


# ---------------------------------------------------------------------------
# Live LLM call (optional) — lazy-imported, fails closed to the fallback.
# ---------------------------------------------------------------------------
def _llm_call(payload: dict, system: str = SYSTEM_PROMPT) -> tuple[str, str] | None:
    """Return (text, model_id) from an LLM, or None if unavailable / errored."""
    import json

    if "question" in payload:
        user_msg = (
            "Answer the marketer's question using ONLY this optimizer context "
            "(JSON). If it isn't answerable from the context, say so.\n\n"
            + json.dumps(payload, indent=2)
        )
    else:
        user_msg = (
            "Explain this budget recommendation to a marketer. Use ONLY these numbers:\n\n"
            + json.dumps(payload, indent=2)
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        try:
            from openai import OpenAI

            model = os.getenv("LLM_MODEL", DEFAULT_OPENAI_MODEL)
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content.strip(), model
        except Exception:
            return None

    if anthropic_key:
        try:
            import anthropic

            model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text.strip(), model
        except Exception:
            return None

    return None


def llm_available() -> bool:
    """True if an API key is configured (the live path can be attempted)."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def active_model() -> str:
    """Name of the model that will be used for the live call (for UI display)."""
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("LLM_MODEL", DEFAULT_OPENAI_MODEL)
    if os.getenv("ANTHROPIC_API_KEY"):
        return os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
    return "none"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def explain(rec: dict) -> dict:
    """Return {text, source, model} — LLM narration with rule-based fallback."""
    result = _llm_call(_facts(rec))
    if result is not None:
        text, model = result
        return {"text": text, "source": "llm", "model": model}
    return {"text": rule_based(rec), "source": "rule-based", "model": None}


def answer_question(rec: dict, question: str) -> dict:
    """Grounded Q&A — answer a marketer's question using only optimizer numbers.

    Requires an LLM. Without one, returns a graceful, honest message rather than
    guessing (we never fabricate an answer in the fallback path).
    """
    if not llm_available():
        return {
            "text": (
                "Grounded Q&A needs an API key (set OPENAI_API_KEY or ANTHROPIC_API_KEY). "
                "Without it, the prototype falls back to the deterministic explanation above "
                "so it never fabricates an answer."
            ),
            "source": "rule-based",
            "model": None,
        }
    payload = _facts(rec)
    payload["question"] = question
    result = _llm_call(payload, system=QA_SYSTEM_PROMPT)
    if result is not None:
        text, model = result
        return {"text": text, "source": "llm", "model": model}
    return {
        "text": (
            "The AI service couldn't be reached just now (check the API key or network). "
            "The grounded explanation above still fully applies — every figure there comes "
            "straight from the optimizer."
        ),
        "source": "rule-based",
        "model": None,
    }
