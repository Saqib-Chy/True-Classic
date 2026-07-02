# Module 3 — Budget Optimization Engine (Design)

> Design-only document. No implementation yet. This describes *what* the optimizer
> does, *how* it consumes Module 2's forecasts, and *how* it renders in the existing
> **Allocation** and **Approval** tabs — sized for a 15-minute live demo, not a
> production optimization platform.
>
> Upstream: Module 1 (canonical dataframe) → Module 2 (forecasts + marginal ROAS).
> Downstream: Allocation view → Approval view → stubbed execution / audit log.

---

## 1. Average ROAS vs Marginal ROAS

This distinction is the heart of the entire system.

- **Average ROAS** = total revenue ÷ total spend. The return on *all dollars spent
  so far*. It's a **scorecard** of the past.
- **Marginal ROAS** = the return on the **next** dollar. It's the **slope** of the
  spend→revenue (saturation) curve at the current spend level. It's a **decision
  signal** about the future.

**Why they differ:** ad channels saturate. The first dollars hit the cheapest, most
intent-rich audiences; later dollars chase more expensive, less responsive ones. So a
channel can have a great *average* ROAS while its *next* dollar barely breaks even.

**Simple worked example — Meta Retargeting:**

| Daily spend | Forecast revenue | Average ROAS | Return on the last $500 (marginal) |
|---|---|---|---|
| $4,000 | $24,000 | 6.0x | — |
| $4,500 | $25,000 | 5.6x | ($25,000−$24,000)/$500 = **2.0x** |

The average ROAS is a stellar **6.0x**, but the **marginal ROAS is only 2.0x** — the
next dollar returns just $2. If your ROAS floor is 4.0x, you should **stop funding the
next dollar here**, even though the channel "looks" like your best performer.

> **One-line version for Q&A:** "You bank average ROAS; you make decisions on marginal
> ROAS. Optimizing on average ROAS over-funds saturated channels."

---

## 2. How Module 3 consumes Module 2's outputs

Module 2 produces, per segment (platform × funnel_stage, optionally × SKU) over the
7-day horizon:

| Module 2 output | How Module 3 uses it |
|---|---|
| Forecast **spend** | Baseline ("current") spend per segment to reallocate from. |
| Forecast **revenue** | Expected return at baseline spend. |
| Forecast **ROAS** (avg) | Reporting / sanity only — *not* the optimization signal. |
| **Marginal ROAS** (slope at current spend) | **The primary ranking signal** — who gets the next dollar. |
| **Confidence band** (P10–P90) | Risk-adjusts marginal ROAS and gates whether a move is allowed. |

Module 2 derives marginal ROAS by evaluating its model at `spend` and `spend ± Δ`, so
Module 3 **does not re-train anything** — it reads a table of
`segment · current_spend · forecast_revenue · marginal_roas · marginal_low · marginal_high`
and runs pure, deterministic allocation math on top of it. Clean contract, independent
modules.

---

## 3. The optimization algorithm, step by step

1. **Load segments** from Module 2 into a table with current spend, forecast revenue,
   marginal ROAS, and the marginal confidence band.
2. **Risk-adjust** each segment's marginal ROAS using its confidence band (e.g., use
   the lower bound, or `marginal × confidence_factor`) so uncertain channels aren't
   trusted blindly.
3. **Rank** segments by risk-adjusted marginal ROAS — highest = best home for the next
   dollar, lowest = first to be cut.
4. **Identify donor & recipient:** donor = lowest marginal ROAS **and** below the ROAS
   floor; recipient = highest marginal ROAS **and** above the floor.
5. **Compute the transfer amount**, bounded by all constraints (Section 5): the most
   the donor may lose (max-change %, min-spend) and the most the recipient may gain
   (max-change %, max-spend).
6. **Estimate impact** using marginal ROAS: revenue lost from the donor, revenue gained
   by the recipient, net revenue change, new blended ROAS (spend held constant).
7. **Emit a recommendation object** with before/after spend, expected deltas, the
   binding constraint, and a confidence label.

The **production** version repeats steps 3–6 **iteratively in small increments**
(water-filling) until marginals equalize or a constraint binds. The **prototype**
version does **one** bounded transfer (see Section 4).

---

## 4. Recommended approach for the prototype

**Two options:**

- **(B) Iterative water-filling allocator (production-oriented).** Move budget in small
  steps (e.g., $100), re-reading marginal ROAS as each curve bends, until all
  marginals equalize or constraints bind. Mathematically ideal — it finds the
  globally balanced allocation. But it's harder to narrate live, more code, more edge
  cases, and a marginal-curve bug can produce a confusing result mid-demo.

- **(A) Simplified one-shot constrained reallocation (recommended for the prototype).**
  Identify the single lowest-marginal segment (below floor) and the single
  highest-marginal segment (above floor), and move **one** constraint-bounded transfer
  between them. One clear arrow, one number, fully explainable in a sentence.

**Recommendation: (A) for the prototype.** It demonstrates the exact business
insight — *cut the saturated/sub-floor channel, feed the channel with headroom* —
with logic an executive grasps instantly and an engineer can audit in seconds. Frame
water-filling as the production evolution (Section 10). See the final section for the
full justification.

---

## 5. Constraints (and the business reason for each)

| Constraint | Rule | Why it exists (business rationale) |
|---|---|---|
| **Total budget constant** | Σ spend after = Σ spend before | Proves the system lifts ROAS through *better allocation*, not by spending more. Keeps the demo honest and the CFO comfortable. |
| **Max daily change** | ≤ **20%** of a segment's current spend | Platform algorithms (Meta CBO, Google smart bidding) re-enter a noisy learning phase after big budget jumps. Small moves protect delivery stability and avoid whipsawing performance. |
| **ROAS floor** | Marginal ROAS ≥ **4.0x** to keep funding | Directly encodes the primary KPI (Blended ROAS ≥ 4.0x). Any dollar whose marginal return is below the floor is destroying blended efficiency and should move. |
| **Confidence threshold** | Only reallocate if recipient's low-bound marginal > donor's high-bound marginal (or confidence ≥ threshold) | Stops the optimizer from acting on a noisy forecast. Money only moves when the forecast is trustworthy enough that the move is very likely net-positive. |
| **Min spend per channel** | Each channel ≥ a floor | Maintains presence/learning signal and market coverage; never zeroes out a channel on one forecast. |
| **Max spend per channel** | Each channel ≤ a cap | Prevents over-concentration into one channel (platform risk, audience fatigue) even if its marginal ROAS looks best today. |

Every constraint maps to a real risk: overspend, delivery instability, platform
concentration, or acting on noise.

---

## 6. Complete numerical example

**Step 1 — Module 2 forecast outputs (next-7-day daily averages):**

| Segment | Current Spend | Forecast Revenue | Avg ROAS | Rev @ +$500 | **Marginal ROAS** |
|---|---|---|---|---|---|
| Meta — Retargeting | $4,000 | $24,000 | 6.0x | $25,000 | **2.0x** |
| Meta — Prospecting | $8,000 | $24,000 | 3.0x | $25,300 | 2.6x |
| Google — Search | $3,000 | $15,000 | 5.0x | $16,750 | 3.5x |
| Google — PMax | $5,000 | $21,000 | 4.2x | $23,400 | **4.8x** |
| **Total** | **$20,000** | **$84,000** | **4.20x blended** | | |

**Step 2 — Marginal ROAS + confidence (risk-adjusted):**
- Lowest marginal: **Meta Retargeting = 2.0x** (P90 ≈ 2.4x) → below 4.0x floor → **donor**.
- Highest marginal: **Google PMax = 4.8x** (P10 ≈ 4.3x) → above floor → **recipient**.

**Step 3 — Confidence gate:** recipient P10 (4.3x) > donor P90 (2.4x) → move is safe. ✅

**Step 4 — Bound the transfer:** max daily change = 20% → Meta Retargeting may lose at
most `0.20 × $4,000 = $800`. Recipient cap not breached. **Transfer = $800.**

**Step 5 — Impact (spend held constant at $20,000):**

| | Spend before | Spend after | Revenue effect |
|---|---|---|---|
| Meta — Retargeting | $4,000 | **$3,200** | ≈ −(2.0 × $800) = **−$1,600** |
| Google — PMax | $5,000 | **$5,800** | ≈ +(4.8 × $800) = **+$3,840** |
| **Net** | $20,000 | $20,000 | **+$2,240 / day** |

- **Blended ROAS:** `$84,000 → $86,240` on `$20,000` → **4.20x → 4.31x**. ✅
- **NC-CPA:** dollars move out of retargeting (≈no new customers) into PMax prospecting
  (new-customer heavy) → new-customer volume up at constant spend → NC-CPA holds/improves. ✅

**Step 6 — Recommendation object (the contract the Approval view renders):**

```
Move $800/day:   Meta Retargeting  →  Google PMax
Why:             Meta Retargeting marginal ROAS 2.0x < 4.0x floor (saturated);
                 Google PMax marginal ROAS 4.8x with headroom.
Expected:        Blended ROAS 4.20x → 4.31x   ·   +$2,240/day revenue
Constraint hit:  capped at 20% max daily change on donor
Confidence:      HIGH (recipient P10 4.3x > donor P90 2.4x)
```

---

## 7. Streamlit "Allocation" page

Executive-readable: the decision and its impact first, the mechanics on demand.
Consistent with the Module 1 dark dashboard styling.

- **Executive summary (top banner):** one sentence —
  *"Move $800/day from Meta Retargeting to Google PMax → blended ROAS 4.20x → 4.31x,
  +$2,240/day."*
- **Impact KPI cards:** Expected Revenue Gain (`+$2,240/day`), Blended ROAS
  (`4.20x → 4.31x`, green delta), Total Spend (`$20,000 — unchanged`), Confidence
  (`HIGH`).
- **Current vs Recommended allocation:** side-by-side table or grouped bar chart of
  per-segment spend before/after, with marginal ROAS shown per row.
- **Budget movement visualization:** a clear donor→recipient arrow (or a small
  before/after bar / waterfall) so the single move is unmistakable.
- **Constraints triggered:** chips/badges listing what bound the result
  (e.g., "20% max-change cap on donor", "ROAS floor breached on Meta Retargeting").
- **Adjustable constraints (sidebar):** sliders for ROAS floor, max daily change %,
  confidence threshold, total budget → **Re-run** recomputes live (sets up the
  Approval interaction).
- **Diminishing-returns callout:** explicit flag for any segment whose marginal ROAS
  fell below the floor.

Keep raw tables in expanders; lead with the summary, cards, and the movement viz.

---

## 8. "Approval" page (human-in-the-loop)

The marketer reviews, adjusts, and approves before anything "executes."

- **Recommendation cards:** one row per recommended move with **Approve / Reject**
  toggles.
- **Editable recommended spend:** the marketer can override the proposed target (e.g.,
  move $600 instead of $800); the expected impact recomputes from the same marginal
  ROAS math.
- **Adjustable constraints + Re-run:** the sidebar sliders from the Allocation page
  let them tighten the floor or change the max-change cap and regenerate.
- **LLM-generated explanation:** a plain-English rationale built **only from optimizer
  outputs** (segments, spend deltas, marginal ROAS, constraint, confidence). The LLM
  *narrates numbers it is given* — it never invents figures. Raw numbers are shown
  alongside; a deterministic rule-based template is the fallback if no API key is
  present. (Hallucination control = closed inputs + visible numbers + fallback.)
- **Stubbed execution + audit log:** **Approve & Execute** does **not** call any ad
  platform API. It appends the decision (timestamp, segments, amounts, who approved,
  resulting KPIs) to `data/audit_log.csv` and shows a confirmation — the accountability
  trail that production execution would build on.

---

## 9. Demo script (≈3–4 minutes)

**0:00–0:30 — Frame the decision.**
- *Click:* open the **Allocation** tab.
- *Appears:* the executive summary banner + impact KPI cards.
- *Say:* "Module 2 forecasts where each channel is heading. Module 3 turns that into a
  budget decision. Here's today's top move and its expected impact — all computed
  live."

**0:30–1:30 — Average vs marginal (the key idea).**
- *Click:* point at the current-vs-recommended table.
- *Appears:* Meta Retargeting (avg ROAS 6.0x) being **cut**; Google PMax being **fed**.
- *Say:* "Meta Retargeting has the *highest* average ROAS, but its *marginal* ROAS is
  only 2.0x — it's saturated, below our 4.0x floor. Google PMax's next dollar returns
  4.8x. So the optimizer cuts the saturated channel and feeds the one with headroom.
  Optimizing on average ROAS would get this exactly backwards."

**1:30–2:30 — Constraints are live.**
- *Click:* drag the **max daily change** slider, then **ROAS floor**; hit **Re-run**.
- *Appears:* the recommended transfer and constraint chips update.
- *Say:* "Every guardrail is real. The move is capped at 20% so we don't throw the
  platform into a learning phase. Tighten the floor and the system flags more channels.
  These aren't decorations — they bound the math."

**2:30–3:30 — Approve with a human in the loop.**
- *Click:* go to **Approval**; read the LLM explanation; edit the amount; click
  **Approve & Execute**.
- *Appears:* recomputed impact, then an audit-log confirmation.
- *Say:* "A marketer never gets a black box. This explanation is an LLM narrating the
  optimizer's actual numbers — never inventing them. They can adjust and approve.
  Execute is stubbed to an audit log; in production this is where the Meta/Google API
  writes go, behind this exact approval gate."

**Reserve for Q&A:** confidence gating (Section 5), why one-shot vs water-filling
(Section 11), and budget-neutrality.

---

## 10. Evolving into a production optimizer (brief)

- **Iterative water-filling** across all segments simultaneously, not a single move.
- **Finer grain:** campaign / ad-set level (thousands of levers) instead of channel.
- **Convex solver** over fitted saturation curves for a provably optimal allocation
  under constraints.
- **Real execution** behind the approval gate: Meta/Google API writes with dry-run,
  change limits, and rollback.
- **Closed loop:** feed realized performance back to Module 2 to recalibrate curves;
  monitor recommendation quality and auto-tune constraints.
- **More objectives:** multi-objective optimization balancing ROAS, NC-CPA, and growth
  targets explicitly.

---

## 11. Recommendation: (A) one-shot vs (B) water-filling

**Recommendation: implement (A), the simplified one-shot constrained reallocation,**
for the prototype.

| Criterion | (A) One-shot | (B) Water-filling |
|---|---|---|
| **Interview scope** | Demonstrates the full business idea (avg vs marginal, constraints, approval) in one clear move. | Same idea, but the extra machinery doesn't add *insight* in 15 minutes. |
| **Implementation complexity** | Low — rank, pick donor/recipient, bound, compute impact. | Higher — increment loop, curve re-evaluation, convergence & edge cases. |
| **Reliability** | Very high — deterministic, few failure modes; demo won't surprise you. | More moving parts → more ways to break live. |
| **Explainability** | One arrow, one number, one sentence — executive-perfect. | Harder to narrate ("it iterated 47 times") without losing the room. |

(B) is the right *production* answer and you should **say so** — but for an interview
judged on clarity, reliability, and storytelling, (A) wins decisively. Build (A), keep
the segment/marginal-ROAS interface clean, and describe (B) as the next step. That
demonstrates judgment: you know the ideal, and you scoped deliberately for the context.

---

## Why not let the LLM optimize the budget?

**Budget allocation is a constrained optimization problem, not a language problem.**
The task is precisely defined: maximize blended ROAS by distributing a fixed pool of
dollars across segments, subject to hard limits (budget constant, ≤20% daily change,
≥4.0x marginal-ROAS floor, min/max per channel, confidence gate). That is numerical
search over a feasible region — the native domain of optimization algorithms. An LLM
predicts likely *text*; it does not minimize an objective function or enforce
inequalities. Framing budget math as a generation task is using the wrong tool for the
shape of the problem.

**A deterministic optimizer is the right tool for financial decisions.** Money
decisions must be correct, repeatable, and explainable to the dollar. A deterministic
optimizer guarantees that every constraint is satisfied by construction and that the
output follows directly from the inputs.

**What "deterministic" means here:** the same inputs (forecasts, current spend,
constraints) always produce the **exact same** recommendation — every run, every
machine. That property delivers two things finance teams require:
- **Reproducibility** — anyone can re-run the optimizer and get the identical answer,
  so results can be verified rather than trusted on faith.
- **Auditability** — every recommended dollar traces back to a specific marginal-ROAS
  value and the binding constraint, so we can answer "*why did we move this $800?*"
  with a precise, defensible chain. Allocating ad budget is spending real money;
  "the model felt this was best" is not an acceptable audit trail.

**Could an LLM be prompted to recommend allocations? Honestly, yes — often
reasonably.** A modern model with a well-structured prompt and the forecast numbers in
context can produce allocations that look sensible and even articulate the average-vs-
marginal intuition. We should not pretend it can't. The objection is not capability —
it's **guarantees**.

**So the architecture separates responsibilities on purpose:**
- The **optimizer** performs *all* numerical optimization and owns the budget numbers.
- The **LLM never decides allocations** — it sees the optimizer's output, not the
  budget knobs.
- The **LLM does what it's genuinely best at**: explaining a recommendation in plain
  English, answering natural-language questions about it ("why was Meta cut if its ROAS
  is highest?"), and supporting the human in the approval loop.

**Practical risks of letting an LLM optimize directly:**
- **No constraint guarantee** — it may quietly breach the ROAS floor, the 20% cap, or
  budget-neutrality; there's no mechanism that *forces* feasibility.
- **Non-deterministic output** — the same inputs can yield different numbers across
  runs, so recommendations can't be reliably reproduced or regression-tested.
- **Reduced auditability** — a number that emerged from a generation step can't be
  cleanly traced to an input and a rule.
- **Hard to justify to stakeholders** — "the optimizer cut this dollar because its
  marginal ROAS was 2.0x, below the 4.0x floor" defends itself; "the LLM suggested it"
  does not survive a CFO's follow-up.

**The governing principle.** This split isn't anti-LLM — it's right-tool-for-the-job
AI engineering:

> Use deterministic algorithms for deterministic problems, and use LLMs where language
> understanding, explanation, and interaction create the most value.

The optimizer makes the decision; the LLM makes the decision *understandable*. Each is
used for exactly what it does best, which is what makes the system both trustworthy and
genuinely AI-native.
