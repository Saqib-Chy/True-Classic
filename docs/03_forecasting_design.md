# Module 2 — AI Performance Forecasting Engine (Design)

> Design-only document. No implementation yet. This explains *what* the
> forecasting module does, *why*, and *how* it plugs into Module 1 (Ingestion)
> upstream and Module 3 (Budget Optimization) downstream — sized for a credible
> 15-minute demo, not a production ML platform.

---

## 1. Objective

**What it predicts.** Given the clean canonical dataframe from Module 1, this
module forecasts **near-term paid-media performance per platform** (and, where
data allows, per SKU): primarily **revenue** and **spend**, from which it derives
**ROAS** (revenue ÷ spend) over a short **forecast horizon (next 7 days)**, with a
**confidence band** around each estimate.

**Why forecast before optimizing.** Budget allocation (Module 3) is a
*forward-looking* decision — you're deciding where tomorrow's dollars go. You
cannot allocate on yesterday's numbers alone: a channel that performed well last
week may be saturating, and a channel that looks expensive today may have
headroom. The optimizer needs an **expected return per channel** and a sense of
**how much to trust it** before it can move money responsibly. The forecast turns
raw history into the forward signal the optimizer consumes.

**The chain.**

```
Historical Data            Forecast                     Budget Allocation
(canonical df)        (Module 2 output)                  (Module 3)
─────────────         ─────────────────                 ───────────────
reconciled daily  →   expected revenue / ROAS / spend  →  shift $ toward the
spend & revenue       per platform for next 7 days,       highest expected
per platform/SKU      with a confidence band              marginal return,
                                                           weighted by confidence
```

- **Historical → Forecast:** the model learns the relationship between spend (and
  time/seasonality) and revenue, then projects it forward.
- **Forecast → Allocation:** the optimizer reads the per-channel expected ROAS and
  uncertainty to decide reallocations. A high-confidence, high-ROAS channel gets
  more; a low-confidence or saturating channel gets less.

The forecast is an **input** to the decision, never the decision itself.

---

## 2. Inputs

The module consumes the **canonical dataframe** produced by Module 1 (already
normalized and SKU-reconciled). Relevant columns:

| Column | Role in forecasting |
|---|---|
| `date` | Time index; source of seasonality features (day-of-week, trend). |
| `platform` | Primary grain — forecasts are produced per platform (Meta vs Google). |
| `funnel_stage` | Optional secondary grain (prospecting vs retargeting) — they behave differently. |
| `sku` | Optional finer grain; reconciled product key for per-SKU views. |
| `spend` | **Core driver feature** — the spend→revenue relationship is what we model. |
| `revenue` | **Primary target** to forecast. |
| `conversions` | Supporting signal / optional secondary target. |
| `clicks` | Engagement signal (feature). |
| `new_customers` | Feeds NC-CPA awareness used later by Module 3. |
| `match_status` | Filter — only `matched` rows enter modeling; `unmatched` are excluded (already flagged in M1). |

**Features actually used in the prototype (kept deliberately small):**

- **`spend`** (the dominant predictor — the lever Module 3 controls).
- **Calendar features** derived from `date`: day-of-week, and a simple time index
  for trend.
- **`platform`** (and optionally `funnel_stage`) as the segmentation key — we train
  per segment so each channel's response is learned independently.
- Lagged/rolling spend & revenue (e.g., 7-day rolling average) **if time permits**;
  optional, not required for the demo.

**Explicitly out of scope for the prototype:** breakdowns by age/gender/placement,
keyword-level data, cross-platform interaction terms. Named as production
extensions, not built now.

---

## 3. Model Selection

We need a model that learns a **nonlinear spend→revenue relationship**, trains in
seconds on a small dataset, and is easy to explain in Q&A.

| Approach | Strengths | Weaknesses for this prototype | Verdict |
|---|---|---|---|
| **Prophet** | Great for seasonality/trend decomposition; clean intervals. | Univariate at heart — models a time series, not spend→response. Awkward to express "revenue as a function of spend." Extra dependency. | Use later for seasonality, not the core. |
| **GradientBoostingRegressor** (scikit-learn) | Already in sklearn (no extra dep), nonlinear, handles tabular features, fast on small data, supports **quantile loss** for intervals. | Slightly slower than LightGBM at scale (irrelevant here). | ✅ **Recommended** |
| **XGBoost** | Powerful, industry standard, fast. | Extra dependency; more knobs to defend; overkill for ~300 rows. | Reserve for scale. |
| **LightGBM** | Very fast, great on large data. | Extra dependency; can be unstable on *tiny* datasets; histogram binning shines at volume we don't have. | Reserve for scale. |

**Recommendation: `GradientBoostingRegressor` (scikit-learn).**

Rationale against the stated priorities:

- **Explainability** — tree-based, easy to describe ("learns how revenue responds
  to spend and day-of-week"); feature importances are inspectable.
- **Speed** — trains in well under a second on ~300 rows; demo never stalls.
- **Robustness** — handles small, noisy, nonlinear data without heavy tuning; no
  risk of a flaky fit mid-demo.
- **Ease of demo** — ships inside scikit-learn (already a light dependency), so no
  new install and nothing exotic to defend.

> **Defensible framing for Q&A:** "I chose `GradientBoostingRegressor` because at
> prototype scale it gives XGBoost-class accuracy with zero extra dependencies and
> full explainability. The architecture swaps in XGBoost/LightGBM unchanged once we
> have production data volume — the model is behind an interface." Prophet is named
> as the seasonality-decomposition add-on, not the core predictor.

---

## 4. Forecast Outputs

For each segment (platform, optionally × funnel_stage / sku) and each day in the
horizon, the module produces:

| Output | Description |
|---|---|
| **Forecasted Spend** | Planned/assumed daily spend the forecast is conditioned on (the lever Module 3 will vary). |
| **Forecasted Revenue** | Model's expected revenue at that spend level. |
| **Forecasted ROAS** | Derived: forecasted revenue ÷ forecasted spend. |
| **Confidence Interval** | Lower/upper band (e.g., P10–P90) around forecasted revenue/ROAS. |
| **Forecast Horizon** | Next **7 days** (configurable). |
| **Accuracy metric** | Backtest error (e.g., MAPE) per segment, so trust is quantified. |

**Output shape (conceptual):** a tidy table —
`date · platform · [funnel_stage] · forecast_spend · forecast_revenue · forecast_roas · roas_low · roas_high`.

**How Module 3 consumes it:**

- Aggregates the per-day forecast to a **per-channel expected ROAS** over the
  horizon — the core input to the allocator.
- Uses the **confidence band** to weight decisions: tighter band → optimizer trusts
  the channel and shifts more aggressively; wider band → it stays conservative.
- Combined with the **spend→response (saturation) curve** (the same model evaluated
  at multiple spend levels), the optimizer reads **marginal ROAS** — the return on
  the *next* dollar — which is what actually drives reallocation and
  diminishing-returns detection.

> Note: the saturation curve is produced by **querying this same model at a range of
> spend values**, so Module 2 owns both the point forecast and the response curve;
> Module 3 only consumes them.

---

## 5. Confidence (lightweight)

The prototype needs an *honest, simple* uncertainty estimate — not a full Bayesian
treatment. Two prototype-appropriate options:

1. **Quantile regression (recommended).** `GradientBoostingRegressor` supports
   `loss="quantile"`. Train three quick models — median (P50), lower (P10), upper
   (P90) — and use the P10/P90 predictions as the band. Native to the chosen model,
   no extra math.
2. **Residual-based band (fallback).** Compute residuals on a holdout/backtest,
   take their standard deviation (or empirical quantiles), and apply `± k·σ` around
   the point forecast. Trivial to implement and explain.

Either way, also report a **backtest MAPE** per segment (train on early dates,
predict the held-out tail, compare) so the UI can state plainly "Meta forecast is
~X% accurate; trust it more than Google's at Y%."

**Deliberately avoided for the prototype:** conformal prediction, Bayesian
structural models, bootstrapped ensembles — named as production upgrades only.

---

## 6. Streamlit UI — Forecast page

Executive-friendly: headline numbers first, one clear chart, controls on the side,
detail on demand.

**Controls (top or sidebar):**
- **Platform selector** — Meta / Google / All.
- **SKU selector** — All SKUs or a specific reconciled SKU.
- (Optional) funnel-stage toggle and horizon slider (default 7 days).

**KPI cards (row across the top):**
- **Forecasted Revenue (next 7d)** — with delta vs the prior 7 days.
- **Forecasted ROAS** — with a green/red chip vs the 4.0x target.
- **Forecasted Spend (next 7d)**.
- **Model Accuracy (MAPE)** — so viewers know how much to trust it.

**Historical vs forecast chart (centerpiece):**
- Line of **historical revenue** (solid) transitioning into **forecasted revenue**
  (dashed), split by a "today" marker.
- **Confidence band** shaded around the forecast (P10–P90).
- Toggle to view ROAS instead of revenue.

**Model summary panel:**
- Plain-language card: model name (`GradientBoostingRegressor`), features used,
  horizon, backtest accuracy, and the confidence method — so the stack is legible
  without reading code.

**Forecast table (collapsed expander):**
- The tidy per-day output (`date · forecast_spend · forecast_revenue · forecast_roas
  · roas_low · roas_high`) for interviewers who want the raw numbers.

Keep the page clean on open: KPIs + chart visible, table tucked in an expander,
consistent with the Module 1 dark dashboard styling.

---

## 7. Demo Flow (≈3–4 minutes)

**0:00–0:30 — Set up the "why."**
- *Click:* open the **Forecast** tab.
- *Appears:* KPI cards + the historical-vs-forecast chart for the default (All
  platforms).
- *Say:* "Module 1 gave us clean unified data. Before we can move budget, we need to
  know where each channel is *heading*. This is the forecast that feeds the
  optimizer."

**0:30–1:30 — Show a per-channel forecast.**
- *Click:* set **Platform = Meta**.
- *Appears:* Meta's history flows into a dashed 7-day forecast with a shaded
  confidence band; KPI cards update (forecasted revenue, ROAS vs 4.0x, MAPE).
- *Say:* "This is a `GradientBoostingRegressor` trained live on the Meta history —
  it learns how revenue responds to spend and day-of-week. The shaded band is the
  P10–P90 confidence range; the MAPE card tells you how accurate it's been in
  backtest."

**1:30–2:30 — Contrast channels.**
- *Click:* switch **Platform = Google**.
- *Appears:* a different curve shape, different ROAS, different band width.
- *Say:* "Notice Meta and Google forecast differently — different spend-response and
  different confidence. That difference is exactly what the optimizer needs; it
  shouldn't treat the channels identically."

**2:30–3:30 — Tie to the next module + show the numbers.**
- *Click:* expand the **forecast table**; optionally pick a single **SKU**.
- *Appears:* the per-day forecast rows with spend, revenue, ROAS, and the band.
- *Say:* "Every row here — expected ROAS plus its confidence — becomes an input to
  Module 3. A high-ROAS, tight-confidence channel earns more budget; a saturating or
  uncertain one earns less. That's the hand-off to the allocator."

**Reserve for Q&A:** why GradientBoosting over Prophet/XGBoost (Section 3), how the
confidence band is computed (Section 5), and how the saturation curve is derived
from the same model (Section 4).

---

## 8. Production Improvements (brief)

- **Retraining** — scheduled retrains (e.g., nightly) on fresh data via the
  orchestration layer; promote models through a registry.
- **Feature store** — centralized, versioned features (lagged spend, promo
  calendar, price, seasonality) shared by training and serving to prevent skew.
- **Model monitoring** — track live forecast accuracy (MAPE/bias) per channel and
  alert when it degrades.
- **Drift detection** — monitor input distribution and spend→response shifts; flag
  when relationships change (e.g., after a platform algorithm update) and trigger
  retraining.
- **Scale-out models** — swap `GradientBoostingRegressor` for XGBoost/LightGBM and
  add Prophet-based seasonality decomposition once data volume justifies it.
