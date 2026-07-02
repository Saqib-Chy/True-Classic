# Architecture — True Classic Paid Media Intelligence Prototype

A concise architecture for the prototype. The detailed, runnable build steps live in
`02_prototype_build_plan.md`; this document explains the *shape* of the system and the
reasoning behind it.

---

## 1. Business problem

True Classic is a global DTC apparel brand spending millions annually across **Meta, Google,
Amazon, and Microsoft**. The paid media operation drives the majority of DTC revenue and is the
single largest controllable cost lever — but it is fragmented:

- **Nothing is unified.** Each platform has its own schema, auth, attribution window, and
  reporting cadence.
- **Attribution conflicts.** Meta (7-day click / 1-day view), Google (data-driven or
  last-click), Amazon (last-touch, 14-day), and Shopify/GA4 all disagree on what drove a sale.
- **Taxonomies don't match.** Campaign naming, SKU/product IDs, and funnel definitions differ
  across platforms.
- **Thousands of daily levers.** Hundreds of campaigns × objectives × audiences × budgets.

The system must make sense of the mess, **forecast performance**, **optimize budget allocation
across channels**, and **automate execution recommendations** — balancing the two-sided risk of
overspending on inefficient placements vs underfunding channels with room to scale.

**Success criteria (priority order):**
1. **Maximize Blended ROAS** (target ≥ 4.0x) — surface diminishing returns, recommend lifts.
2. **Control New-Customer CPA** (≤ target) — separate prospecting from retargeting.
3. **Eliminate waste & underspend** — flag both over-budget inefficiency and caps-hit-by-noon
   underspend.

---

## 2. System flow

```
  Meta CSV ─┐
            ├─▶ Ingestion ─▶ Normalize ─▶ Reconcile ─▶ Unified Dataframe
  Google ───┘                (canonical    (SKU         (one canonical
   CSV                         schema)       crosswalk     fact table +
                                             + flag        quality flags)
                                             mismatch)
                                                   │
                                                   ▼
                                            Forecast (per channel)
                                       XGBoost + seasonality + confidence
                                                   │
                                                   ▼
                                       Saturation / response curves
                                       (spend → revenue, marginal ROAS)
                                                   │
                                                   ▼
                                       Optimization (constrained allocator)
                                       maximize blended ROAS s.t. budget,
                                       ROAS floor, NC-CPA ceiling, max change
                                                   │
                                                   ▼
                                       Recommendations + KPIs + LLM rationale
                                                   │
                                                   ▼
                                       Marketer Approval (human-in-the-loop)
                                       adjust constraints · approve/reject
                                                   │
                                                   ▼
                                       Execution (STUBBED → audit log)
```

The **Overview tab** sits on top as a read-only command center, composing the KPI and
recommendation outputs into an executive landing view.

---

## 3. Module responsibilities

| Module | Responsibility |
|---|---|
| `ingestion` | Load raw Meta + Google CSV exports as-is; no transformation. |
| `normalize` | Map each platform's schema into one **canonical fact table** (`spend`/`cost` → `spend`, `date`/`day` → `date`, etc.); parse `funnel_stage` (prospecting vs retargeting) from campaign names. |
| `reconcile` | Resolve product identity across mismatched keys (Meta catalog ID vs Google item_id) via a crosswalk; **detect and surface** unreconciled SKUs instead of dropping them. |
| `forecast` | Build features and train a per-channel model (XGBoost/GBR) to forecast revenue/ROAS, with a confidence band and backtest accuracy. |
| `curves` | Fit a spend→response **saturation curve** per channel; expose marginal ROAS at current spend. |
| `optimize` | Constrained **marginal-ROAS allocator**: move each next dollar to the highest-return channel until a constraint binds; detect diminishing returns below the ROAS floor. |
| `kpis` | Compute Blended ROAS, NC-CPA, and budget utilization from the unified data. |
| `explain` | Generate a plain-English rationale for each recommendation via LLM, with a deterministic rule-based fallback. |
| `audit_log` | Persist approved/rejected decisions (stubbed execution trail). |
| `streamlit_app` | Thin UI glue: 6 tabs (Overview, Ingestion, Forecast, Curves, Allocation, Approval) wiring the modules together. |

Each module exposes a small, typed function surface so it can be built and tested
independently — the canonical dataframe is the contract between M1 and everything downstream.

---

## 4. Real vs mocked

| Component | Status | Notes |
|---|---|---|
| Source CSVs | **Mocked** | Seeded synthetic generator; deterministic and repeatable. |
| Schema normalization | **Real** | Genuine column mapping + funnel-stage parsing. |
| SKU reconciliation | **Real** | Crosswalk join + fallback; real detection of one intentional mismatch. |
| Attribution | **Simplified** | Uses platform-reported revenue; production reconciles to Shopify. |
| Forecast model | **Real** | Trained live on the synthetic history. |
| Confidence interval | **Real** | Residual/quantile-based band. |
| Saturation curves | **Real** | Curve fit per channel. |
| Optimizer | **Real** | Constrained marginal-ROAS allocator. |
| Diminishing-returns detection | **Real** | Threshold on marginal ROAS. |
| KPI math | **Real** | Computed from unified data. |
| LLM explanation | **Real call, safe** | Narrates given numbers; rule-based fallback. |
| Platform execution | **Stubbed** | Writes to local audit log; no real API call. |

---

## 5. Why the LLM explains, but never does budget math

The budget allocation is a **constrained optimization problem**, and it is handled by a
deterministic optimizer — *not* the LLM. The reasoning:

- **Auditability.** Money decisions must be reproducible and traceable to inputs (forecast,
  curve, constraint). An optimizer produces the same answer every run; an LLM does not.
- **Hard constraint satisfaction.** ROAS floors, NC-CPA ceilings, and budget caps are
  inviolable. Optimizers guarantee feasibility; LLMs approximate and can violate constraints.
- **Hallucination risk.** LLMs can fabricate plausible-but-wrong numbers — unacceptable when
  every dollar must be accountable.

The LLM is used where it genuinely adds value: **turning the optimizer's numeric output into a
clear, human-readable rationale** ("shift $1,800 from Google to Meta — Google retargeting
marginal ROAS fell to 2.1x, below your 4.0x floor; Meta has headroom"). Hallucination controls:

- The LLM only **narrates numbers passed to it** — it never originates figures.
- Raw numbers are shown **alongside** the narration for verification.
- A **rule-based template fallback** runs if no API key is present or the call fails, so the
  demo never depends on the model being up.

---

## 6. Production next steps

With more time, the prototype hardens into the production architecture:

- **Real platform APIs.** Replace CSV ingestion with live connectors for Meta Marketing API and
  Google Ads API (per-platform auth, rate limits, pagination, schema-drift handling), behind the
  same canonical-schema contract.
- **Shopify/GA4 attribution as source of truth.** Reconcile platform-reported revenue against
  Shopify (real DTC revenue), with new-vs-returning segmentation; this is where an MMM/MTA layer
  or Northbeam/Triple Whale plugs in to settle attribution conflicts.
- **Amazon + Microsoft.** Add Amazon Ads (ASIN-level, 14-day window, TACOS metric) and Microsoft
  Ads connectors — the canonical schema and optimizer generalize without rework.
- **Looker integration.** Publish the gold marts to Looker (their internal BI standard) so
  reporting and the platform agree on the same numbers.
- **Execution at scale.** Move from channel-level to campaign/ad-set grain; wire real platform
  writes behind the approval gate with dry-run, change limits, and rollback.
- **Monitoring & observability.** Data-quality event stream, model drift detection, forecast
  accuracy tracking, and an audit trail of every recommendation, override, and execution.
- **Orchestration.** Schedule ingestion → reconcile → forecast → optimize on a recurring DAG
  (Airflow/Dagster/Prefect) with retries and watermarks.
