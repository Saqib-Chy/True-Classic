# True Classic — Paid Media Intelligence Prototype

A Streamlit prototype for optimizing Meta and Google ad spend. It ingests mock platform exports, unifies them into a canonical dataset, forecasts channel performance, recommends budget reallocations using marginal ROAS, and puts a human in the loop with grounded LLM explanations and an audit trail.

Built as an AI Solutions Architect case study — deterministic optimization, optional LLM narration only.

## Quick start

```bash
pip install -r requirements.txt
python3 -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Mock CSVs are generated automatically on first run if missing.

### Optional: OpenAI (grounded Q&A + LLM narrative)

Create a `.env` file in the project root (never commit it):

```
OPENAI_API_KEY=sk-...
```

Without a key, the app still runs using deterministic rule-based explanations.

## Pipeline

```
Mock CSVs → Ingest → Normalize → Reconcile → Forecast → Optimize → Explain → Approve → Audit log
```

| Module | File | Role |
|---|---|---|
| Ingestion | `app/ingestion.py` | Load Meta/Google exports + SKU catalog |
| Normalization | `app/normalize.py` | Canonical schema, funnel stage parsing |
| Reconciliation | `app/reconcile.py` | SKU crosswalk, flag unmatched keys |
| Forecasting | `app/forecast.py` | `GradientBoostingRegressor` per platform, 7-day horizon |
| Optimization | `app/optimize.py` | One-shot marginal-ROAS reallocation under constraints |
| Explanation | `app/explain.py` | Grounded LLM narrative + Q&A (optional) |
| Audit log | `app/audit_log.py` | Append-only decision trail (`data/audit_log.csv`) |
| UI | `app/streamlit_app.py` | Six tabs, sidebar constraints, orchestration |

**Optimizer grain:** platform × funnel stage (4 channels). **Execution is stubbed** — no live ad-platform API writes.

## App tabs

- **Overview** — executive KPIs and top recommendation
- **Ingestion & Unification** — raw data, data quality, SKU mismatch
- **Forecast** — 7-day revenue/ROAS with confidence bands
- **Efficiency Curves** — spend→response visualization (UI only, not optimizer input)
- **Allocation** — optimizer ranking, constraints, recommendation
- **Approval** — review, modify amount, approve/reject with required comment, audit log

## Deploy (Streamlit Cloud)

- **Main file:** `app/streamlit_app.py`
- **Branch:** `main`
- **Secrets:** add `OPENAI_API_KEY` under Advanced settings if you want live LLM features

## Docs

Design and architecture notes live in `docs/`:

- `01_architecture.md` — production-style architecture overview
- `02_prototype_build_plan.md` — build plan and demo script
- `03_forecasting_design.md` — Module 2 design
- `04_optimization_design.md` — Module 3 design

## Tech stack

Python · pandas · NumPy · scikit-learn · Altair · Streamlit · OpenAI (optional)
