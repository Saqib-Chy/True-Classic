"""True Classic — Paid Media Intelligence prototype.

End-to-end slice across four modules:
  M1 Ingestion & Unification · M2 Forecasting · M3 Budget Optimization ·
  M4 Approval workflow + LLM explanation (human-in-the-loop, stubbed execution).

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import audit_log
import explain
import forecast
import ingestion
import normalize
import optimize
import reconcile

# On Streamlit Community Cloud the OpenAI key is supplied via the Secrets manager
# (st.secrets), while locally it comes from a .env file. Bridge st.secrets into the
# environment so explain.py (which reads os.getenv) behaves identically in both.
try:
    if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(
    page_title="True Classic — Paid Media Intelligence",
    page_icon="📊",
    layout="wide",
)

# ----- Lightweight styling (presentation only) -----
st.markdown(
    """
    <style>
      .tc-status {
        padding: 12px 14px; border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        display: flex; align-items: center; gap: 10px; height: 100%;
      }
      .tc-status .ico { font-size: 18px; line-height: 1; }
      .tc-status .txt { font-size: 0.9rem; font-weight: 600; color: #e6e6e6; }
      .tc-ok   { border-left: 4px solid #2ecc71; }
      .tc-warn { border-left: 4px solid #f1c40f; }

      .tc-flow {
        display: flex; align-items: stretch; justify-content: space-between;
        gap: 6px; margin: 4px 0 2px 0;
      }
      .tc-step {
        flex: 1; text-align: center; padding: 14px 8px; border-radius: 10px;
        background: linear-gradient(180deg, rgba(99,102,241,0.16), rgba(99,102,241,0.04));
        border: 1px solid rgba(99,102,241,0.30);
        font-weight: 600; color: #e6e6e6; font-size: 0.92rem;
      }
      .tc-step .sub {
        display: block; font-size: 0.72rem; font-weight: 400;
        color: #9aa0a6; margin-top: 4px;
      }
      .tc-arrow { display: flex; align-items: center; color: #6b7280; font-size: 22px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _status_card(icon: str, text: str, ok: bool = True) -> str:
    cls = "tc-ok" if ok else "tc-warn"
    return (
        f'<div class="tc-status {cls}">'
        f'<span class="ico">{icon}</span><span class="txt">{text}</span></div>'
    )


def _marginal_vs_avg_chart(channels_df: pd.DataFrame, roas_floor: float):
    """Grouped bars: average vs marginal ROAS per channel + the ROAS floor line.

    This is the visual heart of Module 3 — it shows channels that look healthy on
    *average* ROAS while their *marginal* (next-dollar) ROAS has fallen below the
    floor, which is exactly what the optimizer acts on.
    """
    long = pd.concat(
        [
            channels_df[["channel", "forecast_roas"]]
            .rename(columns={"forecast_roas": "roas"})
            .assign(kind="Average ROAS"),
            channels_df[["channel", "marginal_roas"]]
            .rename(columns={"marginal_roas": "roas"})
            .assign(kind="Marginal ROAS (next $)"),
        ],
        ignore_index=True,
    )
    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("channel:N", title=None, sort="-y", axis=alt.Axis(labelAngle=-15)),
            y=alt.Y("roas:Q", title="ROAS (x)"),
            xOffset=alt.XOffset("kind:N"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Average ROAS", "Marginal ROAS (next $)"],
                    range=["#9aa0a6", "#6366f1"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["channel", "kind", alt.Tooltip("roas:Q", format=".2f")],
        )
    )
    floor_rule = (
        alt.Chart(pd.DataFrame({"y": [roas_floor]}))
        .mark_rule(color="#f1c40f", strokeDash=[5, 4], size=2)
        .encode(y="y:Q")
    )
    floor_text = (
        alt.Chart(pd.DataFrame({"y": [roas_floor], "label": [f"{roas_floor:.1f}x floor"]}))
        .mark_text(align="left", dx=4, dy=-6, color="#f1c40f", fontWeight="bold")
        .encode(y="y:Q", text="label:N")
    )
    return (bars + floor_rule + floor_text).properties(height=280, width="container")


def _solve_sat_x(b: float) -> float | None:
    """Solve e^x - 1 = x / b for x > 0 (0 < b < 1); None when ~linear (b→1).

    Used to shape a saturating curve revenue = Rmax * (1 - e^(-spend/τ)) whose
    tangent slope at today's spend equals the channel's marginal ROAS. The
    smaller b (marginal/average), the more the curve bends — genuine diminishing
    returns, not a cosmetic tweak.
    """
    if b >= 0.97:
        return None
    lo, hi = 1e-4, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.exp(mid) - 1 - mid / b > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _channel_curve(row, roas_floor: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (curve_df, point_df, status) for one channel.

    curve_df has 100 (spend, revenue) points swept across 20%→180% of today's
    spend using a smooth diminishing-returns curve revenue = Rmax·(1 − e^(−k·spend)).
    The curve is anchored to Module 2/3 outputs: it passes through today's
    (spend, forecast_revenue) and its slope there equals the channel's marginal
    ROAS. Pure visualization — no model changes.
    """
    cur = float(row.current_spend)
    avg = float(row.forecast_roas)
    marg = float(row.marginal_roas)
    rev = float(row.forecast_revenue)

    b = float(np.clip(marg / avg, 0.35, 0.98)) if avg > 0 else 0.85
    spend_range = np.linspace(cur * 0.2, cur * 1.8, 100)
    x = _solve_sat_x(b)
    if x is None:
        revenue = np.clip(rev + marg * (spend_range - cur), 0, None)  # ~linear
    else:
        k = x / cur                       # rate constant; k·cur = x
        r_max = rev / (1 - np.exp(-x))     # revenue ceiling
        revenue = r_max * (1 - np.exp(-k * spend_range))

    curve_df = pd.DataFrame({"spend": spend_range, "revenue": revenue})
    point_df = pd.DataFrame({"spend": [cur], "revenue": [rev],
                             "label": [f"marginal {marg:.2f}x"]})
    status = "Headroom" if marg >= roas_floor else "Approaching Saturation"
    return curve_df, point_df, status


def _channel_chart(row, roas_floor: float):
    """Standalone spend-response chart for one channel (no faceting)."""
    curve_df, point_df, status = _channel_curve(row, roas_floor)
    color = "#2ecc71" if status == "Headroom" else "#e67e22"

    x_enc = alt.X(
        "spend:Q", title="Daily Spend ($)",
        scale=alt.Scale(domain=[float(curve_df["spend"].min()),
                                float(curve_df["spend"].max())], nice=False),
        axis=alt.Axis(format="$,.0f", grid=True, tickCount=5),
    )
    y_enc = alt.Y("revenue:Q", title="Expected Revenue ($)",
                  scale=alt.Scale(zero=True), axis=alt.Axis(format="$,.0f", grid=True))

    line = alt.Chart(curve_df).mark_line(color="#6366f1", strokeWidth=3).encode(x=x_enc, y=y_enc)
    rule = (alt.Chart(point_df).mark_rule(color="#f1c40f", strokeDash=[5, 4], size=1.5)
            .encode(x="spend:Q"))
    dot = (
        alt.Chart(point_df).mark_point(color="#f1c40f", size=170, filled=True)
        .encode(x="spend:Q", y="revenue:Q",
                tooltip=[alt.Tooltip("spend:Q", title="Today's spend", format="$,.0f"),
                         alt.Tooltip("revenue:Q", title="Expected revenue", format="$,.0f"),
                         alt.Tooltip("label:N", title="")])
    )
    txt = (alt.Chart(point_df).mark_text(align="center", dy=-14, color="#f1c40f", fontWeight="bold")
           .encode(x="spend:Q", y="revenue:Q", text="label:N"))
    return (
        (line + rule + dot + txt)
        .properties(height=300, width="container",
                    title=alt.TitleParams(f"{row.channel}  ·  {status}", color=color))
    )


@st.cache_data(show_spinner=False)
def run_pipeline(_cache_key: int = 0):
    """Ingest -> normalize -> reconcile. Returns every artifact for the UI."""
    meta_raw = ingestion.load_meta()
    google_raw = ingestion.load_google()
    catalog = ingestion.load_catalog()

    unified, impute_report = normalize.normalize_all(meta_raw, google_raw)
    reconciled, recon_report = reconcile.reconcile(unified, catalog)
    forecast_result = forecast.run_forecast(reconciled)

    return {
        "meta_raw": meta_raw,
        "google_raw": google_raw,
        "catalog": catalog,
        "unified": unified,
        "reconciled": reconciled,
        "impute_report": impute_report,
        "recon_report": recon_report,
        "forecast": forecast_result,
    }


# ----- Sidebar -----
with st.sidebar:
    st.header("True Classic")
    st.caption("Paid Media Intelligence & Optimization")
    if "cache_key" not in st.session_state:
        st.session_state.cache_key = 0
    if st.button("Regenerate mock data (new scenario)", width="stretch"):
        seed = ingestion.regenerate(base_jitter=0.30)
        st.session_state.cache_key += 1
        st.session_state.last_seed = seed
        st.cache_data.clear()
    if st.session_state.get("last_seed"):
        st.caption(f"Active scenario seed: `{st.session_state.last_seed}` (random ROAS bases)")

    st.divider()
    # Shared optimizer constraints — drive BOTH the Allocation and Approval tabs
    # so the recommendation a marketer approves is the one they were shown.
    st.subheader("Optimizer constraints")
    st.caption("Live guardrails — the recommendation recomputes as you move these.")
    roas_floor = st.slider("ROAS floor", 2.0, 8.0, 4.0, 0.5)
    max_change = st.slider("Max daily change %", 5, 50, 20, 5) / 100.0
    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05)

    st.divider()
    llm_on = explain.llm_available()
    st.caption(
        f"LLM explanation: {'🟢 live (API key found)' if llm_on else '🟠 rule-based fallback'}\n\n"
        "Modules 1–4 · Ingest → Forecast → Optimize → Approve. "
        "Optimization is deterministic; the LLM only narrates the numbers."
    )

data = run_pipeline(st.session_state.cache_key)

# ----- Shared Module 3 recommendation (computed once, used by M3 + M4 tabs) -----
constraints = optimize.Constraints(
    roas_floor=roas_floor, max_change_pct=max_change, confidence_threshold=conf_thr
)
channels = forecast.channel_summary(data["forecast"])
rec = optimize.optimize(channels, constraints)

st.title("Paid Media Intelligence")
st.caption("AI Solutions Architect case study · Meta + Google · ingest → forecast → optimize → approve")

tab_overview, tab_ingest, tab_forecast, tab_curves, tab_alloc, tab_approve = st.tabs(
    ["Overview", "Ingestion & Unification", "Forecast", "Efficiency Curves", "Allocation", "Approval"]
)


# =========================================================================
# Tab — Ingestion & Unification  (M1, the only implemented tab)
# =========================================================================
with tab_ingest:
    meta_raw = data["meta_raw"]
    google_raw = data["google_raw"]
    catalog = data["catalog"]
    unified = data["unified"]
    reconciled = data["reconciled"]
    impute_report = data["impute_report"]
    recon = data["recon_report"]

    # ----- Derived figures (no backend changes) -----
    meta_days = pd.to_datetime(meta_raw["date"]).dt.date.nunique()
    google_days = pd.to_datetime(google_raw["day"]).dt.date.nunique()
    matched_skus = recon["matched_skus"]
    unmatched_ids = len(recon["unmatched_keys"])
    imputed = impute_report["total_imputed_cells"]

    issues = []
    if unmatched_ids > 0:
        issues.append("unmatched SKU")
    if imputed > 0:
        issues.append("missing values")
    if meta_days != google_days:
        issues.append("date-range mismatch")
    num_issues = len(issues)

    st.markdown(
        "Fragmented ad data from **two platforms** is ingested, normalized to one "
        "schema, and reconciled to canonical SKUs — with every data problem surfaced, "
        "not hidden."
    )

    # ===== 1. Status banner =====
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_status_card("✅", "Meta CSV Loaded", ok=True), unsafe_allow_html=True)
    s2.markdown(_status_card("✅", "Google CSV Loaded", ok=True), unsafe_allow_html=True)
    s3.markdown(_status_card("✅", "Canonical Dataframe Created", ok=True), unsafe_allow_html=True)
    if unmatched_ids > 0:
        s4.markdown(
            _status_card("⚠️", f"{unmatched_ids} SKU Requires Review", ok=False),
            unsafe_allow_html=True,
        )
    else:
        s4.markdown(_status_card("✅", "All SKUs Reconciled", ok=True), unsafe_allow_html=True)

    st.write("")

    # ===== 2. KPI cards =====
    k = st.columns(6)
    kpis = [
        ("Meta Rows", f"{len(meta_raw):,}"),
        ("Google Rows", f"{len(google_raw):,}"),
        ("Unified Rows", f"{len(unified):,}"),
        ("Matched SKUs", f"{matched_skus}"),
        ("Unmatched SKUs", f"{unmatched_ids}"),
        ("Data Quality Issues", f"{num_issues}"),
    ]
    for col, (label, value) in zip(k, kpis):
        with col:
            with st.container(border=True):
                st.metric(label, value)

    st.write("")

    # ===== 3. Workflow strip =====
    st.markdown("##### Pipeline")
    st.markdown(
        """
        <div class="tc-flow">
          <div class="tc-step">Raw Exports<span class="sub">Meta + Google CSV</span></div>
          <div class="tc-arrow">→</div>
          <div class="tc-step">Normalize<span class="sub">One canonical schema</span></div>
          <div class="tc-arrow">→</div>
          <div class="tc-step">Reconcile<span class="sub">SKU crosswalk</span></div>
          <div class="tc-arrow">→</div>
          <div class="tc-step">Canonical DF<span class="sub">Unified fact table</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ===== 4. Data quality =====
    st.markdown("##### Data Quality")
    dq_left, dq_right = st.columns(2)
    with dq_left:
        st.success(
            f"**Schema normalization complete** — 2 platforms unified into one "
            f"canonical schema ({len(unified):,} rows)."
        )
        st.success(
            f"**SKU reconciliation** — {matched_skus} of "
            f"{matched_skus + unmatched_ids} product IDs mapped to canonical SKUs "
            f"({100 * recon['matched_rows'] / max(recon['total_rows'], 1):.0f}% of rows)."
        )
    with dq_right:
        if unmatched_ids > 0:
            keys = recon["unmatched_keys"]
            id_list = ", ".join(
                f"{r.platform} `{r.platform_product_id}`" for r in keys.itertuples()
            )
            st.warning(
                f"**{unmatched_ids} SKU requires review** — {id_list} has no crosswalk "
                f"entry (${recon['unmatched_spend']:,.0f} spend across "
                f"{recon['unmatched_rows']} rows). Flagged, not dropped."
            )
        if imputed > 0:
            cols = ", ".join(
                f"`{c}`" for c in impute_report["missing_cells_by_column"]
            )
            st.info(
                f"**Missing values handled** — imputed {imputed} cell(s) in {cols}. "
                f"Rule: {impute_report['rule']} (never invents revenue)."
            )
        if meta_days != google_days:
            st.info(
                f"**Inconsistent date ranges** — Meta covers {meta_days} days, Google "
                f"{google_days}. Windows must align before cross-channel comparison."
            )

    if unmatched_ids > 0:
        with st.expander("Inspect flagged keys"):
            st.dataframe(recon["unmatched_keys"], width="stretch", hide_index=True)
            st.caption(
                "Production behavior: route to a human review queue or attempt a "
                "fuzzy / embedding match against the catalog before exclusion."
            )

    st.divider()

    # ===== 5. Prototype transparency panel =====
    st.markdown("##### Prototype Transparency")
    with st.container(border=True):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**🟢 Real — live logic**")
            st.markdown(
                "- Schema normalization & funnel parsing\n"
                "- SKU identity resolution (crosswalk)\n"
                "- Data-quality detection & flagging"
            )
        with t2:
            st.markdown("**🟠 Mocked / stubbed**")
            st.markdown(
                "- Source CSVs — seeded synthetic data\n"
                "- Live platform APIs — file exports for now\n"
                "- Forecast & optimizer — later modules"
            )

    st.divider()

    # ===== 6. Raw data (collapsed by default) =====
    st.markdown("##### Inspect the data")
    st.caption("Collapsed by default — expand to audit the raw and unified tables.")

    with st.expander(f"View Raw Meta Export  ·  {len(meta_raw):,} rows"):
        st.caption("Meta-style schema: `spend`, `purchase_value`, `product_id` (FB-####).")
        st.dataframe(meta_raw, width="stretch", hide_index=True)

    with st.expander(f"View Raw Google Export  ·  {len(google_raw):,} rows"):
        st.caption("Google-style schema: `cost`, `conv_value`, `item_id` (G-####).")
        st.dataframe(google_raw, width="stretch", hide_index=True)

    with st.expander(f"View Unified Canonical Dataframe  ·  {len(reconciled):,} rows"):
        st.caption("Both schemas mapped to one fact table with `funnel_stage` and `match_status`.")
        st.dataframe(reconciled, width="stretch", hide_index=True)

    with st.expander(f"View SKU Catalog  ·  {len(catalog):,} SKUs"):
        st.caption("Source-of-truth crosswalk. Note the blank `google_item_id` for POLO-BLK.")
        st.dataframe(catalog, width="stretch", hide_index=True)


# =========================================================================
# Placeholder tabs — implemented in later modules
# =========================================================================
with tab_overview:
    st.markdown(
        "**Command center.** A single executive view composed from all four modules — "
        "ingestion, forecasting, optimization, and the approval workflow. Every number "
        "below is reused directly from the live pipeline; nothing is recomputed here."
    )

    # ----- Figures reused from the pipeline (no recalculation) -----
    horizon = data["forecast"]["horizon"]
    total_spend = rec["total_spend"]
    daily_revenue = float(channels["forecast_revenue"].sum())
    revenue_7d = daily_revenue * horizon
    forecast_roas = rec["blended_roas_before"]

    recon = data["recon_report"]
    unmatched_ids = len(recon["unmatched_keys"])
    imputed = data["impute_report"]["total_imputed_cells"]
    dq_issues = (1 if unmatched_ids else 0) + (1 if imputed else 0)

    # ===== KPI cards =====
    k = st.columns(4)
    with k[0]:
        with st.container(border=True):
            st.metric("Current total spend", f"${total_spend:,.0f}/day")
    with k[1]:
        with st.container(border=True):
            st.metric(f"Forecast revenue ({horizon}-day)", f"${revenue_7d:,.0f}")
    with k[2]:
        with st.container(border=True):
            st.metric("Forecast ROAS (blended)", f"{forecast_roas:.2f}x")
    with k[3]:
        with st.container(border=True):
            if rec["status"] == "reallocate":
                st.metric("Expected revenue gain", f"+${rec['expected_revenue_gain']:,.0f}/day")
            else:
                st.metric("Expected revenue gain", "$0/day")

    st.write("")

    # ===== Executive summary =====
    st.markdown("##### Executive summary")
    with st.container(border=True):
        if rec["status"] == "reallocate":
            st.markdown(
                f"<div class='tc-flow'>"
                f"<div class='tc-step'>{rec['donor']}"
                f"<span class='sub'>marginal {rec['donor_marginal_roas']:.2f}x</span></div>"
                f"<div class='tc-arrow'>${rec['budget_shift']:,.0f}/day →</div>"
                f"<div class='tc-step'>{rec['recipient']}"
                f"<span class='sub'>marginal {rec['recipient_marginal_roas']:.2f}x</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            e = st.columns(3)
            e[0].metric("Recommendation",
                        f"Move ${rec['budget_shift']:,.0f}/day",
                        f"{rec['donor']} → {rec['recipient']}", delta_color="off")
            e[1].metric("Blended ROAS",
                        f"{rec['blended_roas_before']:.2f}x → {rec['blended_roas_after']:.2f}x",
                        delta_color="off")
            e[2].metric("Confidence", str(rec["confidence"]),
                        f"score {rec['confidence_score']:.2f}", delta_color="off")
            st.caption(
                f"Reallocation bounded by the **{rec['binding_constraint']}**. "
                "Review, adjust, and approve in the **Approval** tab."
            )
        else:
            st.info(f"**No reallocation recommended.** {rec['reason']}")
            e = st.columns(2)
            e[0].metric("Blended ROAS", f"{rec['blended_roas_before']:.2f}x", delta_color="off")
            e[1].metric("Total spend", f"${total_spend:,.0f}/day")

    st.write("")

    # ===== Pipeline status strip =====
    st.markdown("##### Pipeline status")
    st.markdown(
        """
        <div class="tc-flow">
          <div class="tc-step">Ingest ✅<span class="sub">Meta + Google unified</span></div>
          <div class="tc-arrow">→</div>
          <div class="tc-step">Forecast ✅<span class="sub">7-day revenue & ROAS</span></div>
          <div class="tc-arrow">→</div>
          <div class="tc-step">Optimize ✅<span class="sub">marginal-ROAS reallocation</span></div>
          <div class="tc-arrow">→</div>
          <div class="tc-step">Approve ✅<span class="sub">human-in-the-loop</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ===== System status =====
    c1, c2 = st.columns(2)
    with c1:
        if dq_issues == 0:
            st.markdown(_status_card("✅", "Data Quality — all checks passed", ok=True),
                        unsafe_allow_html=True)
        else:
            bits = []
            if unmatched_ids:
                bits.append(f"{unmatched_ids} unmatched SKU")
            if imputed:
                bits.append(f"{imputed} imputed cell(s)")
            st.markdown(
                _status_card("⚠️", "Data Quality — " + ", ".join(bits) + " (surfaced)", ok=False),
                unsafe_allow_html=True,
            )
    with c2:
        if llm_on:
            st.markdown(
                _status_card("🟢", f"LLM — live ({explain.active_model()})", ok=True),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _status_card("🟠", "LLM — rule-based fallback (no API key)", ok=False),
                unsafe_allow_html=True,
            )

with tab_forecast:
    fc = data["forecast"]
    reconciled = data["reconciled"]

    st.markdown(
        "A **GradientBoostingRegressor** trained live per platform forecasts the next "
        "**7 days** of revenue and ROAS, with a P10–P90 confidence band. These outputs "
        "feed the Module 3 optimizer."
    )

    # ----- Selectors -----
    sku_options = ["All"] + sorted(
        reconciled.loc[reconciled["match_status"] == "matched", "sku"].dropna().unique().tolist()
    )
    sc1, sc2 = st.columns(2)
    sel_platform = sc1.selectbox("Platform", ["All", "Meta", "Google"], index=0)
    sel_sku = sc2.selectbox("SKU (filter only — not a separate model)", sku_options, index=0)

    hist, fut = forecast.timeline(fc, sel_platform, sel_sku)
    summary = forecast.summarize(fc, sel_platform, sel_sku)
    mape_sel = forecast.mape_for(fc, sel_platform)

    rev_7d = float(fut["revenue"].sum())
    spend_7d = float(fut["spend"].sum())
    roas_7d = rev_7d / spend_7d if spend_7d else float("nan")
    prior_7d_rev = float(hist.sort_values("date")["revenue"].tail(7).sum())
    rev_delta = rev_7d - prior_7d_rev

    # ----- KPI cards -----
    k = st.columns(4)
    with k[0]:
        with st.container(border=True):
            st.metric("Forecast Revenue (7d)", f"${rev_7d:,.0f}", f"${rev_delta:,.0f} vs prior 7d")
    with k[1]:
        with st.container(border=True):
            roas_str = f"{roas_7d:.2f}x" if roas_7d == roas_7d else "—"
            st.metric("Forecast ROAS", roas_str, "vs 4.0x target", delta_color="off")
    with k[2]:
        with st.container(border=True):
            st.metric("Forecast Spend (7d)", f"${spend_7d:,.0f}")
    with k[3]:
        with st.container(border=True):
            acc = f"{(1 - mape_sel) * 100:.0f}%" if mape_sel == mape_sel else "—"
            mape_str = f"MAPE {mape_sel * 100:.1f}%" if mape_sel == mape_sel else ""
            st.metric("Model Accuracy", acc, mape_str, delta_color="off")

    # ----- History vs forecast chart with confidence band -----
    line_df = pd.concat(
        [hist[["date", "revenue", "type"]], fut[["date", "revenue", "type"]]],
        ignore_index=True,
    )
    color_scale = alt.Scale(domain=["History", "Forecast"], range=["#9aa0a6", "#6366f1"])
    band = (
        alt.Chart(fut)
        .mark_area(opacity=0.18, color="#6366f1")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("revenue_low:Q", title="Revenue ($)"),
            y2="revenue_high:Q",
        )
    )
    line = (
        alt.Chart(line_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("revenue:Q", title="Revenue ($)"),
            color=alt.Color("type:N", scale=color_scale, legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash(
                "type:N",
                scale=alt.Scale(domain=["History", "Forecast"], range=[[1, 0], [6, 4]]),
                legend=None,
            ),
        )
    )
    today_rule = (
        alt.Chart(pd.DataFrame({"date": [hist["date"].max()]}))
        .mark_rule(color="#f1c40f", strokeDash=[3, 3])
        .encode(x="date:T")
    )
    chart = (band + line + today_rule).properties(height=320, width="container")
    st.altair_chart(chart, theme="streamlit")
    st.caption(
        "Solid grey = actuals · dashed indigo = forecast · shaded band = P10–P90 "
        "confidence · yellow line = today."
    )

    # ----- Confidence indicator -----
    if not summary.empty:
        label = (
            "High" if mape_sel == mape_sel and mape_sel < 0.15
            else "Medium" if mape_sel == mape_sel and mape_sel < 0.30
            else "Low" if mape_sel == mape_sel else "Unknown"
        )
        msg = (
            f"**Forecast confidence: {label}** — backtest MAPE "
            f"{mape_sel * 100:.1f}% on held-out days. Band shows the P10–P90 range."
        )
        if label == "High":
            st.success(msg)
        elif label == "Medium":
            st.info(msg)
        else:
            st.warning(msg)

    # ----- Model summary -----
    st.markdown("##### Model summary")
    with st.container(border=True):
        ms = st.columns(4)
        ms[0].markdown("**Model**\n\nGradientBoostingRegressor\n(scikit-learn)")
        ms[1].markdown("**Features**\n\nspend · day-of-week · time-index")
        ms[2].markdown("**Horizon**\n\n7 days")
        ms[3].markdown("**Confidence**\n\nQuantile P10–P90\n+ backtest MAPE")

    # ----- Module-3 summary + forecast table (expanders) -----
    with st.expander("Per-platform summary (Module 3 input)"):
        st.caption(
            "`current_spend` & `forecast_revenue` are daily averages over the horizon. "
            "`marginal_roas` = return on the next dollar (probed at +10% spend) — the "
            "signal the optimizer ranks on."
        )
        st.dataframe(summary, width="stretch", hide_index=True)

    with st.expander("7-day forecast detail"):
        detail = fut.copy()
        detail["roas"] = (detail["revenue"] / detail["spend"]).round(2)
        detail = detail.rename(
            columns={
                "spend": "forecast_spend",
                "revenue": "forecast_revenue",
                "revenue_low": "revenue_p10",
                "revenue_high": "revenue_p90",
            }
        )
        cols = ["date", "forecast_spend", "forecast_revenue", "roas", "revenue_p10", "revenue_p90"]
        st.dataframe(
            detail[cols].round({"forecast_spend": 0, "forecast_revenue": 0,
                                "revenue_p10": 0, "revenue_p90": 0}),
            width="stretch",
            hide_index=True,
        )

with tab_curves:
    st.markdown(
        "**Why the optimizer thinks in *marginal* ROAS.** Each curve shows how a channel's "
        "revenue responds as daily spend rises. Revenue always keeps growing — but the slope "
        "flattens, so each extra dollar returns less. The **slope at today's spend** (yellow "
        "marker + dashed line) *is* the channel's marginal ROAS: the return on the next dollar."
    )

    ec = channels.copy()
    total_spend = float(ec["current_spend"].sum())
    total_rev = float(ec["forecast_revenue"].sum())
    blended = total_rev / total_spend if total_spend > 0 else float("nan")
    hi = ec.loc[ec["marginal_roas"].idxmax()]
    lo = ec.loc[ec["marginal_roas"].idxmin()]

    k = st.columns(4)
    k[0].metric("Highest marginal ROAS", f"{hi['marginal_roas']:.2f}x", hi["channel"], delta_color="off")
    k[1].metric("Lowest marginal ROAS", f"{lo['marginal_roas']:.2f}x", lo["channel"], delta_color="off")
    k[2].metric("Current total spend", f"${total_spend:,.0f}/day")
    k[3].metric("Current blended ROAS", f"{blended:.2f}x")

    st.caption(
        f"🟢 **Headroom** = marginal ROAS at/above the {roas_floor:.1f}x floor (a good home for "
        "more budget)  ·  🟠 **Approaching Saturation** = marginal ROAS below the floor "
        "(the next dollar underperforms). Curves span 20–180% of today's spend."
    )

    ch_rows = list(ec.itertuples())
    grid = st.columns(2)
    for i, ch_row in enumerate(ch_rows):
        with grid[i % 2]:
            st.altair_chart(_channel_chart(ch_row, roas_floor), theme="streamlit")

    st.info(
        "As spend increases, revenue still grows, but each extra dollar returns less. "
        "The optimizer uses **marginal ROAS** — the return on the next dollar — to decide "
        "where budget should move. That's why a channel with a high *average* ROAS can still "
        "be a donor: its *next-dollar* return has already flattened."
    )

    with st.expander("View channel summary (raw numbers behind the curves)"):
        st.dataframe(
            ec[["channel", "current_spend", "forecast_revenue", "forecast_roas",
                "marginal_roas", "confidence_label"]]
            .rename(columns={
                "channel": "Channel", "current_spend": "Current spend ($/day)",
                "forecast_revenue": "Forecast revenue ($/day)", "forecast_roas": "Average ROAS",
                "marginal_roas": "Marginal ROAS", "confidence_label": "Confidence",
            }),
            width="stretch", hide_index=True,
        )

with tab_alloc:
    st.markdown(
        "A **deterministic one-shot optimizer** ranks channels by **marginal ROAS** "
        "(return on the *next* dollar) and recommends a single budget-neutral move from "
        "the lowest-marginal channel below the floor to the highest-marginal channel with "
        "headroom — bounded by business constraints. Adjust the guardrails in the **sidebar**."
    )

    channel_df = channels.copy()

    # ===== The core insight: average vs marginal ROAS (always shown) =====
    st.markdown("##### Average vs marginal ROAS")
    st.altair_chart(_marginal_vs_avg_chart(channel_df, roas_floor), theme="streamlit")
    st.caption(
        "Grey = average ROAS (the past scorecard) · indigo = marginal ROAS (return on the "
        "next dollar) · yellow line = your ROAS floor. The optimizer cuts channels whose "
        "**marginal** ROAS sits below the floor — even when their **average** still looks great."
    )

    below = channel_df[channel_df["marginal_roas"] < roas_floor]
    if not below.empty:
        items = " · ".join(f"{r.channel} ({r.marginal_roas:.2f}x)" for r in below.itertuples())
        st.warning(f"**Diminishing returns flagged** — below the {roas_floor:.1f}x floor: {items}.")

    st.divider()

    # ===== Optimizer ranking (transparency — shown for every run) =====
    st.markdown("##### Optimizer ranking")
    st.caption(
        "Channels ranked by marginal ROAS. Donor = lowest marginal below the floor; "
        "recipient = highest marginal above the floor that clears the confidence "
        "threshold. Channel names are derived from the data — never hardcoded."
    )
    rank_df = pd.DataFrame(rec["ranking"])
    sel_icon = {"donor": "🔻 donor", "recipient": "🔺 recipient"}
    rank_view = pd.DataFrame({
        "Channel": rank_df["channel"],
        "Current Spend": rank_df["current_spend"].map(lambda v: f"${v:,.0f}/day"),
        "Marginal ROAS": rank_df["marginal_roas"].map(lambda v: f"{v:.2f}x"),
        "Confidence": rank_df.apply(
            lambda r: f"{r['confidence']:.2f} ({r['confidence_label']})"
            if pd.notna(r["confidence"]) else "—", axis=1),
        "Donor?": rank_df["eligible_donor"].map({True: "✓", False: "—"}),
        "Recipient?": rank_df["eligible_recipient"].map({True: "✓", False: "—"}),
        "Selected": rank_df["selected"].map(lambda v: sel_icon.get(v, "")),
    })
    st.dataframe(rank_view, width="stretch", hide_index=True)

    st.divider()

    if rec["status"] != "reallocate":
        st.info(f"**No reallocation recommended.** {rec['reason']}")
        with st.expander("Recommendation object (Module 4 input)"):
            st.json(rec)
    else:
        # ===== Decision banner =====
        st.markdown(
            f"<div class='tc-flow'>"
            f"<div class='tc-step'>{rec['donor']}"
            f"<span class='sub'>marginal {rec['donor_marginal_roas']:.2f}x · below floor</span></div>"
            f"<div class='tc-arrow'>${rec['budget_shift']:,.0f}/day →</div>"
            f"<div class='tc-step'>{rec['recipient']}"
            f"<span class='sub'>marginal {rec['recipient_marginal_roas']:.2f}x · headroom</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ===== Executive summary =====
        st.markdown("##### Executive summary")
        e = st.columns([1.6, 1, 1.1, 1])
        with e[0]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:1.5rem;font-weight:700'>Move ${rec['budget_shift']:,.0f}/day</div>"
                    f"<div style='color:#9aa0a6;margin-top:6px'>{rec['donor']} "
                    f"<span style='color:#f1c40f'>→</span> {rec['recipient']}</div>",
                    unsafe_allow_html=True,
                )
        with e[1]:
            with st.container(border=True):
                st.metric("Expected Revenue Gain", f"+${rec['expected_revenue_gain']:,.0f}/day")
        with e[2]:
            with st.container(border=True):
                st.metric(
                    "Blended ROAS",
                    f"{rec['blended_roas_before']:.2f}x → {rec['blended_roas_after']:.2f}x",
                    f"+{rec['blended_roas_after'] - rec['blended_roas_before']:.2f}x",
                )
        with e[3]:
            with st.container(border=True):
                st.metric("Confidence", rec["confidence"], f"{rec['confidence_score']:.2f}", delta_color="off")

        st.divider()

        alloc_df = pd.DataFrame(rec["allocation"])

        # ===== Current vs Recommended (grouped horizontal bars) =====
        st.markdown("##### Current vs recommended allocation")
        long = pd.concat([
            alloc_df[["channel", "current_spend"]].rename(columns={"current_spend": "spend"}).assign(kind="Current"),
            alloc_df[["channel", "recommended_spend"]].rename(columns={"recommended_spend": "spend"}).assign(kind="Recommended"),
        ], ignore_index=True)
        bars = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                y=alt.Y("channel:N", title=None, sort="-x"),
                x=alt.X("spend:Q", title="Daily spend ($)"),
                yOffset=alt.YOffset("kind:N"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["Current", "Recommended"], range=["#9aa0a6", "#6366f1"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["channel", "kind", alt.Tooltip("spend:Q", format="$,.0f")],
            )
            .properties(height=240, width="container")
        )
        st.altair_chart(bars, theme="streamlit")

        with st.expander("Per-channel allocation (with marginal ROAS)"):
            show = alloc_df[["channel", "current_spend", "recommended_spend", "delta", "marginal_roas"]]
            st.dataframe(show, width="stretch", hide_index=True)

        # ===== Budget movement (only changed channels) =====
        st.markdown("##### Budget movement")
        movers = alloc_df[alloc_df["delta"].abs() > 1e-6]
        mcols = st.columns(max(len(movers), 1))
        for col, (_, m) in zip(mcols, movers.iterrows()):
            arrow = "▲" if m["delta"] > 0 else "▼"
            color = "#2ecc71" if m["delta"] > 0 else "#f1c40f"
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"**{m['channel']}**<br>"
                        f"<span style='color:#9aa0a6'>${m['current_spend']:,.0f}/day</span> → "
                        f"<b>${m['recommended_spend']:,.0f}/day</b><br>"
                        f"<span style='color:{color}'>{arrow} ${abs(m['delta']):,.0f}/day</span>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ===== Constraints triggered =====
        st.markdown("##### Constraints triggered")
        icon = {"satisfied": "✓", "binding": "•", "triggered": "•"}
        ccols = st.columns(len(rec["triggered_constraints"]))
        for col, ct in zip(ccols, rec["triggered_constraints"]):
            with col:
                with st.container(border=True):
                    st.markdown(f"{icon.get(ct['status'], '•')} **{ct['name']}**")
                    st.caption(ct["detail"])

        st.divider()

        # ===== Recommendation details =====
        st.markdown("##### Recommendation details")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(
                f"- **Donor:** {rec['donor']} — marginal ROAS **{rec['donor_marginal_roas']:.2f}x** "
                f"(below {roas_floor:.1f}x floor)\n"
                f"- **Recipient:** {rec['recipient']} — marginal ROAS **{rec['recipient_marginal_roas']:.2f}x**\n"
                f"- **Budget shift:** ${rec['budget_shift']:,.0f}/day (bounded by {rec['binding_constraint']})"
            )
        with d2:
            st.markdown(
                f"- **Expected revenue gain:** +${rec['expected_revenue_gain']:,.0f}/day\n"
                f"- **Blended ROAS:** {rec['blended_roas_before']:.2f}x → "
                f"{rec['blended_roas_after']:.2f}x\n"
                f"- **Total spend:** ${rec['total_spend']:,.0f}/day (unchanged — budget neutral)"
            )
        st.caption(rec["reason"])
        st.info("Take this recommendation to the **Approval** tab to review, adjust, and execute.")

        # ===== Structured recommendation object (feeds Module 4) =====
        with st.expander("Recommendation object (Module 4 input)"):
            st.json(rec)


# =========================================================================
# Tab — Approval  (M4: human-in-the-loop + LLM explanation + stubbed exec)
# =========================================================================
def _render_audit_log() -> None:
    """The accountability trail — shown at the bottom of the Approval tab."""
    st.markdown("##### Audit log")
    st.caption(
        "Every decision is recorded: who decided, what the optimizer recommended, "
        "what was actually approved, and why."
    )
    log_df = audit_log.load_log()
    if log_df.empty:
        st.caption("No decisions logged yet. Approve or reject above to create the trail.")
        return

    view = pd.DataFrame({
        "Timestamp": log_df["timestamp"],
        "User": log_df["approver"],
        "Recommendation": log_df.apply(
            lambda r: f"{r['donor']} → {r['recipient']}  (${r['optimizer_shift']:,.0f}/day)", axis=1),
        "Final approved": log_df["budget_shift"].map(lambda v: f"${v:,.0f}/day"),
        "Decision": log_df["action"].map(_DECISION_LABEL).fillna(log_df["action"]),
        "Comment": log_df["note"],
    })
    st.dataframe(view, width="stretch", hide_index=True)
    lc1, lc2 = st.columns([1, 4])
    with lc1:
        if st.button("Clear log"):
            audit_log.clear_log()
            st.rerun()
    with lc2:
        st.caption(f"`data/audit_log.csv` · {len(log_df)} decision(s) recorded.")


_DECISION_LABEL = {
    "approved": "✅ Approved",
    "rejected": "❌ Rejected",
    "modified_approved": "✏️ Modified then Approved",
}


with tab_approve:
    st.markdown(
        "**Human-in-the-loop governance.** The optimizer proposes; a paid-media manager "
        "reviews the numbers and a grounded explanation, optionally adjusts the amount, and "
        "makes the final call **with a required comment**. The decision is then **frozen** and "
        "written to an audit trail. *Execution is simulated* — no ad-platform API is called."
    )

    if rec["status"] != "reallocate":
        st.info(
            f"**Nothing to approve.** {rec['reason']} "
            "Adjust the constraints in the sidebar to surface a move."
        )
        st.divider()
        _render_audit_log()
    else:
        # Identity of THIS recommendation. If it changes (constraints/data), any prior
        # frozen decision is stale and the workflow resets to an open state.
        dec_key = (
            rec["donor"], rec["recipient"],
            round(float(rec["budget_shift"]), 2),
            roas_floor, max_change, conf_thr,
        )
        frozen = st.session_state.get("decision")
        if frozen and frozen.get("key") != dec_key:
            frozen = None
            st.session_state.pop("decision", None)

        optimizer_shift = float(rec["budget_shift"])
        spread = rec["recipient_marginal_roas"] - rec["donor_marginal_roas"]

        # ===== 1 · Recommendation summary (the optimizer's original proposal) =====
        st.markdown("##### 1 · Recommendation")
        s = st.columns(5)
        s[0].metric("Move", f"${optimizer_shift:,.0f}/day")
        s[1].metric("From", rec["donor"], f"{rec['donor_marginal_roas']:.2f}x marginal", delta_color="off")
        s[2].metric("To", rec["recipient"], f"{rec['recipient_marginal_roas']:.2f}x marginal", delta_color="off")
        s[3].metric("Expected revenue gain", f"+${rec['expected_revenue_gain']:,.0f}/day")
        s[4].metric(
            "Expected ROAS", f"{rec['blended_roas_before']:.2f}x → {rec['blended_roas_after']:.2f}x",
            delta_color="off",
        )

        # ===== 2 · Grounded explanation (LLM narrates the optimizer's numbers) =====
        st.markdown("##### 2 · Why the optimizer recommends this")
        exp_col, ctrl_col = st.columns([3, 1])
        with ctrl_col:
            regen = st.button("Regenerate", width="stretch")
        exp_key = (rec["donor"], rec["recipient"], round(optimizer_shift, 2),
                   roas_floor, max_change, conf_thr)
        if regen or st.session_state.get("exp_key") != exp_key:
            with st.spinner("Generating grounded explanation…"):
                st.session_state.explanation = explain.explain(rec)
                st.session_state.exp_key = exp_key
        explanation = st.session_state.get("explanation") or explain.explain(rec)
        sections = explain.structured(rec)
        with exp_col:
            with st.container(border=True):
                st.caption(
                    "Executive summary generated from the **current** optimizer output — "
                    "every figure is grounded in the numbers above; nothing is hardcoded."
                )
                for sec in sections:
                    st.markdown(f"**{sec['title']}**")
                    st.markdown(sec["body"])
                if explanation["source"] == "llm":
                    with st.expander(f"AI narrative ({explanation['model']})"):
                        st.write(explanation["text"])
        with st.expander("Ask a question about this recommendation (grounded Q&A)", expanded=True):
            llm_on = explain.llm_available()
            if llm_on:
                st.caption(
                    f"🟢 Connected to OpenAI (`{explain.active_model()}`). Answers are grounded "
                    "strictly in the optimizer output above — the model never invents figures "
                    "or makes budget decisions."
                )
            else:
                st.warning(
                    "Grounded Q&A needs an API key. Add `OPENAI_API_KEY=...` to a `.env` file "
                    "in the project root, then restart the app."
                )

            # Suggested questions (set the input, answered on next run).
            suggestions = [
                "Why did you cut Meta if its average ROAS is higher?",
                "Why wasn't Google Prospecting selected?",
                f"Why only move ${rec['budget_shift']:,.0f}/day?",
                "Which constraint limited the recommendation?",
            ]
            sc = st.columns(2)
            for i, sug in enumerate(suggestions):
                if sc[i % 2].button(sug, key=f"sug_{i}", width="stretch", disabled=not llm_on):
                    st.session_state.qa_input = sug

            q = st.text_input(
                "Your question",
                key="qa_input",
                placeholder="Ask about the donor, recipient, amount, ROAS, or constraints…",
            )
            ask = st.button("Ask", type="primary", disabled=not llm_on)

            if ask and q.strip():
                with st.spinner("Thinking — grounded in the optimizer's numbers…"):
                    st.session_state.qa = {"q": q.strip(), "a": explain.answer_question(rec, q.strip())}

            qa = st.session_state.get("qa")
            if qa:
                with st.container(border=True):
                    st.markdown(f"**You:** {qa['q']}")
                a = qa["a"]
                badge = (f"OpenAI · {a['model']}" if a["source"] == "llm"
                         else "Grounded fallback")
                with st.container(border=True):
                    st.markdown(
                        f"**Assistant**&nbsp;&nbsp;"
                        f"<span style='opacity:0.55;font-size:0.8rem'>{badge}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(a["text"])

        st.divider()

        # ===== 3 · Decision (open) OR frozen status =====
        if not frozen:
            st.markdown("##### 3 · Review & decide")

            modify = st.toggle(
                "Modify budget (keep the same donor → recipient, change the amount)",
                value=False,
            )
            if modify:
                approved_shift = st.number_input(
                    f"Manager-approved amount:  {rec['donor']} → {rec['recipient']} ($/day)",
                    min_value=0.0, max_value=round(optimizer_shift, 2),
                    value=round(optimizer_shift, 2),
                    step=max(1.0, round(optimizer_shift / 20, 0)),
                    help="The optimizer's amount is the constraint-bounded maximum. "
                         "You can only move the same or less — never more.",
                )
            else:
                approved_shift = optimizer_shift

            is_modified = abs(approved_shift - optimizer_shift) > 1e-6

            # Re-derive impact deterministically from the SAME marginal-ROAS math.
            adj_gain = spread * approved_shift
            adj_blended_after = (
                rec["blended_roas_before"] + adj_gain / rec["total_spend"]
                if rec["total_spend"] else rec["blended_roas_before"]
            )

            if is_modified:
                m1, m2 = st.columns(2)
                m1.metric("Optimizer recommended", f"Move ${optimizer_shift:,.0f}/day")
                m2.metric("Manager approving", f"Move ${approved_shift:,.0f}/day",
                          f"{approved_shift - optimizer_shift:,.0f}/day", delta_color="off")
                st.caption(
                    f"Adjusted impact: +${adj_gain:,.0f}/day · blended ROAS "
                    f"{rec['blended_roas_before']:.2f}x → {adj_blended_after:.2f}x"
                )

            with st.container(border=True):
                approver = st.text_input("Approver", value="demo_user")
                comment = st.text_area(
                    "Approval comment (required)",
                    placeholder='e.g. "Approved after reviewing forecast." or '
                                '"Reject due to upcoming promotion."',
                )
                comment_ok = bool(comment.strip())
                if not comment_ok:
                    st.caption("⚠️ A comment is required before you can approve or reject.")
                b1, b2, _ = st.columns([1.2, 1, 2])
                approve_clicked = b1.button(
                    "Approve", type="primary", width="stretch", disabled=not comment_ok)
                reject_clicked = b2.button(
                    "Reject", width="stretch", disabled=not comment_ok)

            if approve_clicked or reject_clicked:
                action = ("rejected" if reject_clicked
                          else "modified_approved" if is_modified else "approved")
                row = audit_log.append_decision({
                    "action": action,
                    "approver": approver,
                    "donor": rec["donor"],
                    "recipient": rec["recipient"],
                    "optimizer_shift": optimizer_shift,
                    "budget_shift": approved_shift,
                    "modified": is_modified,
                    "expected_revenue_gain": round(adj_gain, 2),
                    "blended_roas_before": rec["blended_roas_before"],
                    "blended_roas_after": round(adj_blended_after, 2),
                    "confidence": rec["confidence"],
                    "binding_constraint": rec["binding_constraint"],
                    "note": comment.strip(),
                    "constraints": rec["constraints_used"],
                    "explanation_source": explanation["source"],
                })
                st.session_state.decision = {"key": dec_key, "row": row}
                st.rerun()
        else:
            # ---- Frozen decision ----
            row = frozen["row"]
            action = row["action"]
            st.markdown("##### 3 · Decision (frozen)")
            status = _DECISION_LABEL.get(action, action)
            if action == "rejected":
                st.error(f"### {status}")
            else:
                st.success(f"### {status}")

            d = st.columns([1, 1, 1, 1])
            d[0].metric("Decided by", row["approver"])
            d[1].metric("Timestamp", row["timestamp"].split("T")[-1])
            d[2].metric("Optimizer recommended", f"${row['optimizer_shift']:,.0f}/day")
            d[3].metric("Final decision", f"${row['budget_shift']:,.0f}/day"
                        if action != "rejected" else "$0/day (no change)")

            if action == "modified_approved":
                st.info(
                    f"**Modified.**  Optimizer recommended **Move ${row['optimizer_shift']:,.0f}/day**  ·  "
                    f"Manager approved **Move ${row['budget_shift']:,.0f}/day** "
                    f"({row['donor']} → {row['recipient']})."
                )

            st.markdown(f"**Comment:** _{row['note']}_")

            if action in ("approved", "modified_approved"):
                st.info(
                    "**Execution simulated.** In production this would submit budget changes "
                    "through the advertising platform APIs."
                )
            else:
                st.caption("No budget change was submitted.")

            if st.button("Start a new review"):
                st.session_state.pop("decision", None)
                st.rerun()

        st.divider()
        _render_audit_log()
