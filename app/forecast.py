"""Module 2 — AI Performance Forecasting Engine.

Consumes the canonical (reconciled) dataframe from Module 1 and produces a
7-day per-platform forecast of revenue / ROAS with a confidence band, plus the
marginal-ROAS table that Module 3 (Budget Optimization) will consume later.

Design reference: docs/03_forecasting_design.md
Model: scikit-learn GradientBoostingRegressor, trained per platform on
features [spend, day-of-week, time-index]. Deterministic (fixed random_state).
All ML lives here — the Streamlit layer only calls these functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = ["spend", "dow", "t_index", "funnel"]

HORIZON_DEFAULT = 7        # forecast 7 days ahead
SPEND_BUMP = 0.10          # +10% spend probe to estimate marginal ROAS
LOOKBACK_SPEND_DAYS = 7    # assumed future spend = mean of last N days per segment
HOLDOUT_DAYS = 3           # tail days held out for backtest MAPE

# Modest, deterministic settings — fast and overfit-resistant on small data.
GBR_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)

# Stable segment identity used for per-segment future spend + SKU filtering.
SEGMENT_KEYS = ["platform", "sku", "funnel_stage", "campaign", "platform_product_id"]


# ----------------------------------------------------------------------------
# Feature prep & backtest
# ----------------------------------------------------------------------------
def _prepare(reconciled: pd.DataFrame) -> pd.DataFrame:
    """Matched rows only, with engineered calendar features."""
    df = reconciled[reconciled["match_status"] == "matched"].copy()
    df = df.dropna(subset=["revenue", "spend"])
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["t_index"] = (df["date"] - df["date"].min()).dt.days
    # Segment flag so the per-platform model learns prospecting vs retargeting
    # response separately (their spend<->ROAS relationships are inverted).
    df["funnel"] = (df["funnel_stage"].astype(str) == "retargeting").astype(int)
    return df


def _fit(X: pd.DataFrame, y: pd.Series, loss: str = "squared_error", alpha: float | None = None):
    params = dict(GBR_PARAMS)
    if loss == "quantile":
        return GradientBoostingRegressor(loss="quantile", alpha=alpha, **params).fit(X, y)
    return GradientBoostingRegressor(loss=loss, **params).fit(X, y)


def _backtest_mape(platform_df: pd.DataFrame) -> float:
    """Time-based holdout MAPE: train on early dates, score the last few days."""
    dates = sorted(platform_df["date"].unique())
    if len(dates) <= HOLDOUT_DAYS + 2:
        return float("nan")
    cutoff = dates[-HOLDOUT_DAYS]
    train = platform_df[platform_df["date"] < cutoff]
    test = platform_df[platform_df["date"] >= cutoff]
    model = _fit(train[FEATURES], train["revenue"])
    pred = model.predict(test[FEATURES])
    y = test["revenue"].to_numpy()
    mask = y > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])))


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def run_forecast(reconciled: pd.DataFrame, horizon: int = HORIZON_DEFAULT) -> dict:
    """Train per-platform models and produce the forecast artifacts.

    Returns a dict with:
      - history:          matched actuals (segment-day grain) for charting/filtering
      - segment_forecast: per segment x future-day predictions (+ bumped-spend probe)
      - model_info:       per-platform {mape, n_train}
      - horizon, future_dates
    """
    d = _prepare(reconciled)
    min_date = d["date"].min()
    max_date = d["date"].max()
    future_dates = [max_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    recent_cut = max_date - pd.Timedelta(days=LOOKBACK_SPEND_DAYS - 1)

    seg_rows: list[dict] = []
    model_info: dict[str, dict] = {}

    for platform, dp in d.groupby("platform"):
        point = _fit(dp[FEATURES], dp["revenue"])
        lo = _fit(dp[FEATURES], dp["revenue"], loss="quantile", alpha=0.1)
        hi = _fit(dp[FEATURES], dp["revenue"], loss="quantile", alpha=0.9)
        model_info[platform] = {"mape": _backtest_mape(dp), "n_train": int(len(dp))}

        for keys, seg in dp.groupby(SEGMENT_KEYS):
            recent = seg[seg["date"] >= recent_cut]
            assumed = float((recent if len(recent) else seg)["spend"].mean())
            _, sku, funnel, campaign, ppid = keys
            funnel_flag = int(str(funnel) == "retargeting")
            for fd in future_dates:
                base = pd.DataFrame([{"spend": assumed, "dow": fd.dayofweek,
                                      "t_index": (fd - min_date).days, "funnel": funnel_flag}])
                bump = pd.DataFrame([{"spend": assumed * (1 + SPEND_BUMP), "dow": fd.dayofweek,
                                      "t_index": (fd - min_date).days, "funnel": funnel_flag}])
                pr = float(point.predict(base)[0])
                pl = float(lo.predict(base)[0])
                ph = float(hi.predict(base)[0])
                prb = float(point.predict(bump)[0])
                seg_rows.append({
                    "platform": platform, "sku": sku, "funnel_stage": funnel,
                    "campaign": campaign, "platform_product_id": ppid, "date": fd,
                    "assumed_spend": assumed,
                    "pred_revenue": max(pr, 0.0),
                    "pred_low": max(min(pl, pr), 0.0),   # keep band ordered
                    "pred_high": max(ph, pr),
                    "pred_revenue_bump": max(prb, 0.0),
                })

    history = d[["date", "platform", "sku", "funnel_stage", "spend", "revenue"]].copy()
    return {
        "history": history,
        "segment_forecast": pd.DataFrame(seg_rows),
        "model_info": model_info,
        "horizon": horizon,
        "future_dates": future_dates,
    }


# ----------------------------------------------------------------------------
# UI-facing aggregation helpers (filters: platform + optional SKU)
# ----------------------------------------------------------------------------
def _filter(df: pd.DataFrame, platform: str, sku: str) -> pd.DataFrame:
    if platform != "All":
        df = df[df["platform"] == platform]
    if sku != "All":
        df = df[df["sku"] == sku]
    return df


def timeline(result: dict, platform: str = "All", sku: str = "All") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (history_daily, forecast_daily) aggregated for the chart."""
    h = _filter(result["history"], platform, sku)
    f = _filter(result["segment_forecast"], platform, sku)

    hist = h.groupby("date", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
    hist["type"] = "History"

    fut = f.groupby("date", as_index=False).agg(
        spend=("assumed_spend", "sum"),
        revenue=("pred_revenue", "sum"),
        revenue_low=("pred_low", "sum"),
        revenue_high=("pred_high", "sum"),
    )
    fut["type"] = "Forecast"
    return hist, fut


def _confidence_label(score: float) -> str:
    if score != score:  # NaN
        return "Unknown"
    if score >= 0.85:
        return "High"
    if score >= 0.70:
        return "Medium"
    return "Low"


def summarize(result: dict, platform: str = "All", sku: str = "All") -> pd.DataFrame:
    """Per-platform summary — the contract Module 3 will consume.

    Columns: platform, current_spend, forecast_revenue, forecast_roas,
             confidence, confidence_label, marginal_roas.
    `current_spend` and `forecast_revenue` are daily averages over the horizon.
    """
    f = _filter(result["segment_forecast"], platform, sku)
    horizon = result["horizon"]
    rows = []
    for plat, g in f.groupby("platform"):
        spend_total = float(g["assumed_spend"].sum())
        rev_total = float(g["pred_revenue"].sum())
        bump_total = float(g["pred_revenue_bump"].sum())
        marginal = (bump_total - rev_total) / (spend_total * SPEND_BUMP) if spend_total > 0 else float("nan")
        roas = rev_total / spend_total if spend_total > 0 else float("nan")
        mape = result["model_info"].get(plat, {}).get("mape", float("nan"))
        conf = float(np.clip(1 - mape, 0, 1)) if mape == mape else float("nan")
        rows.append({
            "platform": plat,
            "current_spend": round(spend_total / horizon, 2),
            "forecast_revenue": round(rev_total / horizon, 2),
            "forecast_roas": round(roas, 2),
            "confidence": round(conf, 2) if conf == conf else float("nan"),
            "confidence_label": _confidence_label(conf),
            "marginal_roas": round(marginal, 2),
        })
    return pd.DataFrame(rows)


DEFAULT_ELASTICITY = 0.85  # fallback when a channel lacks spend variation to fit


def _fit_elasticity(spend: pd.Series, revenue: pd.Series) -> float:
    """Estimate saturation elasticity b in revenue = a * spend^b via log-log OLS.

    b is the ratio of marginal ROAS to average ROAS. Clipped to a sane band and
    falls back to a default when the data is too thin / flat to fit reliably.
    """
    s = pd.to_numeric(spend, errors="coerce")
    r = pd.to_numeric(revenue, errors="coerce")
    mask = (s > 0) & (r > 0)
    s, r = s[mask], r[mask]
    if len(s) < 5 or float(np.std(np.log(s))) < 0.02:
        return DEFAULT_ELASTICITY
    b = float(np.polyfit(np.log(s), np.log(r), 1)[0])
    return float(np.clip(b, 0.55, 0.95))


def channel_summary(result: dict, sku: str = "All") -> pd.DataFrame:
    """Aggregate the forecast to platform x funnel_stage 'channels' for Module 3.

    This is the integration contract the optimizer consumes. Columns:
      channel · platform · funnel_stage · current_spend · forecast_revenue ·
      forecast_roas · marginal_roas · marginal_roas_raw · confidence · confidence_label

    `current_spend` / `forecast_revenue` are daily averages over the horizon.

    Marginal ROAS = saturation elasticity x average ROAS. We fit each channel's
    diminishing-returns curve (revenue = a * spend^b) by log-log regression on its
    history; the slope `b` (elasticity, < 1 under diminishing returns) is the ratio
    of marginal to average ROAS. This is far more stable than a single point-probe
    on thin data, and it encodes the core idea: the next dollar returns less than
    the average dollar (marginal < average).
    """
    f = _filter(result["segment_forecast"], "All", sku)
    h = _filter(result["history"], "All", sku)
    horizon = result["horizon"]
    rows = []
    for (platform, funnel), g in f.groupby(["platform", "funnel_stage"]):
        spend_total = float(g["assumed_spend"].sum())
        rev_total = float(g["pred_revenue"].sum())
        roas = rev_total / spend_total if spend_total > 0 else float("nan")

        hist = h[(h["platform"] == platform) & (h["funnel_stage"] == funnel)]
        elasticity = _fit_elasticity(hist["spend"], hist["revenue"])
        marginal = max(0.0, elasticity * roas) if roas == roas else float("nan")

        mape = result["model_info"].get(platform, {}).get("mape", float("nan"))
        conf = float(np.clip(1 - mape, 0, 1)) if mape == mape else float("nan")
        rows.append({
            "channel": f"{platform} {str(funnel).title()}",
            "platform": platform,
            "funnel_stage": funnel,
            "current_spend": round(spend_total / horizon, 2),
            "forecast_revenue": round(rev_total / horizon, 2),
            "forecast_roas": round(roas, 2),
            "marginal_roas": round(marginal, 2),
            "elasticity": round(elasticity, 2),
            "confidence": round(conf, 2) if conf == conf else float("nan"),
            "confidence_label": _confidence_label(conf),
        })
    return pd.DataFrame(rows)


def mape_for(result: dict, platform: str = "All") -> float:
    """MAPE for the selection (n_train-weighted average when platform == All)."""
    info = result["model_info"]
    if platform != "All":
        return info.get(platform, {}).get("mape", float("nan"))
    num = den = 0.0
    for v in info.values():
        if v["mape"] == v["mape"]:
            num += v["mape"] * v["n_train"]
            den += v["n_train"]
    return num / den if den else float("nan")
