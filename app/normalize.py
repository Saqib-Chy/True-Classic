"""Layer 1a — Schema normalization.

Maps each platform's idiosyncratic export into ONE canonical fact table, parses
the funnel stage (prospecting vs retargeting) from the campaign name, and
imputes missing numeric values while reporting exactly what was touched.

Canonical schema:
    date | platform | campaign | funnel_stage | platform_product_id |
    spend | impressions | clicks | conversions | revenue | new_customers
"""
from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS = [
    "date",
    "platform",
    "campaign",
    "funnel_stage",
    "platform_product_id",
    "spend",
    "impressions",
    "clicks",
    "conversions",
    "revenue",
    "new_customers",
]

NUMERIC_COLUMNS = ["spend", "impressions", "clicks", "conversions", "revenue", "new_customers"]


def _funnel_stage(campaign_name: str) -> str:
    name = str(campaign_name).lower()
    if "retarget" in name or "remarket" in name:
        return "retargeting"
    return "prospecting"


def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"]).dt.date
    out["platform"] = "Meta"
    out["campaign"] = df["campaign_name"]
    out["funnel_stage"] = df["campaign_name"].apply(_funnel_stage)
    out["platform_product_id"] = df["product_id"].astype(str)
    out["spend"] = df["spend"]
    out["impressions"] = df["impressions"]
    out["clicks"] = df["clicks"]
    out["conversions"] = df["purchases"]
    out["revenue"] = df["purchase_value"]
    out["new_customers"] = df["new_customers"]
    return out[CANONICAL_COLUMNS]


def normalize_google(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["day"]).dt.date
    out["platform"] = "Google"
    out["campaign"] = df["campaign"]
    out["funnel_stage"] = df["campaign"].apply(_funnel_stage)
    out["platform_product_id"] = df["item_id"].astype(str)
    out["spend"] = df["cost"]
    out["impressions"] = df["impr"]
    out["clicks"] = df["clicks"]
    out["conversions"] = df["conversions"]
    out["revenue"] = df["conv_value"]
    out["new_customers"] = df["new_cust"]
    return out[CANONICAL_COLUMNS]


def normalize_all(meta_raw: pd.DataFrame, google_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (unified_df, impute_report).

    impute_report counts the missing numeric cells (per column) found *before*
    imputation, so the UI can surface what got filled rather than hiding it.
    """
    unified = pd.concat(
        [normalize_meta(meta_raw), normalize_google(google_raw)],
        ignore_index=True,
    )

    missing_before = {col: int(unified[col].isna().sum()) for col in NUMERIC_COLUMNS}
    missing_before = {k: v for k, v in missing_before.items() if v > 0}

    # Imputation rule for the prototype: missing numeric -> 0 (conservative;
    # never invents revenue). Production would impute from history.
    unified[NUMERIC_COLUMNS] = unified[NUMERIC_COLUMNS].fillna(0)

    impute_report = {
        "missing_cells_by_column": missing_before,
        "total_imputed_cells": int(sum(missing_before.values())),
        "rule": "missing numeric values filled with 0",
    }
    return unified, impute_report
