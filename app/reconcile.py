"""Layer 1b — SKU identity resolution.

Resolves each platform's product key (Meta catalog ID like FB-1001, Google
item_id like G-2001) to a single canonical SKU via the crosswalk in
sku_catalog.csv.

Rows whose key has no crosswalk entry are NOT dropped — they are tagged
`unmatched_no_crosswalk` and surfaced, so the data problem is visible (the
prototype intentionally leaves POLO-BLK / G-2005 unmapped on Google).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATCHED = "matched"
UNMATCHED = "unmatched_no_crosswalk"


def build_crosswalk(catalog: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide catalog into long (platform, platform_product_id) -> sku."""
    rows = []
    for _, r in catalog.iterrows():
        meta_id = str(r.get("meta_product_id", "")).strip()
        google_id = str(r.get("google_item_id", "")).strip()
        if meta_id:
            rows.append(
                {
                    "platform": "Meta",
                    "platform_product_id": meta_id,
                    "sku": r["sku"],
                    "product_name": r["product_name"],
                }
            )
        if google_id:
            rows.append(
                {
                    "platform": "Google",
                    "platform_product_id": google_id,
                    "sku": r["sku"],
                    "product_name": r["product_name"],
                }
            )
    return pd.DataFrame(rows, columns=["platform", "platform_product_id", "sku", "product_name"])


def reconcile(unified: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (reconciled_df, recon_report).

    reconciled_df gains `sku`, `product_name`, and `match_status` columns.
    """
    crosswalk = build_crosswalk(catalog)
    merged = unified.merge(crosswalk, on=["platform", "platform_product_id"], how="left")
    merged["match_status"] = np.where(merged["sku"].isna(), UNMATCHED, MATCHED)

    unmatched = merged[merged["match_status"] == UNMATCHED]
    unmatched_keys = (
        unmatched.groupby(["platform", "platform_product_id"])
        .agg(rows=("spend", "size"), spend=("spend", "sum"), revenue=("revenue", "sum"))
        .reset_index()
        .sort_values("spend", ascending=False)
    )

    recon_report = {
        "total_rows": int(len(merged)),
        "matched_rows": int((merged["match_status"] == MATCHED).sum()),
        "unmatched_rows": int(len(unmatched)),
        "matched_skus": int(merged.loc[merged["match_status"] == MATCHED, "sku"].nunique()),
        "unmatched_keys": unmatched_keys,
        "unmatched_spend": float(unmatched["spend"].sum()),
    }
    return merged, recon_report
