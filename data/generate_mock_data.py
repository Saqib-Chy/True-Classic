"""Seeded synthetic ad-data generator for the True Classic prototype.

Produces three CSVs in this folder:
  - meta_ads_export.csv     (Meta-style schema:   spend, purchase_value, product_id, ...)
  - google_ads_export.csv   (Google Ads-style:    cost,  conv_value,    item_id,    ...)
  - sku_catalog.csv         (canonical SKU list + cross-platform crosswalk)

The two exports use deliberately different column names and product-ID
formats so the normalize/reconcile steps have real work to do.

Intentional, demo-able data problems baked in:
  1. SKU mismatch    -> POLO-BLK (Google G-2005) has NO crosswalk entry.
  2. Date ranges     -> Google starts 2 days later than Meta.
  3. Missing values  -> a handful of Google conv_value cells are blank.

Deterministic (fixed seed) so the live demo is repeatable.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SEED = 42

# sku, product_name, meta_product_id, google_item_id
PRODUCTS = [
    ("TEE-CREW-BLK", "Crew Neck Tee - Black", "FB-1001", "G-2001"),
    ("TEE-CREW-WHT", "Crew Neck Tee - White", "FB-1002", "G-2002"),
    ("TEE-VNECK-NVY", "V-Neck Tee - Navy", "FB-1003", "G-2003"),
    ("HENLEY-GRY", "Henley - Grey", "FB-1004", "G-2004"),
    ("POLO-BLK", "Polo - Black", "FB-1005", "G-2005"),
]

# POLO-BLK is intentionally left OUT of the Google crosswalk even though Google
# still reports spend for G-2005. Reconciliation must detect and flag it.
GOOGLE_GAP_SKU = "POLO-BLK"

# (stage label, daily spend base, ROAS base, new-customer share)
#
# ROAS bases are deliberately spread so the optimizer's constraints are each
# observable when the marketer moves a slider in the demo:
#   * Marginal ROAS (≈ ROAS base) lands ABOVE and BELOW the 4.0x floor, so the
#     ROAS-floor slider toggles whether a move is possible.
#   * The HIGHEST-marginal channel (Google Remarketing, 6.0x) sits on the
#     NOISIER, lower-confidence platform, while the runner-up (Meta Retargeting,
#     5.0x) is on the cleaner, higher-confidence platform — so raising the
#     confidence slider flips the recipient from Google to Meta.
# ROAS base = AVERAGE ROAS near the segment's typical spend. With saturation
# (SAT_B < 1) the *marginal* ROAS ≈ SAT_B × base, so the values below place the
# four channels' marginals deliberately around the 4.0x floor.
META_SEGMENTS = [
    ("Prospecting", 120.0, 2.8, 0.75),   # marginal ≈ 2.4x -> below floor (donor)
    ("Retargeting", 60.0, 5.6, 0.10),    # marginal ≈ 4.8x -> above floor, HIGH confidence
]
GOOGLE_SEGMENTS = [
    ("Prospecting", 90.0, 4.2, 0.70),    # marginal ≈ 3.6x -> below floor (donor)
    ("Remarketing", 45.0, 7.0, 0.08),    # marginal ≈ 6.0x -> highest, LOW confidence
]

# Diminishing returns: revenue grows sub-linearly with spend (concave), so the
# marginal ROAS is genuinely below the average ROAS. SAT_B in (0,1); marginal ≈ SAT_B × avg.
SAT_B = 0.85

# Per-platform noise: Meta forecasts cleanly (high confidence), Google is noisy
# (low confidence). This is what makes the confidence threshold slider bite.
META_NOISE = 0.08
GOOGLE_NOISE = 0.22


def _prefix(sku: str) -> str:
    return sku.split("-")[0]


def generate(seed: int = SEED, base_jitter: float = 0.0) -> dict[str, pd.DataFrame]:
    """Generate the mock CSVs.

    base_jitter: fractional random shift applied to each segment's ROAS base
    (e.g. 0.3 = +/-30%). 0.0 (default) keeps the tuned, deterministic demo
    scenario. A non-zero jitter with a varying seed lets the "Regenerate" button
    produce genuinely different marginal-ROAS orderings -> different (still fully
    data-driven) optimizer recommendations.
    """
    rng = np.random.default_rng(seed)
    end = date.today()

    def _jitter(roas: float) -> float:
        if base_jitter <= 0:
            return roas
        return float(round(roas * (1 + rng.uniform(-base_jitter, base_jitter)), 2))

    meta_segments = [(s, sp, _jitter(r), nc) for (s, sp, r, nc) in META_SEGMENTS]
    google_segments = [(s, sp, _jitter(r), nc) for (s, sp, r, nc) in GOOGLE_SEGMENTS]
    # Small, deterministic dataset (~300 rows total):
    #   Meta:   5 SKUs x 2 segments x 16 days = 160 rows
    #   Google: 5 SKUs x 2 segments x 14 days = 140 rows  (starts 2 days later)
    meta_dates = [end - timedelta(days=i) for i in range(16, 0, -1)]
    google_dates = [end - timedelta(days=i) for i in range(14, 0, -1)]  # starts 2 days later

    # ----- Meta-style export -----
    meta_rows = []
    for sku, _name, fb_id, _g_id in PRODUCTS:
        for stage, spend_base, roas_base, nc_share in meta_segments:
            campaign = f"TC_{stage}_{_prefix(sku)}"
            for d in meta_dates:
                spend = float(max(5.0, rng.normal(spend_base, spend_base * META_NOISE)))
                # Concave response: effective ROAS decays as spend rises above base.
                roas_eff = roas_base * (spend / spend_base) ** (SAT_B - 1)
                roas = float(max(0.3, rng.normal(roas_eff, roas_eff * META_NOISE)))
                revenue = spend * roas
                impressions = int(spend / 12.0 * 1000)  # ~$12 CPM
                clicks = int(impressions * rng.uniform(0.008, 0.015))
                aov = rng.uniform(45, 65)
                purchases = int(max(0, revenue / aov))
                meta_rows.append(
                    {
                        "date": d.isoformat(),
                        "campaign_name": campaign,
                        "adset_name": f"{stage}_{sku}",
                        "product_id": fb_id,
                        "spend": round(spend, 2),
                        "impressions": impressions,
                        "clicks": clicks,
                        "purchases": purchases,
                        "purchase_value": round(revenue, 2),
                        "new_customers": int(purchases * nc_share),
                    }
                )
    meta = pd.DataFrame(meta_rows)

    # ----- Google Ads-style export (different column names + ID format) -----
    google_rows = []
    for sku, _name, _fb_id, g_id in PRODUCTS:
        for stage, spend_base, roas_base, nc_share in google_segments:
            campaign = f"{stage}_{_prefix(sku)}"
            for d in google_dates:
                cost = float(max(5.0, rng.normal(spend_base, spend_base * GOOGLE_NOISE)))
                roas_eff = roas_base * (cost / spend_base) ** (SAT_B - 1)
                roas = float(max(0.3, rng.normal(roas_eff, roas_eff * GOOGLE_NOISE)))
                revenue = cost * roas
                impr = int(cost / 10.0 * 1000)  # ~$10 CPM
                clicks = int(impr * rng.uniform(0.01, 0.02))
                aov = rng.uniform(45, 65)
                conversions = int(max(0, revenue / aov))
                google_rows.append(
                    {
                        "day": d.isoformat(),
                        "campaign": campaign,
                        "ad_group": f"{stage}_{sku}",
                        "item_id": g_id,
                        "cost": round(cost, 2),
                        "impr": impr,
                        "clicks": clicks,
                        "conversions": conversions,
                        "conv_value": round(revenue, 2),
                        "new_cust": int(conversions * nc_share),
                    }
                )
    google = pd.DataFrame(google_rows)

    # Inject a few missing revenue cells to exercise missing-data handling.
    missing_idx = rng.choice(google.index, size=5, replace=False)
    google.loc[missing_idx, "conv_value"] = np.nan

    # ----- Catalog / crosswalk (with the intentional Google gap) -----
    catalog_rows = []
    for sku, name, fb_id, g_id in PRODUCTS:
        catalog_rows.append(
            {
                "sku": sku,
                "product_name": name,
                "meta_product_id": fb_id,
                "google_item_id": "" if sku == GOOGLE_GAP_SKU else g_id,
            }
        )
    catalog = pd.DataFrame(catalog_rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_csv(DATA_DIR / "meta_ads_export.csv", index=False)
    google.to_csv(DATA_DIR / "google_ads_export.csv", index=False)
    catalog.to_csv(DATA_DIR / "sku_catalog.csv", index=False)

    return {"meta": meta, "google": google, "catalog": catalog}


if __name__ == "__main__":
    out = generate()
    print("Generated mock data in", DATA_DIR)
    for name, df in out.items():
        print(f"  {name:8s}: {len(df):4d} rows, {len(df.columns)} cols")
    print(f"Intentional gap: Google {GOOGLE_GAP_SKU} (G-2005) has no crosswalk entry.")
