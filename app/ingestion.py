"""Layer 0 — Ingestion.

Loads the raw Meta + Google CSV exports and the SKU catalog *as-is*, with no
transformation. If the mock CSVs don't exist yet, they're generated on demand
so the app is runnable on a fresh checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
META_CSV = DATA_DIR / "meta_ads_export.csv"
GOOGLE_CSV = DATA_DIR / "google_ads_export.csv"
CATALOG_CSV = DATA_DIR / "sku_catalog.csv"


def ensure_data_exists(force: bool = False) -> None:
    """Generate the mock CSVs if any are missing (or if force=True)."""
    have_all = META_CSV.exists() and GOOGLE_CSV.exists() and CATALOG_CSV.exists()
    if have_all and not force:
        return
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    import generate_mock_data  # noqa: E402  (path inserted above)

    generate_mock_data.generate()


def regenerate(base_jitter: float = 0.30, seed: int | None = None) -> int:
    """Regenerate the mock CSVs as a NEW random scenario.

    Uses a random seed and jittered ROAS bases so each regeneration can yield a
    different marginal-ROAS ordering (and thus a different, still fully
    data-driven, optimizer recommendation). Returns the seed used.
    """
    import random

    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    import generate_mock_data  # noqa: E402

    if seed is None:
        seed = random.randint(1, 1_000_000)
    generate_mock_data.generate(seed=seed, base_jitter=base_jitter)
    return seed


def load_meta(force_regen: bool = False) -> pd.DataFrame:
    ensure_data_exists(force=force_regen)
    return pd.read_csv(META_CSV)


def load_google(force_regen: bool = False) -> pd.DataFrame:
    ensure_data_exists(force=force_regen)
    return pd.read_csv(GOOGLE_CSV)


def load_catalog(force_regen: bool = False) -> pd.DataFrame:
    ensure_data_exists(force=force_regen)
    # Keep IDs as strings; the intentional blank google_item_id stays empty.
    return pd.read_csv(CATALOG_CSV, dtype=str, keep_default_na=False)
