"""
Apply stateless feature transformers (geometry + freeze frame) to the
raw shots parquet, save as data/shots_features.parquet.

Categorical encoding (one-hot) is intentionally NOT done here.
It is done in Phase 3 AFTER train/test split, fit on training data only.
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_stateless_features_pipeline

DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "shots_raw.parquet"
OUTPUT_PATH = DATA_DIR / "shots_features.parquet"


def main() -> None:
    print(f"Loading {INPUT_PATH}...", flush=True)
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(df):,} shots, {df.shape[1]} columns.", flush=True)

    print("Building stateless feature pipeline...", flush=True)
    pipeline = build_stateless_features_pipeline()

    print("Applying transforms (this may take a minute)...", flush=True)
    t0 = time.time()
    result = pipeline.fit_transform(df)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s.  Output shape: {result.shape}", flush=True)

    new_cols = [c for c in result.columns if c not in df.columns]
    print(f"\nAdded {len(new_cols)} new columns:", flush=True)
    for c in new_cols:
        print(f"  - {c}", flush=True)

    print("\nValidation:", flush=True)
    print(f"  geom_distance:  min={result['geom_distance'].min():.2f}  "
          f"max={result['geom_distance'].max():.2f}  "
          f"mean={result['geom_distance'].mean():.2f}", flush=True)
    print(f"  geom_angle_deg: min={result['geom_angle_deg'].min():.2f}  "
          f"max={result['geom_angle_deg'].max():.2f}  "
          f"mean={result['geom_angle_deg'].mean():.2f}", flush=True)

    has_frame_rate = result["ff_has_frame"].mean()
    has_gk_rate = result["ff_has_gk"].mean()
    print(f"  ff_has_frame: {has_frame_rate*100:.1f}% of shots have a freeze frame.", flush=True)
    print(f"  ff_has_gk:    {has_gk_rate*100:.1f}% of shots have a visible goalkeeper.", flush=True)

    print("\nNull counts (new feature columns only):", flush=True)
    for c in new_cols:
        nulls = result[c].isnull().sum()
        if nulls > 0:
            print(f"  {c}: {nulls:,} nulls ({nulls/len(result)*100:.1f}%)", flush=True)

    print(f"\nSaving to {OUTPUT_PATH}...", flush=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved. Final shape: {result.shape}", flush=True)


if __name__ == "__main__":
    main()
