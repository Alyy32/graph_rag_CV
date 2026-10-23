from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

FILES = [
    "skillSkillRelations",
    "broaderRelationsSkillPillar",
    "broaderRelationsOccPillar",
    "skillGroups",
    "skillsHierarchy",
    "ISCOGroups",
]

for language in ("en", "fr"):
    folder = ROOT / "data" / language / f"ESCO_{language.upper()}"

    for name in FILES:
        path = folder / f"{name}_{language}.csv"
        print(f"\n--- {name} ({language}) ---")

        if not path.is_file():
            print("FILE NOT FOUND")
            continue

        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

        print(f"Rows: {len(df):,}")
        print("Columns:", df.columns.tolist())

        # Truncate long cells to keep the output readable.
        sample = df.head(2).map(
            lambda value: value[:150] + "…" if len(value) > 150 else value
        )
        print(sample.to_string(index=False))