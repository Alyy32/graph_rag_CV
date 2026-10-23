from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "data"
datasets = {}

for language in ("en", "fr"):
    folder = BASE / language / f"ESCO_{language.upper()}"

    for name in ("occupations", "skills", "occupationSkillRelations"):
        path = folder / f"{name}_{language}.csv"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        datasets[language, name] = df

        print(f"\n--- {name} ({language}) ---")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(2).to_string(index=False))

for name in ("occupations", "skills"):
    en = datasets["en", name]
    fr = datasets["fr", name]

    if "conceptUri" not in en.columns or "conceptUri" not in fr.columns:
        print(f"\n{name}: conceptUri column missing; inspect headers.")
        continue

    en_ids = set(en["conceptUri"]) - {""}
    fr_ids = set(fr["conceptUri"]) - {""}

    print(f"\n--- Language alignment: {name} ---")
    print(f"Common URIs: {len(en_ids & fr_ids):,}")
    print(f"English only: {len(en_ids - fr_ids):,}")
    print(f"French only: {len(fr_ids - en_ids):,}")
    print(f"Duplicate English URIs: {en['conceptUri'].duplicated().sum()}")
    print(f"Duplicate French URIs: {fr['conceptUri'].duplicated().sum()}")
for language in ("en", "fr"):
    for name in ("occupations", "skills"):
        df = datasets[language, name]
        duplicates = df[df.duplicated("conceptUri", keep=False)]

        print(f"\n--- Duplicate details: {name} ({language}) ---")
        print(f"Entirely identical extra rows: {df.duplicated().sum()}")

        for uri, group in duplicates.groupby("conceptUri"):
            different_columns = [
                column
                for column in df.columns
                if group[column].nunique(dropna=False) > 1
            ]
            print(f"URI: {uri}")
            print(f"Different columns: {different_columns}")

    relations = datasets[language, "occupationSkillRelations"]
    occupation_ids = set(datasets[language, "occupations"]["conceptUri"])
    skill_ids = set(datasets[language, "skills"]["conceptUri"])

    print(f"\n--- Relation checks ({language}) ---")
    print("Relation types:", relations["relationType"].value_counts().to_dict())
    print(
        "Unknown occupations:",
        (~relations["occupationUri"].isin(occupation_ids)).sum(),
    )
    print(
        "Unknown skills:",
        (~relations["skillUri"].isin(skill_ids)).sum(),
    )
    print(
        "Duplicate links:",
        relations.duplicated(
            ["occupationUri", "skillUri", "relationType"]
        ).sum(),
    )