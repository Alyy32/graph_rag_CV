from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "processed"


def load(name, language):
    path = (
        ROOT
        / "data"
        / language
        / f"ESCO_{language.upper()}"
        / f"{name}_{language}.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def clean_concepts(df, name, language):
    required = {"conceptUri", "modifiedDate"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} ({language}): missing columns {missing}")

    if df["conceptUri"].str.strip().eq("").any():
        raise ValueError(f"{name} ({language}): empty concept URI.")

    # Duplicate concepts must have identical content apart from dates.
    duplicates = df[df.duplicated("conceptUri", keep=False)]

    for uri, group in duplicates.groupby("conceptUri"):
        content = group.drop(columns="modifiedDate").drop_duplicates()

        if len(content) > 1:
            raise ValueError(
                f"{name} ({language}): conflicting duplicate {uri}"
            )

    dates = pd.to_datetime(
        df["modifiedDate"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    invalid_count = int(dates.isna().sum())

    if invalid_count:
        print(
            f"{name} ({language}): "
            f"{invalid_count} missing or unrecognised dates."
        )

    # Valid dates sort after missing dates, so keep the newest valid row.
    # If all dates are invalid, keep the last occurrence.
    cleaned = (
        df.assign(_date=dates)
        .sort_values("_date", na_position="first", kind="stable")
        .drop_duplicates("conceptUri", keep="last")
        .drop(columns="_date")
    )

    print(
        f"{name} ({language}): {len(df):,} rows -> "
        f"{len(cleaned):,} unique concepts"
    )

    return cleaned


def main():
    prepared = {}
    concept_ids = {}

    for name in ("occupations", "skills"):
        en = clean_concepts(load(name, "en"), name, "en")
        fr = clean_concepts(load(name, "fr"), name, "fr")

        en_ids = set(en["conceptUri"])
        fr_ids = set(fr["conceptUri"])

        if en_ids != fr_ids:
            raise ValueError(f"Language URI mismatch: {name}")

        prepared[name] = en.merge(
            fr,
            on="conceptUri",
            suffixes=("_en", "_fr"),
            validate="one_to_one",
        )

        concept_ids[name] = en_ids

    columns = ["occupationUri", "skillUri", "relationType"]
    relation_frames = {}

    for language in ("en", "fr"):
        df = load("occupationSkillRelations", language)

        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"Relationships ({language}): missing columns {missing}"
            )

        links = df[columns].copy()

        if links.apply(lambda column: column.str.strip().eq("")).any().any():
            raise ValueError(f"Relationships ({language}): empty values.")

        if not links["relationType"].isin(["essential", "optional"]).all():
            raise ValueError(
                f"Relationships ({language}): unexpected relation type."
            )

        if not links["occupationUri"].isin(concept_ids["occupations"]).all():
            raise ValueError(
                f"Relationships ({language}): unknown occupation URI."
            )

        if not links["skillUri"].isin(concept_ids["skills"]).all():
            raise ValueError(
                f"Relationships ({language}): unknown skill URI."
            )

        relation_frames[language] = links.drop_duplicates()

    en_keys = set(
        relation_frames["en"].itertuples(index=False, name=None)
    )
    fr_keys = set(
        relation_frames["fr"].itertuples(index=False, name=None)
    )

    if en_keys != fr_keys:
        raise ValueError("Relationships differ between English and French.")

    prepared["occupation_skill_relations"] = relation_frames["en"]

    # Write outputs only after all validation checks pass.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, df in prepared.items():
        path = OUTPUT / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {path.name}: {len(df):,} rows")

    print("\nPreparation successful! Files saved in data/processed/")


if __name__ == "__main__":
    main()