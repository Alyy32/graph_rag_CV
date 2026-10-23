import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 500

load_dotenv(ROOT / ".env")


def load_groups(name, language):
    path = (
        ROOT / "data" / language / f"ESCO_{language.upper()}"
        / f"{name}_{language}.csv"
    )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    if df["conceptUri"].str.strip().eq("").any():
        raise ValueError(f"{name} ({language}): empty URI.")

    # Only dates may differ between repeated concepts.
    duplicates = df[df.duplicated("conceptUri", keep=False)]

    for uri, group in duplicates.groupby("conceptUri"):
        content = group.drop(
            columns=["modifiedDate"], errors="ignore"
        ).drop_duplicates()

        if len(content) > 1:
            raise ValueError(f"Conflicting duplicate: {uri}")

    if "modifiedDate" in df.columns:
        dates = pd.to_datetime(
            df["modifiedDate"],
            format="mixed",
            utc=True,
            errors="coerce",
        )
        df = (
            df.assign(_date=dates)
            .sort_values("_date", na_position="first", kind="stable")
            .drop(columns="_date")
        )

    return df.drop_duplicates("conceptUri", keep="last")


def prepare_groups(name):
    en = load_groups(name, "en")
    fr = load_groups(name, "fr")

    if set(en["conceptUri"]) != set(fr["conceptUri"]):
        raise ValueError(f"English/French URI mismatch: {name}")

    return en.merge(
        fr,
        on="conceptUri",
        suffixes=("_en", "_fr"),
        validate="one_to_one",
    )


def main():
    # Prepare and validate both datasets before database writes.
    datasets = [
        ("SkillGroup", prepare_groups("skillGroups")),
        ("ISCOGroup", prepare_groups("ISCOGroups")),
    ]

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"],
        ),
    ) as driver:
        driver.verify_connectivity()

        with driver.session(
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        ) as session:
            # A URI identifies one concept across all ESCO node types.
            session.run("""
                CREATE CONSTRAINT esco_uri_unique IF NOT EXISTS
                FOR (n:ESCO) REQUIRE n.uri IS UNIQUE
            """).consume()

            for label, df in datasets:
                rows = []

                for record in df.to_dict("records"):
                    uri = record.pop("conceptUri")
                    rows.append({
                        "uri": uri,
                        "properties": record,
                    })

                # Labels come only from the fixed list above.
                query = f"""
                    UNWIND $rows AS row
                    MERGE (n:ESCO {{uri: row.uri}})
                    SET n:{label}
                    SET n += row.properties
                """

                for start in range(0, len(rows), BATCH_SIZE):
                    batch = rows[start:start + BATCH_SIZE]
                    session.run(query, rows=batch).consume()

                print(f"{label}: {len(rows):,} concepts imported")

    print("Group import completed successfully!")


if __name__ == "__main__":
    main()