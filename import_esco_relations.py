import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 1000

load_dotenv(ROOT / ".env")


def load(name, language):
    path = (
        ROOT / "data" / language / f"ESCO_{language.upper()}"
        / f"{name}_{language}.csv"
    )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def prepare(name, columns):
    frames = {}

    for language in ("en", "fr"):
        df = load(name, language)[columns].copy()

        for column in columns:
            if df[column].str.strip().eq("").any():
                raise ValueError(
                    f"{name} ({language}): empty value in {column}"
                )

        frames[language] = df.drop_duplicates()

    en_keys = set(frames["en"].itertuples(index=False, name=None))
    fr_keys = set(frames["fr"].itertuples(index=False, name=None))

    if en_keys != fr_keys:
        raise ValueError(f"{name}: English/French relations differ.")

    return frames["en"]


def import_batches(session, name, query, rows):
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        with session.begin_transaction() as tx:
            processed = tx.run(query, rows=batch).single()["processed"]

            if processed != len(batch):
                raise ValueError(
                    f"{name}: missing or duplicate endpoint nodes."
                )

            tx.commit()

        print(f"{name}: {start + len(batch):,}/{len(rows):,}")


def main():
    # Prepare every dataset before writing anything.
    datasets = []

    for name, pillar in (
        ("broaderRelationsSkillPillar", "skills"),
        ("broaderRelationsOccPillar", "occupations"),
    ):
        df = prepare(name, ["conceptUri", "broaderUri"])

        rows = [
            {
                "source": row["conceptUri"],
                "target": row["broaderUri"],
                "pillar": pillar,
            }
            for row in df.to_dict("records")
        ]

        datasets.append((name, "hierarchy", rows))

    df = prepare(
        "skillSkillRelations",
        ["originalSkillUri", "relatedSkillUri", "relationType"],
    )

    rows = [
        {
            "source": row["originalSkillUri"],
            "target": row["relatedSkillUri"],
            "relationType": row["relationType"],
        }
        for row in df.to_dict("records")
    ]

    datasets.append(("skillSkillRelations", "skill_links", rows))

    queries = {
        "hierarchy": """
            UNWIND $rows AS row
            MATCH (a:ESCO {uri: row.source})
            MATCH (b:ESCO {uri: row.target})
            MERGE (a)-[:HAS_BROADER {pillar: row.pillar}]->(b)
            RETURN count(*) AS processed
        """,
        "skill_links": """
            UNWIND $rows AS row
            MATCH (a:ESCO:Skill {uri: row.source})
            MATCH (b:ESCO:Skill {uri: row.target})
            MERGE (a)-[:RELATED_SKILL {
                relationType: row.relationType
            }]->(b)
            RETURN count(*) AS processed
        """,
    }

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
            known = {
                record["uri"]
                for record in session.run(
                    "MATCH (n:ESCO) RETURN n.uri AS uri"
                )
            }

            skill_ids = {
                record["uri"]
                for record in session.run(
                    "MATCH (n:ESCO:Skill) RETURN n.uri AS uri"
                )
            }

            # Check all endpoints before importing any relationships.
            for name, kind, rows in datasets:
                allowed = skill_ids if kind == "skill_links" else known

                endpoints = {
                    uri
                    for row in rows
                    for uri in (row["source"], row["target"])
                }

                missing = endpoints - allowed

                if missing:
                    examples = "\n".join(sorted(missing)[:10])
                    raise ValueError(
                        f"{name}: {len(missing)} unknown URIs:\n"
                        f"{examples}"
                    )

                print(f"Validated {name}: {len(rows):,} unique links")

            for name, kind, rows in datasets:
                import_batches(session, name, queries[kind], rows)

    print("\nStructure relationships imported successfully!")


if __name__ == "__main__":
    main()