import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
BATCH_SIZE = 1000

load_dotenv(ROOT / ".env")


def read_csv(filename):
    return pd.read_csv(
        DATA / filename,
        dtype=str,
        keep_default_na=False,
    )


def import_nodes(session, df, label):
    # Labels come only from the fixed values in main().
    query = f"""
    UNWIND $rows AS row
    MERGE (n:ESCO:{label} {{uri: row.uri}})
    SET n += row.properties
    """

    rows = []
    for record in df.to_dict("records"):
        uri = record.pop("conceptUri")
        rows.append({"uri": uri, "properties": record})

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        session.run(query, rows=batch).consume()
        print(f"{label}: {start + len(batch):,}/{len(rows):,}")


def import_relationships(session, df):
    query = """
    UNWIND $rows AS row
    MATCH (o:ESCO:Occupation {uri: row.occupationUri})
    MATCH (s:ESCO:Skill {uri: row.skillUri})
    MERGE (o)-[:REQUIRES_SKILL {importance: row.relationType}]->(s)
    RETURN count(*) AS processed
    """

    rows = df.to_dict("records")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        with session.begin_transaction() as tx:
            processed = tx.run(query, rows=batch).single()["processed"]

            if processed != len(batch):
                raise ValueError("Missing or duplicate relationship endpoints.")

            tx.commit()

        print(f"Relationships: {start + len(batch):,}/{len(rows):,}")


def main():
    occupations = read_csv("occupations.csv")
    skills = read_csv("skills.csv")
    relations = read_csv("occupation_skill_relations.csv")

    # Validate identifiers before making database changes.
    for name, df in (("occupations", occupations), ("skills", skills)):
        if df["conceptUri"].str.strip().eq("").any():
            raise ValueError(f"Empty URI in {name}.")
        if df["conceptUri"].duplicated().any():
            raise ValueError(f"Duplicate URI in {name}.")

    if not relations["occupationUri"].isin(occupations["conceptUri"]).all():
        raise ValueError("Unknown occupation in relationships.")

    if not relations["skillUri"].isin(skills["conceptUri"]).all():
        raise ValueError("Unknown skill in relationships.")

    if not relations["relationType"].isin(["essential", "optional"]).all():
        raise ValueError("Unexpected relationship type.")

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
            for label in ("Occupation", "Skill"):
                session.run(
                    f"""
                    CREATE CONSTRAINT esco_{label.lower()}_uri IF NOT EXISTS
                    FOR (n:{label}) REQUIRE n.uri IS UNIQUE
                    """
                ).consume()

            import_nodes(session, occupations, "Occupation")
            import_nodes(session, skills, "Skill")
            import_relationships(session, relations)

    print("\nESCO core import completed successfully!")


if __name__ == "__main__":
    main()