import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).with_name(".env"))

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
        message = session.run(
            "RETURN 'Neo4j connection successful!' AS message"
        ).single()["message"]

        print(message)