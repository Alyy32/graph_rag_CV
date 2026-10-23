import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DECISIONS = {
    "a": "accepted",
    "r": "rejected",
    "u": "uncertain",
}


def main():
    parser = argparse.ArgumentParser(description="Review ESCO skill candidates.")
    parser.add_argument("cv_id")
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review previously reviewed candidates too.",
    )
    args = parser.parse_args()

    if not ObjectId.is_valid(args.cv_id):
        parser.error("Invalid CV ID.")

    for name in ("MONGODB_URI", "MONGODB_DATABASE"):
        if not os.getenv(name):
            parser.error(f"Missing {name} in .env.")

    identity = {
        "_id": ObjectId(args.cv_id),
        "owner_id": args.owner,
    }

    with MongoClient(
        os.environ["MONGODB_URI"],
        serverSelectionTimeoutMS=5000,
        tz_aware=True,
    ) as client:
        cvs = client[os.environ["MONGODB_DATABASE"]]["cvs"]
        cv = cvs.find_one(identity)

        if cv is None:
            raise ValueError("CV not found for this owner.")

        if cv.get("status") != "extracted":
            raise ValueError("CV is not ready for review.")

        expiry = cv.get("expires_at")
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError("This CV has expired.")

        analysis = cv.get("analysis", {})
        candidates = analysis.get("literal_matching", {}).get("candidates", [])
        reviews = analysis.get("manual_review", [])

        if not isinstance(reviews, list):
            raise ValueError("Unexpected manual_review format; no changes made.")

        if not candidates:
            print("No candidates. Run match_cv_skills.py first.")
            return

        review_exists = "manual_review" in analysis
        reviewed = {item["uri"] for item in reviews}

        print("a = accept | r = reject | u = uncertain | s = skip | q = quit")
        print("Each decision is saved immediately.\n")

        for number, candidate in enumerate(candidates, start=1):
            uri = candidate["uri"]

            if uri in reviewed and not args.all:
                continue

            print(f"\n[{number}/{len(candidates)}] {candidate['label_en']}")
            print(f"French: {candidate['label_fr']}")

            previous = next(
                (item["decision"] for item in reviews if item["uri"] == uri),
                None,
            )
            if previous:
                print(f"Previous decision: {previous}")

            for match in candidate.get("matches", []):
                print(f"  Term: {match['matched_term']}")
                if match.get("ambiguous"):
                    print("  Warning: this term matches several ESCO concepts.")
                for context in match.get("contexts", []):
                    print(f"    ...{context}...")

            while True:
                try:
                    choice = input("\nDecision [a/r/u/s/q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nStopped. Previous decisions are saved.")
                    return

                if choice in {"a", "r", "u", "s", "q"}:
                    break
                print("Enter a, r, u, s, or q.")

            if choice == "q":
                break
            if choice == "s":
                continue

            updated_reviews = [
                item for item in reviews if item["uri"] != uri
            ]
            updated_reviews.append({
                "uri": uri,
                "label_en": candidate["label_en"],
                "label_fr": candidate["label_fr"],
                "decision": DECISIONS[choice],
                "source": "manual",
                "reviewed_at": datetime.now(timezone.utc),
            })

            # Prevent overwriting decisions changed by another process.
            query = {
                **identity,
                "status": "extracted",
                "analysis.manual_review": (
                    reviews if review_exists else {"$exists": False}
                ),
            }

            result = cvs.update_one(
                query,
                {"$set": {"analysis.manual_review": updated_reviews}},
            )

            if result.matched_count != 1:
                raise RuntimeError(
                    "The CV or its reviews changed. Restart the command."
                )

            reviews = updated_reviews
            review_exists = True
            reviewed.add(uri)
            print(f"Saved: {DECISIONS[choice]}")

        candidate_uris = {item["uri"] for item in candidates}
        current_reviews = [
            item for item in reviews if item["uri"] in candidate_uris
        ]

        print("\nReview summary:")
        for decision in DECISIONS.values():
            count = sum(
                item["decision"] == decision for item in current_reviews
            )
            print(f"  {decision}: {count}")

        print(f"  unreviewed: {len(candidate_uris - reviewed)}")


if __name__ == "__main__":
    main()