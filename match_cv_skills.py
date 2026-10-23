import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CV_PATH = ROOT / "data" / "extracted" / "cv_ali.txt"
SKILLS_PATH = ROOT / "data" / "processed" / "skills.csv"
OUTPUT_PATH = ROOT / "data" / "extracted" / "cv_ali_skills.json"


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(r"\s+", " ", text).strip()


def split_labels(value):
    # Les fichiers peuvent utiliser de vrais retours à la ligne
    # ou les caractères littéraux \n.
    return [
        label.strip()
        for label in re.split(r"\r?\n|\\n", value)
        if label.strip()
    ]


def main():
    cv_text = CV_PATH.read_text(encoding="utf-8")

    if not cv_text.strip():
        raise ValueError("Le texte du CV est vide.")

    normalized_cv = normalize(cv_text)

    concepts = {}
    terms = defaultdict(set)

    with SKILLS_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        required = {
            "conceptUri",
            "preferredLabel_en",
            "preferredLabel_fr",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Colonnes ESCO manquantes : {missing}")

        for row in reader:
            uri = row["conceptUri"]

            concepts[uri] = {
                "uri": uri,
                "label_en": row["preferredLabel_en"],
                "label_fr": row["preferredLabel_fr"],
            }

            for language in ("en", "fr"):
                labels = [row[f"preferredLabel_{language}"]]

                labels += split_labels(
                    row.get(f"altLabels_{language}", "")
                )

                for label in labels:
                    term = normalize(label)

                    # Écarter les termes très courts et ambigus.
                    if len(term) < 3 and term not in {"c#", "c++"}:
                        continue

                    terms[term].add(uri)

    matches = defaultdict(list)

    for term, uris in terms.items():
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"

        if re.search(pattern, normalized_cv):
            for uri in uris:
                matches[uri].append({
                    "matched_term": term,
                    "ambiguous": len(uris) > 1,
                })

    results = []

    for uri, evidence in matches.items():
        results.append({
            **concepts[uri],
            "matches": sorted(
                evidence,
                key=lambda item: item["matched_term"],
            ),
            "method": "literal_esco_label_match",
            "review_status": "pending",
        })

    results.sort(key=lambda item: item["label_en"].casefold())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Compétences ESCO candidates : {len(results)}\n")

    for number, result in enumerate(results, start=1):
        print(f"{number}. {result['label_en']}")
        print(f"   Français : {result['label_fr']}")
        print(
            "   Termes trouvés :",
            ", ".join(
                item["matched_term"]
                for item in result["matches"]
            ),
        )

        if any(item["ambiguous"] for item in result["matches"]):
            print("   À vérifier : terme associé à plusieurs concepts.")

    print(f"\nRésultat enregistré : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()