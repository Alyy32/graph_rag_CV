import json
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "cvs" / "cv_ali.pdf"
OUTPUT_DIR = ROOT / "data" / "extracted"

LANGUAGE_CODES = {
    Language.FRENCH: "fr",
    Language.ENGLISH: "en",
}


def extract_text(pdf_path):
    if not pdf_path.is_file():
        raise FileNotFoundError(f"CV not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    empty_pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            empty_pages.append(number)

        pages.append(text)

    text = "\n\n".join(pages).strip()

    if not text:
        raise ValueError(
            "No text extracted. This PDF may require OCR."
        )

    return text, len(reader.pages), empty_pages


def detect_language(text):
    detector = (
        LanguageDetectorBuilder
        .from_languages(Language.FRENCH, Language.ENGLISH)
        .with_minimum_relative_distance(0.20)
        .build()
    )

    confidence_values = detector.compute_language_confidence_values(text)
    detected = detector.detect_language_of(text)

    # Starting rule to evaluate on our CV examples.
    enough_text = sum(character.isalpha() for character in text) >= 80

    if not enough_text:
        detected = None
        status = "insufficient_text"
    elif detected is None:
        status = "uncertain"
    else:
        status = "detected"

    return {
        "dominant_language": LANGUAGE_CODES.get(detected, "und"),
        "status": status,
        "candidate_languages": ["fr", "en"],
        "scores": {
            LANGUAGE_CODES[item.language]: round(item.value, 6)
            for item in confidence_values
        },
        "esco_matching_languages": ["fr", "en"],
    }


def main():
    text, page_count, empty_pages = extract_text(PDF_PATH)
    language = detect_language(text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text_path = OUTPUT_DIR / f"{PDF_PATH.stem}.txt"
    metadata_path = OUTPUT_DIR / f"{PDF_PATH.stem}_language.json"

    text_path.write_text(text, encoding="utf-8")

    metadata = {
        "source_file": PDF_PATH.name,
        "page_count": page_count,
        "character_count": len(text),
        "pages_without_extracted_text": empty_pages,
        **language,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Pages: {page_count}")
    print(f"Extracted characters: {len(text):,}")
    print(f"Dominant language: {language['dominant_language']}")
    print(f"Language status: {language['status']}")

    for code, score in language["scores"].items():
        print(f"Score {code}: {score:.4f}")

    if empty_pages:
        print(f"Pages with no extracted text: {empty_pages}")

    print(f"\nText saved: {text_path}")
    print(f"Metadata saved: {metadata_path}")
    print("\n--- Full extracted text ---\n")
    print(text)


if __name__ == "__main__":
    main()