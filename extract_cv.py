import json
from pathlib import Path

from src.cv_pipeline.language import detect_language
from src.cv_pipeline.pdf import extract_pdf_content


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "cvs" / "cv_ali.pdf"
OUTPUT_DIR = ROOT / "data" / "extracted"

def extract_text(pdf_path):
    return extract_pdf_content(pdf_path)


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