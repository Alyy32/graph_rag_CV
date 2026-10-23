from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


def extract_pdf_content(
    pdf_source: Path | str | BinaryIO | bytes,
) -> tuple[str, int, list[int]]:
    """Extract text and page metadata from a PDF path or binary source."""
    if isinstance(pdf_source, (Path, str)):
        path = Path(pdf_source)
        if not path.is_file():
            raise FileNotFoundError(f"CV not found: {path}")
        reader = PdfReader(path)
    elif isinstance(pdf_source, bytes):
        reader = PdfReader(BytesIO(pdf_source))
    else:
        reader = PdfReader(pdf_source)

    pages = []
    empty_pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty_pages.append(number)
        pages.append(text)

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("No text extracted. This PDF may require OCR.")

    return text, len(pages), empty_pages
