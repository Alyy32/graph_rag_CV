import argparse
import hashlib
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from src.cv_pipeline.language import detect_language
from src.cv_pipeline.pdf import extract_pdf_content


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _parse_args():
    parser = argparse.ArgumentParser(description="Store and extract a CV.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--owner", required=True)
    return parser.parse_args()


def _find_existing(cvs, identity):
    existing = cvs.find_one(identity)
    if existing:
        print(f"CV already registered: {existing['_id']}")
        print(f"Status: {existing['status']}")
    return existing


def main():
    args = _parse_args()
    path = args.pdf.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    retention_days = int(os.environ["CV_RETENTION_DAYS"])
    if retention_days <= 0:
        raise ValueError("CV_RETENTION_DAYS must be positive.")

    pdf_bytes = path.read_bytes()
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    text, page_count, empty_pages = extract_pdf_content(pdf_bytes)
    language = detect_language(text)
    now = datetime.now(timezone.utc)
    identity = {"owner_id": args.owner, "sha256": digest}

    with MongoClient(
        os.environ["MONGODB_URI"],
        serverSelectionTimeoutMS=5000,
    ) as client:
        client.admin.command("ping")
        db = client[os.environ["MONGODB_DATABASE"]]
        cvs = db["cvs"]
        bucket = GridFSBucket(db, bucket_name="cv_files")

        cvs.create_index([("owner_id", 1), ("sha256", 1)], unique=True)
        cvs.create_index("expires_at")

        if _find_existing(cvs, identity):
            return

        cv_id = ObjectId()
        try:
            cvs.insert_one({
                "_id": cv_id,
                **identity,
                "original_filename": path.name,
                "uploaded_at": now,
                "expires_at": now + timedelta(days=retention_days),
                "status": "uploading",
            })
        except DuplicateKeyError:
            _find_existing(cvs, identity)
            return

        try:
            bucket.upload_from_stream_with_id(
                cv_id,
                path.name,
                BytesIO(pdf_bytes),
                metadata={
                    "cv_id": cv_id,
                    "owner_id": args.owner,
                    "contentType": "application/pdf",
                },
            )
            cvs.update_one(
                {"_id": cv_id},
                {"$set": {
                    "pdf_file_id": cv_id,
                    "text": text,
                    "page_count": page_count,
                    "character_count": len(text),
                    "pages_without_text": empty_pages,
                    "language": language,
                    "status": "extracted",
                    "processed_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:
            cvs.update_one(
                {"_id": cv_id},
                {"$set": {"status": "failed"}},
            )
            raise

    print(f"CV registered: {cv_id}")
    print(f"Pages: {page_count}")
    print(f"Extracted characters: {len(text):,}")
    print(f"Language: {language['dominant_language']}")
    print(f"Expiration: {retention_days} days")


if __name__ == "__main__":
    main()
