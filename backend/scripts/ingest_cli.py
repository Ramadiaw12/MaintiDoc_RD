import sys

from app.ingestion.pipeline import ingest_pdf

print(ingest_pdf(sys.argv[1]))