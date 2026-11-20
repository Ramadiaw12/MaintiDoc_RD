# File pour ingérer un PDF via la ligne de commande 
import sys

from app.ingestion.pipeline import ingest_pdf

print(ingest_pdf(sys.argv[1]))