import sys
from pathlib import Path

from src.preprocessing.text_extractor import TextExtractor

pdf_path = Path("data/raw/1-s2.0-S2211383524004507-main.pdf")

extractor = TextExtractor()

print("=== Trying GROBID ===\n")

from grobid_client.grobid_client import GrobidClient
client = GrobidClient(grobid_server='http://localhost:8070')
_, status, tei_xml = client.process_pdf(
    service='processFulltextDocument',
    pdf_file=str(pdf_path),
    generateIDs=True, consolidate_header=1, consolidate_citations=0,
    include_raw_citations=False, include_raw_affiliations=False,
    tei_coordinates=False, segment_sentences=False
)
xml_out = Path("data/test_output.xml")
xml_out.write_text(tei_xml, encoding='utf-8')
print(f"Raw TEI XML saved to {xml_out}\n")

sections = extractor.extract_with_grobid(pdf_path)

if sections and any(sections.values()):
    print("GROBID succeeded\n")
else:
    print("GROBID returned nothing, falling back to PyMuPDF\n")
    sections = extractor.extract_with_pymupdf(pdf_path)

for name, text in sections.items():
    if not text:
        continue
    print(f"{'='*60}")
    print(f"SECTION: {name.upper()}")
    print(f"{'='*60}")
    print(text)
    print(f"\n[chars: {len(text)}]\n")
