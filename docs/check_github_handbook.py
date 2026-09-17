"""Check PDF structure and build contact sheets for human visual review."""
from pathlib import Path
import json

import pdfplumber
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'output/pdf/STS2026_GitHub_Team_Handbook.pdf'
QA = ROOT / 'tmp/pdfs/handbook-qa'

reader = PdfReader(PDF)
assert len(reader.pages) == 23, 'Unexpected page count; index must be updated'
assert len(reader.outline) == 23, 'Missing navigation bookmarks'
issues = []
with pdfplumber.open(PDF) as pdf:
    for number, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ''
        if f'{number:02d} / STS2026 FIELD GUIDE' not in text:
            issues.append(f'Page {number}: chapter header mismatch')
        for char in page.chars:
            if char['x0'] < 40 or char['x1'] > page.width - 40:
                issues.append(f'Page {number}: horizontal overflow: {char["text"]}')
            if char['top'] < 15 or char['bottom'] > page.height - 15:
                issues.append(f'Page {number}: vertical overflow')
        if '\ufffd' in text or '\x00' in text:
            issues.append(f'Page {number}: invalid extracted character')

links = reader.pages[1].get('/Annots', [])
assert len(links) == 21, 'Scenario index must have 21 links'
for number, annotation in enumerate(links, 3):
    dest = annotation.get_object().get('/Dest')
    assert dest is not None, f'Missing destination for page {number}'
    assert dest[0] == reader.pages[number - 1].indirect_reference, 'Wrong index target'

images = sorted(QA.glob('page-*.png'))
assert len(images) == 23, 'Render all pages before visual review'
for start in range(0, len(images), 6):
    sheet = Image.new('RGB', (1260, 1800), '#dbe3e9')
    draw = ImageDraw.Draw(sheet)
    for slot, path in enumerate(images[start:start + 6]):
        with Image.open(path) as source:
            thumb = source.convert('RGB')
            thumb.thumbnail((600, 550))
            x = 20 + (slot % 2) * 630 + (600 - thumb.width) // 2
            y = 30 + (slot // 2) * 590
            sheet.paste(thumb, (x, y))
            draw.text((x, y - 18), f'Page {start + slot + 1}', fill='black')
    sheet.save(QA / f'contact-{start // 6 + 1}.png')

report = {'pages': len(reader.pages), 'bookmarks': len(reader.outline),
          'index_links_verified': len(links), 'rendered_pages': len(images),
          'layout_issues': issues, 'size_bytes': PDF.stat().st_size}
print(json.dumps(report, indent=2))
assert not issues, 'PDF layout bounds failed'
