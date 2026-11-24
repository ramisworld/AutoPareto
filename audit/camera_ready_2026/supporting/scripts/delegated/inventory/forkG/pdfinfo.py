#!/usr/bin/env python3
"""Read-only PDF metadata + text extraction (PyMuPDF). Writes text only under OUTDIR.
Usage: pdfinfo.py OUTDIR PDF [PDF ...]"""
import sys, os, hashlib, time
import fitz  # PyMuPDF

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for p in sys.argv[2:]:
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    st = os.stat(p)
    doc = fitz.open(p)
    md = doc.metadata or {}
    txt = []
    for i, page in enumerate(doc):
        txt.append(f'\n<<<PAGE {i+1}>>>\n' + page.get_text())
    name = p.replace('/', '__').lstrip('_')
    with open(os.path.join(out, name + '.txt'), 'w') as f:
        f.write(''.join(txt))
    first = ' '.join(doc[0].get_text().split())[:160] if len(doc) else ''
    print(f"{p}\n  sha256={h[:16]} bytes={st.st_size} mtime={time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))} pages={len(doc)} "
          f"title={md.get('title')!r} creator={md.get('creator')!r} producer={md.get('producer')!r} created={md.get('creationDate')!r} mod={md.get('modDate')!r}\n  first: {first}")
