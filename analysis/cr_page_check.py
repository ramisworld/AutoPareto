#!/usr/bin/env python3
"""Per-part page accounting for the camera-ready PDF (workshop limits: Part 1 <= 4, Part 2 <= 4, Part 3 <= 1).

Reads the page numbers of the labels partone/partoneend, parttwo/parttwoend, partthree/partthreeend
from paper/main.aux (tectonic --keep-intermediates) and writes paper/camera_ready/page_check.json.
Optionally rasterises each page with macOS `sips` for visual inspection (--render DIR).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
LIMITS = {'Part 1': ('partone', 'partoneend', 4), 'Part 2': ('parttwo', 'parttwoend', 4),
          'Part 3': ('partthree', 'partthreeend', 1)}


def label_pages(aux: Path):
    pages = {}
    for line in aux.read_text(errors='replace').splitlines():
        m = re.match(r'\\newlabel\{([^}]*)\}\{\{[^}]*\}\{(\d+)\}', line)
        if m:
            pages[m.group(1)] = int(m.group(2))
    return pages


def main():
    aux = PAPER / 'main.aux'
    pdf = Path(sys.argv[sys.argv.index('--pdf') + 1]) if '--pdf' in sys.argv else PAPER / 'main.pdf'
    pages = label_pages(aux)
    import pypdf
    n_pages = len(pypdf.PdfReader(str(pdf)).pages)
    out = {'pdf': str(pdf), 'total_pages': n_pages, 'parts': {}}
    ok = True
    for part, (a, b, lim) in LIMITS.items():
        n = pages[b] - pages[a] + 1
        out['parts'][part] = {'first_page': pages[a], 'last_page': pages[b], 'pages': n, 'limit': lim, 'within_limit': n <= lim}
        ok &= n <= lim
    out['all_within_limits'] = ok
    out['appendix_and_references_from_page'] = pages['partthreeend'] + 1
    (ROOT / 'paper/camera_ready').mkdir(exist_ok=True)
    print(json.dumps(out, indent=2))
    if '--render' in sys.argv:
        d = Path(sys.argv[sys.argv.index('--render') + 1])
        d.mkdir(parents=True, exist_ok=True)
        r = pypdf.PdfReader(str(pdf))
        for i, page in enumerate(r.pages, 1):
            w = pypdf.PdfWriter()
            w.add_page(page)
            one = d / f'page{i:02d}.pdf'
            with one.open('wb') as f:
                w.write(f)
            subprocess.run(['sips', '-s', 'format', 'png', '--resampleWidth', '1400', str(one), '--out',
                            str(d / f'page{i:02d}.png')], check=True, capture_output=True)
    return out


if __name__ == '__main__':
    res = main()
    if '--write' in sys.argv:
        (ROOT / 'paper/camera_ready/page_check.json').write_text(json.dumps(res, indent=2) + '\n')
