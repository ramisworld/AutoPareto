import sys, fitz
needles = ['The autonomous search produced', 'Humans de', 'Each iteration begins', 'Each attempt runs as human']
for p in sys.argv[1:]:
    doc = fitz.open(p); found = {}
    for pno, page in enumerate(doc):
        for b in page.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                for s in l['spans']:
                    for n in needles:
                        if s['text'].startswith(n) and n not in found:
                            found[n] = (pno + 1, round(s['size'], 2))
    print(p.split('/')[-2] + '/' + p.split('/')[-1], found)
