import sys, pymupdf
src, dst = sys.argv[1], sys.argv[2]
doc = pymupdf.open(src)
with open(dst, "w") as f:
    for i, page in enumerate(doc):
        f.write(f"\n===== PAGE {i+1} =====\n")
        f.write(page.get_text())
print(doc.page_count, doc.metadata)
