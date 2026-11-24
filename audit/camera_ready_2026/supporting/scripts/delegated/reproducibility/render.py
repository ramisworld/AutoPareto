import sys, pymupdf
doc = pymupdf.open(sys.argv[1])
for spec in sys.argv[3:]:
    pno, x0, y0, x1, y1, zoom = spec.split(',')
    page = doc[int(pno)-1]
    r = page.rect
    clip = pymupdf.Rect(float(x0)*r.width, float(y0)*r.height, float(x1)*r.width, float(y1)*r.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(float(zoom), float(zoom)), clip=clip)
    out = f"{sys.argv[2]}_p{pno}_{x0}_{y0}.png"
    pix.save(out); print(out, pix.width, pix.height)
