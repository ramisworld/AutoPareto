#!/usr/bin/env python3
"""Read-only directory statistics: size, file counts by extension, mtime range.
Usage: dirstats.py ROOT DEPTH [--skip name ...]
Prints one record per directory at depth<=DEPTH (relative to ROOT)."""
import os, sys, time, collections, json

root = os.path.abspath(sys.argv[1])
depth = int(sys.argv[2])
skip = set(sys.argv[3:])

def fmt(ts):
    return time.strftime('%Y-%m-%d', time.localtime(ts)) if ts else '-'

def stats(path):
    size = 0; n = 0; ext = collections.Counter(); lo = None; hi = None; ndirs = 0
    for dp, dns, fns in os.walk(path):
        ndirs += len(dns)
        for f in fns:
            p = os.path.join(dp, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            size += st.st_size; n += 1
            e = os.path.splitext(f)[1].lower() or '(none)'
            if f.startswith('._'):
                e = '._AppleDouble'
            ext[e] += 1
            m = st.st_mtime
            lo = m if lo is None or m < lo else lo
            hi = m if hi is None or m > hi else hi
    return size, n, ndirs, ext, lo, hi

out = []
def walk(path, d):
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if name in skip or not os.path.isdir(p) or os.path.islink(p):
            continue
        s, n, nd, ext, lo, hi = stats(p)
        rel = os.path.relpath(p, root)
        top = ', '.join(f'{k}:{v}' for k, v in ext.most_common(8))
        print(f'{rel}\t{s/1048576:.1f}MB\tfiles={n}\tdirs={nd}\t{fmt(lo)}..{fmt(hi)}\t{top}')
        if d < depth:
            walk(p, d + 1)

walk(root, 1)
