import sqlite3, sys, json
db = sys.argv[1]; sql = sys.argv[2]
c = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
cur = c.execute(sql)
cols = [d[0] for d in cur.description]
print('\t'.join(cols))
for r in cur.fetchall():
    print('\t'.join('' if v is None else (str(v)[:120]) for v in r))
