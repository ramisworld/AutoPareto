import sqlite3, sys
for path in sys.argv[1:]:
    c = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    print("=====", path)
    for (name, sql) in c.execute("select name, sql from sqlite_master where type='table' order by name"):
        n = c.execute(f'select count(*) from "{name}"').fetchone()[0]
        cols = [r[1] for r in c.execute(f'pragma table_info("{name}")')]
        print(f"{name}: rows={n} cols={len(cols)}: {', '.join(cols)[:900]}")
    c.close()
