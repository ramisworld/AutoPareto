import sqlite3, sys, json
def conn(path):
    c = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    return c
DB = "/Users/Rami/Documents/dev/autopareto/results/results.db"
SNAP = "/Users/Rami/Documents/dev/autopareto/results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db"
if __name__ == "__main__":
    path = {"main": DB, "snap": SNAP}.get(sys.argv[1], sys.argv[1])
    c = conn(path)
    sql = sys.argv[2]
    rows = c.execute(sql).fetchall()
    for r in rows:
        print(json.dumps({k: r[k] for k in r.keys()}, default=str))
