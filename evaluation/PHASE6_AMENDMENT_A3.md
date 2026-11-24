# Phase 6 amendment A3: run the frozen P1 downstream stage

Approved by the author on 2026-11-16 (received 20:30 UTC). The author chose "Approve A3 fix
(Recommended)":

> Run the frozen downstream function unchanged through a small wrapper that adds only Phase 5A's
> row_factory line. Frozen files stay unchanged. Rename the empty database aside rather than
> deleting it. Run P0b, controls and anchors first, since they don't depend on downstream, then
> downstream, the A1 switch, A2 and the A1 pilot.

- **Written before** A3 ran.
- **Parent plan:** sha256 dc1e605e..., unchanged.
- **A3 plan:** `evaluation/phase6_amendment_a3_plan.json`. It records:
  - the hashes of this file and of the A3 code;
  - the hash and row counts of the empty database.

## 1. Why

- **What happened.** At 13:35 UTC the parent runner's P1 downstream stage exited 1 within a second,
  before evaluating any checkpoint (`PHASE6_INCIDENTS.md`, incident 4).
- **The bug.** `execute_downstream` in `scripts/run_phase6.py` opens its new database with
  `sqlite3.connect(db_path)`, then calls `autopareto_db.migrate`. `migrate` reads rows by column
  name, which needs `db.row_factory = sqlite3.Row`.
- **Where the missing line comes from.** Phase 5A's function of the same purpose, in
  `scripts/run_phase5a_opus_fixed_tokens.py`, sets that line. Phase 6's copy omits it.
- **What the crash left.** A database with its tables created and 0 rows in every table.

## 2. What A3 does

- **Code.** `scripts/run_phase6_a3.py` calls the parent's `execute_downstream` unchanged, for the
  full parent plan. During that call, `sqlite3.connect` in the parent module returns connections
  with `row_factory = sqlite3.Row`. Nothing else is changed.
- **The empty database.** It is renamed to `phase6_downstream.db.incident4-empty`, so the stage
  creates its database as designed. Nothing is deleted.
- **Unchanged.** The evaluator and suite hashes, which the stage checks itself; the checkpoints; and
  the 23 models evaluated. MOTPE seed 8 is added later by A2.
- **Effect on later code.** The rest of the function reads rows only by index, which works the same
  with `sqlite3.Row`.
- **Order.** A3 runs before A2's downstream stage. A2 then finds an existing, migrated database, so
  it does not reach the failing call.
