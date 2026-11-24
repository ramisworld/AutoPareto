# Phase 6 amendment A1: local storage for P2 (DRAFT, not in force)

- **Status:** draft, awaiting the author's approval.
- **Scope:** nothing here applies to the P0 and P1 runs in progress. Those keep the frozen plan
  dc1e605e unchanged.
- **If approved:** the code changes are frozen in a new plan hash before P2 starts.

## Evidence (pod5, 2026-11-16, 08:15 to 08:30 UTC)

**1. Checkpoint reads are not the bottleneck while the network volume is healthy.**
- A cold read of a 159 MB checkpoint from the volume took 1.2 s. A warm read took 0.26 s, and a
  read from local disk 0.2 s.
- The CMA-ES held-out evaluation took 274 s. In 55 samples taken every 5 s during it:
  - the evaluator was waiting on I/O in 4 of them;
  - the GPU was busy 54% of the time on average;
  - about 4 CPU cores were busy throughout.

**2. Python startup from the volume is slow.**
- Importing torch, numpy, pyarrow and tiktoken from the Env A virtual environment, which lives on
  the volume, takes 10 to 14 s each time.
- The same imports from local disk take 2.4 s (Env C, same torch 2.9.1 build).
- A training run starts about 3 such processes: the uv launcher, the trainer and the evaluator.

**3. Training data is read from the volume inside the timed window.**
- In GPT A24 seed 6 (P1), 23 steps between steps 1,051 and 1,154 took 1.1 to 2.9 s each, against a
  median of 0.20 s. They fell every 4 to 5 steps over 51 s and cost about 33 s in total.
- This pattern fits data reads stalling on the volume. That is not proven.
- No other P1 run so far has a step longer than twice its median. Neither has any E1 run on pod4.
- **Why it matters for P2:** under the fixed 300 s budget, a stall like this cuts the tokens trained
  and so changes the result.

## What would change, for the P2 fixed-time stage and for lm-eval

1. **Env A.** A second copy of the same locked environment is created on local disk with
   `uv sync --locked` from the same `uv.lock`. It is checked identical to the volume copy with
   `uv pip freeze`.
2. **Training data and tokenizer.** These are copied to local disk. Every file's SHA-256 is checked
   against the registered cache manifest before the stage starts.
3. **Run directories.** These are written to local disk.
   - After each run, the whole directory is copied to the volume and every file's SHA-256 is
     compared.
   - The run record keeps both the local and the volume path.
   - A local copy is removed only after its verified volume copy exists.
4. **Checkpoints read by evaluators.** These are staged on local disk with SHA verification, as P0
   inference already does.

## What does not change

- Code, package versions, data bytes, checkpoints, seeds, timing rules, gates and metrics all stay
  the same.
- **Expected effect on results:** none. The one exception is intended: storage stalls no longer fall
  inside fixed-time training windows.

## Comparability check

- The 6 reused E1 fixed-time runs (AR-ref and GPT A22) ran on pod4 against its network volume.
- The pre-registered AR-ref seed-6 fixed-time drift control would run first under this amendment.
- **Stop rule:** if its training tokens or held-out BPB fall outside the range of E1's three seeds,
  the stage stops and is reported.

## Expected saving

- Staging checkpoint reads alone: about 0.1 to 0.2 h over P2.
- With the environment, data and run directories also on local disk: about 0.5 h of wall time in
  total, plus avoiding storage stalls like those on 2026-11-16.
