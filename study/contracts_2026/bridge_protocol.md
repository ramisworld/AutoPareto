# Trusted Scientist–A40 bridge

The Scientist is a local process only. It is launched in a fresh macOS sandbox
containing exactly `train.py`, `prepare.py`, `study_runtime.py`, `program.md`, and
`campaign_view.json`; only `train.py` is writable. The Scientist request contains no
SSH configuration, remote path, credential, or historical result.

The trusted orchestration sequence is:

1. Stage a fresh five-file local context.
2. Run the Scientist in the local sandbox and capture its hypothesis, intended change,
   candidate hash, prompt hash, and all visible-context hashes.
3. Stage a transfer payload containing exactly `train.py` and
   `registered_metadata.json`.
4. Use the orchestrator-owned SSH transport to send those two files to the registered
   A40 run directory.
5. Invoke the trusted A40 runner, which supplies protected `prepare.py` and
   `study_runtime.py` from its preinstalled frozen checkout and runs the protected
   launcher/evaluator.
6. Validate campaign and contract identity, then append only the protected
   campaign-local result to the local history and `campaign_view.json`.

The bridge never copies SSH credentials into the Scientist context. It strips
inherited SSH/RunPod environment variables, rejects remote-access fields in a
Scientist proposal, and has no unsandboxed fallback. A result from another campaign
or a result containing sealed downstream metrics is rejected before the view changes.

The remote transport is intentionally not invoked by readiness tests. A future
campaign operator must explicitly call the transfer and launch methods after the
environment, isolation, baseline, and calibration gates are satisfied.
