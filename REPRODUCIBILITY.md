# Reproducibility

## Reproducible from this release

- The experiment, inference, and deployment tables in the README.
- The three released plots from the public CSV data.
- The named-candidate parameter snapshots and their source-file SHA-256 values.
- Public-data row counts and headline result rows.

Install the plotting dependency and regenerate the plots:

```bash
python3 -m pip install -r requirements-release.txt
python3 scripts/generate_release_plots.py
python3 scripts/verify_release.py
```

The generated SVG and PNG files are written to `plots/`. The verification script does not require third-party Python packages.

## Reproducible from the private evidence repository only

The following requires the private audited research repository and its `results/results.db` ledger:

```bash
python3 scripts/export_public_data.py \
  --database /path/to/autopareto/results/results.db \
  --output-dir /path/to/AutoPareto/data
```

This reproduces the public CSV exports byte-for-byte. It requires the `sqlite3` command-line client and does not modify the source database.

## Not reproducible from this release yet

- Full autonomous-search code and protected evaluation harness.
- Candidate source files beyond the exported parameter snapshots.
- Checkpoints and model weights.
- The 12-run confirmation panel.
- Downstream evaluation, quantization, larger-scale transfer, and cross-seed classical controls.

These are intentionally not represented as completed work. The full code release is planned with the paper.
