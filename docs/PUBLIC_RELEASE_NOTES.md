# Public release notes (camera-ready, November 2026)

This release is one snapshot commit of the camera-ready research branch (local commit `1da8261`). It also includes the
final poster edits: a QR code to the paper and the footer links. The full history stays with the author because it
contains infrastructure details. No measurement, record value, plan or result was changed.

## Removed

- `paper/AUTHOR_METADATA_PRIVATE.md`: private submission metadata.
- `poster/out_v5/autopareto_poster_v4_v5_reference.png`: a layout reference that contains another author's poster.

## Redacted

- **RunPod pod IDs:** five IDs are replaced by `<pod-id>` in 65 text files.
- **SSH addresses:** addresses of the form `<pod-id>-<suffix>@ssh.runpod.io` are replaced by that placeholder text.
- **Where they appeared:** in snapshot folder names such as `results/snapshots/runpod-<pod-id>-shutdown-20260901`, in
  audit notes, and in two default host values in `study/contracts_2026/{bridge,campaign_controller}.py`.

The pods no longer exist; this only keeps infrastructure identifiers out of a public repository.

Twelve of the redacted files also have a SHA-256 recorded in an inventory or manifest written before this release. Those
recorded hashes describe the unredacted originals:

- `PROJECT_STATE.md`
- `paper/generated/artifact_inventory.json`
- both `paper_backups/*/generated/artifact_inventory.json`
- `results/camera_ready_2026/phase2/phase2_verification.json`
- `study/contracts_2026/provenance/{claim_source_artifact_map.json, experiment_manifest_index.json, release_artifact_index.md, reproduction_commands.md, source_registry.json}`
- `study/contracts_2026/tools/preflight_{retrieve_file,transfer_checkpoint}.exp`

No code checks these twelve hashes at run time.

## Added

| File | What it is |
|---|---|
| `README.md` | the five-step replication guide (the August 2026 README is kept in `docs/`) |
| `uv.lock` | the lockfile of the pinned training environment ("Env A"), sha256 `262a4d55…`, previously on a local branch only |
| `assets/build_assets.sh`, `assets/install_assets.sh` | rebuild and install the data and kernel assets from pinned public revisions (previously a private bundle) |
| `study/contracts_2026/provenance/public_base_manifest_opus_campaign.json` | the Opus campaign's final frozen manifest, prepared as the `--base` for a new campaign (see below) |
| `study/contracts_2026/provenance/baseline_registration_new_a40.json` | the baseline registration record the Opus campaign launched with, copied unchanged |
| `scripts/compare_with_published.py` | prints a rerun's metrics next to the published run's |

### The public base manifest

`public_base_manifest_opus_campaign.json` is the Opus campaign's last frozen manifest: the timeout-900 continuation
manifest after the attempt-9 amendment, source SHA-256 `35e50181…`.

**What differs from that manifest:**
- the SSH host, pod IDs and operator-local paths are replaced by placeholders;
- the pinned hashes of the nine deployed files whose pod IDs were redacted are updated to this release's files:
  - `study/contracts_2026/{bridge,campaign_controller}.py`;
  - the five provenance files and two `.exp` files listed above.

**Unchanged:** every other pinned file hash, the contracts, schedule, data, inference protocol and scientist settings.
All 122 pinned files of the original manifest match the camera-ready code byte for byte.

**Check before release:** `derive_claude_campaign_manifest.py` was run with this base on a laptop, using the offline
fake Claude CLI and a placeholder GPU, and the controller's own launch gate accepted the derived manifest.

## Changed

- `analysis/cr_p6.py` writes `unified_13_rows.csv` with LF line endings. A rerun is then byte-identical to the
  committed file on every operating system; only the line endings changed.

## Not included

- Model checkpoints, published only as hashes.
- Raw training logs and the SQLite evidence database.
- Agent session logs, which contain credentials.
