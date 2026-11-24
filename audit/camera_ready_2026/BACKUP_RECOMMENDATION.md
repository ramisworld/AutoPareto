# Backup status and recommendation (2026-11-02)

## Current state (accepted by the author on 2026-11-02)
* `/Users/Rami/Documents/dev/autopareto_evidence_backup_20261102/` is read-only. It holds:
  * an APFS clone of the full repository: 15,094 files, with 1,596 evidence files (11.71 GB) verified
    by SHA-256 and 21 LFS objects verified;
  * git state files;
  * a verified bundle of the `autopareto/*` branches (26.5 MB).
* **Limits:**
  * An APFS clone shares physical blocks with the original until a file changes. It protects against
    accidental edits or deletion, not against disk failure, theft or loss of the Mac.
  * The bundle predates the camera-ready branch commits (7551f83 onward).
  * Everything is on one internal disk. No off-machine copy exists. iCloud is full (0 bytes free),
    the GitHub repository is public, and the AWS SSO profiles appear organisational, so none of these
    was used.

## Recommendation (in priority order)
1. **External SSD, at least 64 GB free, before the GPU phase if possible.** About 30 GB is needed today:
   ```bash
   B=/Volumes/<DRIVE>/autopareto_offsite_$(date +%Y%m%d) && mkdir "$B"
   git -C /Users/Rami/Documents/dev/autopareto bundle create "$B/all_refs.bundle" --all
   tar -C /Users/Rami/Documents/dev/autopareto -cf "$B/lfs_objects.tar" .git/lfs/objects
   tar -C /Users/Rami/Documents/dev/autopareto -cf "$B/untracked_evidence.tar" --exclude .venv --exclude __pycache__ --exclude .pytest_cache results runs paper_backups audit analysis paper prelaunch_audit queue AUTOPARETO_TRAJECTORY_AUDIT.md FORENSIC_RESEARCH_AUDIT.md autopareto_overleaf.zip autopareto_final_overleaf.zip
   (cd "$B" && shasum -a 256 * > SHA256SUMS)
   ```
   Verify the copy:
   * `git bundle verify`;
   * a test `git clone` from the bundle, checking out `autopareto/camera-ready-opus`;
   * `shasum -c SHA256SUMS`;
   * a check of `tar -tf` against `state/evidence_sha256_manifest.txt`.
2. **A second, encrypted copy in personal cloud storage**, e.g. iCloud+ at 200 GB, or a personal
   Backblaze B2 / S3 bucket. Encrypt the archives first, for example with `age`. Do not use the public
   GitHub repository or organisational cloud accounts.
3. **During the GPU phase:**
   * pull every stage's results back to the Mac when the stage ends (E0, every campaign attempt,
     confirmation, E1, the extension);
   * add them to the external copy;
   * never rely on pod storage. Container disk is wiped when a pod stops, and the volume when it is
     terminated.
4. **Scan before any upload.** A scan on 2026-11-02 found no keys or tokens in the working tree or in
   any branch tip. RunPod SSH endpoints appear in about 69 files and must be redacted before any
   public release.
