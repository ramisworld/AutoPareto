#!/usr/bin/env bash
# Back up durable pod state to the /workspace volume every 15 minutes during the campaign.
# Additive only (no --delete): earlier mirrored files are never removed. Same rsync as
# `fresh_a40_bootstrap.sh mirror`; the mirror is a backup, not a trust boundary.
set -u
install -d -m 0700 /workspace/mirror
while true; do
  if rsync -a --no-owner --no-group /var/lib/autopareto/ /workspace/mirror/var_lib_autopareto/; then
    date -u +%FT%TZ > /workspace/mirror/last_sync_utc.txt
  else
    echo "[$(date -u +%FT%TZ)] rsync failed with $?" >> /workspace/mirror/mirror_loop_errors.log
  fi
  sleep 900
done
