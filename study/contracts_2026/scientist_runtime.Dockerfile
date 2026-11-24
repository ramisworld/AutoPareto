FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global --no-audit --no-fund @openai/codex@0.153.0 \
    && test "$(codex --version)" = "codex-cli 0.153.0" \
    && npm cache clean --force \
    && printf '%s\n' '#!/bin/sh' 'set -eu' 'mkdir -p /tmp/scientist-home /tmp/codex-home' 'ln -sf /run/codex/auth.json /tmp/codex-home/auth.json' 'export HOME=/tmp/scientist-home CODEX_HOME=/tmp/codex-home' 'exec /usr/local/bin/codex "$@"' > /usr/local/bin/autopareto-codex-entrypoint \
    && chmod 0555 /usr/local/bin/autopareto-codex-entrypoint

ENTRYPOINT ["/usr/local/bin/autopareto-codex-entrypoint"]
