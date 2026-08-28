#!/usr/bin/env bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname patchpilot \
  --file /opt/patchpilot/001_queue.sql
