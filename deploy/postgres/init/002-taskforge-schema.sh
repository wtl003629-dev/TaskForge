#!/usr/bin/env bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname taskforge \
  --file /opt/taskforge/002_taskforge_runtime.sql
