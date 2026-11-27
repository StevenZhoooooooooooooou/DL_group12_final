#!/bin/bash
# Stop every job started with start_job.sh (or only the named ones: scripts/stop_all.sh NAME...).
cd "$(dirname "$0")/.."
names=("$@"); [ ${#names[@]} -eq 0 ] && names=($(ls outputs/jobs/*.pgid 2>/dev/null | xargs -n1 basename | sed 's/.pgid$//'))
for n in "${names[@]}"; do
  g=$(cat "outputs/jobs/$n.pgid" 2>/dev/null) || continue
  kill -TERM -- -"$g" 2>/dev/null && echo "stopped $n (group $g)" || echo "$n not running"
done
