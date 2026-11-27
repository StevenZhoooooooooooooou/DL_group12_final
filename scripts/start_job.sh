#!/bin/bash
# usage: scripts/start_job.sh NAME COMMAND...   - runs COMMAND detached in its own process group,
# logs to outputs/jobs/NAME.log, records the group id in outputs/jobs/NAME.pgid
cd "$(dirname "$0")/.."
name=$1; shift
setsid nohup bash -c "$*; echo \"[job $name] exited with \$? at \$(date)\"" > "outputs/jobs/$name.log" 2>&1 < /dev/null &
sleep 1
ps -o pgid= -p $! | tr -d ' ' > "outputs/jobs/$name.pgid"
echo "started $name (process group $(cat outputs/jobs/$name.pgid))"
