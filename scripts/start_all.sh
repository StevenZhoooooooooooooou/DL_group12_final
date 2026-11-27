#!/bin/bash
# Launch run_all.sh in its own process group so stop_all.sh can end everything at once.
cd "$(dirname "$0")/.."
setsid nohup scripts/run_all.sh >> outputs/run_all.log 2>&1 < /dev/null &
sleep 1
ps -o pgid= -p $! | tr -d ' ' > outputs/run_all.pgid
echo "started, process group $(cat outputs/run_all.pgid)"
