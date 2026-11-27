#!/bin/bash
# Emit each new important log line once (results, finished runs, errors); exit when no job is running.
cd "$(dirname "$0")/.."
SEEN=outputs/.watch_seen; touch "$SEEN"
while true; do
  {
    grep -hE "EVAL|finished|recon AUROC|trained \(|SZU-style|\[ucsd seed|Traceback|Error|Killed|out of memory" \
         outputs/mvtec_supp_train.log outputs/ucsd/supp_seed*.log outputs/reeval/supp_*.log \
         outputs/dsprites/logs/*.log 2>/dev/null
    grep -hE "_supp_s[0-9]\] it (10000|30000) " outputs/dsprites/logs/*.log 2>/dev/null
    grep -hE "it +20000/20000" outputs/ucsd/supp_seed*.log 2>/dev/null
    grep -h "^\[job " outputs/jobs/*.log 2>/dev/null
  } | sort -u > outputs/.watch_now
  comm -13 "$SEEN" outputs/.watch_now
  mv outputs/.watch_now "$SEEN"
  alive=0
  for f in outputs/jobs/*.pgid; do pgrep -g "$(cat "$f")" > /dev/null && alive=1; done
  [ $alive -eq 0 ] && { echo "[watch] no job running"; exit 0; }
  sleep 60
done
