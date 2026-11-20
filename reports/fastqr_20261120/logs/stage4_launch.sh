#!/bin/bash
# Stage 4 (fast-QR study, 2026-11-20): C6 front-loaded schedule, S = 300 and 1000, seed 204, 10k calls.
# One exclusive gpu-claim per bundle (2 claims), queued with --wait. Run only after the user approves (REPORT.md
# section 9), and after stage 3 has finished.
B=/workspace/data/fastqr_20261120
L=/workspace/TurboSOAP/reports/fastqr_20261120/logs
source /venv/main/bin/activate
pids=()
for name in stage4_seed204_bf16_fl300_w10_qr50_c10000 stage4_seed204_bf16_fl1000_w10_qr50_c10000; do
  gpu-claim run --owner fastqr --job $name --wait --timeout 172800 -- \
    python $B/$name/source/benchmarks/fastqr_training.py campaign --bundle $B/$name > $L/$name.log 2>&1 &
  pids+=($!)
  echo "$(date -u +%FT%TZ) queued $name pid $!"
  sleep 20
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "$(date -u +%FT%TZ) all done rc=$rc"
exit $rc
