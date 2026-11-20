#!/bin/bash
# Stage 2 (fast-QR study, 2026-11-20): 2 arms x seeds 204/205 x 10k calls, one exclusive gpu-claim per bundle.
# Each campaign runs the bundle's frozen runner copy; claims queue with --wait (long timeout: GPUs may be busy).
B=/workspace/data/fastqr_20261120
L=/workspace/TurboSOAP/reports/fastqr_20261120/logs
source /venv/main/bin/activate
pids=()
for seed in 204 205; do for arm in bf16_hh_qr10 bf16_w10_cq50_ns1; do
  name=stage2_seed${seed}_${arm}_c10000
  gpu-claim run --owner fastqr --job $name --wait --timeout 172800 -- \
    python $B/$name/source/benchmarks/fastqr_training.py campaign --bundle $B/$name > $L/$name.log 2>&1 &
  pids+=($!)
  echo "$(date -u +%FT%TZ) queued $name pid $!"
  sleep 20
done; done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "$(date -u +%FT%TZ) all done rc=$rc"
exit $rc
