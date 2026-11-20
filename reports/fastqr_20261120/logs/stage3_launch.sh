#!/bin/bash
# Stage 3 (fast-QR study, 2026-11-20): BF16 QR50 x seeds 204/205 x 10k calls, and FP32-storage Householder QR10,
# seed 204, 2k calls. One exclusive gpu-claim per bundle (3 claims), queued with --wait behind other projects.
# Prepared before approval; run only after the user approves (REPORT.md section 8).
B=/workspace/data/fastqr_20261120
L=/workspace/TurboSOAP/reports/fastqr_20261120/logs
source /venv/main/bin/activate
pids=()
for name in stage3_seed204_bf16_hh_qr50_c10000 stage3_seed205_bf16_hh_qr50_c10000 stage3_seed204_fp32s_hh_qr10_c2000; do
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
