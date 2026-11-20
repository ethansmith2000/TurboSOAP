#!/bin/bash
# GPU checks + timing for bf16_soap_fastqr, one exclusive gpu-claim (fast-QR study 2026-11-20)
set -e
cd /workspace/TurboSOAP
L=reports/fastqr_20261120/logs
echo "claim $CUDA_VISIBLE_DEVICES $(date -u) load $(cat /proc/loadavg)"
test -f reports/fastqr_20261120/unit_cuda_compiled.json || python benchmarks/fastqr_checks.py unit --device cuda --compiled > $L/unit_cuda_compiled.log 2>&1
echo "unit done $(date -u)"
python benchmarks/fastqr_timing.py > $L/timing.log 2>&1
echo "timing done $(date -u) load $(cat /proc/loadavg)"
python benchmarks/fastqr_checks.py compiled > $L/compiled_ieee_check.log 2>&1
echo "compiled check done $(date -u)"
for w in 512 1024 2048; do
  python benchmarks/fastqr_checks.py trained --device cuda --width $w > $L/trained_d${w}_cuda.log 2>&1
  echo "trained d$w done $(date -u)"
done
echo "done $(date -u)"
