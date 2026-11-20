"""Diagnostic: run the runner's own 'compiled_fp32' case (bf16_storage_screen as frozen here, unpatched) for 12 calls
in the paper's launch environment, to compare with the cached bf16_learning_20261026 compiled_fp32 losses."""
import json, os, sys
from pathlib import Path
B = Path(__file__).resolve().parent
sys.path[:0] = [str(B / 'source/benchmarks'), str(B / 'source')]
import fastqr_training as ft  # sets the launch environment before torch loads
os.environ.setdefault('GPU_CLAIM_INDICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(B / 'compiler_cache_compiled_fp32')
import bf16_storage_screen as bss
m = json.loads((B / "manifest.json").read_text())
bss.run(B, "compiled_fp32", 12, protocol=m["protocols"]["fp32s_hh_qr10"])
