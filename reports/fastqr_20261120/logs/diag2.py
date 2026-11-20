"""Localize the FP32-storage nondeterminism (12 calls each, same claim):
asis: fp32s_hh_qr10 through the runner's install(); bssparity: same but the runner's own cuda_parity restored;
cfp32_myrunner: the compiled_fp32 schedule (repair 50) built through install() with bss.cuda_parity."""
import json, os, sys
from pathlib import Path
D = Path(__file__).resolve().parent
sys.path[:0] = [str(D / 'source/benchmarks'), str(D / 'source')]
import fastqr_training as ft
os.environ.setdefault('GPU_CLAIM_INDICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
variant = sys.argv[1]
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(D / variant / 'compiler_cache')
import bf16_storage_screen as bss
m = json.loads((D / 'manifest.json').read_text())
protocol = m['protocols']['fp32s_hh_qr10']
own_parity = bss.cuda_parity
if variant == 'cfp32_myrunner':
    ft.ARMS['cfp32_myrunner'] = dict(cls='production', frequency=10, repair=50, state_dtype='float32', basis_dtype='float32')
    arm = 'cfp32_myrunner'
else:
    arm = 'fp32s_hh_qr10'
ft.install(bss, arm)
if variant in ('bssparity', 'cfp32_myrunner'):
    bss.cuda_parity = lambda case: own_parity('compiled_fp32')
bss.run(D / variant, arm, 12, protocol=protocol)
