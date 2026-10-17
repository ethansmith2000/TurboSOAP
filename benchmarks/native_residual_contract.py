"""Fresh explicit native policy feasibility; prior protocols remain immutable."""
from pathlib import Path
from hybrid_width_contract import recipe as width_recipe, protocol as width_protocol, SNAPSHOT_CAP, PARAMETERS
from hybrid_training_contract import expected_counts

PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_hybrid_width2048_learning_seed145_frozen_20261011'),
         Path('/workspace/optimizer_replay_results/soap_prefix_pure_frozen_20261012'))
SEED=150
ARMS=('qr20','current_hybrid20','highest_residual20')
POLICIES=ARMS[1:]
ORDER=('highest_residual20','current_hybrid20','qr20')
HORIZON=420
WARMUP=245


def recipe(base):
    c=width_recipe(base);c.update(seed=SEED,max_wall_seconds=1800)
    c['matched_comparison']=dict(arms=list(ARMS),steps_per_arm=HORIZON,execution_order=list(ORDER),
        scope='Native actual-state feasibility and inherited-M/V adaptive preview, not learning qualification')
    return c


def protocol():
    p=width_protocol();p.update(seed=SEED,arms=list(ARMS),execution_order=list(ORDER),
        evaluation_calls=[],steps_per_arm=HORIZON,disposable_calls_per_arm=WARMUP)
    for phase,calls in (('primary',HORIZON),('disposable',WARMUP)):
        p['expected'][phase]={a:expected_counts(calls,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS}
    p['expected'].update(hybrid_kernel_calls=464,hybrid_calls_by_dimension={'2048':290,'6144':58,'8192':116},
        policy_calls_by_dimension={a:{'2048':145,'6144':29,'8192':58} for a in POLICIES},
        diagnostic_reference_calls=32,full_moment_transactions=16)
    p['precision']=dict(basis_state='FP32',covariance_state='FP32',
        policies={'current_hybrid20':dict(prefix='high',candidate='BF16 Q@A',retraction='old highest FP32 cubic2'),
                  'highest_residual20':dict(prefix='highest',candidate='BF16 Q@A',retraction='BF16 residual correction, highest FP32 Gram, cubic2')},
        momentum='highest FP32 unchanged',variance='carry warm; QR permutation unchanged',
        default_compile_fullgraph=True,dynamic=False,common_backend_delta={'allow_bf16_reduced_precision_reduction':False},
        reference='Each compiled hook versus its own matched eager BF16-candidate policy; counterfactual QR/current/candidate on identical actual C,Q,M,V')
    p['gates'].update(own_eager_basis_relative=1e-4,common_strict_basis_overlay=1e-4,
        full_moment_diagnostic_nonmutation=True,full_moment_registered_release=True,
        physical_m_reconstruction_limit=.01)
    p['artifacts']['probe'].update(purpose='Selected actual factor parity and common C,Q,M,V full physical-M/inherited-adaptive preview',
        expected_max_extra_bytes=16*8192**2*4+12*8192*2048*4,
        retention='RAM-only: release each parameter transaction before actual live refresh; compact scalar evidence retained')
    p['artifacts']['compiler_cache']['expected_bytes']=250*2**20
    p['bounds']['maximum_evidence_bytes']=100*2**20
    p['implementation']['entry']='source/benchmarks/cifar_native_residual.py'
    p['implementation']['diagnostic_adapter']='Instance refresh wrapper before original refresh; original ResearchSOAP M/V/QR/step unchanged'
    p['full_moment_diagnostic']=dict(calls=[21,281],source_arms=list(POLICIES),parameters_per_call=4,
        common_state='Post-current-step covariance/M/V, before actual refresh; exact parameter/gradient/state/RNG digests before/after',
        policies=['qr_counterfactual','current_hybrid20','highest_residual20'],
        preview='Transport actual M through world at highest, carry V for warm or permute V for QR, reconstruct inherited-state adaptive vector before any future gradient',
        not_a_future_step=True,no_cross_moment_history=True,stage_attribution='Validated eager-only raw norm/cap, never compiled intermediate attribution',
        registered_gpu_ram_limit_extra_bytes=16*2**30,
        expected_extra_bytes=p['artifacts']['probe']['expected_max_extra_bytes'],
        cpu_hash_workspace_bytes=2**20,check_rng=True,no_tensor_serialization=True)
    p['timing']['diagnostic_calls_excluded']=[21,281]
    p['bound_rationale']='Fresh explicit3arm420/245 native comparison,2020updates incl25QRsmoke; prior2048pilot317.791s/5.059GiB and pure-prefix300.965s/8.511GiB.1800/1860s,24GiB,512MiB CPUinitialpurpose,100MiBcompact/512MiBcache prospective; extra fullM probe RAM bounded~4.75GiB and retained onlypertransaction.'
    return p


def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared native residual policy/recipe delta')
