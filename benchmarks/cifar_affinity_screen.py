"""Narrow timing intervention: pin only this process's main/autograd threads."""
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]


def main():
    import cifar_timing_screen as trainer
    bundle=Path(sys.argv[sys.argv.index('--bundle')+1])
    original_config,original_build=trainer.screen_arm_config,trainer.build_optimizer
    original_affinities={}
    records=[]
    def config(c,arm):
        result=original_config(c,arm)
        result['timing_affinity']=arm['timing_affinity']
        result['timing_label']=arm['label']
        return result
    def build(model,c,policy):
        plan=c.get('timing_affinity')
        if plan is not None:
            workers=[int(p.name) for p in Path('/proc/self/task').iterdir()
                     if (p/'comm').read_text().strip()=='pt_autograd_0']
            assert len(workers)==1,workers
            pairs=[(os.getpid(),plan['main']),(workers[0],plan['autograd'])]
            for tid,cpu in pairs:
                if tid not in original_affinities:original_affinities[tid]=os.sched_getaffinity(tid)
                assert cpu in original_affinities[tid]
                os.sched_setaffinity(tid,{cpu})
                assert os.sched_getaffinity(tid)=={cpu}
            records.append(dict(label=c['timing_label'],threads=pairs))
            (bundle/'affinity_records.json').write_text(json.dumps(records,indent=2)+'\n')
        return original_build(model,c,policy)
    trainer.screen_arm_config,trainer.build_optimizer=config,build
    try:trainer.main()
    finally:
        for tid,affinity in original_affinities.items():
            try:os.sched_setaffinity(tid,affinity)
            except ProcessLookupError:pass
        (bundle/'affinity_restored.json').write_text(json.dumps({str(k):sorted(v) for k,v in original_affinities.items()},indent=2)+'\n')


if __name__=='__main__':main()
