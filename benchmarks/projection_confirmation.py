"""Bound the unchanged188runner to one fresh189confirmation entrypoint."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import projection_learning as runner
from projection_learning import backend_record
from projection_learning_contract import ARMS as SUPPORTED_ARMS
from projection_confirmation_contract import ARMS
from prepare_projection_confirmation import verify,verify_launch,verify_backend

class Runtime(runner.Runtime):
    def before_smoke(self):
        record=backend_record();verify_backend(self.bundle,record)
        self.emit('backend_before_smoke',dict(record=record,passed=True,execution_contract=self.execution_contract()))
    def after_measured(self):
        record=backend_record();verify_backend(self.bundle,record)
        self.emit('backend_after_measured',dict(record=record,passed=True,execution_contract=self.execution_contract()))
        self.retain_code('after_measured');self.budget(resources=True)

def bind_runner():
    if set(ARMS)!=set(SUPPORTED_ARMS):raise ValueError('Confirmation cannot add policy')
    # These bindings belong only to this process/imported frozen runner module.
    # The source files, algorithm, trajectory and kernel functions remain unchanged.
    runner.ARMS=ARMS
    runner.Runtime=Runtime
    runner.verify=verify
    runner.verify_launch=verify_launch
    runner.verify_backend=verify_backend
    return runner

if __name__=='__main__':bind_runner().main()
