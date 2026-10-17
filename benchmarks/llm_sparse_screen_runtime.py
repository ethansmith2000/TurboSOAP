"""Qualified common model/evaluator with sparse-screen launch gates and one J1 kernel."""
import shutil
import torch
from llm_math_learning_runtime import Runtime as PreviousRuntime
from execution_fingerprints import backend_record
from matched_training_runtime import streaming_digest
from llm_sparse_screen_kernel import Router


class Runtime(PreviousRuntime):
    def initialize(self):
        super().initialize()
        self.kernel = Router(self.emit, lambda: streaming_digest(self.streams.capture()))

    def budget(self, resources=False):
        super().budget(resources=False)
        if resources:
            bounds = self.protocol['bounds']; cache = self.bundle/'compile_cache'
            compact = sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and cache not in p.parents)
            cached = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
            if (torch.cuda.max_memory_allocated() >= bounds['maximum_peak_allocated_bytes'] or
                compact >= bounds['maximum_evidence_bytes'] or cached >= bounds['maximum_compiler_cache_bytes'] or
                shutil.disk_usage(self.bundle).free < bounds['minimum_free_disk_bytes']):
                raise RuntimeError('Frozen resource bound exceeded')

    def before_smoke(self):
        from prepare_llm_sparse_screen import verify_backend
        record = backend_record(); verify_backend(self.bundle, record)
        self.emit('backend_before_smoke', dict(record=record, passed=True, execution_contract=self.execution_contract()))

    def after_measured(self):
        from prepare_llm_sparse_screen import verify_backend
        record = backend_record(); verify_backend(self.bundle, record)
        self.emit('backend_after_measured', dict(record=record, passed=True, execution_contract=self.execution_contract()))
        self.retain_code('after_measured'); self.budget(resources=True)
