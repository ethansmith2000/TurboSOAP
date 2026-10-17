"""Same187 projection adapter; native QR and original own-history M sampler."""
from llm_sparse_screen_kernel import build_arm as native_arm,common_geometry_guard
from projection_profile_kernel import install
from reuse_learning_kernel import qualify_actual_m
from projection_learning_contract import ARMS

def build_arm(model,config,arm,router):
    if arm not in ARMS:raise ValueError('Undeclared projection-learning arm')
    opt=native_arm(model,config,'qr20',router)
    if arm=='bf1620':install(opt.matrix,'bf16')
    return opt
