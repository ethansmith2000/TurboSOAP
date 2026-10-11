"""Exercise a policy on disposable state, then restore the measured start."""
import time


def discarded_warmup(*, steps, model, initial_model, initial_rng,
                     make_optimizer, train_step, restore_rng, rng_state,
                     digest, synchronize, validate, counters, telemetry):
    if steps <= 0:
        raise ValueError('Warmup must contain updates')
    model.load_state_dict(initial_model)
    model.train()
    restore_rng(initial_rng)
    initial_model_hash = digest(model.state_dict())
    initial_rng_hash = digest(rng_state())
    before = telemetry()
    synchronize()
    started = time.monotonic()
    opt = make_optimizer()
    try:
        for index in range(steps):
            train_step(index, opt)
        synchronize()
        result = dict(steps=steps, seconds=time.monotonic()-started,
                      model_sha256=digest(model.state_dict()),
                      optimizer_sha256=digest(opt.state_dict()),
                      optimizer_counters=counters(opt), guard=validate(opt),
                      telemetry_before=before, telemetry_after=telemetry())
    finally:
        del opt
        model.load_state_dict(initial_model)
        model.train()
        restore_rng(initial_rng)
        synchronize()
    result['restored_model_sha256'] = digest(model.state_dict())
    result['restored_rng_sha256'] = digest(rng_state())
    assert result['restored_model_sha256'] == initial_model_hash
    assert result['restored_rng_sha256'] == initial_rng_hash
    return result
