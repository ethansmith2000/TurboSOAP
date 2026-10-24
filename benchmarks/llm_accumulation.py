"""Equal-sized microbatch accumulation; one gradient clip before one optimizer call."""
import torch


def backward_accumulated(model, batches, loss_fn, *, count, clip_norm):
    batches = tuple(batches)
    if type(count) is not int or count < 1 or len(batches) != count:
        raise ValueError('Exact positive microbatch count required')
    if not batches or any(len(batch) != len(batches[0]) for batch in batches) or len(batches[0]) < 1:
        raise ValueError('Equal nonempty microbatches required for mean accumulation')
    model.zero_grad(set_to_none=True)
    total = None
    for batch in batches:
        loss = loss_fn(model, batch)
        (loss / count).backward()
        value = loss.detach() / count
        total = value if total is None else total + value
        del loss, value
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, foreach=True)
    return total, norm.detach()


def weighted_cost(means, arm, key):
    weights = {'qr10': {'ordinary': .9, 'qr': .1},
               'fast10': {'ordinary': .9, 'warm': .08, 'qr': .02}}[arm]
    if any(kind not in means or means[kind]['samples'] < 1 for kind in weights):
        raise ValueError('Every cadence event needs measured samples')
    return sum(weight * means[kind][key] for kind, weight in weights.items())
