import numpy as np
import config

"""Ramp functions for the Mean Teacher method."""

"""Ramp up and down for the learnng rate."""
def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = config.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, config.lr_rampup) * (config.lr - config.initial_lr) + config.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if config.lr_rampdown_epochs:
        assert config.lr_rampdown_epochs >= config.epochs
        lr *= cosine_rampdown(epoch, config.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


"""Sigmoid ramp up for the consistency loss."""
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def get_current_consistency_weight(epoch):
    return config.consistency * sigmoid_rampup(epoch, config.consistency_rampup)