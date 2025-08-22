
import config
import ramps
import torch
import torch.nn as nn


from data import NO_LABEL
import state
from loss import softmax_mse_loss

device = torch.device("cpu")

def train(train_loader, model, ema_model, optimizer, epoch):

    # average meters to record the training statistics
    running_loss = 0.0     

    # Loss function of student with reduction 'sum' to devide by the total number of labeled samples
    # in the minibatch instead of the total batch size 
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL)     

    # Loss function of consistency loss size between student and teacher
    consistency_criterion = softmax_mse_loss   

    # switch to train mode
    model.train()
    ema_model.train()

    # Iterate over the training data for one epoch
    # train loader provides both labeled and unlabeled data
    for i, (input, target) in enumerate(train_loader): 

        # skip the last incomplete batch
        if input.size(0) != config.batch_size:
            continue

        # Adjust the learning rate 
        ramps.adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # Move input and target to device (CPU)
        input_var = torch.autograd.Variable(input).to(device)
        target_var = torch.autograd.Variable(target).to(device)

        minibatch_size = input.size(0)
        labeled_minibatch_size = target_var.ne(NO_LABEL).sum()   

        # Make sure we have at least one labeled sample in the minibatch
        assert labeled_minibatch_size > 0, "No labeled samples in the minibatch"

        # Compute output of student model
        model_output = model(input_var)

        # Compute the classification loss only over the labeled samples
        classification_loss = class_criterion(model_output, target_var) / labeled_minibatch_size

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = config.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, config.lr_rampup) * (config.lr - config.initial_lr) + config.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if config.lr_rampdown_epochs:
        assert config.lr_rampdown_epochs >= config.epochs
        lr *= ramps.cosine_rampdown(epoch, config.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return config.consistency * ramps.sigmoid_rampup(epoch, config.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res