
import torch
from types import SimpleNamespace
import os
""" Configuration file for Mean Teacher Method """

BASE_DIR = os.path.dirname(__file__)

BN = False # whether to use batch normalization or not
sntg = False # whether to use sntg loss or not

# save on CPU by default
device = torch.device("cpu") 

# model initialization
model = 'cifar_shakeshake26'  # Options: ResNet32x32, ResNet224x224, cifar_shakeshake26, resnext152

# learning rate configuration
lr = 0.1
lr_rampup = 0
initial_lr = 0
lr_rampdown_epochs = None

# optimizer parameters
momentum = 0.9
weight_decay = 0.1e-4
optim = 'SGD'

# Evaluation
evaluate = True
evaluation_epochs = 1

# Training parameters
start_epoch = 0
epochs = 100

# Checkpointing
checkpoint_epochs = 25  # Set to None to disable checkpointing
saveX = True
save_path = 'experiments' # base directory to save the model
dataName = 'cifar10'

# directories and data
traindir = 'data/dataset/train'
testdir = 'data/dataset/test'
labels = "data/dataset/labels/cifar10/00.txt" # path to the file containing 1000 labels for a subset of training data


# type of learnong: fully supervised or semi-supervised
supervised = False # if True, only use labeled data for training (fully supervised learning)

# if supervised is False, set labeled_batch_size to the desired batch size of labeled data (semi-supervised learning)
labeled_batch_size = 62 # if None, use standard random sampler (no constraint on
batch_size = 256 # total batch size (labeled + unlabeled) for semi-supervised learning, or batch size for fully supervised learning

# DataLoader parameters
num_workers = 4
pin_memory = False

# consistency loss parameters
consistency = 50.0/4
consistency_rampup = 5 # in the original paper, they used 30 for imagenet 


# multi-head architecture parameters
logit_distance_cost = 1 # for supervised = True, set to -1

# Exponential Moving Average (EMA) parameters
ema_decay = 0.999

# config.py
from types import SimpleNamespace

def build_config():
    return SimpleNamespace(
        device=device,
        model=model,
        lr=lr,
        lr_rampup=lr_rampup,
        initial_lr=initial_lr,
        lr_rampdown_epochs=lr_rampdown_epochs,
        momentum=momentum,
        weight_decay=weight_decay,
        optim=optim,
        evaluate=evaluate,
        evaluation_epochs=evaluation_epochs,
        start_epoch=start_epoch,
        epochs=epochs,
        checkpoint_epochs=checkpoint_epochs,
        saveX=saveX,
        save_path=save_path,
        dataName=dataName,
        traindir=traindir,
        testdir=testdir,
        labels=labels,
        supervised=supervised,
        labeled_batch_size=labeled_batch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        consistency=consistency,
        consistency_rampup=consistency_rampup,
        logit_distance_cost=logit_distance_cost,
        ema_decay=ema_decay,
        BN=BN,
        sntg=sntg,
        pretrained=False,  # Whether to use a pretrained model
    )






