
import torch

""" Configuration file for Mean Teacher Method """

# save on CPU by default
device = torch.device("cpu") 

# model initialization
model = 'ResNet32x32'  # Options: ResNet32x32, ResNet224x224, cifar_shakeshake26, resnext152

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
save_path = 'Mean_Teacher_Method/experiments' # base directory to save the model
dataName = 'cifar10'

# directories and data
traindir = 'Mean_Teacher_Method/data/dataset/cifar10/train'
testdir = 'Mean_Teacher_Method/data/dataset/cifar10/test'
labels = "Mean_Teacher_Method/data/dataset/labels/cifar10.txt" # path to the file containing labels for a subset of training data


# type of learnong: fully supervised or semi-supervised
supervised = False # if True, only use labeled data for training (fully supervised learning)

# if supervised is False, set labeled_batch_size to the desired batch size of labeled data (semi-supervised learning)
labeled_batch_size = 31 # if None, use standard random sampler (no constraint on
batch_size = 128 # total batch size (labeled + unlabeled) for semi-supervised learning, or batch size for fully supervised learning

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





