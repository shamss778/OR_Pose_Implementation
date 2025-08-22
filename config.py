
traindir = 'Mean_Teacher_Method/data/dataset/cifar10/train'
batch_size = 64
num_workers = 4
pin_memory = False
testdir = 'Mean_Teacher_Method/data/dataset/cifar10/test'
labels = "Mean_Teacher_Method/data/dataset/labels/cifar10"
consistency = 1.0
consistency_rampup = 100
lr = 0.002
# lr_rampup = 0
# initial_lr = 0.002
# lr_rampdown_epochs = 200
logit_distance_cost = 1
supervised = False
ema_decay = 0.99
epochs = 200
weight_decay = 0.0005
