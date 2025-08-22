
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