from .data import TransformTwice, GaussianNoise, myDataset, relabel_dataset, TwoStreamBatchSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchvision import transforms
from .. import config
from torch.utils.data import DataLoader



train_transform = TransformTwice(transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomAffine(degrees=0, translate=(2/32, 2/32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    GaussianNoise(mean=0. , sigma=0.15, clip=True), 
    transforms.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
]))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std= (1, 1, 1))
])

train_dataset = myDataset(root_dir=config.traindir, transform=train_transform)
test_dataset = myDataset(root_dir=config.testdir, transform=test_transform)

if config.labels:
    with open(config.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
    labeled_idxs, unlabeled_idxs = relabel_dataset(train_dataset, labels)

if config.exclude_unlabeled:
    sampler = SubsetRandomSampler(labeled_idxs)
    batch_sampler = BatchSampler(sampler, config.batch_size, drop_last=True)
elif config.labeled_batch_size:
    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, config.batch_size, config.labeled_batch_size)
else:
    assert False, "labeled batch size {}".format(config.labeled_batch_size)

train_loader = DataLoader(
    train_dataset,
    batch_sampler= batch_sampler,
    num_workers= config.num_workers,
    pin_memory= config.pin_memory,
)

test_loader = DataLoader(
    test_dataset,
    batch_size= config.batch_size,
    shuffle=False,
    num_workers= config.num_workers,
    pin_memory= config.pin_memory,
    drop_last=False
)



