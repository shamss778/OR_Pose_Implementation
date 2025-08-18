from .data import TransformTwice, GaussianNoise, myDataset
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

train_loader = DataLoader(
    train_dataset,
    batch_size= config.batch_size,
    shuffle=True,
    num_workers= config.num_workers,
    pin_memory= config.pin_memory,
)

test_loader = DataLoader(
    test_dataset,
    batch_size= config.batch_size,
    shuffle=False,
    num_workers= config.num_workers,
    pin_memory= config.pin_memory,
)

