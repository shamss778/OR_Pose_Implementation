import itertools
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler


class GaussianNoise(object):
    def __init__(self, mean=0., sigma=0.15, clip=True):
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.sigma + self.mean
        noisy_tensor = tensor + noise
        if self.clip:
            noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)
        return noisy_tensor

class myDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        # Gather class names from subdirectory names
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        # Map class name to integer label
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        # Build list of (image_path, label) tuples
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input):
        out1 = self.transform(input)
        out2 = self.transform(input)
        return out1, out2

def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, -1
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices. (unlabeled)
    During the epoch, the secondary indices are iterated through
    as many times as needed. (labeled)
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)