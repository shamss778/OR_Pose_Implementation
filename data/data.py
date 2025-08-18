import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch

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
    

