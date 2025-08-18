import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import GaussianNoise

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input):
        out1 = self.transform(input)
        out2 = self.transform(input)
        return out1, out2
    
train_transform = TransformTwice(transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomAffine(degrees=0, translate=(2/32, 2/32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.GaussianNoise(mean=0. , sigma=0.15, clip=True), 
    transforms.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
]))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std= (1, 1, 1))
])

