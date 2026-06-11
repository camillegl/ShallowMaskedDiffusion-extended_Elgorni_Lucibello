
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import Subset
import random

# UniformIsingDataset, RandomFeaturesDataset, BinarizedMNIST

class UniformIsingDataset(Dataset):
    """
        UniformIsingDataset(L, num_samples)
    
    Dataset that returns vectors x in {-1, +1} of length L. Number of samples M=round(alpha*L).
    """

    def __init__(self, L: int, num_samples: int):
        self.L = L
        self.num_samples = num_samples
        rand_ints = torch.randint(0, 2, (self.num_samples, self.L))
        self.data = (rand_ints.to(torch.float32) * 2.0) - 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def to_dict(self):
        return {"data": self.data, "L": self.L, "num_samples": self.num_samples}
    
    @classmethod
    def from_dict(cls, d):
        obj = cls(d["L"], d["num_samples"])
        obj.data = d["data"]
        return obj


class RandomFeaturesDataset(Dataset):
    """
        RandomFeaturesDataset(n_visible, n_hidden, num_samples, act = torch.sign)

    Dataset that returns vectors x as x = act(z @ F.T) where z ~ N(0, I) and F is a random matrix.
    """
    
    def __init__(self, n_visible: int, n_hidden: int, num_samples: int, act = torch.sign):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.num_samples = num_samples
        self.act = act
        self.F = torch.randn((n_visible, n_hidden)) / n_hidden**0.5
        self.z = torch.randn((self.num_samples, self.n_hidden))
        self.data = self.act(self.z @ self.F.t())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
    
    def to_dict(self):
        return {"data": self.data, "n_visible": self.n_visible, "n_hidden": self.n_hidden, 
                "num_samples": self.num_samples, "F": self.F, "z": self.z, "act": self.act}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["n_visible"], d["n_hidden"], d["num_samples"], act=d["act"])
        obj.data = d["data"]
        obj.F = d["F"]
        obj.z = d["z"]
        return obj
    
class BinarizedMNIST(Dataset):
    """
        BinarizedMNIST(root, train=True)

    Dataset that returns binarized MNIST images.
    """
    
    def __init__(self, root: str = "datasets/", train: bool = True, 
                 num_samples: int = None):

        self.mnist = datasets.MNIST(root=root, train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Lambda(binarize)
                                    ]))
        if num_samples is None:
            num_samples = len(self.mnist)
        if num_samples < len(self.mnist):
            self.mnist = Subset(self.mnist, list(range(num_samples)))
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        return self.mnist[idx][0].view(-1)  # flatten the image to a vector

def binarize(x): # have to defined outside of class for serialization
    return 2 * (x > 0.5).float() - 1