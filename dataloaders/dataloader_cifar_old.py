from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from typing import Optional, Callable
import os
import numpy as np
from PIL import Image


class Cifar100NoisyFine(Dataset):
    def __init__(self, root: str, mode: str, transform: Optional[Callable], probabilities: list | None =None):
        cifar100 = CIFAR100(
            root=root, train=mode != 'test', transform=None, download=False
        )
        self.data = cifar100.data
        self.targets = cifar100.targets
        self.transform = transform
        self.probabilities = probabilities
        self.mode = mode

        if mode != 'test':
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} needs {noisy_label_file} to be used!"
                )
            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], self.targets):
                raise RuntimeError("Clean labels do not match!")
            self.targets = noise_file["noisy_label"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        if self.mode == 'labeled':
            img, target, prob = self.data[i], self.targets[i], self.probabilities[i]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.data[i]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img, target = self.data[i], self.targets[i]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, i
        elif self.mode == 'test':
            img, target = self.data[i], self.targets[i]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
