import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100


class CachedCIFAR100:
    def __init__(self, root: str, train: bool, transform=None, download: bool = False, noisy_label_file: str = None):
        self.transform = transform
        self.data, self.clean_labels = self._load_and_cache_data(root, train, download)

        # Use noisy labels if provided
        if train and noisy_label_file:
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(f"Noisy label file not found: {noisy_label_file}")
            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], self.clean_labels):
                raise RuntimeError("Mismatch between clean labels and dataset targets.")
            self.labels = noise_file["noisy_label"]
        else:
            self.labels = self.clean_labels  # Use clean labels for test mode

    def _load_and_cache_data(self, root: str, train: bool, download: bool):
        cifar100 = CIFAR100(root=root, train=train, download=download, transform=None)
        return cifar100.data, cifar100.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


class Cifar100Dataset(Dataset):

    def __init__(self, cached_dataset, mode: str, pred=None, probability=None):
        self.cached_dataset = cached_dataset
        self.mode = mode
        self.pred = pred
        self.probability = probability

        if mode in ['labeled', 'unlabeled']:
            assert pred is not None, "Prediction indices required for labeled/unlabeled mode."
            if mode == 'labeled':
                self.indices = pred.nonzero()[0]
            elif mode == 'unlabeled':
                self.indices = (1 - pred).nonzero()[0]
        else:
            self.indices = np.arange(len(self.cached_dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]
        img1, target = self.cached_dataset[real_index]

        if self.mode == 'labeled':
            prob = self.probability[index]
            img2, _ = self.cached_dataset[real_index]
            return img1, img2, target, prob
        if self.mode == 'unlabeled':
            img2, _ = self.cached_dataset[real_index]
            return img1, img2
        if self.mode == 'all':
            return img1, target, index  # Sper ca nu e index aicea
        return img1, target


class CifarLoader:

    def __init__(self, root_dir, batch_size, num_workers, transform_train, transform_test, noisy_label_file):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.noisy_label_file = noisy_label_file

        # Initialize cached datasets
        self.cached_train = CachedCIFAR100(
            root=root_dir,
            train=True,
            transform=transform_train,
            download=True,
            noisy_label_file=noisy_label_file
        )
        self.cached_test = CachedCIFAR100(
            root=root_dir,
            train=False,
            transform=transform_test,
            download=True
        )

    def run(self, mode, pred=None, prob=None):
        if mode == 'warmup':
            dataset = Cifar100Dataset(cached_dataset=self.cached_train, mode='all')
            return DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=True, num_workers=self.num_workers)

        elif mode == 'train':
            labeled_dataset = Cifar100Dataset(cached_dataset=self.cached_train, mode='labeled', pred=pred, probability=prob)
            unlabeled_dataset = Cifar100Dataset(cached_dataset=self.cached_train, mode='unlabeled', pred=pred)
            labeled_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader

        elif mode == 'test':
            dataset = Cifar100Dataset(cached_dataset=self.cached_test, mode='test')
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        elif mode == 'eval_train':
            dataset = Cifar100Dataset(cached_dataset=self.cached_train, mode='all')
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


