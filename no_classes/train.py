import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from typing import Optional, Callable
import os
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from models.PreActResNet18 import PreActResNet18
import torch.amp as amp
import time
from PIL import Image


device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True  # Disable for CPU, it is slower!
scaler = GradScaler("cuda", enabled=enable_half)


class Cifar100NoisyFine(Dataset):
    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool):
        cifar100 = CIFAR100(
            root=root, train=train, transform=None, download=download
        )
        self.data = cifar100.data
        self.targets = cifar100.targets
        self.transform = transform

        if train:
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
        return len(self.targets)

    def __getitem__(self, i: int):
        img, target = self.data[i], self.targets[i]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

train_transforms = v2.Compose([
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                        v2.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ])

test_transforms = v2.Compose([
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                        v2.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ])

train_set = Cifar100NoisyFine('./../datasets', train=True, transform=train_transforms, download=False)
test_set = Cifar100NoisyFine('./../datasets', train=False, transform=test_transforms, download=False)

train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = PreActResNet18()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

metrics = {"epoch": [], "train_acc": [], "train_loss": [], "test_acc": [], "test_loss": [], "gpu_memory": []}


def train():
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with amp.autocast('cuda', enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

    return 100.0 * total_correct / total_samples, total_loss / len(train_loader)


@torch.no_grad()
def val():
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with amp.autocast('cuda', enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

    return 100.0 * total_correct / total_samples, total_loss / len(test_loader)


@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


@torch.no_grad()
def measure_final_inference_time():
    model.eval()
    start_time = time.time()

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with amp.autocast('cuda', enabled=enable_half):
            model(inputs)

    return time.time() - start_time


if __name__ == "__main__":
    best_val_acc = 0.0
    epochs = 200

    start_training_time = time.time()

    for epoch in range(epochs):
        train_acc, train_loss = train()
        val_acc, val_loss = val()

        gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        metrics["epoch"].append(epoch)
        metrics["train_acc"].append(train_acc)
        metrics["train_loss"].append(train_loss)
        metrics["test_acc"].append(val_acc)
        metrics["test_loss"].append(val_loss)
        metrics["gpu_memory"].append(gpu_memory)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, GPU Memory: {gpu_memory:.2f} MB")

    total_training_time = time.time() - start_training_time
    final_inference_time = measure_final_inference_time()

    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Final Inference Time: {final_inference_time:.2f} seconds")

    df = pd.DataFrame(metrics)
    df.to_csv("metrics_baseline.csv", index=False)
    with open("info_baseline.txt", "w") as f:
        f.write(f"Total Training Time: {total_training_time:.2f} seconds")
        f.write(f"Final Inference Time: {final_inference_time:.2f} seconds")

    data = {
        "ID": [],
        "target": []
    }

    for i, label in enumerate(inference()):
        data["ID"].append(i)
        data["target"].append(label)

    df = pd.DataFrame(data)
    df.to_csv("/kaggle/working/submission_baseline.csv", index=False)

