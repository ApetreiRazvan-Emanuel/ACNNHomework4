import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
import os

# Path to the dataset and noise file
root = "./datasets"
noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")

# Load noise file
if not os.path.isfile(noisy_label_file):
    raise FileNotFoundError(f"{noisy_label_file} does not exist!")

noise_file = np.load(noisy_label_file)

# Extract clean and noisy labels
clean_labels = noise_file["clean_label"]
noisy_labels = noise_file["noisy_label"]

# Check if clean and noisy labels match
mismatch_indices = np.where(clean_labels != noisy_labels)[0]
num_noisy = len(mismatch_indices)

print(f"Total samples: {len(clean_labels)}")
print(f"Noisy samples: {num_noisy}")
print(f"Noise ratio: {num_noisy / len(clean_labels):.2%}")

# Visualize some noisy samples
dataset = CIFAR100(root=root, train=True, transform=ToTensor(), download=False)
for i, idx in enumerate(mismatch_indices[:5]):  # Display first 5 noisy samples
    image, _ = dataset[idx]
    clean_label = clean_labels[idx]
    noisy_label = noisy_labels[idx]

    plt.figure(figsize=(2, 2))
    plt.imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    plt.title(f"Clean: {clean_label}, Noisy: {noisy_label}")
    plt.axis("off")
    plt.show()
