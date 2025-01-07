import os
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torch import nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dataloaders.dataloader_cifar import CifarLoader
from models.PreActResNet18 import PreActResNet18
from configuration import Config


class CifarTrainer:
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.device = torch.device(f'cuda:{self.config.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.model1 = self.create_model()
        self.model2 = self.create_model()

        self.optimizer1 = SGD(self.model1.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.optimizer2 = SGD(self.model2.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.CEloss = nn.CrossEntropyLoss(reduction="none")
        self.semi_loss = self.SemiLoss()
        self.conf_penalty = self.NegEntropy() if self.config.noise_mode == "asym" else None
        self.all_loss = [[], []]

        self.loader = CifarLoader(
            root_dir=self.config.root_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            transform_train=self.get_transform("train"),
            transform_test=self.get_transform("test"),
            noisy_label_file=os.path.join(self.config.root_dir, "CIFAR-100-noisy.npz"),
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir="tensorboard_logs")

    def create_model(self):
        return PreActResNet18(num_classes=self.config.num_classes).to(self.device)

    def get_transform(self, mode: str):
        if mode == "train":
            return v2.Compose([
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
            ])
        elif mode == "test":
            return v2.Compose([
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
            ])

    class SemiLoss:
        def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
            probs_u = torch.softmax(outputs_u, dim=1)
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
            Lu = torch.mean((probs_u - targets_u) ** 2)
            return Lx, Lu, self.linear_rampup(epoch, warm_up)

        @staticmethod
        def linear_rampup(current, warm_up, rampup_length=16):
            current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
            return current * 25.0

    class NegEntropy:
        def __call__(self, outputs):
            probs = torch.softmax(outputs, dim=1)
            return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def warmup(self, model, optimizer, train_loader, epoch):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.CEloss(outputs, labels).mean()
            if self.config.noise_mode == "asym":
                penalty = self.conf_penalty(outputs)
                loss -= penalty
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.writer.add_scalar(f"Warmup Loss/Epoch", avg_loss, epoch)
        print(f"Warmup Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    def train(self, epoch, model1, model2, optimizer, labeled_loader, unlabeled_loader):
        model1.train()
        model2.eval()
        total_labeled_loss = 0
        total_unlabeled_loss = 0
        unlabeled_iter = iter(unlabeled_loader)

        for batch_idx, (inputs_x1, inputs_x2, labels_x, w_x) in enumerate(labeled_loader):
            try:
                inputs_u1, inputs_u2 = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                inputs_u1, inputs_u2 = next(unlabeled_iter)

            batch_size = inputs_x1.size(0)
            labels_x = torch.zeros(batch_size, self.config.num_classes).scatter_(1, labels_x.view(-1, 1), 1).to(self.device)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor).to(self.device)

            inputs_x1, inputs_x2 = inputs_x1.to(self.device), inputs_x2.to(self.device)
            inputs_u1, inputs_u2 = inputs_u1.to(self.device), inputs_u2.to(self.device)

            # Compute pseudo-labels and refine labels
            with torch.no_grad():
                outputs_u11, outputs_u12 = model1(inputs_u1), model1(inputs_u2)
                outputs_u21, outputs_u22 = model2(inputs_u1), model2(inputs_u2)
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                      torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                targets_u = (pu ** (1 / self.config.temperature)) / pu.sum(dim=1, keepdim=True)

                outputs_x1, outputs_x2 = model1(inputs_x1), model1(inputs_x2)
                px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                targets_x = (px ** (1 / self.config.temperature)) / px.sum(dim=1, keepdim=True)

            # MixMatch
            l = max(np.random.beta(self.config.alpha, self.config.alpha), 1 - np.random.beta(self.config.alpha, self.config.alpha))
            all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
            idx = torch.randperm(all_inputs.size(0))
            mixed_input = l * all_inputs + (1 - l) * all_inputs[idx]
            mixed_target = l * all_targets + (1 - l) * all_targets[idx]

            # Forward pass
            logits = model1(mixed_input)
            logits_x, logits_u = logits[:batch_size * 2], logits[batch_size * 2:]
            Lx, Lu, lamb = self.semi_loss(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:], epoch, self.config.warm_up)

            prior = torch.ones(self.config.num_classes).to(self.device) / self.config.num_classes
            penalty = torch.sum(prior * torch.log(prior / logits.mean(0)))

            loss = Lx + lamb * Lu + penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_labeled_loss += Lx.item()
            total_unlabeled_loss += Lu.item()

        avg_labeled_loss = total_labeled_loss / len(labeled_loader)
        avg_unlabeled_loss = total_unlabeled_loss / len(labeled_loader)
        self.writer.add_scalar("Labeled Loss/Epoch", avg_labeled_loss, epoch)
        self.writer.add_scalar("Unlabeled Loss/Epoch", avg_unlabeled_loss, epoch)

    def test(self, model1, model2, test_loader, epoch):
        model1.eval()
        model2.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs1, outputs2 = model1(inputs), model2(inputs)
                predicted = (outputs1 + outputs2).argmax(dim=1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        acc = 100.0 * correct / total
        self.writer.add_scalar("Test Accuracy/Epoch", acc, epoch)
        print(f"Test Accuracy: {acc:.2f}%")

    @torch.inference_mode()
    def inference(self, test_loader):
        self.model1.eval()
        self.model2.eval()
        labels = []
        for inputs, _ in test_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            outputs1 = self.model1(inputs)
            outputs2 = self.model2(inputs)
            outputs = outputs1 + outputs2
            predicted = outputs.argmax(dim=1).tolist()
            labels.extend(predicted)
        return labels

    def run(self):
        for epoch in range(self.config.epochs + 1):
            lr = self.config.learning_rate
            if epoch >= 150:
                lr /= 10
            for param_group in self.optimizer1.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer2.param_groups:
                param_group["lr"] = lr

            test_loader = self.loader.run("test")
            eval_loader = self.loader.run("eval_train")

            if epoch < self.config.warm_up:
                warmup_loader = self.loader.run("warmup")
                print(f"Warmup Epoch {epoch} - Model 1")
                self.warmup(self.model1, self.optimizer1, warmup_loader, epoch)
                print(f"Warmup Epoch {epoch} - Model 2")
                self.warmup(self.model2, self.optimizer2, warmup_loader, epoch)
            else:
                prob1 = self.eval_train(self.model1, eval_loader)
                prob2 = self.eval_train(self.model2, eval_loader)
                pred1 = prob1 > self.config.p_threshold
                pred2 = prob2 > self.config.p_threshold

                labeled_loader, unlabeled_loader = self.loader.run("train", pred=pred2)
                self.train(epoch, self.model1, self.model2, self.optimizer1, labeled_loader, unlabeled_loader)

                labeled_loader, unlabeled_loader = self.loader.run("train", pred=pred1)
                self.train(epoch, self.model2, self.model1, self.optimizer2, labeled_loader, unlabeled_loader)

            self.test(self.model1, self.model2, test_loader, epoch)

        labels = self.inference(test_loader)
        data = {"ID": list(range(len(labels))), "target": labels}
        df = pd.DataFrame(data)
        df.to_csv("submission.csv", index=False)
