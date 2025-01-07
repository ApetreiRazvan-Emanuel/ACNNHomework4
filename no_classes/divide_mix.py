import os

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from dataloaders.dataloader_cifar import CifarLoader
from models.PreActResNet18 import PreActResNet18
from torch import nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from metrics_tracker import MetricsTracker


device = torch.device('cuda')
ROOT_DIR = "./../datasets"
NOISY_LABEL_FILE = os.path.join(ROOT_DIR, "CIFAR-100-noisy.npz")
EPOCHS = 200
LEARNING_RATE = 0.02
WARM_UP = 30
INITIAL_LEARNING_RATE = 0.02
NOISE_MODE = "asym"
BATCH_SIZE = 128
NUM_CLASSES = 100
P_THRESHOLD = 0.5
T = 0.5
ALPHA = 4
LAMBDA_U = 150


def create_model():
    return PreActResNet18().to(device)


def train(epoch, model1, model2, optimizer, labeled_trainloader, unlabeled_trainloader):
    model1.train()
    model2.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // BATCH_SIZE) + 1
    for batch_idx, (inputs_x1, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u1, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u1, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x1.size(0)

        labels_x = torch.zeros(batch_size, NUM_CLASSES).scatter_(1, labels_x.view(-1, 1), 1).to(device)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor).to(device)

        inputs_x1, inputs_x2, labels_x, w_x = inputs_x1.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u1, inputs_u2 = inputs_u1.to(device), inputs_u2.to(device)

        with torch.no_grad():
            outputs_u11 = model1(inputs_u1)
            outputs_u12 = model1(inputs_u2)
            outputs_u21 = model2(inputs_u1)
            outputs_u22 = model2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) +
                  torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) +
                  torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / T)

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            outputs_x1 = model1(inputs_x1)
            outputs_x2 = model1(inputs_x2)

            px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / T)

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        l = np.random.beta(ALPHA, ALPHA)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = model1(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, WARM_UP)

        prior = torch.ones(NUM_CLASSES) / NUM_CLASSES
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def test(epoch, model1, model2, test_loader, best_acc):
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs = outputs1 + outputs2

            # Calculate combined loss
            loss = CEloss(outputs, targets).item()  # Using CEloss for consistency
            total_loss += loss

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    test_loss = total_loss / len(test_loader)

    print(f"Test epoch: {epoch}, accuracy: {test_acc:.2f}, loss: {test_loss:.4f}")

    # Save the best models if accuracy is higher than 55%
    if test_acc > 55 and test_acc > best_acc:
        torch.save(model1.state_dict(), f"best_model1_{test_acc}_epoch{epoch}.pth")
        print(f"New best accuracy for Model 1: {test_acc:.2f}. Saved model.")
        torch.save(model2.state_dict(), f"best_model2_{test_acc}_epoch{epoch}.pth")
        print(f"New best accuracy for Model 2: {test_acc:.2f}. Saved model.")
        best_acc = test_acc

    return best_acc, test_acc, test_loss


def eval_train(model, all_loss, eval_loader):
    model.eval()
    losses = torch.zeros(50000)
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = CE(outputs, targets)

            # Store individual losses for GMM
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]

            # Calculate accuracy and total loss
            total_loss += loss.sum().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate train metrics
    train_acc = 100. * correct / total
    train_loss = total_loss / total

    # Normalize losses for GMM
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    # GMM fitting
    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    return prob, all_loss, train_acc, train_loss


def warmup(model: torch.nn.Module, optimizer, train_loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels, path) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = CEloss(outputs, labels)

        if NOISE_MODE == "asym":
            penalty = conf_penalty(outputs)
            loss -= penalty

        loss.backward()
        optimizer.step()

        # Calculate accuracy and accumulate loss
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print(f"Warmup batch {batch_idx}, loss {loss}")

    # Calculate final metrics
    train_acc = 100. * correct / total
    train_loss = total_loss / len(train_loader)

    return train_loss, train_acc


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return LAMBDA_U * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


@torch.inference_mode()
def inference():
    model1.eval()
    model2.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=True):
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs = outputs1 + outputs2

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


if __name__ == '__main__':
    transform_train = v2.Compose([
                        v2.RandomCrop(32, padding=4),
                        v2.RandomHorizontalFlip(),
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                        v2.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                      ])
    transform_test = v2.Compose([
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                        v2.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                     ])

    loader = CifarLoader(ROOT_DIR, BATCH_SIZE, 5, transform_train, transform_test, NOISY_LABEL_FILE)

    model1 = create_model()
    model2 = create_model()

    criterion = SemiLoss()

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=INITIAL_LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=INITIAL_LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()

    if NOISE_MODE == "asym":
        conf_penalty = NegEntropy()

    all_loss = [[], []]  # save the history of losses from two networks

    # Initialize best accuracies
    best_acc = 0.0

    tracker = MetricsTracker()
    tracker.start_training()

    for epoch in range(EPOCHS + 1):
        lr = LEARNING_RATE
        if epoch >= 100:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < WARM_UP:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            train_loss1, train_acc1 = warmup(model1, optimizer1, warmup_trainloader)
            print('\nWarmup Net2')
            train_loss2, train_acc2 = warmup(model2, optimizer2, warmup_trainloader)

            train_loss = (train_loss1 + train_loss2) / 2
            train_acc = (train_acc1 + train_acc2) / 2

        else:
            prob1, all_loss[0], train_acc1, train_loss1 = eval_train(model1, all_loss[0], eval_loader)
            prob2, all_loss[1], train_acc2, train_loss2 = eval_train(model2, all_loss[1], eval_loader)

            train_acc = (train_acc1 + train_acc2) / 2
            train_loss = (train_loss1 + train_loss2) / 2

            pred1 = (prob1 > P_THRESHOLD)
            pred2 = (prob2 > P_THRESHOLD)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, model1, model2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, model2, model1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

        best_acc, test_acc, test_loss = test(epoch, model1, model2, test_loader, best_acc)
        tracker.update_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc
        )

    test_loader = loader.run('test')
    tracker.measure_inference_time(model1, model2, test_loader, device)
    tracker.end_training()

    data = {
        "ID": [],
        "target": []
    }

    for i, label in enumerate(inference()):
        data["ID"].append(i)
        data["target"].append(label)

    df = pd.DataFrame(data)
    df.to_csv("/kaggle/working/submission.csv", index=False)
