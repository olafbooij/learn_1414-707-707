from __future__ import print_function
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
from external.focal_loss_pytorch import focalloss


#def resnet_mnist():
#    resnet = resnet18(num_classes=10)
#    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
#    return resnet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def determine_class_weights(train_loader):
    number_per_target = np.zeros(10)
    # print(len(train_loader.dataset))
    for _, targets in train_loader:
        number_per_target += np.bincount(targets, minlength=10)
    weights =  number_per_target.sum() / number_per_target
    # weights /= weights.mean() # needed ?
    return weights


def train(args, model, device, train_loader, optimizer, weights, epoch, use_focal_loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if use_focal_loss:
            loss_f = focalloss.FocalLoss(gamma=2.0, alpha=torch.Tensor(weights))
            loss = loss_f(output, target)
        else:
            loss = F.cross_entropy(output, target, weight=torch.Tensor(weights))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_labels(filename):
    with open(filename) as labels_file:
      labels_raw = json.load(labels_file)
    labels_raw= [label for label in labels_raw if label['newLabel'] not in "psud-"] # remove non-integer input
    labels = {label['index']: label['newLabel'] for label in labels_raw} # remove duplicates, keeping last
    return labels


def relabel_set(dataset, filename, use_original_labels):
    labels = load_labels(filename)
    dataset.data = [dataset.data[index] for index in labels]
    dataset.targets = [
      dataset.targets[index] if use_original_labels
      else labels[index]
      for index in labels
    ]
    return dataset


def unbalance_set(dataset, ratios):
    unbalanced_data_targets = [
        data_target
        for data_target in zip(dataset.data, dataset.targets)
        if random.random() < ratios[data_target[1]]
    ]
    dataset.data, dataset.targets = zip(*unbalanced_data_targets)
    return dataset


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--label_file', type=str, default='mnist_labels/test2894.json', metavar='FILE',
                        help='filename with new mnist labels (default: mnist_labels/test2894.json)')
    parser.add_argument('--use_original_labels', action='store_true',
                        help='use original mnist labels instead of the provided labels (but do pick trainset according to new labels)')
    parser.add_argument('--unbalance_dataset', action='store_true',
                        help='make dataset unbalanced by dropping samples with certain labels (using hardcoded ratios)')
    parser.add_argument('--do_class_weighting', action='store_true',
                        help='use class weighting during training (relevant for unbalanced datasets)')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='use focal loss instead of cross entropy loss')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size,
                    'shuffle'   : True
                   }
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset_test = datasets.MNIST('../data', train=False,
                       transform=transform)
    relabel_set(dataset_train, args.label_file, args.use_original_labels)
    if args.unbalance_dataset:
      unbalance_set(dataset_train, [.1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    weights = determine_class_weights(train_loader)
    if not args.do_class_weighting:
        weights = np.ones(weights.size) / float(weights.size)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, weights, epoch, args.use_focal_loss)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
