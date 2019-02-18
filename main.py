import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from CNNC import CNNC
from FocalLoss import FocalLoss
from MyDataSet import MyDataSet

# Configuration

cuda = torch.cuda.is_available()

root_dir = 'assign1'
classes = 2

batch_size = 64
num_workers = batch_size // 8
validation_split = .2
random_seed = 42

epochs = 80
learning_rate = 1e-2
step_size = 40
gamma = 0.1

# Set device

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

# Dataset

train_dataset = datasets.ImageFolder(root=root_dir + '/trainset', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]))
test_dataset = MyDataSet(root=root_dir + '/testset/test', transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
]))

# Create validation split with shuffle enabled

indices = list(range(len(train_dataset)))
split = int(np.floor(validation_split * len(train_dataset)))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Data loader

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Set the model

model = CNNC(num_classes=classes)

if cuda:
    model = model.cuda()

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Print total parameters of the model

total_params = sum(p.numel() for p in model.parameters())
total_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
tqdm.write('Number of parameters : {}'.format(total_params))
tqdm.write('Number of trainable parameters : {}'.format(total_grad_params))


def train(_epoch):
    scheduler.step()
    model.train()
    num_images = 0
    running_loss = 0.
    running_accuracy = 0.
    monitor = tqdm(train_loader, desc='Training')
    for i, (train_images, train_labels) in enumerate(monitor):
        train_images, train_labels = train_images.to(device), train_labels.to(device)

        outputs = model(train_images)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, train_labels)

        num_images += train_images.size(0)
        running_loss += loss.item() * train_images.size(0)
        running_accuracy += (preds == train_labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        monitor.set_postfix(epoch=_epoch, loss=running_loss / num_images, accuracy=running_accuracy / num_images)

    epoch_loss = running_loss / num_images
    epoch_accuracy = running_accuracy / num_images

    return epoch_loss, epoch_accuracy


def valid(_epoch):
    model.eval()
    with torch.no_grad():
        num_images = 0
        running_loss = 0.
        running_accuracy = 0.
        monitor = tqdm(valid_loader, desc='Validating')
        for i, (valid_images, valid_labels) in enumerate(monitor):
            valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)

            outputs = model(valid_images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, valid_labels)

            num_images += valid_images.size(0)
            running_loss += loss.item() * valid_images.size(0)
            running_accuracy += (preds == valid_labels).sum().item()

            monitor.set_postfix(epoch=_epoch, loss=running_loss / num_images, accuracy=running_accuracy / num_images)

        epoch_loss = running_loss / num_images
        epoch_accuracy = running_accuracy / num_images

    return epoch_loss, epoch_accuracy


def test():
    model.eval()
    with torch.no_grad():
        num_images = 0
        columns = ['id', 'label']
        all_preds = pd.DataFrame(columns=columns)
        monitor = tqdm(test_loader, desc='Testing')
        for i, (test_images, _) in enumerate(monitor):
            test_images = test_images.to(device)

            outputs = model(test_images)
            _, preds = torch.max(outputs.data, 1)

            for j, (pred) in enumerate(preds):
                all_preds = all_preds.append({'id': num_images + j + 1, 'label': 'Cat' if pred == 0 else 'Dog'},
                                             ignore_index=True)

            num_images += test_images.size(0)

    return all_preds


def log_results(file_name, losses, accuracies):
    file = open(file_name, 'w+')
    for x, y in zip(losses, accuracies):
        file.write('{:.3f},'.format(x))
        file.write('{:.3f}'.format(y))
        file.write('\n')
    file.close()


if __name__ == '__main__':

    # Lists to save loss and accuracy for each epoch

    train_loss_list = []
    train_acc_list = []

    valid_loss_list = []
    valid_acc_list = []

    best_acc = 0.

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # Valid
        valid_loss, valid_acc = valid(epoch)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        # Save the best model parameters

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'CNN_C.pt')

        tqdm.write('')

    log_results('CNN_C_TRAIN' + '.csv', train_loss_list, train_acc_list)
    log_results('CNN_C_VALID' + '.csv', valid_loss_list, valid_acc_list)

    # Load the best model parameters

    model.load_state_dict(torch.load('CNN_C.pt'))

    # Make inference

    submission = test()
    submission.to_csv('submissionC.csv', index=False)
