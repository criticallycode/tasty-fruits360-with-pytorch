import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Set main directory for use in getting training/testing data

main_dir = "C:/Users/Daniel/Downloads/Fruits/fruits-360/"

# Declare transforms for train and test data

train_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5)),
])

test_transforms = transforms.Compose([transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5)),])

# Set the directories for the training and testing data, create the datasets with the transforms

train_data = datasets.ImageFolder(os.path.join(main_dir, 'Train/'), transform=train_transforms)
test_data = datasets.ImageFolder(os.path.join(main_dir, 'Test/'), transform=test_transforms)

# Use the dataloaders to create iterable objects out of the datasets

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Set the number of classes
classes = 118

# Specify the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the Model to use for training

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(8*8*16, 300)
        self.fc2 = nn.Linear(300, classes)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

    # Can combine layers and activations if you want:
    # Ex. X = self.conv1_bn(self.conv1(X))

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv1_bn(X)
        X = F.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.conv2_bn(X)
        X = F.relu(X)
        X = self.pool(X)
        X = X.view(-1, 8*8*16)
        X = self.fc1(X)
        X = F.relu(X)
        X = F.dropout(X)
        X = self.fc2(X)

        return X

# Set number of training epochs and learning rate
# May also want to use learning rate scheduler

epochs = 10
learning_rate = 0.0001

# Declare the model, loss criterion, and chosen optimizer

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.0001)

# Declare variables to hold the training and testing loss
# Which will be updated over the course of the epoch

epoch_loss_train = []
epoch_loss_test = []

# For our chosen number of epochs, train and check performance on test set

for i in range(epochs):

    # Declare variables to hold metrics

    train_loss = 0.0
    train_total = 0
    train_correct = 0

    # For the features and labels in the train loader
    for x, y in train_loader:

        # Declare the inputs and labels, zero the gradients,
        # run the inputs through the model and calculate the loss
        inputs, labels = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Get the most likely prediction
        # Do backprop to calculate gradient and optimize
        _, prediction = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()

        # Update the values
        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (prediction == labels).sum().item()

    else:

        test_loss = 0.0
        test_total = 0
        test_correct = 0

        # Set to no_grad for purposes of evaluation

        with torch.no_grad():
            for test_images, test_targets in test_loader:
                images, targets = test_images.to(device), test_targets.to(device)
                test_outputs = model(images)
                difference = criterion(test_outputs, targets)
                test_loss += difference.item()
                _, test_pred = torch.max(test_outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (test_pred == targets).sum().item()

        epoch_loss_train.append(train_loss/len(train_loader))
        epoch_loss_test.append(test_loss/len(test_loader))

        print("---------")
        print("End Epoch")
        print('Training Accuracy: {:.2f}%'.format(100 * train_correct / train_total))
        print('Testing Accuracy: {:.2f}%'.format(100 * test_correct / test_total))

